from __future__ import annotations

import argparse
import json
import sys
import uuid
import copy
from dataclasses import replace
from pathlib import Path

from .adapter import DatasetAdapter
from .core import GuardPlanError, Settings, fingerprint, outside_source, read_json, write_json
from .evaluation import score, summarize
from .ingest import Ingestor
from .llm import CloudPlanner, LocalModel
from .planner import GuardPlan
from .privacy import Privacy


def parser():
    package = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="GuardPlan M6; source dataset is never modified")
    p.add_argument("command", choices=["inspect", "ingest", "run", "evaluate"])
    p.add_argument("--data-root", type=Path, default=package.parent.parent / "Project Data")
    p.add_argument("--workspace", type=Path, default=package / "artifacts")
    p.add_argument("--persona", default="persona_01_maya")
    p.add_argument("--reference-time", default="2026-11-07T14:00:00")
    p.add_argument("--session", default="default")
    p.add_argument("--task")
    p.add_argument("--query")
    p.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    p.add_argument("--quantization", default="Q4_K_M")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--candidate-pool", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--variant", choices=["full", "no-privacy", "no-gate", "no-checker", "no-state-filters"], default="full")
    p.add_argument("--cloud-host", help="Optional HTTPS Ollama-compatible endpoint for sanitized planning only")
    p.add_argument("--cloud-model", help="Explicit fixed cloud planner model/version")
    return p


def main():
    args = parser().parse_args()
    workspace = outside_source(args.data_root, args.workspace)
    before = fingerprint(args.data_root)
    workspace.mkdir(parents=True, exist_ok=True)
    settings = Settings(model=args.model, quantization=args.quantization, max_tokens=args.max_tokens,
                        temperature=args.temperature, candidate_pool=args.candidate_pool, max_turns=args.max_turns,
                        host=args.host, k=args.k, seed=args.seed, variant=args.variant)
    model = LocalModel(settings)
    if bool(args.cloud_host) != bool(args.cloud_model):
        raise GuardPlanError("Supply both --cloud-host and --cloud-model")
    planner_model = CloudPlanner(replace(settings, host=args.cloud_host, model=args.cloud_model)) if args.cloud_host else model
    adapter = DatasetAdapter(args.data_root, args.persona, workspace, args.reference_time, args.session)
    exit_code = 0
    try:
        if not 1 <= settings.k <= settings.candidate_pool or settings.max_tokens < 1 or settings.max_turns < 1:
            raise GuardPlanError("Require 1 <= k <= candidate_pool and positive max-tokens/max-turns")
        if args.command != "inspect":
            model.verify_model()
        if args.command == "inspect":
            result = {"adapter": type(adapter.connector).__name__, "persona": args.persona,
                      "calendar_events": len(adapter.search_events()), "messages_as_of_reference": len(adapter.search_messages()),
                      "local_files": adapter.list_files(), "tasks": [t["task_id"] for t in adapter.tasks()],
                      "source_read_only": True, "workspace": str(workspace)}
            print(json.dumps(result, indent=2))
        elif args.command == "ingest":
            records = Ingestor(adapter, model).records()
            print(json.dumps({"persona": args.persona, "memory_records": len(records), "workspace": str(workspace)}))
        elif args.command == "run":
            task = next((t for t in adapter.tasks() if t["task_id"] == args.task), None)
            if args.task and task is None:
                raise GuardPlanError("Unknown task ID")
            if task:
                adapter = DatasetAdapter(args.data_root, args.persona, workspace, task["reference_time"], args.session)
            query = args.query or (task["user_query"] if task else None)
            if not query:
                raise GuardPlanError("run requires --task or --query")
            result = GuardPlan(adapter, model, settings, planner_model).run(query, args.task or "interactive")
            path = workspace / "workflows" / (uuid.uuid4().hex + ".json")
            write_json(path, result)
            print(json.dumps({"status": result["status"], "steps": len(result["steps"]), "workflow": str(path),
                              "error": result.get("error")}, indent=2))
            exit_code = int(result["status"] != "COMPLETE")
        else:
            if args.runs < 1:
                raise GuardPlanError("runs must be positive")
            # Benchmark metadata is intentionally kept outside planner prompts.
            splits = read_json(args.data_root / "splits.json")
            run_id, rows = uuid.uuid4().hex[:12], []
            for persona in splits[args.split + "_personas"]:
                base = DatasetAdapter(args.data_root, persona, workspace, args.reference_time)
                tasks = base.tasks()
                for run in range(args.runs):
                    session = f"eval_{run_id}_{persona}_{run}"
                    # Chronological sequence preserves 02a -> 02b preference state.
                    for task in sorted(tasks, key=lambda t: t["reference_time"]):
                        settings.seed = args.seed
                        planner_model.settings.seed = settings.seed
                        current = DatasetAdapter(args.data_root, persona, workspace, task["reference_time"], session)
                        planner = GuardPlan(current, model, settings, planner_model)
                        profile_before = copy.deepcopy(current.get_profile())
                        events_before = copy.deepcopy(current.search_events())
                        workflow = planner.run(task["user_query"], task["task_id"])
                        workflow["run"] = run
                        evaluation_privacy = Privacy(current.privacy_policy(), current.get_profile())
                        row = score(task, workflow, profile_before, events_before, evaluation_privacy, current.contact_id)
                        row.update({"persona": persona, "run": run, "seed": settings.seed})
                        rows.append(row)
                        write_json(workspace / "evaluations" / run_id / persona / str(run) / (task["task_id"] + ".json"), workflow)
                        print(f"{persona} run={run} {task['task_id']} {workflow['status']}", flush=True)
            write_json(workspace / "evaluations" / run_id / "scores.json", rows)
            write_json(workspace / "evaluations" / run_id / "summary.json", summarize(rows))
    except (GuardPlanError, ImportError, OSError, ValueError) as exc:
        print(f"GuardPlan: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        after = fingerprint(args.data_root)
        if before != after:
            raise GuardPlanError("Source dataset integrity check failed")
        write_json(workspace / "source_integrity.json", {"unchanged": True, "files": before})
    return exit_code
