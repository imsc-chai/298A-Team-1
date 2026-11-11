"""Run M3 over tasks and write one Output-contract JSON per task.

    python -m model3.run --split dev --run 0 --results-dir /content/drive/MyDrive/m3_results
    python -m model3.run --personas persona_01_maya --tasks eval_03_medical_privacy_boundary

Resumable: a task whose result file already exists is skipped (Colab sessions disconnect).
--rerun-errors also re-runs tasks whose saved status is ERROR (e.g. Ollama died mid-run).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from common.contract import ResultBuilder, TaskResult, TaskStatus, build_settings, result_path, write_result
from common.environment import build_retriever, open_task_env
from common.hardware import detect_hardware
from common.llm_client import LLMError, LLMParseError, OllamaClient
from common.retriever import DEFAULT_K, EMBED_MODEL, HybridRetriever
from common.schemas import AgentTask
from common.tasks import load_agent_tasks, personas_for_split
from model3.m3_hierarchical import MODEL_NAME, M3HierarchicalPlanner

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
# Two transport failures in a row almost always mean the Ollama server is gone; stop instead
# of stamping ERROR on every remaining task.
MAX_CONSECUTIVE_LLM_ERRORS = 2


def run_task(task: AgentTask, run: int, llm: OllamaClient, retriever: HybridRetriever, settings: dict, k: int) -> tuple[TaskResult, Exception | None]:
    llm.telemetry.reset()
    builder = ResultBuilder(MODEL_NAME, task.persona_id, task.task_id, run, task.reference_time, settings,
                            known_record_ids={r.id for r in retriever.records})
    start = time.perf_counter()
    error: Exception | None = None
    with open_task_env(task) as env:
        try:
            status, final_response, debug = M3HierarchicalPlanner(llm, retriever, k).run(env, builder)
        except Exception as exc:  # recorded in the result file as ERROR, never swallowed
            error = exc
            status, final_response = TaskStatus.ERROR, None
            debug = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    debug["wall_ms"] = int((time.perf_counter() - start) * 1000)
    debug["llm_call_log"] = [c.__dict__ for c in llm.telemetry.calls]
    return builder.finish(status, llm.telemetry.summary(), final_response, debug), error


def should_skip(path: Path, rerun_errors: bool) -> bool:
    if not path.is_file():
        return False
    if rerun_errors and json.loads(path.read_text()).get("status") == TaskStatus.ERROR.value:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--personas", nargs="+", help="overrides --split")
    ap.add_argument("--tasks", nargs="+", help="task ids (only with a single persona)")
    ap.add_argument("--run", type=int, default=0, help="run index -> run_XX folder")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--rerun-errors", action="store_true")
    args = ap.parse_args(argv)

    personas = args.personas or personas_for_split(args.split)
    if args.tasks and len(personas) != 1:
        ap.error("--tasks needs exactly one persona")
    if args.split == "test" and not args.personas:
        print("WARNING: running TEST personas. Only do this after prompts are frozen (CLAUDE.md rule 3).")

    llm = OllamaClient()
    try:
        llm.check_available()
    except LLMError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    settings = build_settings(llm.settings, EMBED_MODEL, args.k, detect_hardware())
    print(f"M3 run_{args.run:02d} -> {args.results_dir} | {llm.settings.model} @ {llm.settings.host} | gpu={settings['hardware'].get('gpu')}")

    consecutive_llm_errors = 0
    counts: dict[str, int] = {}
    for persona in personas:
        tasks = load_agent_tasks(persona, args.tasks)
        pending = [t for t in tasks if not should_skip(result_path(args.results_dir, args.run, persona, t.task_id), args.rerun_errors)]
        print(f"{persona}: {len(tasks)} tasks, {len(tasks) - len(pending)} already done")
        if not pending:
            continue
        retriever = build_retriever(persona)
        for task in pending:
            print(f"  -> {task.task_id} ...", flush=True)
            result, error = run_task(task, args.run, llm, retriever, settings, args.k)
            path = write_result(args.results_dir, result)
            counts[result.status.value] = counts.get(result.status.value, 0) + 1
            t = result.telemetry
            print(f"     {result.status.value} | {len(result.steps)} steps | {t.llm_calls} LLM calls | "
                  f"{t.latency_ms / 1000:.1f}s LLM | {path}")
            if error is not None:
                print(f"     error: {type(error).__name__}: {error}", file=sys.stderr)
            if isinstance(error, LLMError) and not isinstance(error, LLMParseError):
                consecutive_llm_errors += 1
                if consecutive_llm_errors >= MAX_CONSECUTIVE_LLM_ERRORS:
                    print("ABORT: Ollama keeps failing. Restart it and re-run with --rerun-errors.", file=sys.stderr)
                    return 3
            else:
                consecutive_llm_errors = 0
    print(f"Done. Status counts this invocation: {counts or 'nothing to run'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
