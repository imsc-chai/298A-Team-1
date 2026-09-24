#!/usr/bin/env python3
"""Run OpenJarvis PinchBench with tool auto-approve.

Upstream `jarvis bench skills` builds a non-interactive ToolExecutor, so
shell_exec is blocked (requires confirmation). Table 1-style agent eval
must actually execute tools. This wrapper does not modify OpenJarvis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OJ = ROOT / "third_party" / "OpenJarvis"
sys.path.insert(0, str(OJ / "src"))


def _patch_tool_auto_approve() -> None:
    from openjarvis.tools._stubs import ToolExecutor

    _orig_init = ToolExecutor.__init__

    def _init(self, tools, bus=None, *, interactive=False, confirm_callback=None, **kwargs):
        _orig_init(
            self,
            tools,
            bus,
            interactive=True,
            confirm_callback=(confirm_callback or (lambda _prompt: True)),
            **kwargs,
        )

    ToolExecutor.__init__ = _init  # type: ignore[method-assign]


def _patch_max_turns(max_turns: int) -> None:
    from openjarvis.evals.skill_benchmark import SkillBenchmarkRunner

    _orig = SkillBenchmarkRunner._build_backend_for_condition

    def _build(self, condition):
        backend = _orig(self, condition)
        try:
            backend._system._config.agent.max_turns = max_turns
        except Exception:
            pass
        return backend

    SkillBenchmarkRunner._build_backend_for_condition = _build  # type: ignore[method-assign]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="skills_on")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--engine", default="ollama")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tasks-file",
        default="",
        help="Text file of PinchBench IDs (Table 1 = 23 tasks).",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _patch_tool_auto_approve()
    _patch_max_turns(args.max_turns)

    from openjarvis.evals.core.types import RunConfig as _RunConfig
    from openjarvis.evals import skill_benchmark as skill_mod

    out_jsonl = str(out / "pinchbench.jsonl")
    record_ids = None
    if args.tasks_file:
        record_ids = [
            line.strip()
            for line in Path(args.tasks_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    # If we pin Table 1 IDs, do not also slice to N random tasks.
    max_samples = None if record_ids else (args.max_samples if args.max_samples > 0 else None)

    def _run_config(**kwargs):
        kwargs["max_samples"] = max_samples
        kwargs["output_path"] = out_jsonl
        kwargs["max_turns"] = args.max_turns
        if record_ids:
            kwargs["record_ids"] = record_ids
        return _RunConfig(**kwargs)

    skill_mod.RunConfig = _run_config  # type: ignore[misc]

    from openjarvis.evals.skill_benchmark import SkillBenchmarkConfig, SkillBenchmarkRunner

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = SkillBenchmarkConfig(
        model=args.model,
        engine=args.engine,
        seeds=seeds,
        max_samples=max_samples,
        output_dir=out,
    )
    runner = SkillBenchmarkRunner(cfg)
    result = runner.run_condition(args.condition)

    summary = {
        "condition": result.condition,
        "mean_pass_rate": result.mean_pass_rate,
        "stddev_pass_rate": result.stddev_pass_rate,
        "per_seed_pass_rate": result.per_seed_pass_rate,
        "total_tokens": result.total_tokens,
        "total_runtime_seconds": result.total_runtime_seconds,
        "published_target_percent": 87.9,
        "our_percent": round(result.mean_pass_rate * 100.0, 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
