# Reproduction log — OpenJarvis Table 1 (c)

Date started: 2026-09-21

## Target
87.9% PinchBench, OpenJarvis + Qwen3.5-9B, Hermes Agent row, condition (c).

## Runs

| Date | Commit SHA | Hardware | Seeds | n tasks | Our mean | Published | Gap (pp) | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09-21 | ef00570 | Mac | 42 | 3 (interrupted) | ~6% (0/3 correct) | 87.9 | large | First smoke via `bench skills` blocked `shell_exec` (no confirm callback). Partial jsonl in results/pinchbench_qwen3.5-9b.jsonl. Wrapper added to auto-approve tools for eval. |

## 2026-09-21 13:27 “smoke” accidentally ran the FULL PinchBench (~147 tasks)
Cause: EvalRunner reloads the dataset with RunConfig.max_samples, which the skills bench does not set, so the `--max-samples 2` flag was ignored.
Stopped after **36/147** tasks (~2.4h). Saved to `notes/partial_runs/20260921_132713_36tasks.jsonl`.
Mean score so far ~0.40 (16/36 ≥ 0.5). Not comparable to the paper’s 87.9% yet (different task count, incomplete run).
Fix in `scripts/run_pinchbench.py`: inject max_samples + output_path into RunConfig.

## 2026-09-21 first smoke — failed harness
`jarvis bench skills` blocked `shell_exec` (confirmation required, eval is non-interactive).
Scores in `results/pinchbench_qwen3.5-9b.jsonl`: 0.192 / 0.0 / 0.0. Not a paper result.
Fix: `scripts/run_pinchbench.py` auto-approves tools and uses 40 max turns.
`jarvis bench skills` blocked `shell_exec` (confirmation required, eval is non-interactive).
Scores in `results/pinchbench_qwen3.5-9b.jsonl`: 0.192 / 0.0 / 0.0. Not a paper result.
Fix: `scripts/run_pinchbench.py` auto-approves tools and uses 40 max turns.

## If we cannot hit 87.9
Record: model tag, quantization, Ollama version, judge on/off, timeout, and which tasks failed.
