# Baseline reproduction — OpenJarvis Table 1 (c)

Reproduce **87.9% PinchBench** for **OpenJarvis + Qwen3.5-9B** (Hermes Agent row, condition c).

This is the **published baseline**, not one of our four models.

## Layout

```
baseline-reproduction/
├── TARGET.yaml              locked numbers and protocol
├── configs/                 run settings
├── scripts/                 setup → smoke → full eval → summarize
├── env/                     secrets template (copy to .env, never commit)
├── paper/                   citation only
├── data/                    local task caches (gitignored when large)
├── results/
│   ├── published/           the paper’s 87.9 row
│   ├── runs/                our seed-level outputs
│   └── logs/                stdout/stderr
├── notes/                   what we tried if the number does not match
└── third_party/OpenJarvis/  upstream clone (not our code)
```

## Exact claim

| Item | Value |
|---|---|
| Paper | Saad-Falcon et al., 2026, arXiv:2605.17172 |
| Table | 1 |
| Row | Hermes Agent |
| Condition | (c) OPENJARVIS with Qwen3.5-9B |
| Benchmark | PinchBench (23 tasks) |
| Published | **87.9%** mean over 5 runs |
| Not this | Hermes + Qwen3.5-9B = 68.7% |

## Commands

From this folder:

```bash
# 1. Clone OpenJarvis
bash scripts/01_clone_openjarvis.sh

# 2. Check tools (uv, ollama, disk)
bash scripts/00_check_env.sh

# 3. Pull the 9B model (needs Ollama running)
bash scripts/02_pull_model.sh

# 4. Tiny run (2 tasks, 1 seed) — prove the harness works
bash scripts/03_smoke_eval.sh

# 5. Full Table 1 attempt (23 tasks × 5 seeds). Long. GPU/CPU heavy.
bash scripts/04_full_eval.sh

# 6. Write published vs ours
python3 scripts/05_summarize_results.py
```

## Honest scope

The paper also uses a 2-hour task timeout and GPT-5-mini as judge on some items. If our number is not 87.9, we log hardware, commit SHA, seeds, and the gap in `notes/reproduction_log.md`. A documented miss is a valid 298A result.
