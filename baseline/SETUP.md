# Local baseline: Table 1, condition (c)

Saad-Falcon et al. (2026), Table 1, Hermes Agent block, condition (c): OpenJarvis with Qwen3.5-9B on PinchBench. The paper reports 87.9% against 93.5% for the default cloud setup. This run is the course baseline. Conditions (a) and (b) are out of scope.

The official number is the mean of the per-task scores (0 to 1), not the pass rate OpenJarvis prints as "Accuracy".

## Result

Five runs of the same configuration, seed 42:

| Run | Mean task score |
| --- | --- |
| run_01 | 72.2% |
| run_02 | 72.0% |
| run_03 | 65.7% |
| run_04 | 73.2% |
| run_05 | 74.5% |

**Mean 71.5%, standard deviation 3.4** (sample standard deviation). Unrounded values are in `results/summary_5runs.json`.

`results/test_run_judge2048` is an earlier run with the judge reply limit still at 2048. It is kept as evidence and is not part of the mean.

## What was run

- OpenJarvis v1.0.0, commit `e97088f199cf86ea5f78de921772357d1f0d2cec` (2026-05-16). The OpenJarvis source was not edited.
- PinchBench snapshot `0c419e45aac78f6bdeadf5ac83eca9aa8cd14b49` (2026-03-24): `task_00_sanity` through `task_22_second_brain`, 23 tasks. The current PinchBench repo has 147 tasks and was not used.
- Agent `native_openhands`, the 11 tools in the published `pinchbench-qwen-9b.toml`, temperature 0.6, max tokens 8192, one worker, seed 42. Seed only shuffles task order. It is not passed to the model.
- Judge `gpt-5-mini-2025-08-07` at temperature 0, as in Section 4.1. The published OpenJarvis config uses Claude Opus. This run follows the paper.
- Model `qwen3.5:9b` through Ollama. On this Mac that file is Q4_K_M, about 6.6 GB. The paper used FP16 on vLLM. This machine has 16 GB and no NVIDIA GPU, so FP16 and vLLM do not fit.
- Web search through Tavily. v1.0.0 implements Tavily, with DuckDuckGo as the fallback.
- Turn limit left at the OpenJarvis default of 10. A trial at 50 turns scored lower and was not used.

## Differences from the paper

The paper's 87.9% used a spec-search configuration that is not published. This run uses the published `pinchbench-qwen-9b.toml`, with the model and judge changes above. Three further choices are local:

1. `shell_exec` is auto-approved. The eval path rejects every tool marked `requires_confirmation`, and `shell_exec` is marked that way. The launcher uses the same auto-approve callback OpenJarvis uses for its own server.
2. The judge reply limit is 16000 tokens. `scorers/pinchbench.py` calls the judge with 2048. GPT-5-mini spends part of that budget on hidden reasoning, so long transcripts came back empty and scored 0. The prompt, model, temperature, and rubric are unchanged.
3. Live web pages are from October 2026. The paper's web tasks saw the web in early 2026.

`run_eval.py` is the only local code. It sets those three items and then calls OpenJarvis's eval CLI. The agent sees only the task Prompt section. Expected behavior, the rubric, and automated checks stay with the grader. `episode_mode` is false, so earlier tasks are not shown as examples.

The five runs were debugged on this same 23-task set. There is no held-out split.

The configs in `config/` now use a relative `output_dir` (`results/run_01` and so on). The runs themselves wrote to those same folders through an absolute path. The run logs still show that path.

## Repeat the run

From `baseline/`, with `uv`, Rust, Ollama, and `qwen3.5:9b` already installed.

```bash
git clone https://github.com/open-jarvis/OpenJarvis.git OpenJarvis
git -C OpenJarvis checkout e97088f199cf86ea5f78de921772357d1f0d2cec
(
  cd OpenJarvis
  uv sync --python 3.13 --extra inference-cloud --extra pdf --extra tools-search
  source "$HOME/.cargo/env"
  uv run maturin develop --release -m rust/crates/openjarvis-python/Cargo.toml
)

git clone https://github.com/pinchbench/skill.git pinchbench_repo
mkdir -p pinchbench_23tasks
git -C pinchbench_repo archive 0c419e45aac78f6bdeadf5ac83eca9aa8cd14b49 | tar -x -C pinchbench_23tasks

cp .env.example .env
# Put OPENAI_API_KEY and TAVILY_API_KEY in .env. Do not commit .env.

set -a && source .env && set +a
mkdir -p results/run_01
caffeinate -i uv run --project OpenJarvis python run_eval.py run -c config/condition_c.toml
```

`config/condition_c_run02.toml` through `condition_c_run05.toml` are the same file with `output_dir` set to `results/run_02` through `results/run_05`. One full run takes on the order of two hours. `caffeinate -i` keeps the Mac awake.
