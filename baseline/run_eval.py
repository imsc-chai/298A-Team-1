"""Launch OpenJarvis's eval CLI for PinchBench condition (c).

Three adjustments, nothing else:
1. shell_exec auto-approval. PinchBench runs non-interactively, so OpenJarvis's
   ToolExecutor rejects every tool with requires_confirmation=True. This applies
   the same auto-approval OpenJarvis uses in server/agent_manager_routes.py
   (interactive=True, confirm_callback=lambda _prompt: True).
2. 23-task snapshot. In v1.0.0 suite runs, cli.py builds the PinchBench dataset
   without the TOML subset path, so it would clone the current 147-task repo.
   CACHE_DIR is pointed at pinchbench_23tasks instead.
3. Judge reply limit. scorers/pinchbench.py calls the judge with max_tokens=2048.
   gpt-5-mini is a reasoning model, and that limit covers its hidden reasoning
   plus the reply, so long transcripts sometimes return an empty or cut-off reply
   that scores 0. The limit is raised to 16000; prompt, model, temperature and
   rubric are unchanged.

Usage (from the baseline folder):
    uv run --project OpenJarvis python run_eval.py run -c config/condition_c.toml
"""

import functools
from pathlib import Path

import openjarvis.evals.datasets.pinchbench as pinchbench_dataset
from openjarvis.tools._stubs import ToolExecutor

PINCHBENCH_23_TASKS = Path(__file__).resolve().parent / "pinchbench_23tasks"
assert len(list((PINCHBENCH_23_TASKS / "tasks").glob("task_*.md"))) == 23
pinchbench_dataset.CACHE_DIR = PINCHBENCH_23_TASKS

import openjarvis.evals.scorers.pinchbench as pinchbench_scorer  # noqa: E402

JUDGE_MAX_TOKENS = 16000


class _JudgeWithLargerReplyLimit:
    def __init__(self, backend):
        self._backend = backend

    def generate(self, *args, **kwargs):
        kwargs["max_tokens"] = JUDGE_MAX_TOKENS
        return self._backend.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._backend, name)


_original_grade_llm_judge = pinchbench_scorer._grade_llm_judge


def _grade_llm_judge_larger_reply(record, transcript, workspace_path, judge_backend, judge_model):
    if judge_backend is not None:
        judge_backend = _JudgeWithLargerReplyLimit(judge_backend)
    return _original_grade_llm_judge(record, transcript, workspace_path, judge_backend, judge_model)


pinchbench_scorer._grade_llm_judge = _grade_llm_judge_larger_reply

_original_init = ToolExecutor.__init__


@functools.wraps(_original_init)
def _auto_approve_init(self, *args, **kwargs):
    kwargs["interactive"] = True
    kwargs["confirm_callback"] = lambda _prompt: True
    _original_init(self, *args, **kwargs)


ToolExecutor.__init__ = _auto_approve_init

from openjarvis.evals.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
