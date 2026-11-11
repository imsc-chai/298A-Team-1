"""Per-task reset: every task runs on a fresh temp copy of the persona.

Only agent-visible folders are copied. test_suite/ is never copied, so ground truth is not
reachable from the agent's filesystem at all (CLAUDE.md rules 1 and 2).
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

AGENT_VISIBLE = ("agent_memory", "streams", "local_files")


@contextmanager
def task_sandbox(source_persona_dir: Path) -> Iterator[Path]:
    """Yield a throwaway persona directory; deleted on exit even if the task crashes."""
    source = Path(source_persona_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="model_a_task_") as tmp:
        target = Path(tmp) / source.name
        target.mkdir()
        for name in AGENT_VISIBLE:
            src = source / name
            if not src.is_dir():
                raise FileNotFoundError(f"Persona is missing {name}/: {source}")
            shutil.copytree(src, target / name, ignore=shutil.ignore_patterns(".DS_Store"))
        if (target / "test_suite").exists():  # defensive: must never happen
            raise RuntimeError("test_suite leaked into the task sandbox")
        yield target
