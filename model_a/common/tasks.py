"""Task loading. The only non-scorer code allowed to open eval_tasks.json (CLAUDE.md rule 2).

It strips each task down to user_query + reference_time before anything reaches a model.
"""

from __future__ import annotations

import json

from model_a.common.schemas import AgentTask
from model_a.common.config import load_splits, persona_dir


def load_agent_tasks(persona_id: str, task_ids: list[str] | None = None) -> list[AgentTask]:
    raw = json.loads((persona_dir(persona_id) / "test_suite" / "eval_tasks.json").read_text())
    tasks = []
    for t in raw:
        if "reference_time" not in t:
            raise ValueError(f"{persona_id}/{t.get('task_id')} has no reference_time")
        tasks.append(
            AgentTask(persona_id=persona_id, task_id=t["task_id"], user_query=t["user_query"], reference_time=t["reference_time"])
        )
    if task_ids:
        known = {t.task_id for t in tasks}
        missing = set(task_ids) - known
        if missing:
            raise ValueError(f"Unknown task id(s) for {persona_id}: {sorted(missing)}")
        tasks = [t for t in tasks if t.task_id in task_ids]
    return tasks


def personas_for_split(split: str) -> list[str]:
    splits = load_splits()
    key = {"dev": "dev_personas", "test": "test_personas"}.get(split)
    if key is None:
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    return list(splits[key])
