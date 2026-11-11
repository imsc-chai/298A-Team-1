"""One call to set up everything a model needs for a single task.

Keeps per-task setup identical across models: fresh sandbox, connector, dispatcher and
persona context. The retriever is built once per persona from the pristine data (minus
test_suite) and shared, since its inputs do not change between tasks.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from model_a.common.connectors import LocalBenchmarkConnector, open_connector
from model_a.common.context import PersonaContext
from model_a.common.sandbox import task_sandbox
from model_a.common.schemas import AgentTask
from model_a.common.tools import ToolDispatcher
from model_a.common.config import load_tool_registry, persona_dir
from model_a.common.ingest import build_records
from model_a.common.retriever import BgeEmbedder, Embedder, HybridRetriever


@dataclass
class TaskEnv:
    task: AgentTask
    sandbox_dir: Path
    connector: LocalBenchmarkConnector
    dispatcher: ToolDispatcher
    context: PersonaContext
    registry: dict


@contextmanager
def open_task_env(task: AgentTask) -> Iterator[TaskEnv]:
    registry = load_tool_registry()
    with task_sandbox(persona_dir(task.persona_id)) as sandbox_dir:
        connector = open_connector(sandbox_dir)
        yield TaskEnv(
            task=task,
            sandbox_dir=sandbox_dir,
            connector=connector,
            dispatcher=ToolDispatcher(connector, registry, task.reference_time),
            context=PersonaContext.load(sandbox_dir),
            registry=registry,
        )


def build_retriever(persona_id: str, embedder: Embedder | None = None) -> HybridRetriever:
    """Index over a sandbox copy so the indexer, too, physically cannot read test_suite/."""
    with task_sandbox(persona_dir(persona_id)) as sandbox_dir:
        records = build_records(sandbox_dir, persona_id)
    return HybridRetriever(records, embedder or BgeEmbedder())
