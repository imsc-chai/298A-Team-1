from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class GuardPlanError(RuntimeError):
    pass


class BackendError(GuardPlanError):
    """Infrastructure failure; replanning cannot repair an unavailable model."""


class ModelOutputError(GuardPlanError):
    """An unexecuted model response needs a bounded formatting retry."""


class ContractError(GuardPlanError):
    """A requested outcome or permission is missing; do not replan around it."""


def timestamp(value: str) -> datetime:
    """Naive dataset timestamps use the same persona-local timeline."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def fingerprint(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def outside_source(source: Path, destination: Path) -> Path:
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise GuardPlanError("Derived artifacts must be outside the original dataset")
    return destination


@dataclass
class MemoryRecord:
    id: str
    persona_id: str
    source: str
    type: str
    text: str
    date: str | None = None
    valid_from: str | None = None
    superseded_by: str | None = None
    deleted: bool = False
    sensitivity: str = "none"
    data: dict[str, Any] = field(default_factory=dict)

    def visible(self, reference_time: str) -> bool:
        # Event date is NOT knowledge time: already-known future appointments
        # must remain retrievable. Messages use their timestamp as valid_from.
        return (not self.deleted and self.superseded_by is None and
                (not self.valid_from or timestamp(self.valid_from) <= timestamp(reference_time)))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Settings:
    model: str = "qwen3.5:9b"
    embedding_model: str = "all-minilm"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.6
    max_tokens: int = 8192
    context_tokens: int = 16384
    k: int = 8
    retries: int = 2
    max_depth: int = 2
    max_steps_per_subgoal: int = 8
    max_llm_calls: int = 32
    seed: int = 0
    timeout: int = 600
    variant: str = "full"
