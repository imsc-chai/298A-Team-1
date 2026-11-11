"""Shared data schemas.

The run-output (Workflow) schema is intentionally NOT defined yet: it waits for the team's
Output contract so the exporter and validator are written once, against the real spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RecordType(str, Enum):
    CALENDAR_EVENT = "calendar_event"
    MESSAGE = "message"
    CONTACT = "contact"
    PREFERENCE = "preference"
    ROUTINE = "routine"
    PROFILE = "profile"
    FILE = "file"


class MemoryRecord(BaseModel):
    """One retrievable unit of persona memory (proposal §4.2).

    Deliberately carries no sensitivity label: typed privacy belongs to M6, and putting it in
    the shared record would leak M6 machinery into M1/M3 (CLAUDE.md rule 5).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    persona_id: str
    source: str
    type: RecordType
    text: str
    date: str | None = None
    timestamp: str | None = None  # messages only; used for the reference_time cutoff


class StepStatus(str, Enum):
    EXECUTED = "EXECUTED"
    DRAFT = "DRAFT"
    REFUSED = "REFUSED"  # tool needs confirmation; dispatcher refused (boundedness violation)
    FAILED = "FAILED"  # invalid call or tool error


class ToolResult(BaseModel):
    """What the dispatcher returns for one tool call."""

    tool: str
    args: dict[str, Any]
    status: StepStatus
    observation: str
    evidence_ids: list[str] = Field(default_factory=list)
    violation: str | None = None


class AgentTask(BaseModel):
    """The ONLY task fields a model may see (CLAUDE.md rule 2)."""

    model_config = ConfigDict(frozen=True)

    persona_id: str
    task_id: str
    user_query: str
    reference_time: str
