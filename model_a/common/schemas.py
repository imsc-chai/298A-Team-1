"""Shared internal data schemas. The run-output schema lives in contract.py."""

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
    """Step statuses of the Output contract."""

    EXECUTED = "EXECUTED"
    DRAFT = "DRAFT"
    REQUIRES_CONFIRMATION = "REQUIRES_CONFIRMATION"  # dispatcher refused; logged as boundedness violation
    BLOCKED = "BLOCKED"  # reserved for M6's release gate; the shared dispatcher never emits it
    FAILED = "FAILED"  # well-formed call whose execution failed (bad date, missing file, ...)


class ToolResult(BaseModel):
    """What the dispatcher returns for one tool call."""

    tool: str
    args: dict[str, Any]
    status: StepStatus
    observation: str
    evidence_ids: list[str] = Field(default_factory=list)
    violation: str | None = None
    # False = unknown tool or args not matching the registry. Such calls cannot appear in
    # steps[] (contract: registry tool, exact args), so models log them under debug instead.
    well_formed: bool = True


class AgentTask(BaseModel):
    """The ONLY task fields a model may see (CLAUDE.md rule 2)."""

    model_config = ConfigDict(frozen=True)

    persona_id: str
    task_id: str
    user_query: str
    reference_time: str
