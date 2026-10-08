"""Typed records for personas, events and tasks.

Ground truth lives in these fields, never only in rendered text.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

DAYS = 30
HISTORY_LAST_DAY = 20  # Workbook 1 temporal split: days 1-20 train (history)
VAL_LAST_DAY = 24  # days 21-24 validation, days 25-30 test


def split_for_day(day: int) -> str:
    return "train" if day <= HISTORY_LAST_DAY else "val" if day <= VAL_LAST_DAY else "test"


class Sensitivity(str, Enum):
    NONE = "none"
    HEALTH = "health"
    FINANCE = "finance"
    LOCATION = "location"
    RELATIONSHIP = "relationship"


class EventType(str, Enum):
    CALENDAR = "calendar_event"
    MESSAGE = "message"
    FILE = "file"
    BEHAVIOR = "observed_behavior"
    FEEDBACK = "explicit_feedback"
    TASK = "user_task"
    FACT_UPDATE = "fact_update"
    DELETION = "deletion_request"


# Event types whose text is written by the LLM renderer. All others get templated text in the skeleton.
RENDERED_TYPES = {EventType.MESSAGE, EventType.FILE, EventType.FACT_UPDATE}


class Contact(BaseModel):
    contact_id: str
    name: str
    email: str
    relation: str


class Routine(BaseModel):
    name: str
    weekdays: list[int]  # 0 = Monday; day 1 of every trace is a Monday
    start_minute: int = Field(ge=0, lt=1440)
    jitter_minutes: int = Field(ge=0)
    duration_minutes: int = Field(gt=0)
    skip_prob: float = Field(ge=0, le=1)


class Preference(BaseModel):
    key: str
    options: list[str]
    initial: str
    drift_day: int | None = None
    drift_to: str | None = None

    def value_on(self, day: int) -> str:
        if self.drift_day is not None and day >= self.drift_day:
            return self.drift_to
        return self.initial


class SensitiveAttribute(BaseModel):
    attr_id: str
    category: Sensitivity
    value: str
    allowed_contacts: list[str]
    may_leave_device: bool = False


class Persona(BaseModel):
    persona_id: str
    archetype: str
    name: str
    email: str
    contacts: list[Contact]
    routines: list[Routine]
    preferences: list[Preference]
    sensitive: list[SensitiveAttribute]


class Checker(BaseModel):
    kind: str  # recall | update | deletion | privacy | scheduling
    args: dict[str, Any]


class Event(BaseModel):
    event_id: str
    persona_id: str
    day: int = Field(ge=1, le=DAYS)
    minute: int = Field(ge=0, lt=1440)
    type: EventType
    payload: dict[str, Any] = {}
    fact_id: str | None = None
    fact_value: str | None = None
    preference_key: str | None = None
    preference_value: str | None = None
    sensitivity: Sensitivity = Sensitivity.NONE
    canary: str | None = None
    provenance: str
    checker: Checker | None = None
    text: str | None = None
    render_status: str = "pending"  # pending | ok | render_failed | not_needed
