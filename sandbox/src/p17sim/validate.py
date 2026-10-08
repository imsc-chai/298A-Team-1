"""Schema and invariant checks. Any violation stops the pipeline before data is loaded."""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from p17sim.schema import HISTORY_LAST_DAY, Event, EventType, Sensitivity

ASSERTING = {EventType.MESSAGE, EventType.FILE, EventType.FACT_UPDATE}


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Return human-readable violations; an empty list means the data is valid."""
    violations: list[str] = []
    events: list[Event] = []
    for i, row in enumerate(rows):
        try:
            events.append(Event.model_validate(row))
        except ValidationError as exc:
            first = exc.errors()[0]
            violations.append(f"schema: row {i} ({row.get('event_id')}): {first['loc']} {first['msg']}")

    seen: set[str] = set()
    first_asserted: dict[str, tuple[int, int]] = {}
    for e in sorted(events, key=lambda e: (e.day, e.minute)):
        if e.event_id in seen:
            violations.append(f"duplicate_event_id: {e.event_id}")
        seen.add(e.event_id)
        when = (e.day, e.minute)
        if e.type in ASSERTING and e.fact_id and e.type != EventType.FACT_UPDATE:
            first_asserted.setdefault(e.fact_id, when)
        if e.type in (EventType.DELETION, EventType.FACT_UPDATE):
            if e.fact_id not in first_asserted or first_asserted[e.fact_id] >= when:
                violations.append(f"{e.type.value}_without_prior_fact: {e.event_id} -> {e.fact_id}")
        if e.type == EventType.TASK:
            if e.checker is None:
                violations.append(f"task_without_checker: {e.event_id}")
            if e.day <= HISTORY_LAST_DAY:
                violations.append(f"task_in_history_window: {e.event_id} day {e.day}")
        if e.sensitivity != Sensitivity.NONE and not e.canary:
            violations.append(f"sensitive_without_canary: {e.event_id}")
        if e.render_status == "ok":
            for required in (e.fact_value, e.canary):
                if required and required not in (e.text or ""):
                    violations.append(f"rendered_text_missing_slot: {e.event_id} missing {required!r}")
    return violations
