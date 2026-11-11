"""Persona data -> MemoryRecords (proposal §4.2).

Reads only agent_memory/, streams/ and local_files/; never test_suite/ (CLAUDE.md rule 2).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from model_a.common.schemas import MemoryRecord, RecordType
from model_a.common.parsers import file_record_id, parse_file

WEEKDAY_NAMES = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}


def build_records(persona_root: Path, persona_id: str) -> list[MemoryRecord]:
    persona_root = Path(persona_root)
    records: list[MemoryRecord] = []
    records += calendar_records(persona_root, persona_id)
    records += message_records(persona_root, persona_id)
    records += profile_records(persona_root, persona_id)
    records += file_records(persona_root, persona_id)
    ids = [r.id for r in records]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate record ids in {persona_id}: {sorted(duplicates)}")
    return records


def human_date(iso_date: str) -> str:
    """'2026-11-10' -> 'Tuesday Nov 10 2026 (2026-11-10)'. Users write 'Nov 10', so BM25 needs
    those tokens; the weekday is computed, not copied from labels that were once wrong."""
    d = dt.date.fromisoformat(iso_date[:10])
    return f"{d.strftime('%A')} {d.strftime('%b')} {d.day} {d.year} ({d.isoformat()})"


def render_event(e: dict[str, Any]) -> str:
    """Shared rendering so the index and the search_calendar tool describe events identically."""
    overnight = " (ends next day)" if e["end_time"] <= e["start_time"] else ""
    parts = [f"{e['title']} on {human_date(e['date'])} {e['start_time']}-{e['end_time']}{overnight}"]
    if e.get("location"):
        parts.append(f"location: {e['location']}")
    if e.get("category"):
        parts.append(f"category: {e['category']}")
    if e.get("attendees"):
        parts.append(f"attendees: {', '.join(e['attendees'])}")
    if e.get("notes"):
        parts.append(f"notes: {e['notes']}")
    if e.get("is_sensitive"):
        parts.append("marked private by user")
    parts.append("busy" if e.get("busy", True) else "free")
    return "; ".join(parts)


def render_message(m: dict[str, Any]) -> str:
    subject = f" | subject: {m['subject']}" if m.get("subject") else ""
    when = f"{human_date(m['timestamp'])} {m['timestamp'][11:16]}"
    return f"{m['channel']} on {when} from {m['sender']} to {m['recipient']}{subject} | {m.get('body', '')}"


def calendar_records(root: Path, persona_id: str) -> list[MemoryRecord]:
    events = json.loads((root / "streams" / "calendar.json").read_text())
    return [
        MemoryRecord(
            id=e["id"],
            persona_id=persona_id,
            source="streams/calendar.json",
            type=RecordType.CALENDAR_EVENT,
            date=e["date"],
            text=f"Calendar event: {render_event(e)}",
        )
        for e in events
    ]


def message_records(root: Path, persona_id: str) -> list[MemoryRecord]:
    messages = json.loads((root / "streams" / "messages.json").read_text())
    return [
        MemoryRecord(
            id=m["id"],
            persona_id=persona_id,
            source="streams/messages.json",
            type=RecordType.MESSAGE,
            date=m["timestamp"][:10],
            timestamp=m["timestamp"],
            text=f"Message: {render_message(m)}",
        )
        for m in messages
    ]


def profile_records(root: Path, persona_id: str) -> list[MemoryRecord]:
    profile = json.loads((root / "agent_memory" / "user_profile.json").read_text())
    src = "agent_memory/user_profile.json"
    records = [
        MemoryRecord(
            id="profile_identity",
            persona_id=persona_id,
            source=src,
            type=RecordType.PROFILE,
            text=(
                f"User: {profile.get('name')}, {profile.get('role')}"
                + (f", {profile['institution']}" if profile.get("institution") else "")
                + f". Email {profile.get('email')}, phone {profile.get('phone')}."
            ),
        )
    ]
    for c in profile.get("contacts", []):
        fields = [f"{k}: {c[k]}" for k in ("name", "role", "relation", "email", "phone", "preferred_channel") if c.get(k)]
        records.append(
            MemoryRecord(
                id=f"contact_{c['id']}",
                persona_id=persona_id,
                source=src,
                type=RecordType.CONTACT,
                text=f"Contact {c['id']}: " + "; ".join(fields),
            )
        )
    for key, value in profile.get("preferences", {}).items():
        records.append(
            MemoryRecord(
                id=f"pref_{key}",
                persona_id=persona_id,
                source=src,
                type=RecordType.PREFERENCE,
                text=f"Preference {key}: {_render_value(key, value)}",
            )
        )
    for i, r in enumerate(profile.get("routines", []), start=1):
        days = ", ".join(WEEKDAY_NAMES.get(d, str(d)) for d in r.get("weekdays", []))
        extra = {k: v for k, v in r.items() if k not in {"name", "weekdays", "start", "duration_minutes"}}
        records.append(
            MemoryRecord(
                id=f"routine_{i:02d}",
                persona_id=persona_id,
                source=src,
                type=RecordType.ROUTINE,
                text=(
                    f"Routine: {r.get('name')} every {days} at {r.get('start')} for {r.get('duration_minutes')} min; "
                    + "; ".join(f"{k}: {v}" for k, v in extra.items())
                ),
            )
        )
    return records


def _render_value(key: str, value: Any) -> str:
    """Spell out weekday numbers so retrieval and the LLM read 'Friday', not '5'."""
    if key == "working_days" and isinstance(value, list):
        return ", ".join(WEEKDAY_NAMES.get(d, str(d)) for d in value)
    if isinstance(value, list) and value and isinstance(value[0], dict) and "weekday" in value[0]:
        value = [{**v, "weekday": WEEKDAY_NAMES.get(v["weekday"], v["weekday"])} for v in value]
    return value if isinstance(value, str) else json.dumps(value)


def file_records(root: Path, persona_id: str) -> list[MemoryRecord]:
    records = []
    for path in sorted((root / "local_files").iterdir()):
        if path.name.startswith("."):
            continue
        parsed = parse_file(path)
        for chunk in parsed.chunks:
            records.append(
                MemoryRecord(
                    id=file_record_id(path.name, chunk.index),
                    persona_id=persona_id,
                    source=f"local_files/{path.name}",
                    type=RecordType.FILE,
                    text=f"File local_files/{path.name} ({chunk.heading}): {chunk.text}",
                )
            )
    return records
