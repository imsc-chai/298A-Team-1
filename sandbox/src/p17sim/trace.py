"""Workbook 1 trace: raw rendered events -> cleaned 7-field records + separate labels.

The trace (what models see) follows Workbook 1, Table 6:
event_id, persona_id, timestamp, event_type, text, permission_label, deleted.
Ground truth (facts, canaries, checkers) goes to labels.jsonl, which only the harness reads.
Splits are temporal (Workbook 1, Section 2.1): days 1-20 train, 21-24 val, 25-30 test.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from p17sim.rng import rng_for
from p17sim.schema import DAYS, HISTORY_LAST_DAY, VAL_LAST_DAY, Event, EventType, Persona, Sensitivity, split_for_day

START = datetime(2026, 9, 7)  # a Monday; day 1 of every trace
EVENT_TYPES = ("calendar", "message", "file", "preference", "permission", "conflict",
               "routine_task", "sensitive_task", "deletion_request")
PERMISSIONS = ("read", "draft", "send", "deny")
TOOLS = ("calendar.read", "calendar.create", "mail.read", "mail.send", "files.read")


class TraceRecord(BaseModel):
    event_id: str
    persona_id: str
    timestamp: str
    event_type: Literal[EVENT_TYPES]  # type: ignore[valid-type]
    text: str
    permission_label: Literal[PERMISSIONS]  # type: ignore[valid-type]
    deleted: bool


# ---------- cleaning ----------

PREAMBLE = re.compile(r"^\s*(?:sure[,!.]?\s*)?here(?:'s| is)[^\n:]{0,100}:\s*", re.I)
MARKDOWN = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
EMAIL = re.compile(r"[\w.+-]+@([\w-]+\.)+[a-z]{2,}", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}(?!\w)|(?<![\w-])\d{3}-\d{4}(?![\w-])")
SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")
FICTIONAL_PHONE = re.compile(r"555-01\d\d$")
ALLOWED_EMAIL = ("@example.com", "@example.org")


def pii_findings(text: str) -> list[tuple[str, str]]:
    """Identifiers that could belong to a real person. Synthetic forms (example domains,
    555-01xx numbers) are allowed."""
    found = []
    for m in EMAIL.finditer(text):
        if not m.group(0).lower().endswith(ALLOWED_EMAIL):
            found.append(("email", m.group(0)))
    for m in PHONE.finditer(text):
        if not FICTIONAL_PHONE.search(m.group(0)):
            found.append(("phone", m.group(0)))
    found += [("ssn", m.group(0)) for m in SSN.finditer(text)]
    found += [("card_number", m.group(0)) for m in CARD.finditer(text)]
    return found


def clean_text(text: str, protected: list[str]) -> tuple[str, list[str]]:
    """Return cleaned text and the list of fixes applied. Strings in `protected`
    (fact values, canaries) are never redacted."""
    fixes = []
    t = PREAMBLE.sub("", text, count=1)
    if t != text:
        fixes.append("llm_preamble")
    s = t.strip()
    if len(s) >= 2 and s[0] in "\"“" and s[-1] in "\"”":
        s = s[1:-1].strip()
        fixes.append("wrapping_quotes")
    elif s[:1] in "\"“" and sum(s.count(q) for q in "\"“”") == 1:
        s = s[1:].strip()
        fixes.append("stray_quote")
    unmarked = MARKDOWN.sub(lambda x: x.group(1) or x.group(2), s)
    if unmarked != s:
        s = unmarked
        fixes.append("markdown")
    tidy = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", s))
    if tidy != s:
        s = tidy
        fixes.append("whitespace")
    for kind, value in pii_findings(s):
        if not any(value in p or p in value for p in protected):
            s = s.replace(value, f"[{kind.upper()}]")
            fixes.append(f"redacted_{kind}")
    return s, fixes


# ---------- export ----------

def _timestamp(day: int, minute: int) -> str:
    return (START + timedelta(days=day - 1, minutes=minute)).isoformat()


def _event_type(e: Event) -> str:
    if e.type == EventType.CALENDAR:
        return "calendar"
    if e.type in (EventType.BEHAVIOR, EventType.FEEDBACK):
        return "preference"
    if e.type == EventType.FACT_UPDATE:
        return "conflict"
    if e.type == EventType.DELETION:
        return "deletion_request"
    if e.type == EventType.TASK:
        return "sensitive_task" if e.checker.kind == "privacy" else "routine_task"
    return e.type.value  # message, file


def _permission(e: Event) -> str:
    if e.type == EventType.DELETION:
        return "deny"
    if e.sensitivity != Sensitivity.NONE:
        return "deny"  # may be read on the device, never sent
    if e.type == EventType.TASK:
        return {"privacy": "draft", "scheduling": "send"}.get(e.checker.kind, "read")
    return "read"


def _imputed(e: Event, persona: Persona) -> str:
    who = {c.contact_id: c.name for c in persona.contacts}.get(e.payload.get("from", ""), persona.name)
    parts = [f"{who} wrote about {e.payload.get('topic', 'an update')}."]
    parts += [s for s in (e.fact_value, f"ref {e.canary}" if e.canary else None) if s]
    return " ".join(parts)


def build_trace(personas: list[Persona], raw: list[Event], seed: int) -> dict[str, Any]:
    """Clean raw events into trace records + labels. Returns records, labels and a cleaning log."""
    by_id = {p.persona_id: p for p in personas}
    log: Counter = Counter()
    samples: list[dict] = []
    records, labels = [], []
    seen_text: set[tuple[str, str]] = set()
    asserting: dict[str, str] = {}  # fact_id -> first event_id that asserted it
    deleted_facts = {e.fact_id for e in raw if e.type == EventType.DELETION}

    for p in personas:  # tool permission records, day 1 at 00:00
        rng = rng_for(seed, p.persona_id, "permissions")
        for tool in TOOLS:
            label = ("send" if tool == "calendar.create" else
                     str(rng.choice(["draft", "send"])) if tool == "mail.send" else "read")
            eid = f"{p.persona_id}-perm-{tool}"
            records.append({"event_id": eid, "persona_id": p.persona_id, "timestamp": _timestamp(1, 0),
                            "event_type": "permission", "text": f"{tool}: {label}",
                            "permission_label": label, "deleted": False})
            labels.append({"event_id": eid, "tool": tool, "source": "generator"})

    for e in sorted(raw, key=lambda e: (e.persona_id, e.day, e.minute, e.event_id)):
        log["raw_records"] += 1
        persona = by_id[e.persona_id]
        protected = [s for s in (e.fact_value, e.canary) if s]
        source = "template" if e.render_status == "not_needed" else "llm"
        text = e.text
        if text is None or not text.strip():
            log["missing_text_imputed"] += 1
            text, source = _imputed(e, persona), "imputed"
        cleaned, fixes = clean_text(text, protected)
        for f in fixes:
            log[f] += 1
        if fixes and len(samples) < 8 and source == "llm":
            samples.append({"event_id": e.event_id, "before": text, "after": cleaned, "fixes": fixes})
        if not e.fact_id and e.type in (EventType.MESSAGE, EventType.FILE):
            key = (e.persona_id, cleaned.lower())
            if key in seen_text:
                log["duplicate_dropped"] += 1
                continue
            seen_text.add(key)
        if any(r not in cleaned for r in protected):
            log["required_string_lost"] += 1
        if e.fact_id and e.type in (EventType.MESSAGE, EventType.FILE):
            asserting.setdefault(e.fact_id, e.event_id)
        records.append({"event_id": e.event_id, "persona_id": e.persona_id, "timestamp": _timestamp(e.day, e.minute),
                        "event_type": _event_type(e), "text": cleaned, "permission_label": _permission(e),
                        "deleted": e.fact_id in deleted_facts and e.type in (EventType.MESSAGE, EventType.FILE)})
        labels.append({"event_id": e.event_id, "day": e.day, "split": split_for_day(e.day), "source": source,
                       "fixes": fixes, "fact_id": e.fact_id, "fact_value": e.fact_value, "canary": e.canary,
                       "sensitivity": e.sensitivity.value, "preference_key": e.preference_key,
                       "preference_value": e.preference_value,
                       "checker": e.checker.model_dump() if e.checker else None,
                       "target_event_id": asserting.get(e.fact_id) if e.type in (
                           EventType.DELETION, EventType.FACT_UPDATE) else None})
    log["trace_records"] = len(records)
    log["pii_remaining"] = sum(len(pii_findings(r["text"])) for r in records)
    return {"records": records, "labels": labels, "log": dict(log), "samples": samples}


# ---------- validation, splits, leakage ----------

def validate_trace(records: list[dict], labels: list[dict]) -> list[str]:
    """Workbook 1 checks: schema parse, unique ids, timestamps in window,
    deletion requests reference an existing earlier record, no PII left."""
    v = []
    ids: set[str] = set()
    when: dict[str, str] = {}
    lo, hi = START.isoformat(), (START + timedelta(days=DAYS)).isoformat()
    for i, r in enumerate(records):
        try:
            TraceRecord.model_validate(r)
        except ValidationError as exc:
            v.append(f"schema: line {i} ({r.get('event_id')}): {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}")
            continue
        if r["event_id"] in ids:
            v.append(f"duplicate_event_id: {r['event_id']}")
        ids.add(r["event_id"])
        when[r["event_id"]] = r["timestamp"]
        if not lo <= r["timestamp"] < hi:
            v.append(f"timestamp_outside_window: {r['event_id']} {r['timestamp']}")
        if not r["text"].strip():
            v.append(f"empty_text: {r['event_id']}")
        if pii_findings(r["text"]):
            v.append(f"pii: {r['event_id']} {pii_findings(r['text'])}")
    types = {r["event_id"]: r.get("event_type") for r in records}
    for lab in labels:
        t = lab.get("target_event_id")
        if types.get(lab["event_id"]) == "deletion_request" and t is None:
            v.append(f"deletion_without_target: {lab['event_id']}")
        if t is not None and (t not in when or when[t] >= when.get(lab["event_id"], "")):
            v.append(f"target_missing_or_later: {lab['event_id']} -> {t}")
    return v


def splits_doc(records: list[dict]) -> dict:
    out = {"scheme": "temporal", "source": "Workbook 1, Section 2.1",
           "windows": {"train": [1, HISTORY_LAST_DAY], "val": [HISTORY_LAST_DAY + 1, VAL_LAST_DAY],
                       "test": [VAL_LAST_DAY + 1, DAYS]}, "start_date": START.date().isoformat(), "event_ids": {}}
    for r in records:
        day = (datetime.fromisoformat(r["timestamp"]) - START).days + 1
        out["event_ids"].setdefault(split_for_day(day), []).append(r["event_id"])
    out["counts"] = {k: len(v) for k, v in out["event_ids"].items()}
    return out


def leakage_checks(records: list[dict], labels: list[dict]) -> list[str]:
    """1 tasks only in val/test; 2 every task's evidence is in the train window;
    3 canaries only inside their own persona; 4 no task text contains its own answer."""
    found = []
    rec = {r["event_id"]: r for r in records}
    first_day: dict[str, int] = {}
    for lab in labels:
        if lab.get("fact_id") and rec.get(lab["event_id"], {}).get("event_type") in ("message", "file", "conflict"):
            first_day[lab["fact_id"]] = max(first_day.get(lab["fact_id"], 0), lab["day"])
    canary_owner = {lab["canary"]: rec[lab["event_id"]]["persona_id"] for lab in labels if lab.get("canary")}
    for lab in labels:
        r = rec.get(lab["event_id"])
        if r is None:
            continue
        if lab.get("checker"):
            if lab["split"] == "train":
                found.append(f"task_in_train_window: {lab['event_id']}")
            if lab.get("fact_id") and first_day.get(lab["fact_id"], 99) > HISTORY_LAST_DAY:
                found.append(f"task_evidence_after_train_window: {lab['event_id']}")
            args = lab["checker"]["args"]
            forbidden = args.get("forbidden") or []
            answers = [args.get("expected")] + ([forbidden] if isinstance(forbidden, str) else forbidden)
            if any(a and a in r["text"] for a in answers):
                found.append(f"task_text_contains_answer: {lab['event_id']}")
        for canary, owner in canary_owner.items():
            if canary in r["text"] and r["persona_id"] != owner:
                found.append(f"canary_outside_owner: {canary} in {r['event_id']}")
    return found


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
