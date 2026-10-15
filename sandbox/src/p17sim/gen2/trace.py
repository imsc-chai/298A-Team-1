"""v2 export: rendered events -> cleaned trace records + labels, then validation and leakage checks.

The trace keeps Workbook 1's seven fields and adds three (schema v2):
- `sender`: who wrote it ("me" for the user's own notes and calendar);
- `channel`: message | note | calendar | notification | preference | instruction | permission;
- `end`: end time of calendar entries (null otherwise).
A mailbox shows the sender, and v1 left it out, so an agent could not tell who said what.
Timestamps use each persona's own start date (v2 traces start on different weekdays).
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, time, timedelta
from typing import Any

from datasketch import MinHash, MinHashLSH

from p17sim.gen2 import phrasing as ph
from p17sim.gen2.personas import PersonaExtra
from p17sim.schema import DAYS, Event, EventType, Persona, Sensitivity, split_for_day
from p17sim.trace import PERMISSIONS, clean_text, pii_findings

TOOLS = ("calendar.read", "calendar.create", "mail.read", "mail.send", "files.read", "memory.forget")

# Artifacts the renderers add that v1's cleaner did not know about (mostly phi3).
# A trailing "(15 words, casual tone ...)" note, closed or cut off by the token cap.
META_TAIL = re.compile(r"\n*\s*\((?:[^()]*\b(?:words?|tone|style|note|exact|phrase|word count)\b[^()]*)\)?\s*$",
                       re.I | re.S)
NOTE_TAIL = re.compile(r"\n+\s*(?:Note|Explanation|Word count)\s*:.*$", re.I | re.S)
PLACEHOLDER = re.compile(r"\[(?:Your Name|Name|Recipient(?:'s)? Name|Sender(?:'s)? Name|Date|Your Signature)\]",
                         re.I)


def clean_text_v2(text: str, protected: list[str]) -> tuple[str, list[str]]:
    fixes = []
    t = text
    for _ in range(3):  # meta notes can stack: "(15 words...)\n\n(Note: ...)"
        new = t
        for pattern in (META_TAIL, NOTE_TAIL):
            m = pattern.search(new)
            if m and not any(p in m.group(0) for p in protected):
                new = new[:m.start()].rstrip()
        if new == t:
            break
        t = new
        fixes.append("meta_commentary")
    if PLACEHOLDER.search(t):
        t = PLACEHOLDER.sub("", t)
        t = re.sub(r"\n\s*\n\s*$", "", t).rstrip(" ,\n")
        fixes.append("placeholder")
    cleaned, more = clean_text(t, protected)
    return cleaned, fixes + more


def timestamp(extra: PersonaExtra, day: int, minute: int) -> str:
    start = datetime.combine(ph.day1(extra.start_weekday), time(0, 0))
    return (start + timedelta(days=day - 1, minutes=minute)).isoformat()


def _channel(e: Event) -> str:
    if e.provenance == "filler:notification":
        return "notification"
    if e.type == EventType.CALENDAR:
        return "calendar"
    if e.type == EventType.FILE:
        return "note"
    if e.type in (EventType.BEHAVIOR, EventType.FEEDBACK):
        return "preference"
    if e.type in (EventType.DELETION, EventType.TASK):
        return "instruction"
    return "message"


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
    return e.type.value


def _permission(e: Event) -> str:
    if e.type == EventType.DELETION or e.sensitivity != Sensitivity.NONE:
        return "deny"
    if e.type == EventType.TASK:
        return {"privacy": "draft", "scheduling": "send"}.get(e.checker.kind, "read")
    return "read"


def _imputed(e: Event, persona: Persona) -> str:
    who = {c.contact_id: c.name for c in persona.contacts}.get(e.payload.get("from", ""), persona.name)
    parts = [f"{who} wrote about {e.payload.get('topic', 'an update')}."]
    parts += [s for s in (e.fact_value, f"ref {e.canary}" if e.canary else None) if s]
    return " ".join(parts)


def build(personas: list[Persona], extras: list[PersonaExtra], raw: list[Event],
          splits: dict[str, str]) -> dict[str, Any]:
    by_id = {p.persona_id: p for p in personas}
    x_by_id = {x.persona_id: x for x in extras}
    log: Counter = Counter()
    samples: list[dict] = []
    records, labels = [], []
    seen: set[tuple[str, str]] = set()
    asserting: dict[str, str] = {}
    for p in personas:
        x = x_by_id[p.persona_id]
        for tool in TOOLS:  # mail.send needs the user's OK before sending for some people
            label = "draft" if tool == "mail.send" and x.pref_noise > 0.15 else "send" if tool in (
                "mail.send", "calendar.create", "memory.forget") else "read"
            eid = f"{p.persona_id}-perm-{tool}"
            records.append({"event_id": eid, "persona_id": p.persona_id, "timestamp": timestamp(x, 1, 0),
                            "event_type": "permission", "text": f"{tool}: {label}", "permission_label": label,
                            "deleted": False, "sender": "system", "channel": "permission"})
            labels.append({"event_id": eid, "persona_id": p.persona_id, "tool": tool, "source": "generator",
                           "persona_split": splits[p.persona_id], "day": 1})
    deleted_facts = {e.fact_id for e in raw if e.type == EventType.DELETION}
    for e in sorted(raw, key=lambda e: (e.persona_id, e.day, e.minute, e.event_id)):
        log["raw_records"] += 1
        p, x = by_id[e.persona_id], x_by_id[e.persona_id]
        protected = [s for s in (e.fact_value, e.canary) if s]
        source = "template" if e.render_status == "not_needed" else x.renderer
        text = e.text
        if text is None or not text.strip():
            log["missing_text_imputed"] += 1
            text, source = _imputed(e, p), "imputed"
        cleaned, fixes = clean_text_v2(text, protected)
        for f in fixes:
            log[f] += 1
        if fixes and len(samples) < 12 and source.startswith("ollama"):
            samples.append({"event_id": e.event_id, "renderer": source, "before": text, "after": cleaned,
                            "fixes": fixes})
        if not e.fact_id and e.type in (EventType.MESSAGE, EventType.FILE) and e.provenance != "filler:notification":
            key = (e.persona_id, cleaned.lower())
            if key in seen:
                log["duplicate_dropped"] += 1
                continue
            seen.add(key)
        if any(r not in cleaned for r in protected):
            log["required_string_lost"] += 1
        if e.fact_id and e.type in (EventType.MESSAGE, EventType.FILE, EventType.CALENDAR) \
                and e.provenance.startswith("fact:"):
            asserting.setdefault(e.fact_id, e.event_id)
        contacts = {c.contact_id: c.name for c in p.contacts}
        sender = (contacts.get(e.payload.get("from", "")) or e.payload.get("from_service")
                  or ("me" if e.type in (EventType.FILE, EventType.CALENDAR, EventType.TASK, EventType.DELETION,
                                         EventType.FEEDBACK) else "assistant"))
        records.append({"event_id": e.event_id, "persona_id": e.persona_id, "timestamp": timestamp(x, e.day, e.minute),
                        "event_type": _event_type(e), "text": cleaned, "permission_label": _permission(e),
                        "deleted": bool(e.fact_id in deleted_facts and e.type in (EventType.MESSAGE, EventType.FILE)),
                        "sender": sender, "channel": _channel(e),
                        "end": timestamp(x, e.day, e.minute + e.payload.get("duration_minutes", 30))
                        if e.type == EventType.CALENDAR else None})
        labels.append({"event_id": e.event_id, "persona_id": e.persona_id, "day": e.day, "minute": e.minute,
                       "split": split_for_day(e.day), "persona_split": splits[e.persona_id], "source": source,
                       "fixes": fixes, "provenance": e.provenance, "fact_id": e.fact_id, "fact_value": e.fact_value,
                       "canary": e.canary, "sensitivity": e.sensitivity.value, "preference_key": e.preference_key,
                       "preference_value": e.preference_value,
                       "checker": e.checker.model_dump() if e.checker else None,
                       "wording": e.payload.get("wording"), "fact_kind": e.payload.get("fact_kind"),
                       "target_event_id": asserting.get(e.fact_id) if e.type in (
                           EventType.DELETION, EventType.FACT_UPDATE) else None})
    log["trace_records"] = len(records)
    log["pii_remaining"] = sum(len(pii_findings(r["text"])) for r in records)
    return {"records": records, "labels": labels, "log": dict(log), "samples": samples}


# ------------------------------------------------------------------ checks

def validate(records: list[dict], labels: list[dict], extras: list[PersonaExtra]) -> list[str]:
    """Workbook 1 checks with per-persona windows: schema, unique ids, timestamps inside the
    persona's 30 days, deletions and updates point to an earlier record, no PII left."""
    x_by_id = {x.persona_id: x for x in extras}
    v, ids, when = [], set(), {}
    for i, r in enumerate(records):
        for field in ("event_id", "persona_id", "timestamp", "event_type", "text", "permission_label", "deleted",
                      "sender", "channel"):
            if field not in r:
                v.append(f"schema: line {i} missing {field}")
        if r.get("permission_label") not in PERMISSIONS:
            v.append(f"schema: line {i} bad permission_label {r.get('permission_label')}")
        if r["event_id"] in ids:
            v.append(f"duplicate_event_id: {r['event_id']}")
        ids.add(r["event_id"])
        when[r["event_id"]] = r["timestamp"]
        x = x_by_id[r["persona_id"]]
        lo, hi = timestamp(x, 1, 0), timestamp(x, DAYS + 1, 0)
        if not lo <= r["timestamp"] < hi:
            v.append(f"timestamp_outside_window: {r['event_id']} {r['timestamp']}")
        if not str(r.get("text", "")).strip():
            v.append(f"empty_text: {r['event_id']}")
        if pii_findings(r.get("text", "")):
            v.append(f"pii: {r['event_id']} {pii_findings(r['text'])}")
    for lab in labels:
        t = lab.get("target_event_id")
        if lab.get("provenance", "").startswith("delete:") and t is None:
            v.append(f"deletion_without_target: {lab['event_id']}")
        if t is not None and (t not in when or when[t] >= when.get(lab["event_id"], "")):
            v.append(f"target_missing_or_later: {lab['event_id']} -> {t}")
    return v


def _minhash(text: str) -> MinHash:
    words = text.lower().split()
    m = MinHash(num_perm=128)
    for i in range(max(0, len(words) - 2)):
        m.update(" ".join(words[i:i + 3]).encode())
    return m


def leakage(records: list[dict], labels: list[dict], splits: dict[str, str]) -> list[str]:
    """Checks that stop the pipeline:
    1 no persona in two persona splits;
    2 held-out wordings only in test-window tasks of test/OOD personas;
    3 every task's evidence (the fact, its updates or deletion) comes before the task;
    4 each canary appears only in its own persona's records;
    5 no task text contains its own answer;
    6 no label-carrying text is a near-duplicate of one in another persona split (MinHash, J >= 0.8)."""
    found = []
    owners: dict[str, set[str]] = {}
    for pid, split in splits.items():
        owners.setdefault(pid, set()).add(split)
    found += [f"persona_in_two_splits: {pid}" for pid, s in owners.items() if len(s) > 1]
    rec = {r["event_id"]: r for r in records}
    by_fact: dict[str, list[dict]] = {}
    for lab in labels:
        if lab.get("fact_id") and not lab.get("checker"):
            by_fact.setdefault(lab["fact_id"], []).append(lab)
    canary_owner = {lab["canary"]: lab["persona_id"] for lab in labels if lab.get("canary")}
    for lab in labels:
        ck = lab.get("checker")
        if not ck:
            continue
        if lab.get("wording") == "heldout" and not (lab["persona_split"] in ("test", "ood_test")
                                                     and lab["split"] == "test"):
            found.append(f"heldout_wording_outside_test: {lab['event_id']}")
        if ck["kind"] in ("recall", "update", "deletion"):
            ev = [x for x in by_fact.get(lab["fact_id"], []) if (x["day"], x["minute"]) < (lab["day"], lab["minute"])]
            if not any(x["provenance"].startswith("fact:") for x in ev):
                found.append(f"evidence_after_task: {lab['event_id']}")
        text = rec[lab["event_id"]]["text"].lower()
        answers = [ck["args"].get("expected")] if ck["kind"] in ("recall", "update") else []
        if any(a and a.lower() in text for a in answers):
            found.append(f"task_contains_answer: {lab['event_id']}")
    for r in records:
        for canary, owner in canary_owner.items():
            if canary in r["text"] and r["persona_id"] != owner:
                found.append(f"canary_outside_owner: {canary} in {r['event_id']}")
    lsh, docs = MinHashLSH(threshold=0.8, num_perm=128), {}
    for lab in labels:
        r = rec[lab["event_id"]]
        if lab.get("fact_id") and not lab.get("checker") and len(r["text"].split()) >= 12:
            key = f"{lab['persona_split']}|{lab['event_id']}"
            docs[key] = _minhash(r["text"])
            lsh.insert(key, docs[key])
    pairs = set()
    for key, m in docs.items():
        for other in lsh.query(m):
            if key.split("|")[0] != other.split("|")[0]:
                pairs.add(" <-> ".join(sorted([key, other])))
    found += [f"near_duplicate_across_splits: {p}" for p in sorted(pairs)]
    return found
