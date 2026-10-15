"""The shared tool shell: one persona's local data, the tools every agent calls, and the policy layer.

All agents (baseline, Model A and its ablations) see the same tools, the same records and the same
permissions. Only the method differs.

Policy layer:
- per-tool permissions come from the persona's permission records; `mail.send: draft` means each
  send waits for the user's OK (a simulated user approves and the shell counts one intervention);
- every tool call is written to a provenance log (actor, tool, arguments, result, time);
- `cancel_event` rolls back a created event; sends cannot be rolled back;
- deletion marks records deleted, removes them from the search index and writes a receipt.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from p17sim.schema import Persona
from p17sim.system.text import BM25, contains_value

KIND_TO_CHANNELS = {"all": None, "message": {"message", "notification"}, "note": {"note"},
                    "calendar": {"calendar"}, "preference": {"preference"}, "instruction": {"instruction"}}
HIDDEN_CHANNELS = {"permission"}
MAX_RESULT_CHARS = 400


@dataclass
class Shell:
    persona: Persona
    permissions: dict[str, str]
    now: datetime
    records: dict[str, dict] = field(default_factory=dict)  # visible, in arrival order
    deleted: set[str] = field(default_factory=set)
    receipts: list[dict] = field(default_factory=list)
    created: list[dict] = field(default_factory=list)
    sent: list[dict] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    confirmations: int = 0
    actor: str = "agent"

    def __post_init__(self):
        self.index = BM25()

    # ---------------------------------------------------------------- data in
    def observe(self, record: dict) -> None:
        """A record arrives (message received, note saved, preference signal...)."""
        if record["channel"] in HIDDEN_CHANNELS:
            return
        self.records[record["event_id"]] = record
        self.index.add(record["event_id"], f"{record.get('sender', '')} {record['text']}")

    def visible(self) -> list[dict]:
        return [r for rid, r in self.records.items() if rid not in self.deleted]

    # ---------------------------------------------------------------- helpers
    def resolve(self, name: str) -> tuple[str | None, str]:
        """Contact id for a name the agent wrote. Returns (id, message); id is None when unknown or ambiguous."""
        name_l = (name or "").strip().lower()
        if not name_l:
            return None, "No recipient given."
        exact = [c for c in self.persona.contacts if c.name.lower() == name_l or c.contact_id == name]
        if len(exact) == 1:
            return exact[0].contact_id, exact[0].name
        parts = set(name_l.replace(".", " ").split())
        partial = [c for c in self.persona.contacts if parts & set(c.name.lower().replace(".", " ").split())]
        if len(partial) == 1:
            return partial[0].contact_id, partial[0].name
        if len(partial) > 1:
            return None, "Ambiguous recipient; did you mean: " + ", ".join(c.name for c in partial) + "?"
        close = difflib.get_close_matches(name_l, [c.name.lower() for c in self.persona.contacts], n=1, cutoff=0.75)
        if close:
            c = next(c for c in self.persona.contacts if c.name.lower() == close[0])
            return c.contact_id, c.name
        return None, f"No contact named '{name}'. Use list_contacts to see names."

    def _record(self, tool: str, args: dict, result: str) -> None:
        self.log.append({"actor": self.actor, "tool": tool, "args": args, "result": result[:300],
                         "time": self.now.isoformat()})

    @staticmethod
    def _parse_date(value: str) -> date:
        return datetime.strptime(value.strip()[:10], "%Y-%m-%d").date()

    @staticmethod
    def _parse_time(value: str) -> int:
        v = value.strip().lower().replace(".", "")
        m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", v)
        if not m:
            raise ValueError(f"time '{value}' is not HH:MM")
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        if not (0 <= h < 24 and 0 <= mi < 60):
            raise ValueError(f"time '{value}' is out of range")
        return h * 60 + mi

    def calendar_on(self, day: date) -> list[dict]:
        out = []
        for r in self.visible():
            if r["channel"] != "calendar":
                continue
            start = datetime.fromisoformat(r["timestamp"])
            if start.date() == day:
                end = datetime.fromisoformat(r["end"]) if r.get("end") else start + timedelta(minutes=30)
                out.append({"id": r["event_id"], "start": start.strftime("%H:%M"), "end": end.strftime("%H:%M"),
                            "title": r["text"]})
        for e in self.created:
            if e["date"] == day.isoformat() and not e.get("cancelled"):
                out.append({"id": e["id"], "start": e["start"], "end": e["end"], "title": e["title"]})
        return sorted(out, key=lambda e: e["start"])

    # ---------------------------------------------------------------- tools
    def search_records(self, query: str, kind: str = "all", limit: int = 8) -> str:
        channels = KIND_TO_CHANNELS.get(kind, None)
        allowed = {r["event_id"] for r in self.visible() if channels is None or r["channel"] in channels}
        scored = self.index.scores(query, allowed)
        # Highest score first; on a tie, the newer record first.
        top = sorted(scored, key=lambda rid: (scored[rid], self.records[rid]["timestamp"]),
                     reverse=True)[:max(1, min(int(limit), 20))]
        rows = [{"id": rid, "time": self.records[rid]["timestamp"][:16], "from": self.records[rid].get("sender"),
                 "kind": self.records[rid]["channel"], "permission": self.records[rid]["permission_label"],
                 "text": self.records[rid]["text"][:MAX_RESULT_CHARS]} for rid in top]
        return json.dumps(rows) if rows else "No matching records."

    def read_record(self, id: str) -> str:
        r = self.records.get(id)
        if r is None or id in self.deleted:
            return f"No record {id}."
        return json.dumps({"id": id, "time": r["timestamp"][:16], "from": r.get("sender"), "kind": r["channel"],
                           "permission": r["permission_label"], "text": r["text"]})

    def list_contacts(self) -> str:
        return json.dumps([{"name": c.name, "relation": c.relation} for c in self.persona.contacts])

    def list_calendar(self, date: str) -> str:
        return json.dumps(self.calendar_on(self._parse_date(date))) or "[]"

    def create_event(self, date: str, start: str, duration_minutes: int, with_contact: str, title: str = "") -> str:
        cid, who = self.resolve(with_contact)
        if cid is None:
            return f"Error: {who}"
        d = self._parse_date(date)
        s = self._parse_time(start)
        dur = int(duration_minutes)
        if dur <= 0 or s + dur > 24 * 60:
            return "Error: the meeting must start and end on the same day."
        e = {"id": f"new-{len(self.created) + 1}", "date": d.isoformat(), "start": f"{s // 60:02d}:{s % 60:02d}",
             "end": f"{(s + dur) // 60:02d}:{(s + dur) % 60:02d}", "start_minute": s, "duration": dur,
             "with_id": cid, "with": who, "title": title or f"Meeting with {who}", "created_at": self.now.isoformat()}
        self.created.append(e)
        return f"Created {e['id']}: {e['title']} on {e['date']} {e['start']}-{e['end']}."

    def cancel_event(self, event_id: str) -> str:
        for e in self.created:
            if e["id"] == event_id and not e.get("cancelled"):
                e["cancelled"] = True
                return f"Cancelled {event_id}."
        return f"Error: no event {event_id} created by the assistant."

    def send_email(self, to: str, subject: str = "", body: str = "") -> str:
        cid, who = self.resolve(to)
        if cid is None:
            return f"Error: {who}"
        if self.permissions.get("mail.send", "send") == "deny":
            return "Error: sending mail is not permitted."
        needs_ok = self.permissions.get("mail.send", "send") == "draft"
        if needs_ok:
            self.confirmations += 1  # the simulated user reads the draft and approves it
        self.sent.append({"to_id": cid, "to": who, "subject": subject, "body": body, "time": self.now.isoformat(),
                          "confirmed_by_user": needs_ok})
        return f"Email sent to {who}." + (" (The user approved the draft first.)" if needs_ok else "")

    def delete_records(self, ids: list[str] | str) -> str:
        if isinstance(ids, str):
            ids = [i.strip() for i in re.split(r"[,\s]+", ids) if i.strip()]
        done = []
        for rid in ids:
            if rid in self.records and rid not in self.deleted:
                self.deleted.add(rid)
                self.index.remove(rid)
                self.receipts.append({"record": rid, "time": self.now.isoformat(), "by": self.actor})
                done.append(rid)
        return f"Deleted {len(done)} record(s): {', '.join(done)}." if done else "Nothing deleted (unknown ids)."

    # ---------------------------------------------------------------- dispatch
    TOOLS = {
        "search_records": ("Search the user's messages, notes, calendar entries and preference history by keywords. "
                           "Returns the best matches; newest information is not always first. 'permission: deny' "
                           "marks private records that may be read on the device but never sent to anyone.",
                           {"query": ("string", "keywords to search for"),
                            "kind": ("string", "one of all, message, note, calendar, preference, instruction"),
                            "limit": ("integer", "how many results, at most 20")}, ["query"]),
        "read_record": ("Read one record in full by its id.", {"id": ("string", "record id")}, ["id"]),
        "list_contacts": ("List the user's contacts.", {}, []),
        "list_calendar": ("List calendar entries on a date.", {"date": ("string", "YYYY-MM-DD")}, ["date"]),
        "create_event": ("Create a calendar event with a contact.",
                         {"date": ("string", "YYYY-MM-DD"), "start": ("string", "HH:MM, 24-hour"),
                          "duration_minutes": ("integer", "length in minutes"),
                          "with_contact": ("string", "contact name"), "title": ("string", "event title")},
                         ["date", "start", "duration_minutes", "with_contact"]),
        "cancel_event": ("Cancel an event the assistant created.", {"event_id": ("string", "id from create_event")},
                         ["event_id"]),
        "send_email": ("Send an email to a contact.", {"to": ("string", "contact name"), "subject": ("string", ""),
                                                        "body": ("string", "")}, ["to", "body"]),
        "delete_records": ("Permanently delete records by id, for example when the user asks you to forget something.",
                           {"ids": ("array", "record ids to delete")}, ["ids"]),
    }

    @classmethod
    def tool_schemas(cls) -> list[dict]:
        out = []
        for name, (desc, params, required) in cls.TOOLS.items():
            props = {}
            for p, (typ, d) in params.items():
                props[p] = {"type": typ, "description": d}
                if typ == "array":
                    props[p]["items"] = {"type": "string"}
            out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": {
                "type": "object", "properties": props, "required": required}}})
        return out

    @classmethod
    def tool_descriptions(cls) -> str:
        return "\n".join(f"- {n}({', '.join(p)}): {d}" for n, (d, p, _) in cls.TOOLS.items())

    def execute(self, name: str, args: dict[str, Any]) -> str:
        fn = getattr(self, name, None) if name in self.TOOLS else None
        if fn is None:
            result = f"Error: unknown tool '{name}'."
        else:
            try:
                result = fn(**{k: v for k, v in (args or {}).items() if k in self.TOOLS[name][1]})
            except (TypeError, ValueError) as exc:
                result = f"Error: {exc}"
        self._record(name, args or {}, result)
        return result

    # ---------------------------------------------------------------- audits
    def holds(self, value: str) -> list[str]:
        """Ids of visible records whose text still contains `value` (storage audit)."""
        return [r["event_id"] for r in self.visible() if contains_value(r["text"], value)]
