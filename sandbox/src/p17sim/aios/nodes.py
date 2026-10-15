"""The node library. Each node type says what it does, where it runs (local or cloud), which tool
permission it needs, whether it can be undone, and how its result reads to an agent.

`run(ctx, params)` returns (output, data_class). The engine decides whether the node may run at all
(location and tool policy) and logs it; nodes only do their one job.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable

from p17sim.aios.privacy import max_class

RESULT_CHARS = 400


@dataclass
class RunContext:
    ws: Any  # p17sim.aios.workspace.Workspace
    execution_id: str
    node_id: str = ""
    upstream_class: str = "public"


@dataclass
class NodeSpec:
    type: str
    category: str  # trigger | context | reason | logic | action | human | output | cloud
    description: str
    params: dict[str, str]
    run: Callable[[RunContext, dict], tuple[dict, str]]
    location: str = "local"
    tool: str | None = None
    reversible: bool = False
    undo: Callable[[RunContext, dict], None] | None = None
    display: Callable[[dict], str] = field(default=lambda out: json.dumps(out, default=str))


NODES: dict[str, NodeSpec] = {}


def node(spec: NodeSpec) -> NodeSpec:
    NODES[spec.type] = spec
    return spec


# ------------------------------------------------------------------ helpers

def row_view(r: dict) -> dict:
    return {"id": r["id"], "time": r["ts"][:16], "from": r.get("sender"), "kind": r["channel"],
            "permission": r.get("permission"), "text": r["text"][:RESULT_CHARS]}


def class_of(rows: list[dict]) -> str:
    return max_class("public", *(r.get("data_class") or "personal" for r in rows))


def resolve_contact(contacts: list[dict], name: str) -> tuple[dict | None, str]:
    """A contact for a name the agent or a template wrote. None + a message when unknown or ambiguous."""
    n = (name or "").strip().lower()
    if not n:
        return None, "No recipient given."
    exact = [c for c in contacts if c["name"].lower() == n or c["id"] == name]
    if len(exact) == 1:
        return exact[0], exact[0]["name"]
    parts = set(n.replace(".", " ").split())
    partial = [c for c in contacts if parts & set(c["name"].lower().replace(".", " ").split())]
    if len(partial) == 1:
        return partial[0], partial[0]["name"]
    if len(partial) > 1:
        return None, "Ambiguous recipient; did you mean: " + ", ".join(c["name"] for c in partial) + "?"
    close = difflib.get_close_matches(n, [c["name"].lower() for c in contacts], n=1, cutoff=0.75)
    if close:
        c = next(c for c in contacts if c["name"].lower() == close[0])
        return c, c["name"]
    return None, f"No contact named '{name}'. Use list_contacts to see names."


def parse_date(value: str) -> date:
    return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date()


def parse_time(value: str) -> int:
    v = str(value).strip().lower().replace(".", "")
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


KIND_TO_CHANNELS = {"all": None, "message": {"message", "notification"}, "note": {"note"},
                    "calendar": {"calendar"}, "preference": {"preference"}, "instruction": {"instruction"}}
SEARCHABLE = {"message", "notification", "note", "calendar", "preference", "instruction"}


# ------------------------------------------------------------------ triggers

node(NodeSpec("trigger.chat", "trigger", "A goal typed by the user.", {"goal": "the request"},
              run=lambda ctx, p: ({"goal": p.get("goal", "")}, "personal")))
node(NodeSpec("trigger.time", "trigger", "Fires at a time of day (daily) or on weekdays.",
              {"at": "HH:MM", "weekdays": "list of 0-6, empty = every day"},
              run=lambda ctx, p: ({"time": ctx.ws.now.isoformat()}, "public")))
node(NodeSpec("trigger.event", "trigger", "Fires when a record arrives that matches.",
              {"channel": "message|note|calendar", "sender": "optional sender name"},
              run=lambda ctx, p: ({"record": p.get("record")}, "personal")))


# ------------------------------------------------------------------ context

def _search(ctx: RunContext, p: dict) -> tuple[dict, str]:
    channels = KIND_TO_CHANNELS.get(p.get("kind", "all"), None) or SEARCHABLE
    rows = ctx.ws.store.search(ctx.ws.pid, str(p.get("query", "")), channels, int(p.get("limit", 8) or 8),
                               before=ctx.ws.now.isoformat())
    return {"items": [row_view(r) for r in rows]}, class_of(rows)


def _read(ctx: RunContext, p: dict) -> tuple[dict, str]:
    r = ctx.ws.store.record(str(p.get("id", "")))
    if r is None or r["deleted"] or r["persona_id"] != ctx.ws.pid or r["ts"] >= ctx.ws.now.isoformat():
        return {"record": None, "error": f"No record {p.get('id')}."}, "public"
    return {"record": {**row_view(r), "text": r["text"]}}, class_of([r])


def _recent(ctx: RunContext, p: dict) -> tuple[dict, str]:
    since = (ctx.ws.now - timedelta(days=float(p.get("days", 7)))).isoformat()
    channels = KIND_TO_CHANNELS.get(p.get("kind", "all"), None) or {"message", "note", "calendar"}
    rows = [r for r in ctx.ws.store.visible(ctx.ws.pid, before=ctx.ws.now.isoformat(), channels=channels)
            if r["ts"] >= since]
    if p.get("exclude_private"):
        rows = [r for r in rows if r.get("permission") != "deny"]
    rows = rows[-int(p.get("limit", 20)):]
    return {"items": [row_view(r) for r in rows]}, class_of(rows)


def _calendar(ctx: RunContext, p: dict) -> tuple[dict, str]:
    d = parse_date(p.get("date") or ctx.ws.now.date().isoformat())
    return {"date": d.isoformat(), "entries": ctx.ws.store.calendar_on(ctx.ws.pid, d)}, "personal"


def _contacts(ctx: RunContext, p: dict) -> tuple[dict, str]:
    return {"contacts": [{"name": c["name"], "relation": c["relation"]} for c in ctx.ws.store.contacts(ctx.ws.pid)]}, \
        "personal"


def _preferences(ctx: RunContext, p: dict) -> tuple[dict, str]:
    since = (ctx.ws.now - timedelta(days=float(p.get("days", 14)))).isoformat()
    rows = [r for r in ctx.ws.store.visible(ctx.ws.pid, before=ctx.ws.now.isoformat(), channels={"preference"})
            if r["ts"] >= since and (not p.get("key") or str(p["key"]).replace("_", " ").split()[0] in r["text"].lower())]
    return {"signals": [{"time": r["ts"][:16], "text": r["text"]} for r in rows]}, "personal"


node(NodeSpec("context.search", "context", "Keyword search over messages, notes, calendar and preferences.",
              {"query": "keywords", "kind": "all|message|note|calendar|preference|instruction", "limit": "max rows"},
              run=_search, tool="mail.read", display=lambda o: json.dumps(o["items"]) if o["items"] else
              "No matching records."))
node(NodeSpec("context.read", "context", "Read one record in full.", {"id": "record id"}, run=_read, tool="mail.read",
              display=lambda o: json.dumps(o["record"]) if o["record"] else o["error"]))
node(NodeSpec("context.recent", "context", "Records from the last N days.",
              {"days": "number of days", "kind": "all|message|note|calendar", "exclude_private": "true to skip deny records",
               "limit": "max rows"}, run=_recent, tool="mail.read", display=lambda o: json.dumps(o["items"])))
node(NodeSpec("context.calendar", "context", "Calendar entries on a date.", {"date": "YYYY-MM-DD"}, run=_calendar,
              tool="calendar.read", display=lambda o: json.dumps(o["entries"])))
node(NodeSpec("context.contacts", "context", "The user's contacts.", {}, run=_contacts,
              display=lambda o: json.dumps(o["contacts"])))
node(NodeSpec("context.preferences", "context", "Recent preference signals.", {"key": "e.g. meeting_window",
                                                                                "days": "how far back"},
              run=_preferences, display=lambda o: json.dumps(o["signals"])))


# ------------------------------------------------------------------ reason

def _llm(ctx: RunContext, p: dict) -> tuple[dict, str]:
    msgs = ([{"role": "system", "content": str(p["system"])}] if p.get("system") else []) + \
           [{"role": "user", "content": str(p.get("prompt", ""))}]
    r = ctx.ws.llm.chat(msgs, component=ctx.node_id or "reason.llm", json_mode=bool(p.get("json")),
                        max_tokens=int(p.get("max_tokens", 600)))
    parsed = None
    if p.get("json"):
        try:
            parsed = json.loads(r["content"])
        except json.JSONDecodeError:
            parsed = None
    return {"text": r["content"].strip(), "json": parsed}, ctx.upstream_class


node(NodeSpec("reason.llm", "reason", "Ask the local model (qwen3.5:4b).",
              {"prompt": "text, may reference upstream outputs", "system": "optional", "json": "true for JSON output",
               "max_tokens": "output cap"}, run=_llm, display=lambda o: o["text"]))
node(NodeSpec("reason.agent_step", "reason", "One step of an agent loop (recorded, not re-run).",
              {"turn": "loop turn"}, run=lambda ctx, p: (p, ctx.upstream_class),
              display=lambda o: o.get("content", "")))


# ------------------------------------------------------------------ logic

def _if(ctx: RunContext, p: dict) -> tuple[dict, str]:
    left, op, right = p.get("left"), p.get("op", "eq"), p.get("right")
    tests = {"eq": lambda: left == right, "ne": lambda: left != right,
             "contains": lambda: str(right).lower() in str(left).lower(),
             "empty": lambda: not left, "nonempty": lambda: bool(left),
             "gt": lambda: float(left) > float(right), "lt": lambda: float(left) < float(right)}
    if op not in tests:
        raise ValueError(f"unknown operator {op}")
    return {"result": bool(tests[op]())}, ctx.upstream_class


node(NodeSpec("logic.if", "logic", "Branch on a condition; outgoing edges carry branch true/false.",
              {"left": "value", "op": "eq|ne|contains|empty|nonempty|gt|lt", "right": "value"}, run=_if))
node(NodeSpec("logic.merge", "logic", "Combine upstream lists into one.", {"inputs": "list of references"},
              run=lambda ctx, p: ({"items": [x for part in p.get("inputs", []) for x in (part if isinstance(part, list)
                                                                                          else [part])]},
                                  ctx.upstream_class)))


# ------------------------------------------------------------------ actions

def _send(ctx: RunContext, p: dict) -> tuple[dict, str]:
    contact, who = resolve_contact(ctx.ws.store.contacts(ctx.ws.pid), str(p.get("to", "")))
    if contact is None:
        raise ValueError(who)
    ctx.ws.store.add_outbox({"persona_id": ctx.ws.pid, "ts": ctx.ws.now.isoformat(), "to_id": contact["id"],
                             "to_name": contact["name"], "subject": str(p.get("subject", "")),
                             "body": str(p.get("body", "")), "execution_id": ctx.execution_id,
                             "approved": p.get("_approved", False)})
    return {"sent": True, "to": contact["name"]}, ctx.upstream_class


def _create(ctx: RunContext, p: dict) -> tuple[dict, str]:
    contact, who = resolve_contact(ctx.ws.store.contacts(ctx.ws.pid), str(p.get("with_contact", "")))
    if contact is None:
        raise ValueError(who)
    d, s, dur = parse_date(p.get("date")), parse_time(p.get("start")), int(p.get("duration_minutes", 30))
    if dur <= 0 or s + dur > 24 * 60:
        raise ValueError("the meeting must start and end on the same day")
    eid = f"new-{uuid.uuid4().hex[:8]}"
    title = str(p.get("title") or f"Meeting with {contact['name']}")
    ctx.ws.store.add_event({"id": eid, "persona_id": ctx.ws.pid, "date": d.isoformat(), "start_min": s, "duration": dur,
                            "with_id": contact["id"], "with_name": contact["name"], "title": title,
                            "execution_id": ctx.execution_id, "ts": ctx.ws.now.isoformat()})
    return {"event_id": eid, "date": d.isoformat(), "start": f"{s // 60:02d}:{s % 60:02d}",
            "end": f"{(s + dur) // 60:02d}:{(s + dur) % 60:02d}", "with": contact["name"], "title": title}, \
        ctx.upstream_class


def _cancel(ctx: RunContext, p: dict) -> tuple[dict, str]:
    ok = ctx.ws.store.cancel_event(str(p.get("event_id", "")))
    if not ok:
        raise ValueError(f"no event {p.get('event_id')} created by the assistant")
    return {"cancelled": p.get("event_id")}, "public"


def _delete(ctx: RunContext, p: dict) -> tuple[dict, str]:
    ids = p.get("ids", [])
    if isinstance(ids, str):
        ids = [i for i in re.split(r"[,\s]+", ids) if i]
    receipt = ctx.ws.deletion.delete(ctx.ws.pid, [str(i) for i in ids], str(p.get("request", "")), actor="assistant",
                                     ts=ctx.ws.now.isoformat())
    return {"deleted": receipt["record_ids"], "receipt": receipt["id"]}, "public"


node(NodeSpec("action.send_email", "action", "Send an email to a contact (cannot be undone).",
              {"to": "contact name", "subject": "", "body": ""}, run=_send, tool="mail.send",
              display=lambda o: f"Email sent to {o['to']}."))
node(NodeSpec("action.create_event", "action", "Create a calendar event with a contact.",
              {"date": "YYYY-MM-DD", "start": "HH:MM", "duration_minutes": "minutes", "with_contact": "name",
               "title": "optional"}, run=_create, tool="calendar.create", reversible=True,
              undo=lambda ctx, out: ctx.ws.store.cancel_event(out["event_id"]),
              display=lambda o: f"Created {o['event_id']}: {o['title']} on {o['date']} {o['start']}-{o['end']}."))
node(NodeSpec("action.cancel_event", "action", "Cancel an event the assistant created.", {"event_id": "id"},
              run=_cancel, tool="calendar.create", display=lambda o: f"Cancelled {o['cancelled']}."))
node(NodeSpec("action.delete_records", "action", "Permanently delete records (and every copy of what they said).",
              {"ids": "record ids", "request": "the user's words"}, run=_delete, tool="memory.forget",
              display=lambda o: (f"Deleted {len(o['deleted'])} record(s): {', '.join(o['deleted'])}."
                                 if o["deleted"] else "Nothing deleted (unknown ids).")))


# ------------------------------------------------------------------ human, output, cloud

node(NodeSpec("human.approval", "human", "Wait for the user to approve before continuing.", {"message": "what to approve"},
              run=lambda ctx, p: ({"approved": True, "message": p.get("message", "")}, ctx.upstream_class),
              tool="human.approval"))
node(NodeSpec("output.respond", "output", "The answer shown to the user.", {"text": "answer"},
              run=lambda ctx, p: ({"text": str(p.get("text", ""))}, ctx.upstream_class), display=lambda o: o["text"]))


def _cloud(ctx: RunContext, p: dict) -> tuple[dict, str]:
    url = os.environ.get("P17_CLOUD_URL")
    if not url:
        raise ValueError("no cloud endpoint configured (P17_CLOUD_URL)")
    body = json.dumps({"prompt": str(p.get("prompt", ""))}).encode()
    out = ctx.ws.gateway.post(url, body, purpose="cloud.llm", persona_id=ctx.ws.pid)
    return {"text": out.decode(errors="ignore")}, ctx.upstream_class


node(NodeSpec("cloud.llm", "cloud", "Call a cloud model. Only public data may be sent.", {"prompt": "text"},
              run=_cloud, location="cloud"))
