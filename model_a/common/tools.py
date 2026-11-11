"""Tool dispatcher: the only path from a model's tool call to persona data.

- Validates calls against tool_registry.json (unknown tool / unknown arg / missing arg / wrong type).
- Refuses tools whose permission is "requires_confirmation" without touching the connector,
  and logs a boundedness violation (CLAUDE.md rule 6).
- Applies the time rule: messages after reference_time are invisible; calendar is not filtered.

Every model uses this same dispatcher, so tool semantics never differ between models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from model_a.common.connectors import LocalBenchmarkConnector, persist_profile
from model_a.common.schemas import StepStatus, ToolResult
from model_a.ingestion.ingest import render_event, render_message
from model_a.ingestion.parsers import ParsedFile, file_record_id, parse_file

MAX_OBSERVATION_CHARS = 6000
MAX_MESSAGES = 25

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"the", "and", "for", "with", "from", "about", "any", "all", "my", "our", "meeting", "event", "events"}


@dataclass(frozen=True)
class ArgSpec:
    name: str
    kind: str  # string | boolean | int_array | any
    optional: bool


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    permission: str
    args: dict[str, ArgSpec]


def parse_registry(registry: dict[str, Any]) -> dict[str, ToolSpec]:
    """Turn the registry's prose types ("string (optional, ...)") into checkable specs."""
    specs = {}
    for tool in registry["tools"]:
        args = {}
        for name, desc in tool["arguments"].items():
            d = desc.lower()
            if d.startswith("string"):
                kind = "string"
            elif d.startswith("boolean"):
                kind = "boolean"
            elif d.startswith("array of integers"):
                kind = "int_array"
            elif d.startswith("any"):
                kind = "any"
            else:
                raise ValueError(f"Unrecognized arg type for {tool['name']}.{name}: {desc!r}")
            args[name] = ArgSpec(name=name, kind=kind, optional="optional" in d)
        specs[tool["name"]] = ToolSpec(
            name=tool["name"], description=tool["description"], permission=tool["permission"], args=args
        )
    return specs


def render_tools(registry: dict[str, Any]) -> str:
    """Shared tool listing for prompts (identical text for every model)."""
    lines = []
    for t in registry["tools"]:
        args = ", ".join(f"{k}: {v}" for k, v in t["arguments"].items())
        lines.append(f"- {t['name']}({args}) [permission: {t['permission']}]: {t['description']}")
    return "\n".join(lines)


class ArgError(ValueError):
    pass


def validate_args(spec: ToolSpec, args: Any) -> None:
    if not isinstance(args, dict):
        raise ArgError(f"arguments must be a JSON object, got {type(args).__name__}")
    unknown = set(args) - set(spec.args)
    if unknown:
        raise ArgError(f"unknown argument(s) {sorted(unknown)}; allowed: {sorted(spec.args)}")
    for name, arg in spec.args.items():
        if name not in args or args[name] is None:
            if not arg.optional:
                raise ArgError(f"missing required argument '{name}'")
            continue
        value = args[name]
        ok = {
            "string": isinstance(value, str),
            "boolean": isinstance(value, bool),
            "int_array": isinstance(value, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in value),
            "any": True,
        }[arg.kind]
        if not ok:
            raise ArgError(f"argument '{name}' must be {arg.kind}, got {type(value).__name__}")


class ToolDispatcher:
    def __init__(
        self,
        connector: LocalBenchmarkConnector,
        registry: dict[str, Any],
        reference_time: str,
        parse: Callable[[Path], ParsedFile] = parse_file,
    ) -> None:
        self.conn = connector
        self.specs = parse_registry(registry)
        self.reference_time = reference_time
        self._parse = parse
        self.violations: list[dict[str, Any]] = []
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            "read_local_file": self._read_local_file,
            "search_calendar": self._search_calendar,
            "read_messages": self._read_messages,
            "lookup_contact": self._lookup_contact,
            "create_calendar_event": self._create_calendar_event,
            "draft_email": self._draft_email,
            "draft_sms": self._draft_sms,
            "update_profile": self._update_profile,
        }

    def dispatch(self, tool: str, args: Any) -> ToolResult:
        args_dict = args if isinstance(args, dict) else {"_raw": args}
        spec = self.specs.get(tool)
        if spec is None:
            return self._fail(tool, args_dict, f"unknown tool '{tool}'. Available: {sorted(self.specs)}")
        if spec.permission == "requires_confirmation":
            violation = f"{tool} requires user confirmation; refused by dispatcher"
            self.violations.append({"type": "boundedness", "tool": tool, "args": args_dict, "detail": violation})
            return ToolResult(
                tool=tool,
                args=args_dict,
                status=StepStatus.REFUSED,
                observation=f"REFUSED: {violation}. Outbound actions must stay as drafts.",
                violation=violation,
            )
        try:
            validate_args(spec, args)
        except ArgError as exc:
            return self._fail(tool, args_dict, f"invalid arguments: {exc}")
        handler = self._handlers.get(tool)
        if handler is None:  # registry tool with autonomous permission but no implementation
            return self._fail(tool, args_dict, f"tool '{tool}' is not implemented by the dispatcher")
        try:
            result = handler(args)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            return self._fail(tool, args_dict, f"{type(exc).__name__}: {exc}")
        if len(result.observation) > MAX_OBSERVATION_CHARS:
            result = result.model_copy(
                update={"observation": result.observation[:MAX_OBSERVATION_CHARS] + "\n[... truncated]"}
            )
        return result

    # ---- handlers -------------------------------------------------------------------------

    def _read_local_file(self, args: dict[str, Any]) -> ToolResult:
        rel = args["path"].strip().lstrip("/")
        if rel.startswith("local_files/"):
            rel = rel[len("local_files/") :]
        if "/" in rel or "\\" in rel or rel in {"", ".", ".."}:
            raise ValueError(f"path must name a file directly inside local_files/: {args['path']!r}")
        path = Path(self.conn.local_files_dir) / rel
        if not path.is_file():
            available = ", ".join(self.conn.list_files())
            raise FileNotFoundError(f"local_files/{rel} does not exist. Available: {available}")
        parsed = self._parse(path)
        ids = [file_record_id(path.name, c.index) for c in parsed.chunks]
        return self._ok("read_local_file", args, f"Contents of local_files/{path.name}:\n{parsed.text}", ids)

    def _search_calendar(self, args: dict[str, Any]) -> ToolResult:
        date = args.get("date")
        if date and not _DATE.match(date):
            raise ValueError(f"date must be YYYY-MM-DD, got {date!r}")
        query = args.get("query")
        # Read the connector's live event list (includes events created this task); its own
        # search_events only does single-substring title matching.
        events = [e for e in self.conn._events if not date or e["date"] == date]
        if query:
            events = [e for e in events if _matches(query, " ".join(str(e.get(k, "")) for k in ("title", "notes", "location", "category")))]
        events.sort(key=lambda e: (e["date"], e["start_time"]))
        if not events:
            return self._ok("search_calendar", args, "No calendar events found.", [])
        lines = [f"[{e['id']}] {render_event(e)}" for e in events]
        return self._ok("search_calendar", args, f"{len(events)} event(s):\n" + "\n".join(lines), [e["id"] for e in events])

    def _read_messages(self, args: dict[str, Any]) -> ToolResult:
        raw = self.conn.search_messages(reference_time=self.reference_time)  # rule 7 cutoff
        msgs = [m.__dict__ for m in raw]
        if args.get("contact"):
            terms = self._contact_terms(args["contact"])
            msgs = [m for m in msgs if any(t in f"{m['sender']} {m['recipient']}".lower() for t in terms)]
        if args.get("query"):
            msgs = [m for m in msgs if _matches(args["query"], f"{m['subject']} {m['body']}")]
        if args.get("day_range"):
            rng = args["day_range"]
            if len(rng) != 2:
                raise ValueError("day_range must be [start_day, end_day]")
            msgs = [m for m in msgs if rng[0] <= int(m["timestamp"][8:10]) <= rng[1]]
        msgs.sort(key=lambda m: m["timestamp"])
        note = ""
        if len(msgs) > MAX_MESSAGES:
            note = f" (showing latest {MAX_MESSAGES} of {len(msgs)})"
            msgs = msgs[-MAX_MESSAGES:]
        if not msgs:
            return self._ok("read_messages", args, "No messages found.", [])
        lines = [f"[{m['id']}] {render_message(m)}" for m in msgs]
        return self._ok("read_messages", args, f"{len(msgs)} message(s){note}:\n" + "\n".join(lines), [m["id"] for m in msgs])

    def _lookup_contact(self, args: dict[str, Any]) -> ToolResult:
        contact = self._find_contact(args["name"])
        if contact is None:
            names = ", ".join(c["name"] for c in self.conn.get_profile().get("contacts", []))
            return self._ok("lookup_contact", args, f"No contact matches {args['name']!r}. Known contacts: {names}", [])
        fields = "; ".join(f"{k}: {contact[k]}" for k in ("id", "name", "role", "relation", "email", "phone", "preferred_channel") if contact.get(k))
        return self._ok("lookup_contact", args, f"Contact found: {fields}", [f"contact_{contact['id']}"])

    def _create_calendar_event(self, args: dict[str, Any]) -> ToolResult:
        if not _DATE.match(args["date"]):
            raise ValueError(f"date must be YYYY-MM-DD, got {args['date']!r}")
        for key in ("start_time", "end_time"):
            if not _TIME.match(args[key]):
                raise ValueError(f"{key} must be HH:MM (24h), got {args[key]!r}")
        if args["end_time"] <= args["start_time"]:
            raise ValueError("end_time must be after start_time")
        event = self.conn.create_event(
            title=args["title"], date=args["date"], start_time=args["start_time"],
            end_time=args["end_time"], busy=args["busy"], notes=args.get("notes", ""),
        )
        obs = f"Created calendar event {event.id}: {event.title} on {event.weekday} {event.date} {event.start_time}-{event.end_time}"
        return self._ok("create_calendar_event", args, obs, [event.id])

    def _draft_email(self, args: dict[str, Any]) -> ToolResult:
        if "@" not in args["to"]:
            raise ValueError(f"'to' must be an email address, got {args['to']!r}")
        draft = self.conn.create_draft(to=args["to"], subject=args["subject"], body=args["body"], channel="email")
        return self._draft_result("draft_email", args, draft.draft_id, f"Email draft {draft.draft_id} to {args['to']} saved (DRAFT, not sent).")

    def _draft_sms(self, args: dict[str, Any]) -> ToolResult:
        if sum(ch.isdigit() for ch in args["to"]) < 7:
            raise ValueError(f"'to' must be a phone number, got {args['to']!r}")
        draft = self.conn.create_draft(to=args["to"], subject="", body=args["body"], channel="sms")
        return self._draft_result("draft_sms", args, draft.draft_id, f"SMS draft {draft.draft_id} to {args['to']} saved (DRAFT, not sent).")

    def _update_profile(self, args: dict[str, Any]) -> ToolResult:
        field = args["field"].strip()
        if not field or any(not part for part in field.split(".")):
            raise ValueError(f"field must be a dot path like preferences.meeting_window, got {field!r}")
        self.conn.update_profile(field, args["value"])
        persist_profile(self.conn)
        return self._ok("update_profile", args, f"Profile field {field} set to {args['value']!r}.", ["profile_updated"])

    # ---- helpers --------------------------------------------------------------------------

    def _contacts(self) -> list[dict[str, Any]]:
        return self.conn.get_profile().get("contacts", [])

    def _find_contact(self, query: str) -> dict[str, Any] | None:
        """Match by id, email, phone, or name tokens (the connector alone misses 'Marcus' vs id)."""
        q = query.strip().lower()
        for c in self._contacts():
            if q in {c["id"].lower(), c.get("email", "").lower(), (c.get("phone") or "").lower()}:
                return c
        tokens = [t for t in _WORD.findall(q.replace("dr.", "")) if t not in {"dr", "mr", "ms", "mrs"}]
        if not tokens:
            return None
        for c in self._contacts():
            hay = f"{c['id']} {c['name']} {c.get('role', '')}".lower()
            if all(t in hay for t in tokens):
                return c
        return None

    def _contact_terms(self, query: str) -> list[str]:
        contact = self._find_contact(query)
        if contact is None:
            return [query.strip().lower()]
        return [t.lower() for t in (contact["name"], contact.get("email"), contact.get("phone")) if t]

    def _ok(self, tool: str, args: dict[str, Any], observation: str, ids: list[str]) -> ToolResult:
        return ToolResult(tool=tool, args=args, status=StepStatus.EXECUTED, observation=observation, evidence_ids=ids)

    def _draft_result(self, tool: str, args: dict[str, Any], draft_id: str, observation: str) -> ToolResult:
        return ToolResult(tool=tool, args=args, status=StepStatus.DRAFT, observation=observation, evidence_ids=[draft_id])

    def _fail(self, tool: str, args: dict[str, Any], message: str) -> ToolResult:
        return ToolResult(tool=tool, args=args, status=StepStatus.FAILED, observation=f"ERROR: {message}")


def _matches(query: str, text: str) -> bool:
    """Any meaningful query word appears in text (LLMs pass phrases, not exact titles)."""
    words = [w for w in _WORD.findall(query.lower()) if len(w) > 2 and w not in _STOPWORDS]
    text_l = text.lower()
    if not words:
        return query.lower() in text_l
    return any(w in text_l for w in words)
