"""Wrap the supplied connectors.py; keep writes in a separate derived journal."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import types
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .core import GuardPlanError, outside_source, read_json, timestamp, write_json
from .constraints import minutes


def load_connectors(data_root: Path):
    path = data_root / "connectors.py"
    key = "_guardplan_connectors_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    if key not in sys.modules:
        # Execute the user's adapter without writing __pycache__ in Project Data.
        module = types.ModuleType(key)
        module.__file__ = str(path)
        sys.modules[key] = module
        try:
            exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
        except Exception:
            del sys.modules[key]
            raise
    return sys.modules[key]


class DatasetAdapter:
    """All persona data operations delegate to LocalBenchmarkConnector.

    The supplied API has no privacy-policy/task/registry methods; those metadata
    reads are isolated here. Mutating connector methods modify memory only.
    A reference-time journal makes those mutations persistent outside the source.
    One adapter per run; concurrent writers to a session are not supported.
    """

    def __init__(self, data_root: Path, persona_id: str, workspace: Path,
                 reference_time: str, session: str = "default"):
        self.data_root = data_root.resolve()
        if persona_id not in self.personas(self.data_root):
            raise GuardPlanError("Unknown persona")
        if not session or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in session):
            raise GuardPlanError("Session must contain only letters, digits, underscore or hyphen")
        self.persona_id, self.reference_time = persona_id, reference_time
        timestamp(reference_time)
        self.workspace = outside_source(self.data_root, workspace)
        self.state_dir = self.workspace / "state" / session / persona_id
        self.connector = load_connectors(self.data_root).get_connector(
            persona_id, mode="benchmark", data_root=str(self.data_root))
        self.created_event_ids = {}
        self.journal = []
        journal_path = self.state_dir / "journal.json"
        if journal_path.exists():
            self.journal = read_json(journal_path)
        for entry in self.journal:
            if timestamp(entry["reference_time"]) <= timestamp(reference_time):
                self._apply(entry["tool"], entry["args"])

    @staticmethod
    def personas(root: Path) -> list[str]:
        return sorted(p.name for p in root.glob("persona_*") if p.is_dir())

    def get_profile(self) -> dict:
        return copy.deepcopy(self.connector.get_profile())

    def privacy_policy(self) -> dict:
        return read_json(self.data_root / self.persona_id / "agent_memory/privacy_policy.json")

    def tasks(self) -> list[dict]:
        return read_json(self.data_root / self.persona_id / "test_suite/eval_tasks.json")

    def registry(self) -> dict[str, dict]:
        return {t["name"]: t for t in read_json(self.data_root / "tool_registry.json")["tools"]}

    def list_files(self) -> list[str]:
        return self.connector.list_files()

    def read_file(self, relative_path: str) -> bytes:
        # Tighten the basename-based supplied adapter: reject traversal and symlinks.
        if relative_path not in self.list_files():
            raise GuardPlanError("File must match a listed local_files path")
        base = (self.data_root / self.persona_id / "local_files").resolve()
        resolved = (self.data_root / self.persona_id / relative_path).resolve()
        if base not in resolved.parents:
            raise GuardPlanError("File points outside persona local_files")
        return self.connector.read_file(relative_path)

    def search_events(self, date=None, query=None) -> list[dict]:
        events = [asdict(e) for e in self.connector.search_events(
            date=date, query=query, reference_time=self.reference_time)]
        for event in events:
            identifier = self.created_event_ids.get(self._event_key(event))
            if identifier:
                event["id"] = identifier
        return events

    @staticmethod
    def _event_key(event: dict) -> tuple:
        return tuple(event[k] for k in ("title", "date", "start_time", "end_time"))

    def search_messages(self, contact=None, query=None, day_range=None) -> list[dict]:
        # Apply temporal filtering ourselves too: the supplied connector compares
        # ISO strings, which is sufficient for the dataset's naive timestamps.
        messages = [asdict(m) for m in self.connector.search_messages(
            contact=contact, query=query, reference_time=self.reference_time)]
        messages = [m for m in messages if timestamp(m["timestamp"]) <= timestamp(self.reference_time)]
        if day_range is not None:
            start, end = day_range
            messages = [m for m in messages if start <= timestamp(m["timestamp"]).day <= end]
        return messages

    def unfiltered_messages_for_ablation(self) -> list[dict]:
        return [asdict(m) for m in self.connector.search_messages(reference_time=None)]

    def lookup_contact(self, name: str) -> dict | None:
        try:
            result = self.connector.lookup_contact(name)
        except KeyError:
            # The supplied Contact mapper assumes email exists, although some
            # phone-only contacts have no email. Read its profile API as fallback.
            terms = name.casefold().replace(".", "").split()
            matches = [c for c in self.get_profile().get("contacts", [])
                       if all(t in json.dumps(c).casefold().replace(".", "") for t in terms)]
            return matches[0] if len(matches) == 1 else None
        return asdict(result) if result else None

    def contact_id(self, address: str) -> str | None:
        matches = [c["id"] for c in self.get_profile().get("contacts", [])
                   if address.casefold() in {str(c.get(k, "")).casefold() for k in ("id", "email", "phone")}
                   and address]
        return matches[0] if len(matches) == 1 else None

    def _apply(self, tool: str, args: dict) -> Any:
        if tool == "update_profile":
            if not args["field"].startswith(("preferences.", "routines.", "contacts.")):
                raise GuardPlanError("Profile updates must target preferences, routines or contacts")
            if args["field"] == "preferences.earliest_meeting_time":
                if not isinstance(args["value"], str) or not re.fullmatch(r"\d{2}:\d{2}", args["value"]):
                    raise GuardPlanError("earliest_meeting_time must be HH:MM")
                minutes(args["value"])
            if args["field"] == "preferences.meeting_window":
                value = args["value"]
                if isinstance(value, dict) and {"start", "end"} <= value.keys():
                    if minutes(value["start"]) >= minutes(value["end"]):
                        raise GuardPlanError("Meeting window must have start before end")
                elif value not in ("afternoon", "any", "prep_period_and_after_school"):
                    raise GuardPlanError("Unsupported meeting_window. For a no-meetings-before rule, update preferences.earliest_meeting_time to HH:MM")
            return self.connector.update_profile(args["field"], copy.deepcopy(args["value"]))
        if tool == "create_calendar_event":
            result = asdict(self.connector.create_event(**args))
            # len(events)+1 is not unique when source event IDs have gaps.
            # Map only derived events to stable IDs through the public API.
            key = self._event_key(result)
            identifier = "gp_cal_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
            self.created_event_ids[key] = identifier
            result["id"] = identifier
            return result
        raise GuardPlanError("Unsupported state mutation")

    def mutate(self, tool: str, args: dict) -> Any:
        if self.journal and timestamp(self.reference_time) < max(timestamp(e["reference_time"]) for e in self.journal):
            raise GuardPlanError("Cannot write into a session's past; use a fresh session")
        if tool == "create_calendar_event":
            for event in self.search_events(date=args["date"]):
                if all(event.get(k) == args.get(k) for k in ("title", "start_time", "end_time")):
                    return event  # idempotent calendar blocking
        result = self._apply(tool, args)
        self.journal.append({"reference_time": self.reference_time, "tool": tool, "args": copy.deepcopy(args)})
        write_json(self.state_dir / "journal.json", self.journal)
        return result

    def create_draft(self, args: dict, channel: str) -> dict:
        draft = asdict(self.connector.create_draft(
            to=args["to"], subject=args.get("subject", ""), body=args["body"], channel=channel))
        # The connector counter resets per process; use a unique persisted ID.
        draft["draft_id"] = "draft_" + uuid.uuid4().hex[:12]
        draft["created_at"] = self.reference_time
        write_json(self.state_dir / "drafts" / (draft["draft_id"] + ".json"), draft)
        return draft
