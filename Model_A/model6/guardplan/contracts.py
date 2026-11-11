"""Task permissions and postconditions derived from the original user request."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .constraints import TIME_PATTERN, mentioned_times, time_value
from .core import ContractError


def trusted_request(query: str) -> str:
    quoted = r'"[^"\n]*"|“[^”]*”|(?<!\w)\x27(?:[^\x27]|\x27(?=\w))*\x27(?!\w)|‘[^’]*’'
    return re.sub(quoted, lambda m: " " * len(m.group()), query)


def draft_tools(query: str) -> tuple[str, ...]:
    text = trusted_request(query)
    explicit = re.search(r"\bdraft\s+(?:(?:a|an|the|polite|brief|short)\s+)*"
                         r"(?:(?:reply|response)\s+(?:(?:by|via|as|an?|in)\s+)?)?(email|sms|text)\b|"
                         r"\b(?:reply|respond)\b[^.!?]{0,80}?\b(?:by|via|using)\s+(email|sms|text)\b", text, re.I)
    if explicit and not re.search(r"(?:don't|do not|never)\s*$", text[max(0, explicit.start()-16):explicit.start()], re.I):
        return ("draft_email",) if (explicit.group(1) or explicit.group(2)).lower() == "email" else ("draft_sms",)
    match = re.search(r"\bdraft\b[^.!?]{0,100}?\b(email|sms|text|message|reply|response)\b", text, re.I)
    if match:
        prefix = text[max(0, match.start()-16):match.start()]
        if re.search(r"(?:don't|do not|never)\s*$", prefix, re.I):
            return ()
        noun = match.group(1).lower()
        return ("draft_email",) if noun == "email" else (("draft_sms",) if noun in {"sms", "text"}
                                                          else ("draft_email", "draft_sms"))
    reply = re.search(r"\b(?:reply|respond)\b", text, re.I)
    if reply and not re.search(r"(?:don't|do not|never)\s*$", text[max(0, reply.start()-16):reply.start()], re.I):
        return ("draft_email", "draft_sms")
    direct = re.search(r"(?:^|\bplease\s+|\band\s+)(email|text|message)\s+", text, re.I)
    if direct:
        return ("draft_email",) if direct.group(1).lower() == "email" else ("draft_sms",)
    return ()


@dataclass
class RequestContract:
    query: str
    communication_tools: tuple[str, ...] = ()
    profile_requested: bool = False
    updates: dict = field(default_factory=dict)
    calendar_requested: bool = False

    @classmethod
    def from_query(cls, query: str):
        text = trusted_request(query)
        result = cls(query, draft_tools(query))
        result.profile_requested = bool(re.search(
            r"\b(?:update|change|set|save)\b[^.!?]{0,45}\b(?:profile|preferences?|preferences\.[\w.]+)\b|"
            r"\bremember\b[^.!?]{0,80}\b(?:meetings?|preferences?)\b", text, re.I))
        if result.profile_requested:
            times = []
            for before in re.finditer(r"\bbefore\s+", text, re.I):
                match = TIME_PATTERN.match(text, before.end())
                if match:
                    times.append(time_value(match))
            if not times and re.search(r"earliest(?:_meeting_time| meeting time)?", text, re.I):
                times = list(mentioned_times(text))
            if len(set(times)) == 1:
                result.updates["preferences.earliest_meeting_time"] = times[0]
        result.calendar_requested = bool(re.search(
            r"(?:^|[.!?]\s*|\bplease\s+|\band\s+|\b(?:can|could|would)\s+you\s+(?:please\s+)?)"
            r"(?:schedule|book|create|add|block)\b[^.!?]{0,60}\b(?:calendar|event|meeting|flight)\b", text, re.I))
        return result

    def validate_request(self):
        if self.profile_requested and not self.updates:
            raise ContractError("Profile update needs an explicit supported field/value; specify the earliest meeting time")

    def authorize_write(self, tool: str, args: dict):
        if tool == "create_calendar_event" and not self.calendar_requested:
            raise ContractError("Calendar changes were not requested; only check availability and draft the reply")
        if tool != "update_profile":
            return
        if not self.profile_requested:
            raise ContractError("Profile changes were not requested; scheduling constraints cannot be changed to fit a meeting")
        path, value = args["field"], args["value"]
        if path in self.updates and value == self.updates[path]:
            return
        if path == "preferences.meeting_window_rationale" and isinstance(value, str):
            expected = self.updates.get("preferences.earliest_meeting_time")
            if expected and mentioned_times(value) == {expected} and len(list(TIME_PATTERN.finditer(value))) == 1:
                return
        raise ContractError("Profile change does not match the requested field/value: " + path)

    def assert_complete(self, profile: dict, steps: list[dict]):
        for path, expected in self.updates.items():
            value = profile
            for key in path.split("."):
                value = value.get(key) if isinstance(value, dict) else None
            if value != expected:
                raise ContractError("Requested profile update was not saved: " + path)
        if self.calendar_requested and not any(s["tool"] == "create_calendar_event" and s["status"] == "EXECUTED" for s in steps):
            raise ContractError("Requested calendar event was not created")

    def summary(self):
        return {"draft_tools": list(self.communication_tools), "profile_updates": self.updates,
                "calendar_event_required": self.calendar_requested}
