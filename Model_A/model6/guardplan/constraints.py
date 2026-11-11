from __future__ import annotations

import re
from datetime import date, datetime, timedelta


def minutes(value: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"\d{1,2}:\d{2}", value):
        raise ValueError("Clock time must use HH:MM, for example 15:00")
    hour, minute = map(int, value.split(":"))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("Invalid clock time")
    return hour * 60 + minute


def clock(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def overlap(start, end, other_start, other_end) -> bool:
    return start < other_end and other_start < end


TIME_PATTERN = re.compile(
    r"\b(?:(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(am|pm)|([01]?\d|2[0-3]):([0-5]\d))\b", re.I)


def time_value(match) -> str:
    hour, minute, period, hour24, minute24 = match.groups()
    if period:
        return clock((int(hour) % 12 + (12 if period.lower() == "pm" else 0))*60 + int(minute or 0))
    return clock(int(hour24)*60 + int(minute24))


def normalize_slot(proposal: dict) -> dict:
    """Normalize unambiguous 12/24-hour clock strings; never guess bare hours."""
    result = dict(proposal)
    date.fromisoformat(result["date"])
    for key in ("start_time", "end_time"):
        value = result[key]
        match = TIME_PATTERN.fullmatch(value.strip()) if isinstance(value, str) else None
        if not match:
            raise ValueError(f"{key} must use HH:MM, for example 15:00, or an explicit AM/PM time")
        result[key] = time_value(match)
    return result


class ConstraintChecker:
    """Deterministic scheduling checks; no LLM judgment in the verdict.

    Structured preferences are authoritative. Supported natural-language windows
    are compiled from explicit HH:MM ranges in meeting_window_rationale. Other
    unstructured rules are exposed as unsupported rather than silently verified.
    """

    def __init__(self, profile: dict, events: list[dict], reference_time: str):
        self.profile, self.events, self.reference_time = profile, events, reference_time
        self.preferences = profile.get("preferences", {})
        self.unsupported = [k for k in ("childcare_pickup_deadline", "shift_swap_preferences")
                            if k in self.preferences]

    def windows(self) -> list[tuple[int, int]]:
        pref = self.preferences
        window = pref.get("meeting_window", "any")
        rationale = pref.get("meeting_window_rationale", "")
        ranges = [(minutes(a), minutes(b)) for a, b in re.findall(
            r"(\d{2}:\d{2})\s*[-–]\s*(\d{2}:\d{2})", rationale)]
        if window == "prep_period_and_after_school" and len(ranges) >= 4:
            return ranges[-2:]
        if window == "afternoon":
            # The last explicit range describes client/investor availability.
            if len(ranges) >= 2:
                return [ranges[-1]]
            return [(12*60, 18*60)]
        if isinstance(window, dict):
            return [(minutes(window["start"]), minutes(window["end"]))]
        return [(8*60, 20*60)]

    def check(self, proposal: dict, kind: str = "meeting") -> list[dict]:
        try:
            proposal = normalize_slot(proposal)
            day = date.fromisoformat(proposal["date"])
            start, end = minutes(proposal["start_time"]), minutes(proposal["end_time"])
        except (KeyError, ValueError, TypeError):
            return [{"code": "invalid_schedule"}]
        issues = []
        def issue(code, **details):
            issues.append({"code": code, **details})
        if end <= start:
            return [{"code": "nonpositive_duration"}]
        if datetime.fromisoformat(day.isoformat() + "T" + clock(start)) < datetime.fromisoformat(self.reference_time):
            issue("past_time")
        # Fixed travel blocks intentionally record an existing commitment; the
        # planner must identify the conflicting meetings instead of moving travel.
        if kind == "fixed_event":
            return issues
        pref = self.preferences
        if day.isoweekday() not in pref.get("working_days", [1, 2, 3, 4, 5, 6, 7]):
            issue("nonworking_day")
        if day.strftime("%A").casefold() in {d.casefold() for d in pref.get("focus_days", [])}:
            issue("focus_day")
        earliest = pref.get("earliest_meeting_time")
        if earliest and start < minutes(earliest):
            issue("earliest_meeting_time", earliest=earliest)
        if not any(start >= a and end <= b for a, b in self.windows()):
            issue("meeting_window")
        buffer = pref.get("buffer_between_meetings_minutes", 0)
        if kind == "candidate_interview":
            buffer = max(buffer, pref.get("candidate_interview_buffer_minutes", 0))
        if kind == "parent_conference":
            buffer = max(buffer, pref.get("parent_conference_buffer_minutes", 0))
        same_day = [e for e in self.events if e["date"] == day.isoformat()]
        for e in same_day:
            if not e.get("busy", True):
                continue
            a, b = minutes(e["start_time"]), minutes(e["end_time"])
            if b <= a:  # overnight event starting today
                b += 1440
            if overlap(start, end, a, b):
                issue("calendar_overlap", evidence_id=e["id"])
            elif buffer and overlap(start, end, a-buffer, b+buffer):
                issue("meeting_buffer", evidence_id=e["id"], minutes=buffer)
        previous = day - timedelta(days=1)
        night_end = None
        for e in self.events:
            if e["date"] == previous.isoformat() and minutes(e["end_time"]) <= minutes(e["start_time"]):
                b = minutes(e["end_time"])
                if overlap(start, end, 0, b):
                    issue("overnight_overlap", evidence_id=e["id"])
                night_end = max(night_end or 0, b)
        if night_end and "post_night_shift_sleep_rule" in pref and start < 15*60:
            issue("post_night_shift_sleep", until="15:00")
        for i, routine in enumerate(self.profile.get("routines", [])):
            if day.isoweekday() not in routine.get("weekdays", []) or not routine.get("strict_conflict"):
                continue
            if night_end and "drop-off" in routine.get("name", "").casefold():
                continue
            a = minutes(routine["start"])
            b = a + routine["duration_minutes"]
            if overlap(start, end, a, b):
                issue("routine_overlap", evidence_id=f"routine_{i:03d}")
        for i, slot in enumerate(pref.get("deep_work_focus_slots", [])):
            if day.isoweekday() == slot["weekday"] and overlap(start, end, minutes(slot["start"]), minutes(slot["end"])):
                issue("protected_focus", evidence_id="preference_deep_work_focus_slots")
        meeting_events = [e for e in same_day if re.search(r"meeting|sync|call|interview|conference", e.get("title", ""), re.I)]
        limit = pref.get("max_meetings_per_day", pref.get("max_client_syncs_per_day"))
        if limit is not None and len(meeting_events) >= limit:
            issue("max_meetings_per_day", limit=limit)
        for special, key, pattern in [("investor_call", "max_investor_calls_per_day", "investor"),
                                      ("parent_conference", "max_parent_conferences_per_day", "parent")]:
            if kind == special and key in pref and sum(bool(re.search(pattern, e.get("title", ""), re.I)) for e in same_day) >= pref[key]:
                issue(key, limit=pref[key])
        return issues

    def alternatives(self, proposal: dict, kind: str = "meeting", count: int = 3) -> list[dict]:
        try:
            proposal = normalize_slot(proposal)
            day = date.fromisoformat(proposal["date"])
            duration = minutes(proposal["end_time"]) - minutes(proposal["start_time"])
        except (KeyError, ValueError, TypeError):
            return []
        if duration <= 0:
            return []
        result = []
        for offset in range(8):
            candidate_day = day + timedelta(days=offset)
            for window_start, window_end in self.windows():
                for start in range(window_start, window_end - duration + 1, 5):
                    candidate = {"date": candidate_day.isoformat(), "start_time": clock(start),
                                 "end_time": clock(start + duration)}
                    if not self.check(candidate, kind):
                        result.append(candidate)
                        if len(result) == count:
                            return result
        return result


def mentioned_times(text: str) -> set[str]:
    return {time_value(match) for match in TIME_PATTERN.finditer(text)}


def draft_time_violations(text: str, proposals: list[dict], rejected_times: set[str],
                          earliest: str | None = None) -> list[dict]:
    """Recognize narrow refusal/rule clauses, without whitelisting rejected times.

    A refusal is eligible only after a symbolic check of the user's requested
    slot. Positive proposals still need an accepted schedule_proposal. Ambiguous
    clauses fail for repair; this is a bounded grammar, not a semantic proof.
    """
    from .time_claims import validate_time_text
    return validate_time_text(text, proposals, rejected_times, earliest)
