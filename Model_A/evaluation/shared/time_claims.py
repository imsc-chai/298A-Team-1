"""Validate dated meeting offers separately from evidence-backed time facts."""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from .constraints import TIME_PATTERN, mentioned_times, minutes, time_value


MONTHS = {name.lower(): i for i in range(1, 13) for name in (calendar.month_name[i], calendar.month_abbr[i])}
MONTH_PATTERN = "|".join(sorted(MONTHS, key=len, reverse=True))
NEGATIVE = re.compile(r"\b(?:can't|cannot|can’t|not available|unavailable|doesn't work|doesn’t work|"
                      r"does not work|won't work|won’t work|unable|not free|conflict|not possible|"
                      r"doesn't fit|doesn’t fit|does not fit)\b", re.I)
OFFER = re.compile(r"\b(?:would|could|how about|let's|let’s|can we|can meet|available|works for me|free at|confirmed)\b", re.I)
RULE = re.compile(r"\b(?:no meetings? before|cannot meet before|can't meet before|can’t meet before|"
                  r"earliest (?:meeting |start )?time|earliest meeting|earliest availability|"
                  r"don't schedule.*before|do not schedule.*before)\b", re.I)
REFERENCE = re.compile(r"\b(?:requested time|your (?:request|proposal)|you (?:asked|requested|proposed)|"
                       r"checked (?:my|the) (?:calendar|schedule))\b", re.I)
FACT = re.compile(r"\b(?:flight|train|appointment|lecture|event)\b.*\b"
                  r"(?:depart(?:s|ure)?|arriv(?:es|al)|leaves?|starts?|ends?|at|takes off|lands?)\b", re.I)


def dates_in(text: str, year: int, reference_time: str | None):
    values = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
    for month, day, y in re.findall(r"\b(" + MONTH_PATTERN + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b", text, re.I):
        try:
            values.add(date(int(y or year), MONTHS[month.lower()], int(day)).isoformat())
        except ValueError:
            values.add("INVALID_DATE")
    if reference_time:
        base = date.fromisoformat(reference_time[:10])
        if re.search(r"\btomorrow\b", text, re.I):
            values.add((base + timedelta(days=1)).isoformat())
        if re.search(r"\btoday\b", text, re.I):
            values.add(base.isoformat())
    weekdays = {i for i, name in enumerate(calendar.day_name) if re.search(r"\b" + name + r"\b", text, re.I)}
    return values, weekdays


def validate_time_text(text, proposals, rejected_times, earliest=None, *, factual_times=(),
                       requested_times=(), reference_time=None, requested_dates=()):
    if not text:
        return []
    year = int(reference_time[:4]) if reference_time else (int(proposals[0]["date"][:4]) if proposals else 2026)
    issues = []
    # Commas are retained because they can belong to dates. Split alternative
    # clauses so a refusal cannot authorize a later positive offer.
    clauses = re.split(r"[!?;\n]+|\.(?!\d)|\b(?:but|however|instead)\b", text, flags=re.I)
    for clause in clauses:
        matches = list(TIME_PATTERN.finditer(clause))
        times = {time_value(m) for m in matches}
        explicit_dates, weekdays = dates_in(clause, year, reference_time)
        is_fact = bool(FACT.search(clause)) and not OFFER.search(clause)
        if is_fact and times and times <= set(factual_times):
            continue
        is_rule = bool(RULE.search(clause)) and not OFFER.search(clause)
        # "not available" is a refusal, not a positive availability assertion.
        positive_text = NEGATIVE.sub("", clause)
        negative = bool(NEGATIVE.search(clause)) and not OFFER.search(positive_text)
        reference = bool(REFERENCE.search(clause)) and not OFFER.search(clause)
        if (negative or is_rule or reference) and times:
            allowed = set(rejected_times) if negative else set()
            if is_rule and earliest:
                allowed.add(earliest)
            if reference:
                allowed.update(requested_times)
            if times <= allowed:
                if explicit_dates and requested_dates and not explicit_dates <= set(requested_dates):
                    issues.append({"code": "draft_date_mismatch"})
                continue
        if not matches:
            if proposals and explicit_dates and not explicit_dates <= {p["date"] for p in proposals}:
                issues.append({"code": "draft_date_mismatch"})
            continue
        durations = {int(n) for n in re.findall(r"\b(\d+)\s*[- ]?\s*min(?:ute)?s?\b", clause, re.I)}
        index = 0
        while index < len(matches):
            start = time_value(matches[index])
            end = None
            if index + 1 < len(matches):
                between = clause[matches[index].end():matches[index + 1].start()]
                if re.fullmatch(r"\s*(?:[-–—]|to|until)\s*", between, re.I):
                    end = time_value(matches[index + 1])
            candidates = [p for p in proposals if p["start_time"] == start and (end is None or p["end_time"] == end)]
            if explicit_dates:
                candidates = [p for p in candidates if p["date"] in explicit_dates]
                if not explicit_dates <= {p["date"] for p in candidates}:
                    candidates = []
            if weekdays:
                candidates = [p for p in candidates if date.fromisoformat(p["date"]).weekday() in weekdays]
                if not weekdays <= {date.fromisoformat(p["date"]).weekday() for p in candidates}:
                    candidates = []
            if durations:
                candidates = [p for p in candidates if minutes(p["end_time"]) - minutes(p["start_time"]) in durations]
                if len(durations) != 1:
                    candidates = []
            if not candidates:
                issues.append({"code": "unchecked_draft_time", "time": start,
                               "repair": "Use the exact date, START time, end time and duration of an accepted schedule_proposal. Put factual times only in source-supported factual statements."})
            index += 2 if end is not None else 1
    return issues


def factual_times_from(query: str, records) -> set[str]:
    facts = set()
    for text in [query, *(r.text for r in records)]:
        if FACT.search(text):
            facts.update(mentioned_times(text))
    return facts
