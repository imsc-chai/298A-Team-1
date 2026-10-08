"""Leakage checks across splits. Any finding stops the pipeline."""
from __future__ import annotations

import re

from datasketch import MinHash, MinHashLSH

from p17sim.schema import HISTORY_LAST_DAY, Event, EventType

CANARY_RE = re.compile(r"REF-[0-9a-f]{8}")


def history_view(events: list[Event], upto_day: int = HISTORY_LAST_DAY) -> list[Event]:
    """What an agent may see before the eval window: no eval-day events and no tasks."""
    return [e for e in events if e.day <= upto_day and e.type != EventType.TASK]


def persona_overlap(events_by_split: dict[str, list[Event]]) -> list[str]:
    owner: dict[str, str] = {}
    found = []
    for split, events in events_by_split.items():
        for pid in {e.persona_id for e in events}:
            if pid in owner and owner[pid] != split:
                found.append(f"persona_in_two_splits: {pid} in {owner[pid]} and {split}")
            owner.setdefault(pid, split)
    return found


def eval_window_leak(history: list[Event]) -> list[str]:
    return [f"eval_event_in_history: {e.event_id} day {e.day}"
            for e in history if e.day > HISTORY_LAST_DAY or e.type == EventType.TASK]


def canary_leak(events: list[Event]) -> list[str]:
    """A canary is unique to one persona; seeing it in another persona's text means a pipeline bug."""
    owner = {e.canary: e.persona_id for e in events if e.canary}
    found = []
    for e in events:
        for canary in CANARY_RE.findall(e.text or ""):
            if owner.get(canary, e.persona_id) != e.persona_id:
                found.append(f"canary_outside_owner: {canary} of {owner[canary]} in {e.event_id}")
    return found


def _minhash(text: str, num_perm: int = 128) -> MinHash:
    words = text.lower().split()
    m = MinHash(num_perm=num_perm)
    for i in range(len(words) - 2):
        m.update(" ".join(words[i:i + 3]).encode())
    return m


def near_duplicates(events_by_split: dict[str, list[Event]], threshold: float = 0.8,
                    min_words: int = 12) -> list[str]:
    """Near-duplicate label-carrying text across splits (MinHash on word 3-grams)."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    docs = {}
    for split, events in events_by_split.items():
        for e in events:
            if e.fact_id and e.text and len(e.text.split()) >= min_words:
                key = f"{split}|{e.event_id}"
                docs[key] = _minhash(e.text)
                lsh.insert(key, docs[key])
    found = set()
    for key, m in docs.items():
        for other in lsh.query(m):
            if key.split("|")[0] != other.split("|")[0]:
                found.add(" <-> ".join(sorted([key, other])))
    return [f"near_duplicate_across_splits: {pair}" for pair in sorted(found)]


def value_overlap_count(events_by_split: dict[str, list[Event]]) -> int:
    """Informational: fact values that occur in more than one split by coincidence (e.g. 'Tuesday')."""
    values: dict[str, set[str]] = {}
    for split, events in events_by_split.items():
        for e in events:
            if e.fact_value:
                values.setdefault(e.fact_value, set()).add(split)
    return sum(1 for splits in values.values() if len(splits) > 1)
