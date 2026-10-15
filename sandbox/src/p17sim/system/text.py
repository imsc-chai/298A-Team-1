"""Tokenizing, BM25 and value matching shared by the shell, Model A and the checkers."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
STOP = frozenset("a an and are as at be but by can could do does for from had has have hi hey i if in is it its "
                 "just me my of on or our please so that the their them then there this to up us was we what "
                 "when where which who will with would you your".split())


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    """Okapi BM25 over a growing set of documents. Documents can be removed (deletion)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: dict[str, Counter] = {}
        self.lengths: dict[str, int] = {}
        self.df: Counter = Counter()

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self.docs:
            self.remove(doc_id)
        tf = Counter(tokens(text))
        self.docs[doc_id], self.lengths[doc_id] = tf, sum(tf.values())
        self.df.update(tf.keys())

    def remove(self, doc_id: str) -> None:
        tf = self.docs.pop(doc_id, None)
        if tf is None:
            return
        self.lengths.pop(doc_id)
        self.df.subtract(tf.keys())
        self.df += Counter()  # drop zero counts

    def scores(self, query: str, allowed: set[str] | None = None) -> dict[str, float]:
        q = tokens(query)
        n = len(self.docs) or 1
        avg = (sum(self.lengths.values()) / n) or 1.0
        out: dict[str, float] = {}
        for doc_id, tf in self.docs.items():
            if allowed is not None and doc_id not in allowed:
                continue
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / avg))
            if s > 0:
                out[doc_id] = s
        return out


def value_pattern(value: str) -> re.Pattern:
    """Match a value however it is spaced or punctuated: 'P3-217' matches 'P3 217' and 'p3-217',
    'level 3 spot 217' matches 'Level 3, Spot 217'. Word boundaries stop '4821' matching '14821'."""
    chunks = re.findall(r"[a-z0-9]+", value.lower())
    return re.compile(r"(?<![a-z0-9])" + r"[\W_]*".join(map(re.escape, chunks)) + r"(?![a-z0-9])")


def contains_value(text: str, value: str) -> bool:
    return bool(value) and bool(value_pattern(value).search(text.lower()))
