"""Model A's local memory: hybrid retrieval, a trained reranker, provenance and deletion.

- Every record the user receives is admitted with its provenance (id, time, sender, channel,
  permission). Nothing leaves the device: embeddings come from all-minilm through local Ollama.
- Retrieval takes the union of the BM25 top-k and the embedding top-k and scores each candidate
  with the trained reranker (probability that the record states the fact asked about, in any
  version). Without a reranker it falls back to reciprocal-rank fusion.
- `forget` removes the item, its vector and any cached evidence, and keeps a receipt with the
  user's words, so a later question about it gets "you asked me to forget that".
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np

from p17sim.system.llm import Profile
from p17sim.system.text import BM25, contains_value, tokens

CHANNELS = ["message", "note", "calendar", "notification", "preference", "instruction"]
_CODE = re.compile(r"\b(?=[A-Za-z]*\d)[A-Za-z0-9][A-Za-z0-9#-]{2,}\b")


@dataclass
class Item:
    record: dict
    time: datetime
    vector: np.ndarray | None = None


@dataclass
class Candidate:
    item: Item
    bm25: float = 0.0
    bm25_rank: int = 99
    cosine: float = 0.0
    dense_rank: int = 99
    score: float = 0.0


def features(query: str, cands: list[Candidate], now: datetime) -> np.ndarray:
    """Numeric features per candidate. The text itself goes in through hashed n-grams (see Reranker)."""
    q = set(tokens(query))
    max_bm = max((c.bm25 for c in cands), default=1.0) or 1.0
    by_time = sorted(range(len(cands)), key=lambda i: cands[i].item.time, reverse=True)
    time_rank = {i: r for r, i in enumerate(by_time)}
    rows = []
    for i, c in enumerate(cands):
        r = c.item.record
        t = set(tokens(r["text"]))
        rows.append([
            c.bm25 / max_bm, math.log1p(c.bm25), c.cosine, 1 / (1 + c.bm25_rank), 1 / (1 + c.dense_rank),
            math.log1p(max(0.0, (now - c.item.time).total_seconds() / 86400)), time_rank[i] / max(1, len(cands)),
            len(q & t) / max(1, len(q)), float(r.get("sender") == "me"), float(bool(_CODE.search(r["text"]))),
            math.log1p(len(r["text"].split())), float(r.get("permission_label") == "deny"),
            *[float(r["channel"] == ch) for ch in CHANNELS],
        ])
    return np.array(rows, dtype=float)


class Reranker:
    """Logistic regression on numeric features plus hashed word n-grams of the record and of
    query-record word pairs. Trained in agents/train.py on train personas only."""

    def __init__(self, model: Any = None, scaler: Any = None, n_hash: int = 2 ** 12):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.model, self.scaler = model, scaler
        self.hasher = HashingVectorizer(n_features=n_hash, ngram_range=(1, 2), alternate_sign=False, norm="l2")

    def matrix(self, query: str, cands: list[Candidate], now: datetime):
        from scipy.sparse import csr_matrix, hstack

        num = features(query, cands, now)
        if self.scaler is not None:
            num = self.scaler.transform(num)
        text = self.hasher.transform([c.item.record["text"] for c in cands])
        q = set(tokens(query))
        pairs = self.hasher.transform([" ".join(f"q_{w}" for w in q & set(tokens(c.item.record["text"])))
                                       for c in cands])
        return hstack([csr_matrix(num), text, pairs]).tocsr()

    def score(self, query: str, cands: list[Candidate], now: datetime) -> np.ndarray:
        if not cands:
            return np.array([])
        return self.model.predict_proba(self.matrix(query, cands, now))[:, 1]


class Memory:
    def __init__(self, embed_fn: Callable[[list[str]], list[list[float]]] | None, profile: Profile,
                 reranker: Reranker | None = None, k: int = 30):
        self.embed_fn, self.profile, self.reranker, self.k = embed_fn, profile, reranker, k
        self.items: dict[str, Item] = {}
        self.index = BM25()
        self.receipts: list[dict] = []
        self.cache: dict[str, list[str]] = {}  # query -> evidence ids from earlier turns

    # ---------------------------------------------------------------- admission
    def add(self, record: dict) -> None:
        if record["channel"] == "permission":
            return
        self.items[record["event_id"]] = Item(record, datetime.fromisoformat(record["timestamp"]))
        self.index.add(record["event_id"], f"{record.get('sender', '')} {record['text']}")

    def _embed_missing(self) -> None:
        if self.embed_fn is None:
            return
        todo = [i for i in self.items.values() if i.vector is None]
        with self.profile.time("embed"):
            for s in range(0, len(todo), 64):
                batch = todo[s:s + 64]
                vecs = self.embed_fn([f"{i.record.get('sender', '')}: {i.record['text']}"[:1000] for i in batch])
                for it, v in zip(batch, vecs):
                    a = np.asarray(v, dtype=np.float32)
                    it.vector = a / (np.linalg.norm(a) or 1.0)

    # ---------------------------------------------------------------- retrieval
    def candidates(self, query: str, channels: set[str] | None = None) -> list[Candidate]:
        allowed = {rid for rid, it in self.items.items() if channels is None or it.record["channel"] in channels}
        with self.profile.time("retrieve_bm25"):
            bm = self.index.scores(query, allowed)
        bm_top = sorted(bm, key=bm.get, reverse=True)[: self.k]
        dense: dict[str, float] = {}
        if self.embed_fn is not None and allowed:
            self._embed_missing()
            with self.profile.time("retrieve_dense"):
                qv = np.asarray(self.embed_fn([query])[0], dtype=np.float32)
                qv /= np.linalg.norm(qv) or 1.0
                ids = [rid for rid in allowed if self.items[rid].vector is not None]
                if ids:
                    sims = np.stack([self.items[rid].vector for rid in ids]) @ qv
                    dense = dict(zip(ids, sims.tolist()))
        dense_top = sorted(dense, key=dense.get, reverse=True)[: self.k]
        out = {}
        for r, rid in enumerate(bm_top):
            out[rid] = Candidate(self.items[rid], bm25=bm[rid], bm25_rank=r, cosine=dense.get(rid, 0.0))
        for r, rid in enumerate(dense_top):
            c = out.setdefault(rid, Candidate(self.items[rid], bm25=bm.get(rid, 0.0), cosine=dense[rid]))
            c.dense_rank = r
        return list(out.values())

    def retrieve(self, query: str, now: datetime, channels: set[str] | None = None,
                 use_reranker: bool = True) -> list[Candidate]:
        cands = self.candidates(query, channels)
        if not cands:
            return []
        if use_reranker and self.reranker is not None:
            with self.profile.time("rerank"):
                probs = self.reranker.score(query, cands, now)
            for c, p in zip(cands, probs):
                c.score = float(p)
        else:  # reciprocal-rank fusion of the two lists
            for c in cands:
                c.score = 1 / (60 + c.bm25_rank) + 1 / (60 + c.dense_rank)
        ranked = sorted(cands, key=lambda c: c.score, reverse=True)
        self.cache[query] = [c.item.record["event_id"] for c in ranked[:10]]
        return ranked

    # ---------------------------------------------------------------- deletion
    def forget(self, ids: list[str], request: str, when: datetime) -> list[str]:
        gone = []
        for rid in ids:
            if self.items.pop(rid, None) is not None:
                self.index.remove(rid)
                gone.append(rid)
        for q in list(self.cache):
            if any(rid in gone for rid in self.cache[q]):
                del self.cache[q]
        self.receipts.append({"request": request, "time": when.isoformat(), "deleted": gone})
        return gone

    def holds(self, value: str) -> list[str]:
        """Storage audit: items or cached evidence that still contain `value`."""
        found = [rid for rid, it in self.items.items() if contains_value(it.record["text"], value)]
        return found + [f"cache:{q}" for q, ids in self.cache.items() if any(i in found for i in ids)]

    def forgotten(self, query: str) -> dict | None:
        """The receipt whose request shares the most content words with `query`, if any share two."""
        q = set(tokens(query))
        best, overlap = None, 1
        for r in self.receipts:
            o = len(q & set(tokens(r["request"])))
            if o > overlap:
                best, overlap = r, o
        return best
