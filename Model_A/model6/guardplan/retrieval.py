from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter

from .core import GuardPlanError, MemoryRecord, read_json, write_json


def words(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold())


class LexicalRetriever:
    """Visible records ranked by unique query-token overlap; no embeddings.

    Rank the visible corpus, keep an initial pool of eight, then take six.
    Ties are broken by record ID. The controller calls this once per task.
    """

    def __init__(self, records, *, candidate_pool=8, state_filters=True):
        self.records = records
        self.candidate_pool = candidate_pool
        self.state_filters = state_filters
        self.history = []

    def retrieve(self, query, reference_time, k=6):
        if not 1 <= k <= self.candidate_pool:
            raise GuardPlanError("Retrieval requires 1 <= k <= candidate_pool")
        tokens = set(words(query))
        active = [r for r in self.records if r.type != "request" and
                  (not self.state_filters or r.visible(reference_time))]
        scores = {r.id: len(tokens & set(words(r.text))) for r in active}
        pool = sorted(active, key=lambda r: (-scores[r.id], r.id))[:self.candidate_pool]
        selected = pool[:k]
        self.history.append({"query_hash": hashlib.sha256(query.encode()).hexdigest(),
            "reference_time": reference_time, "method": "lexical_token_overlap",
            "candidate_pool_size": self.candidate_pool, "k": k,
            "candidate_ids": [r.id for r in pool],
            "evidence_ids": [r.id for r in selected]})
        return selected


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise GuardPlanError("Embedding dimensions differ; rebuild index")
    denominator = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / denominator if denominator else 0.0


class Retriever:
    """State filter -> BM25 + local dense ranking -> reciprocal-rank fusion."""

    def __init__(self, records: list[MemoryRecord], embedding_client, workspace=None, state_filters=True):
        self.records, self.embedding_client, self.workspace = records, embedding_client, workspace
        self.state_filters = state_filters
        self.vectors = {}
        self.query_vectors = {}
        self.history: list[dict] = []

    def _embed_records(self, records):
        missing = []
        model = getattr(getattr(self.embedding_client, "settings", None), "embedding_model", "test")
        paths = {}
        for record in records:
            digest = hashlib.sha256((model + "\0pooled240-v1\0" + record.text).encode()).hexdigest()
            if self.workspace:
                path = self.workspace / "embeddings" / (digest + ".json")
                paths[record.id] = path
                if path.exists():
                    self.vectors[record.id] = read_json(path)
            if record.id not in self.vectors:
                missing.append(record)
        if missing:
            vectors = self.embedding_client.embed([r.text for r in missing])
            if len(vectors) != len(missing):
                raise GuardPlanError("Missing dense embeddings")
            for record, vector in zip(missing, vectors):
                self.vectors[record.id] = vector
                if record.id in paths:
                    write_json(paths[record.id], vector)

    def retrieve(self, query: str, reference_time: str, k: int = 8) -> list[MemoryRecord]:
        if k < 1:
            raise GuardPlanError("Retrieval k must be positive")
        active = [r for r in self.records if not self.state_filters or r.visible(reference_time)]
        if not active:
            return []
        # Filtering happens before either branch ranks records.
        self._embed_records(active)
        if query not in self.query_vectors:
            self.query_vectors[query] = self.embedding_client.embed([query])[0]
        query_vector = self.query_vectors[query]
        counts = [Counter(words(r.text)) for r in active]
        lengths = [sum(c.values()) for c in counts]
        avg = sum(lengths) / max(1, len(lengths)) or 1
        document_frequency = Counter(token for count in counts for token in count)
        scores = []
        for index, count in enumerate(counts):
            score = 0.0
            for token in set(words(query)):
                tf, df = count[token], document_frequency[token]
                idf = math.log(1 + (len(active) - df + 0.5) / (df + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths[index] / avg))
            scores.append(score)
        sparse = sorted(range(len(active)), key=lambda i: (-scores[i], active[i].id))
        dense = sorted(range(len(active)), key=lambda i: (-cosine(query_vector, self.vectors[active[i].id]), active[i].id))
        fused = Counter()
        for order in (sparse, dense):
            for rank, index in enumerate(order, start=1):
                fused[index] += 1 / (60 + rank)
        selected = [active[i] for i in sorted(fused, key=lambda i: (-fused[i], active[i].id))[:k]]
        self.history.append({"query_hash": hashlib.sha256(query.encode()).hexdigest(),
                             "reference_time": reference_time, "evidence_ids": [r.id for r in selected]})
        return selected
