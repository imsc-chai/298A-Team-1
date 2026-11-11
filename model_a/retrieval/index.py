"""Hybrid retriever: BM25 + bge-small dense embeddings, fused with reciprocal-rank fusion.

Shared by every model (proposal §4.3); models differ only in WHEN and HOW OFTEN they call it.
BM25 catches exact identifiers (names, codes, dates); embeddings catch paraphrase
("absence reason" -> "symposium"). RRF combines ranks without needing score calibration.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from model_a.common.schemas import MemoryRecord, RecordType
from model_a.config import cache_dir

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge v1.5 recommends this instruction on short queries (not on passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60
DEFAULT_K = 8

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    name: str

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


class BgeEmbedder:
    """Lazy-loaded sentence-transformers wrapper; vectors are L2-normalized for cosine."""

    def __init__(self, model_name: str = EMBED_MODEL, device: str | None = None) -> None:
        self.name = model_name
        self._device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self._device)
        return self._model

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._load().encode(texts, normalize_embeddings=True, batch_size=32), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(
            self._load().encode([BGE_QUERY_PREFIX + text], normalize_embeddings=True)[0], dtype=np.float32
        )


@dataclass(frozen=True)
class Hit:
    record: MemoryRecord
    score: float
    bm25_rank: int | None
    dense_rank: int | None


class HybridRetriever:
    def __init__(self, records: list[MemoryRecord], embedder: Embedder, use_cache: bool = True) -> None:
        if not records:
            raise ValueError("Cannot build an index over zero records")
        self.records = records
        self._by_id = {r.id: r for r in records}
        self._bm25 = BM25Okapi([tokenize(r.text) for r in records])
        self._embedder = embedder
        self._matrix = self._embed_passages(use_cache)

    def get(self, record_id: str) -> MemoryRecord | None:
        return self._by_id.get(record_id)

    def search(self, query: str, reference_time: str, k: int = DEFAULT_K) -> list[Hit]:
        """Top-k records for `query`, hiding messages sent after reference_time (CLAUDE.md rule 7).

        Calendar events are never time-filtered: future events are already-known plans.
        """
        visible = np.array([_visible(r, reference_time) for r in self.records])

        bm25_scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        dense_scores = self._matrix @ self._embedder.encode_query(query)

        bm25_order = [i for i in np.argsort(-bm25_scores, kind="stable") if visible[i] and bm25_scores[i] > 0]
        dense_order = [i for i in np.argsort(-dense_scores, kind="stable") if visible[i]]

        bm25_rank = {i: r for r, i in enumerate(bm25_order, start=1)}
        dense_rank = {i: r for r, i in enumerate(dense_order, start=1)}
        fused: dict[int, float] = {}
        for ranks in (bm25_rank, dense_rank):
            for i, rank in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)

        top = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        return [
            Hit(record=self.records[i], score=fused[i], bm25_rank=bm25_rank.get(i), dense_rank=dense_rank.get(i))
            for i in top
        ]

    def _embed_passages(self, use_cache: bool) -> np.ndarray:
        """Embeddings cached on disk keyed by model + exact record texts."""
        digest = hashlib.sha256(self._embedder.name.encode())
        for r in self.records:
            digest.update(r.id.encode() + b"\0" + r.text.encode() + b"\0")
        path: Path = cache_dir() / "embeddings" / f"{digest.hexdigest()}.npy"
        if use_cache and path.is_file():
            return np.load(path)
        matrix = self._embedder.encode_passages([r.text for r in self.records])
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
        return matrix


def _visible(record: MemoryRecord, reference_time: str) -> bool:
    if record.type == RecordType.MESSAGE:
        return record.timestamp is not None and record.timestamp <= reference_time
    return True


def format_hits(hits: list[Hit]) -> str:
    """Shared rendering of retrieved context for prompts: `[id] text`, one per line."""
    return "\n".join(f"[{h.record.id}] {h.record.text}" for h in hits) or "(no records retrieved)"
