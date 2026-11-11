import hashlib

import numpy as np
import pytest

from common.context import PersonaContext
from common.sandbox import task_sandbox
from common.schemas import MemoryRecord, RecordType
from common.config import persona_dir
from common.ingest import build_records
from common.parsers import parse_file
from common.retriever import HybridRetriever, tokenize

MAYA = "persona_01_maya"


def test_png_ocr_recovers_flight_fields():
    text = parse_file(persona_dir(MAYA) / "local_files" / "flight_ticket.png").text.lower()
    for field in ["skywest", "412", "08:00", "02:15", "nov 12"]:
        assert field in text


def test_markdown_chunked_by_heading():
    parsed = parse_file(persona_dir(MAYA) / "local_files" / "research_notes.md")
    hits = [c for c in parsed.chunks if "Midwest CS Graduate Symposium" in c.text]
    assert hits and hits[0].heading.startswith("Upcoming Deadlines")


def test_csv_one_chunk_per_row():
    parsed = parse_file(persona_dir(MAYA) / "local_files" / "lab_gpu_cluster_inventory.csv")
    assert parsed.chunks[0].text.startswith("server_name=lambda-node-01")


@pytest.mark.parametrize("persona", ["persona_01_maya", "persona_02_elena"])
def test_records_cover_all_sources_with_unique_ids(persona):
    with task_sandbox(persona_dir(persona)) as sb:
        records = build_records(sb, persona)
    types = {r.type for r in records}
    assert types == set(RecordType)
    files = {r.source for r in records if r.type == RecordType.FILE}
    assert len(files) == len([p for p in (persona_dir(persona) / "local_files").iterdir() if not p.name.startswith(".")])


class HashEmbedder:
    """Deterministic bag-of-words embedder so retrieval logic is testable without the model."""

    name = "hash-test"

    def _vec(self, text):
        v = np.zeros(256, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text):
        return self._vec(text)


def rec(id_, type_, text, ts=None):
    return MemoryRecord(id=id_, persona_id="p", source="s", type=type_, text=text, timestamp=ts)


def test_retriever_hides_future_messages_not_future_events():
    records = [
        rec("msg_old", RecordType.MESSAGE, "lab sync tuesday", "2026-11-01T10:00:00"),
        rec("msg_new", RecordType.MESSAGE, "lab sync tuesday", "2026-11-20T10:00:00"),
        rec("cal_future", RecordType.CALENDAR_EVENT, "lab sync tuesday Nov 24"),
    ]
    ids = [h.record.id for h in HybridRetriever(records, HashEmbedder(), use_cache=False).search("lab sync", "2026-11-05T00:00:00", k=5)]
    assert "msg_old" in ids and "cal_future" in ids and "msg_new" not in ids


def test_rrf_prefers_records_ranked_by_both():
    records = [
        rec("a", RecordType.FILE, "gate code locker apartment"),
        rec("b", RecordType.FILE, "unrelated grocery list"),
        rec("c", RecordType.FILE, "apartment"),
    ]
    hits = HybridRetriever(records, HashEmbedder(), use_cache=False).search("apartment gate code", "2026-11-05T00:00:00", k=3)
    assert hits[0].record.id == "a"
    assert hits[0].bm25_rank == 1


@pytest.mark.slow
def test_real_index_finds_hidden_context_for_subgoal():
    """Proposal §5 M3 example: an absence-reason subgoal should surface the symposium note."""
    from common.environment import build_retriever

    retriever = build_retriever(MAYA)
    hits = retriever.search("reason for traveling and missing office hours on Nov 12", "2026-11-07T14:00:00", k=8)
    texts = " ".join(h.record.text for h in hits)
    assert "Midwest CS Graduate Symposium" in texts


def test_persona_context_contains_profile_policy_and_weekday():
    with task_sandbox(persona_dir(MAYA)) as sb:
        text = PersonaContext.load(sb).system_text("2026-11-09T18:00:00")
    assert "Current time: 2026-11-09T18:00:00 (Monday)" in text
    assert "redaction_substitute" in text and "Dr. Arthur Chen" in text
    assert "draft" in text.lower()
