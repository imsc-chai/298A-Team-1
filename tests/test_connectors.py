"""Tests for the shared connectors.py fixes: unique event ids, overnight events, day_range,
DATA_ROOT. Every test works on a sandbox copy so dataset files are never touched."""

import json
import shutil

import pytest

from connectors import duration_minutes, get_connector, is_overnight
from model_a.common.config import load_tool_registry, persona_dir
from model_a.common.connectors import open_connector
from model_a.common.sandbox import task_sandbox
from model_a.common.schemas import StepStatus
from model_a.common.tools import ToolDispatcher


@pytest.fixture
def maya():
    with task_sandbox(persona_dir("persona_01_maya")) as sb:
        yield open_connector(sb)


@pytest.fixture
def david():
    with task_sandbox(persona_dir("persona_03_david")) as sb:
        yield open_connector(sb)


# ---- unique event ids -----------------------------------------------------------------------

def test_new_event_id_does_not_collide_with_existing(maya):
    existing = {e["id"] for e in maya._events}
    assert len(existing) == 47 and "cal_048" in existing  # the old len+1 scheme would reuse cal_048
    ev = maya.create_event("Flight", "2026-11-12", "08:00", "14:15")
    assert ev.id == "cal_049" and ev.id not in existing
    assert maya.create_event("Second", "2026-11-12", "15:00", "16:00").id == "cal_050"


def test_ids_never_reused_after_delete(maya):
    first = maya.create_event("A", "2026-11-12", "08:00", "09:00").id
    assert maya.delete_event(first)
    second = maya.create_event("B", "2026-11-12", "08:00", "09:00").id
    assert second != first
    ids = [e["id"] for e in maya._events]
    assert len(ids) == len(set(ids))


def test_id_keeps_persona_prefix(david):
    assert david.create_event("Swap shift", "2026-11-20", "07:00", "19:30").id == "cal_p3_034"


def test_reset_discards_in_memory_writes(maya):
    maya.create_event("A", "2026-11-12", "08:00", "09:00")
    maya.reset()
    assert len(maya._events) == 47


# ---- overnight events -----------------------------------------------------------------------

def test_overnight_helpers():
    assert is_overnight("19:00", "07:30") and not is_overnight("08:00", "09:00")
    assert duration_minutes("19:00", "07:30") == 750
    assert duration_minutes("08:00", "14:15") == 375


def test_overnight_event_visible_on_next_morning(david):
    nov10 = {e.id: e for e in david.search_events(date="2026-11-10")}
    assert "cal_p3_009" in nov10  # Nov 9 19:00 -> Nov 10 07:30 night shift
    assert nov10["cal_p3_009"].ends_next_day
    assert nov10["cal_p3_009"].duration_minutes == 750


def test_create_overnight_event_has_positive_duration(maya):
    ev = maya.create_event("Red-eye", "2026-11-14", "22:00", "06:00")
    assert ev.duration_minutes == 480 and ev.ends_next_day
    assert ev.id in {e.id for e in maya.search_events(date="2026-11-15")}


def test_zero_length_event_rejected(maya):
    with pytest.raises(ValueError):
        maya.create_event("Nothing", "2026-11-12", "09:00", "09:00")


def test_dispatcher_accepts_overnight_and_shows_it(david):
    d = ToolDispatcher(david, load_tool_registry(), "2026-11-08T12:00:00")
    r = d.dispatch("create_calendar_event", {"title": "Cover night", "date": "2026-11-21", "start_time": "19:00", "end_time": "07:30", "busy": True})
    assert r.status == StepStatus.EXECUTED and "ends next day" in r.observation
    r = d.dispatch("search_calendar", {"date": "2026-11-10"})
    assert "cal_p3_009" in r.evidence_ids and "ends next day" in r.observation


# ---- day_range ------------------------------------------------------------------------------

def test_day_range_filters_by_day_inclusive(maya):
    msgs = maya.search_messages(day_range=[2, 3])
    assert msgs and all(m.timestamp[:10] in {"2026-11-02", "2026-11-03"} for m in msgs)
    all_days = {m.timestamp[:10] for m in maya.search_messages()}
    assert {"2026-11-02", "2026-11-03"} <= all_days and len(all_days) > 2


def test_day_range_combines_with_reference_time(maya):
    msgs = maya.search_messages(day_range=[1, 30], reference_time="2026-11-02T00:00:00")
    assert msgs and all(m.timestamp <= "2026-11-02T00:00:00" for m in msgs)


@pytest.mark.parametrize("bad", [[5], [5, 3], [1, 2, 3]])
def test_invalid_day_range_rejected(maya, bad):
    with pytest.raises(ValueError):
        maya.search_messages(day_range=bad)


def test_dispatcher_day_range_goes_through_connector(maya):
    d = ToolDispatcher(maya, load_tool_registry(), "2026-11-30T23:59:00")
    assert d.dispatch("read_messages", {"day_range": [5, 3]}).status == StepStatus.FAILED
    r = d.dispatch("read_messages", {"day_range": [1, 1]})
    assert r.evidence_ids and all(i in {"msg_001", "msg_002"} for i in r.evidence_ids)


# ---- DATA_ROOT ------------------------------------------------------------------------------

def test_get_connector_uses_data_root_env(tmp_path, monkeypatch):
    with task_sandbox(persona_dir("persona_01_maya")) as sb:
        shutil.copytree(sb, tmp_path / "persona_01_maya")
    marker = tmp_path / "persona_01_maya" / "agent_memory" / "user_profile.json"
    profile = json.loads(marker.read_text())
    profile["name"] = "Env Root Maya"
    marker.write_text(json.dumps(profile))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert get_connector("persona_01_maya").get_profile()["name"] == "Env Root Maya"


def test_explicit_data_root_beats_env(tmp_path, monkeypatch):
    real_root = str(persona_dir("persona_01_maya").parent)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))  # empty dir: would fail if used
    conn = get_connector("persona_01_maya", data_root=real_root)
    assert conn.get_profile()["name"] == "Maya Lin"


def test_missing_persona_under_data_root_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        get_connector("persona_01_maya")
