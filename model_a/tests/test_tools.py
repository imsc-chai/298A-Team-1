import json

import pytest

from model_a.common.connectors import open_connector
from model_a.common.sandbox import task_sandbox
from model_a.common.schemas import StepStatus
from model_a.common.tools import ToolDispatcher, parse_registry, render_tools
from model_a.config import load_tool_registry, persona_dir

REF_TIME = "2026-11-04T09:10:00"


@pytest.fixture
def env():
    with task_sandbox(persona_dir("persona_01_maya")) as sb:
        conn = open_connector(sb)
        yield ToolDispatcher(conn, load_tool_registry(), REF_TIME), conn, sb


def test_registry_parses_all_tools():
    specs = parse_registry(load_tool_registry())
    assert specs["read_messages"].args["day_range"].kind == "int_array"
    assert specs["read_messages"].args["day_range"].optional
    assert not specs["draft_email"].args["to"].optional
    assert specs["send_email"].permission == "requires_confirmation"
    assert "send_sms" in render_tools(load_tool_registry())


@pytest.mark.parametrize("tool,args", [("send_email", {"draft_id": "draft_0001"}), ("send_sms", {"draft_id": "x"}), ("delete_calendar_event", {"event_id": "cal_001"})])
def test_confirmation_tools_refused_and_logged(env, tool, args):
    d, conn, _ = env
    n_events = len(conn._events)
    r = d.dispatch(tool, args)
    assert r.status == StepStatus.REFUSED
    assert d.violations and d.violations[-1]["type"] == "boundedness"
    assert len(conn._events) == n_events  # connector never called


@pytest.mark.parametrize(
    "tool,args,fragment",
    [
        ("teleport", {}, "unknown tool"),
        ("search_calendar", {"day": "2026-11-10"}, "unknown argument"),
        ("draft_email", {"to": "a.chen@univ.edu", "subject": "x"}, "missing required"),
        ("create_calendar_event", {"title": "t", "date": "2026-11-12", "start_time": "08:00", "end_time": "09:00", "busy": "yes"}, "must be boolean"),
        ("read_messages", {"day_range": [1, "3"]}, "must be int_array"),
        ("search_calendar", {"date": "Nov 10"}, "YYYY-MM-DD"),
        ("create_calendar_event", {"title": "t", "date": "2026-11-12", "start_time": "9:00", "end_time": "10:00", "busy": True}, "HH:MM"),
        ("draft_sms", {"to": "Marcus", "body": "hi"}, "phone number"),
        ("read_local_file", {"path": "../agent_memory/user_profile.json"}, "directly inside local_files"),
        ("read_local_file", {"path": "local_files/nope.pdf"}, "does not exist"),
    ],
)
def test_invalid_calls_fail_cleanly(env, tool, args, fragment):
    r = env[0].dispatch(tool, args)
    assert r.status == StepStatus.FAILED
    assert fragment in r.observation


def test_messages_hidden_after_reference_time(env):
    d, _, sb = env
    all_msgs = json.loads((sb / "streams" / "messages.json").read_text())
    r = d.dispatch("read_messages", {})
    future = [m["id"] for m in all_msgs if m["timestamp"] > REF_TIME]
    assert future, "fixture should have future messages"
    assert not set(future) & set(r.evidence_ids)
    # The cap on returned messages may hide old ones, so check visibility via day_range too.
    assert all(m["timestamp"] <= REF_TIME for m in all_msgs if m["id"] in r.evidence_ids)


def test_calendar_not_time_filtered(env):
    r = env[0].dispatch("search_calendar", {"date": "2026-11-10"})
    assert "cal_015" in r.evidence_ids  # Nov 10 is after REF_TIME but must be visible
    assert "Private Health Appt" in r.observation


def test_calendar_query_matches_any_word(env):
    r = env[0].dispatch("search_calendar", {"query": "deep research focus"})
    assert "cal_011" in r.evidence_ids


def test_read_messages_resolves_contact_id_and_day_range(env):
    r = env[0].dispatch("read_messages", {"contact": "c_dr_chen", "day_range": [1, 1]})
    assert r.status == StepStatus.EXECUTED
    assert "msg_001" in r.evidence_ids


def test_lookup_contact_by_partial_name(env):
    r = env[0].dispatch("lookup_contact", {"name": "Marcus"})
    assert "+1-555-014-9932" in r.observation
    assert r.evidence_ids == ["contact_c_marcus"]


def test_read_pdf_returns_parsed_text_and_chunk_ids(env):
    r = env[0].dispatch("read_local_file", {"path": "local_files/clinic_bill.pdf"})
    assert r.status == StepStatus.EXECUTED
    assert "L5-S1" in r.observation
    assert r.evidence_ids[0] == "file_clinic_bill_001"


def test_drafts_are_draft_status(env):
    d = env[0]
    r1 = d.dispatch("draft_sms", {"to": "+1-555-014-9932", "body": "Can we do 3pm?"})
    r2 = d.dispatch("draft_email", {"to": "a.chen@univ.edu", "subject": "s", "body": "b"})
    assert r1.status == r2.status == StepStatus.DRAFT
    assert r1.evidence_ids == ["draft_0001"]


def test_create_event_and_update_profile_stay_in_sandbox(env):
    d, conn, sb = env
    source_profile = (persona_dir("persona_01_maya") / "agent_memory" / "user_profile.json").read_text()
    r = d.dispatch("create_calendar_event", {"title": "Flight", "date": "2026-11-12", "start_time": "08:00", "end_time": "14:15", "busy": True})
    assert r.status == StepStatus.EXECUTED
    r = d.dispatch("update_profile", {"field": "preferences.earliest_meeting_time", "value": "10:30"})
    assert r.status == StepStatus.EXECUTED
    saved = json.loads((sb / "agent_memory" / "user_profile.json").read_text())
    assert saved["preferences"]["earliest_meeting_time"] == "10:30"
    assert (persona_dir("persona_01_maya") / "agent_memory" / "user_profile.json").read_text() == source_profile
