from datetime import datetime, timedelta

import pytest
from conftest import make_workspace

from p17sim.aios.engine import queue_approval
from p17sim.aios.generators import NotTrained, catalog, generator, routine_reference, suggestion_reference
from p17sim.aios.templates import TEMPLATES, build


def ready_ws(ds, replies=None):
    pid = next(p for p, s in sorted(ds.splits.items()) if s == "val")
    ws = make_workspace(ds, pid, replies)
    ws.advance_to(datetime.fromisoformat(ds.records[pid][0]["timestamp"]).replace(hour=0) + timedelta(days=24))
    return ws


def test_baseline_run_becomes_a_workflow(v2ds):
    ws = ready_ws(v2ds, [{"tool_calls": [{"id": "1", "name": "search_records", "arguments": {"query": "code"}}]},
                         {"content": "It is 1234."}])
    res = generator("baseline").run("What is my gate code?", ws)
    types = [n["type"] for n in res.workflow["nodes"]]
    assert types == ["trigger.chat", "reason.agent_step", "context.search", "reason.agent_step", "output.respond"]
    assert res.answer == "It is 1234." and res.status == "succeeded"
    assert ws.store.execution(res.execution_id)["generator"] == "baseline"


def test_baseline_failure_is_recorded_not_raised(v2ds):
    ws = ready_ws(v2ds)

    def boom(*a, **k):
        raise TimeoutError("model server timed out")
    ws.llm.chat = boom
    res = generator("baseline").run("anything", ws)
    assert res.status == "failed" and "timed out" in res.error


@pytest.mark.parametrize("tid,params", [("morning_briefing", {}), ("weekly_recap", {"to": "x"}),
                                        ("book_meeting", {"contact": "x", "date": "2026-09-30"}), ("forget", {"query": "x"})])
def test_templates_are_valid_graphs(tid, params):
    wf = build(tid, params)
    assert wf.order()[0].type.startswith("trigger.")


def test_weekly_recap_waits_for_approval_then_sends(v2ds):
    ws = ready_ws(v2ds, [{"content": "Busy week, lots of meetings!"}])
    to = ws.persona.contacts[0].name
    res = generator("template:weekly_recap").run("", ws, approver=queue_approval, params={"to": to})
    assert res.status == "waiting_approval" and ws.store.outbox(ws.pid) == []
    done = ws.engine.resume(res.approval_id, "approve")
    assert done.status == "succeeded" and ws.store.outbox(ws.pid)[0]["body"] == "Busy week, lots of meetings!"


def test_pending_models_say_so():
    with pytest.raises(NotTrained):
        generator("model_a").run("x", None)
    statuses = {g["name"]: g["status"] for g in catalog()}
    assert statuses["baseline"] == "ready" and statuses["model_b"] == "pending training"
    assert all(f"template:{t}" in statuses for t in TEMPLATES)


def test_routine_reference_finds_weekly_routines(v2ds):
    ws = ready_ws(v2ds)
    found = routine_reference(ws)
    assert found and all(f["occurrences"] >= 3 for f in found)


def test_suggestion_reference_picks_the_most_accepted_option(v2ds):
    ws = ready_ws(v2ds)
    sugg = {s["key"]: s for s in suggestion_reference(ws)}
    assert "meeting_window" in sugg and sugg["meeting_window"]["suggest"] in ("morning", "afternoon", "evening")


def test_template_prompts_receive_the_upstream_data(v2ds):
    ws = ready_ws(v2ds, [{"content": '{"start": "07:00"}'}])
    contact = ws.persona.contacts[0].name
    day = ws.now.date().isoformat()
    generator("template:book_meeting").run("", ws, params={"contact": contact, "date": day})
    prompt = ws.llm.seen[0][-1]["content"]
    assert "{{" not in prompt and "}}" not in prompt and '{"start": "HH:MM"}' in prompt
    assert "Calendar: [" in prompt
