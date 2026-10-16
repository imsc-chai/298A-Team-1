from datetime import datetime, timedelta

from conftest import make_workspace

from p17sim.aios.workflow import Edge, Node, Workflow


def first(ds, split="val"):
    return next(p for p, s in sorted(ds.splits.items()) if s == split)


def test_replay_delivers_only_the_past(v2ds):
    pid = first(v2ds)
    ws = make_workspace(v2ds, pid)
    start = datetime.fromisoformat(v2ds.records[pid][0]["timestamp"]).replace(hour=0, minute=0)
    until = start + timedelta(days=24)
    ws.advance_to(until)
    expected = [r for r in v2ds.records[pid] if r["timestamp"] < until.isoformat()]
    assert len(ws.store._rows("SELECT id FROM records WHERE persona_id = ?", (pid,))) == len(expected)
    assert ws.now == until
    assert all(r["ts"] < until.isoformat() for r in ws.store.visible(pid))


def test_instructions_reach_the_handler_and_questions_can_be_hidden(v2ds):
    pid = first(v2ds)
    ws = make_workspace(v2ds, pid)
    seen = []
    ws.advance_to(datetime(2026, 12, 31), on_instruction=seen.append,
                  skip=lambda r: bool(v2ds.labels[r["event_id"]].get("checker")))
    tasks = [r for r in v2ds.records[pid] if v2ds.labels[r["event_id"]].get("checker")]
    assert all(ws.store.record(t["event_id"]) is None for t in tasks)
    assert seen and all(v2ds.labels[r["event_id"]]["provenance"].startswith("delete:") for r in seen)


def test_time_trigger_fires_once_per_day(v2ds):
    pid = first(v2ds)
    ws = make_workspace(v2ds, pid)
    start = datetime.fromisoformat(v2ds.records[pid][0]["timestamp"]).replace(hour=0, minute=0)
    ws.advance_to(start + timedelta(days=3))
    wf = Workflow(id="brief", name="Briefing", source="template:test",
                  nodes=[Node(id="t", type="trigger.time", params={"at": "08:00"}),
                         Node(id="o", type="output.respond", params={"text": "Good morning"})],
                  edges=[Edge(source="t", target="o")])
    ws.add_trigger(wf, "time", {"at": "08:00"})
    fired = ws.advance_to(start + timedelta(days=5))
    assert len(fired) == 2 and all(f.answer == "Good morning" for f in fired)


def test_event_trigger_fires_on_matching_arrivals(v2ds):
    pid = first(v2ds)
    ws = make_workspace(v2ds, pid)
    sender = next(r["sender"] for r in v2ds.records[pid] if r["channel"] == "message" and r["sender"] not in
                  ("assistant", "me", "system"))
    wf = Workflow(id="ping", name="Ping", source="template:test",
                  nodes=[Node(id="t", type="trigger.event"), Node(id="o", type="output.respond",
                                                                  params={"text": "New from {{t.record.sender}}"})],
                  edges=[Edge(source="t", target="o")])
    ws.add_trigger(wf, "event", {"channel": "message", "sender": sender})
    fired = ws.advance_to(datetime(2026, 12, 31))
    n = sum(1 for r in v2ds.records[pid] if r["channel"] == "message" and sender.lower() in (r["sender"] or "").lower())
    assert len(fired) == n and fired[0].answer == f"New from {sender}"


def test_agent_tools_run_as_engine_nodes(v2ds):
    pid = first(v2ds)
    ws = make_workspace(v2ds, pid)
    ws.advance_to(datetime(2026, 12, 31))
    eid = ws.engine.open("find gym", "baseline")
    tools = ws.tools(eid)
    out = tools.execute("search_records", {"query": "meeting"})
    assert out.startswith("[") or out == "No matching records."
    assert "Error" in tools.execute("send_email", {"to": "Nobody Known Here", "body": "x"})
    res = ws.engine.close(eid, "done")
    assert [n["type"] for n in res.workflow["nodes"]][1:3] == ["context.search", "action.send_email"]
    assert ws.store.egress_log(pid) == []  # no model calls were made, and nothing left the device
