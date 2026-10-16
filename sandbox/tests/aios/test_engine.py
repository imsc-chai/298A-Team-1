from p17sim.aios.engine import Engine, auto_approve, queue_approval
from p17sim.aios.nodes import NODES, NodeSpec
from p17sim.aios.workflow import Edge, Node, Workflow
from p17sim.system.llm import ScriptedChat


def wf(nodes, edges):
    return Workflow(id="w1", name="t", source="test", nodes=[Node(**n) for n in nodes],
                    edges=[Edge(source=a, target=b, branch=br) for a, b, br in edges])


def set_tool(ws, tool, value):
    ws.store.set_policy("q1", "tool", tool, value)


def test_chain_runs_and_carries_the_data_class(ws):
    ws.llm = ScriptedChat([{"content": "Your gate code is 4821."}])
    w = wf([{"id": "t", "type": "trigger.chat", "params": {"goal": "gate code?"}},
            {"id": "s", "type": "context.search", "params": {"query": "gate code"}},
            {"id": "l", "type": "reason.llm", "params": {"prompt": "Answer from {{s.items}}"}},
            {"id": "o", "type": "output.respond", "params": {"text": "{{l.text}}"}}],
           [("t", "s", None), ("s", "l", None), ("l", "o", None)])
    res = Engine(ws).run(w, "test")
    assert res.status == "succeeded" and res.answer == "Your gate code is 4821."
    assert [r["data_class"] for r in res.node_runs] == ["personal"] * 4
    assert "4821" in ws.llm.seen[0][-1]["content"]


def test_cloud_node_with_personal_input_is_blocked(ws):
    w = wf([{"id": "s", "type": "context.search", "params": {"query": "gate code"}},
            {"id": "c", "type": "cloud.llm", "params": {"prompt": "{{s.items}}"}}], [("s", "c", None)])
    res = Engine(ws).run(w, "test")
    assert res.status == "blocked"
    assert res.node_runs[-1]["status"] == "blocked" and "4821" not in (res.node_runs[-1]["input"] or "")
    assert any(a["action"] == "blocked_location" for a in ws.store.audits("q1"))
    assert ws.store.egress_log("q1") == []  # nothing even tried to leave


def test_ask_policy_with_the_harness_approver_sends_and_counts(ws):
    set_tool(ws, "mail.send", "ask")
    w = wf([{"id": "m", "type": "action.send_email", "params": {"to": "Ben", "body": "hi"}}], [])
    res = Engine(ws).run(w, "test", approver=auto_approve)
    assert res.status == "succeeded" and res.interventions == 1 and len(ws.store.outbox("q1")) == 1


def test_ask_policy_in_the_app_waits_then_resumes(ws):
    set_tool(ws, "mail.send", "ask")
    w = wf([{"id": "m", "type": "action.send_email", "params": {"to": "Ben", "body": "hi"}},
            {"id": "o", "type": "output.respond", "params": {"text": "sent to {{m.to}}"}}], [("m", "o", None)])
    eng = Engine(ws)
    res = eng.run(w, "test", approver=queue_approval)
    assert res.status == "waiting_approval" and ws.store.outbox("q1") == []
    assert ws.store.approvals("q1", "pending")[0]["request"]["params"]["to"] == "Ben"
    done = eng.resume(res.approval_id, "approve")
    assert done.status == "succeeded" and done.answer == "sent to Ben Okafor" and len(ws.store.outbox("q1")) == 1


def test_rejected_action_does_not_run(ws):
    set_tool(ws, "mail.send", "ask")
    w = wf([{"id": "m", "type": "action.send_email", "params": {"to": "Ben", "body": "hi"}}], [])
    res = Engine(ws).run(w, "test", approver=lambda r: "rejected")
    assert res.status == "rejected" and ws.store.outbox("q1") == []


def test_failure_rolls_back_reversible_actions(ws):
    NODES["test.boom"] = NodeSpec("test.boom", "logic", "fails", {}, run=lambda ctx, p: (_ for _ in ()).throw(
        ValueError("boom")))
    w = wf([{"id": "e", "type": "action.create_event", "params": {"date": "2026-09-26", "start": "10:00",
                                                                   "duration_minutes": 30, "with_contact": "Ana"}},
            {"id": "b", "type": "test.boom"}], [("e", "b", None)])
    res = Engine(ws).run(w, "test")
    assert res.status == "failed" and ws.store.events("q1")[0]["cancelled"] == 1
    assert any(a["action"] == "rollback" for a in ws.store.audits("q1"))
    del NODES["test.boom"]


def test_deny_policy_stops_the_action(ws):
    set_tool(ws, "calendar.create", "deny")
    w = wf([{"id": "e", "type": "action.create_event", "params": {"date": "2026-09-26", "start": "10:00",
                                                                   "duration_minutes": 30, "with_contact": "Ana"}}], [])
    res = Engine(ws).run(w, "test")
    assert res.status == "denied" and ws.store.events("q1") == []


def test_if_branches_skip_the_other_side(ws):
    w = wf([{"id": "s", "type": "context.search", "params": {"query": "nothing matches this"}},
            {"id": "i", "type": "logic.if", "params": {"left": "{{s.items}}", "op": "empty"}},
            {"id": "y", "type": "output.respond", "params": {"text": "I don't have that information."}},
            {"id": "n", "type": "output.respond", "params": {"text": "{{s.items.0.text}}"}}],
           [("s", "i", None), ("i", "y", "true"), ("i", "n", "false")])
    res = Engine(ws).run(w, "test")
    assert res.answer == "I don't have that information."
    assert [r["status"] for r in res.node_runs][-1] == "skipped"


def test_stepwise_execution_records_an_agent_run_as_a_workflow(ws):
    eng = Engine(ws)
    eid = eng.open("What is my gate code?", "baseline")
    eng.record(eid, "reason.agent_step", {"content": "", "tool_calls": ["search"]})
    status, out = eng.step(eid, "context.search", {"query": "gate code"})
    assert status == "ok" and out["items"]
    status, err = eng.step(eid, "action.send_email", {"to": "Nobody At All", "body": "x"})
    assert status == "failed" and "No contact" in err
    res = eng.close(eid, "Your gate code is 4821.")
    assert [n["type"] for n in res.workflow["nodes"]] == ["trigger.chat", "reason.agent_step", "context.search",
                                                          "action.send_email", "output.respond"]
    assert res.answer == "Your gate code is 4821." and res.status == "succeeded"


def test_resumed_workflow_still_asks_for_later_actions(ws):
    """Regression (found in the UI): after one approval, later 'ask' actions were auto-approved."""
    set_tool(ws, "mail.send", "ask")
    w = wf([{"id": "ok", "type": "human.approval", "params": {"message": "Send this recap?"}},
            {"id": "m", "type": "action.send_email", "params": {"to": "Ben", "body": "hi"}}], [("ok", "m", None)])
    eng = Engine(ws)
    first = eng.run(w, "test", approver=queue_approval)
    second = eng.resume(first.approval_id, "approve")
    assert second.status == "waiting_approval" and ws.store.outbox("q1") == []
    third = eng.resume(second.approval_id, "approve")
    assert third.status == "succeeded" and len(ws.store.outbox("q1")) == 1
