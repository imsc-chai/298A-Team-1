import pytest

from p17sim.aios.nodes import NODES, RunContext, resolve_contact
from p17sim.aios.workflow import Edge, Node, Workflow, refs, resolve
from p17sim.system.llm import ScriptedChat


def chain(*types):
    nodes = [Node(id=f"n{i}", type=t) for i, t in enumerate(types)]
    return Workflow(id="w", name="w", source="test", nodes=nodes,
                    edges=[Edge(source=f"n{i}", target=f"n{i + 1}") for i in range(len(types) - 1)])


def test_order_is_topological_and_cycles_fail():
    wf = chain("trigger.chat", "context.search", "reason.llm", "output.respond")
    assert [n.id for n in wf.order()] == ["n0", "n1", "n2", "n3"]
    wf.edges.append(Edge(source="n3", target="n0"))
    with pytest.raises(ValueError, match="cycle"):
        wf.order()


def test_references_resolve_whole_values_and_inside_text():
    outs = {"s": {"items": [{"text": "gate 4821"}]}}
    assert resolve("{{ s.items.0.text }}", outs) == "gate 4821"
    assert resolve("{{s.items}}", outs) == [{"text": "gate 4821"}]
    assert resolve("Answer from: {{s.items.0.text}}", outs) == "Answer from: gate 4821"
    assert refs({"a": "{{s.items}} and {{t.x}}"}) == {"s", "t"}
    with pytest.raises(KeyError):
        resolve("{{s.nothing}}", outs)


def test_search_marks_sensitive_results(ws):
    out, cls = NODES["context.search"].run(RunContext(ws, "x"), {"query": "clinic chart"})
    assert out["items"][0]["id"] == "q1-e2" and cls == "sensitive"
    out, cls = NODES["context.search"].run(RunContext(ws, "x"), {"query": "gate code"})
    assert out["items"][0]["text"].endswith("4821") and cls == "personal"


def test_search_does_not_see_the_future(ws):
    out, _ = NODES["context.calendar"].run(RunContext(ws, "x"), {"date": "2026-09-26"})
    assert out["entries"][0]["title"] == "Gym"  # calendar shows planned entries
    out, _ = NODES["context.search"].run(RunContext(ws, "x"), {"query": "gym", "kind": "calendar"})
    assert out["items"] == []  # but search only returns what has already arrived


def test_create_event_writes_the_calendar_and_can_be_undone(ws):
    spec = NODES["action.create_event"]
    out, _ = spec.run(RunContext(ws, "x"), {"date": "2026-09-26", "start": "10:00", "duration_minutes": 30,
                                            "with_contact": "ana"})
    assert out["with"] == "Ana Silva" and ws.store.events("q1")[0]["start_min"] == 600
    spec.undo(RunContext(ws, "x"), out)
    assert [e["start"] for e in ws.store.calendar_on("q1", __import__("datetime").date(2026, 9, 26))] == ["18:00"]


def test_contact_resolution_reports_ambiguity():
    contacts = [{"id": "1", "name": "Ana Silva"}, {"id": "2", "name": "Ana Souza"}]
    c, msg = resolve_contact(contacts, "Ana")
    assert c is None and "Ambiguous" in msg
    assert resolve_contact(contacts, "ana souza")[0]["id"] == "2"


def test_llm_node_returns_text_and_json(ws):
    ws.llm = ScriptedChat([{"content": '{"slot": "10:00"}'}])
    out, cls = NODES["reason.llm"].run(RunContext(ws, "x", upstream_class="personal"), {"prompt": "x", "json": True})
    assert out["json"] == {"slot": "10:00"} and cls == "personal"
