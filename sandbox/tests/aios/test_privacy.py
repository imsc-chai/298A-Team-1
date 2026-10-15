import pytest

from p17sim.aios.privacy import DeletionManager, EgressBlocked, EgressGateway, Policy, classify_record, max_class
from p17sim.aios.store import Store
from test_store import rec


def test_classes_order_and_classification():
    assert max_class("public", "sensitive", "personal") == "sensitive"
    assert classify_record({"permission_label": "deny"}) == "sensitive"
    assert classify_record({"permission_label": "read"}) == "personal"


def test_location_policy_keeps_personal_data_local():
    p = Policy(Store(), "q1")
    assert p.may_process("local", "sensitive")[0]
    assert not p.may_process("cloud", "personal")[0]
    assert p.may_process("cloud", "public")[0]


def test_tool_policy_follows_permission_records_and_overrides():
    s = Store()
    for tool, label in [("mail.send", "draft"), ("calendar.create", "send"), ("memory.forget", "deny")]:
        s.add_record(rec(tool, f"{tool}: {label}", channel="permission", perm=label), "public")
    p = Policy(s, "q1")
    assert (p.tool("mail.send"), p.tool("calendar.create"), p.tool("memory.forget")) == ("ask", "allow", "deny")
    assert p.tool("calendar.read") == "allow"
    p.set_tool("mail.send", "allow")
    assert Policy(s, "q1").tool("mail.send") == "allow"


def test_gateway_blocks_hosts_off_the_allowlist():
    s = Store()
    gw = EgressGateway(s)
    with pytest.raises(EgressBlocked):
        gw.post("https://api.example.com/v1/chat", b"{}", purpose="test", persona_id="q1")
    row = s.egress_log("q1")[0]
    assert row["host"] == "api.example.com" and row["allowed"] == 0


def test_gateway_counts_persona_text_on_allowed_calls(monkeypatch):
    s = Store()
    gw = EgressGateway(s)
    monkeypatch.setattr(gw, "_send", lambda url, body, timeout: b"{}")
    gw.post("http://localhost:11434/api/chat", b'{"x": "my gate code is 4821 at home"}', purpose="llm",
            persona_id="q1", persona_texts=["my gate code is 4821 at home"])
    row = s.egress_log("q1")[0]
    assert row["allowed"] == 1 and row["persona_bytes"] == len("my gate code is 4821 at home")


def test_deletion_removes_the_value_from_every_store():
    s = Store()
    s.add_record(rec(1, "Hi from Ana: your building gate code is 4821."), "personal")
    s.add_record(rec(2, "Ana says dinner is at 7."), "personal")
    s.add_node_run({"execution_id": "x1", "persona_id": "q1", "node_id": "n1", "type": "context.search",
                    "status": "ok", "location": "local", "data_class": "personal", "input": '{"q": "gate"}',
                    "output": '[{"id": "q1-e1", "text": "Hi from Ana: your building gate code is 4821."}]',
                    "started": "t", "wall_s": 0.1})
    s.add_approval({"id": "a1", "persona_id": "q1", "execution_id": "x1", "node_id": "n2", "tool": "mail.send",
                    "request": {"body": "The code is 4821"}, "ts": "t"})
    dm = DeletionManager(s)
    receipt = dm.delete("q1", ["q1-e1"], "Please forget my gate code", actor="user", ts="2026-09-20T09:00:00")
    assert dm.storage_audit("q1", "4821") == []
    assert receipt["redactions"] >= 2
    assert "Ana" in s.record("q1-e2")["text"]  # shared tokens are not redacted
    assert s.search("q1", "gate code") == []
