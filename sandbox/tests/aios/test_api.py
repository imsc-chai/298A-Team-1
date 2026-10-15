import pytest
from fastapi.testclient import TestClient

from p17sim.aios.api import create_app
from p17sim.system.llm import ScriptedChat


@pytest.fixture
def client(v2ds, tmp_path):
    replies = [{"content": "A calm week with a few dinners."}] * 50
    app = create_app(v2ds, store_dir=tmp_path / "ws", results=tmp_path / "results",
                     llm_factory=lambda: ScriptedChat(list(replies)))
    return TestClient(app)


def val_pid(c):
    return next(p["id"] for p in c.get("/api/personas").json() if p["split"] == "val")


def test_personas_and_load(client):
    ps = client.get("/api/personas").json()
    assert ps[0]["split"] == "test" and ps[0]["contacts"]
    pid = val_pid(client)
    st = client.post(f"/api/ws/{pid}/load", json={"day": 25, "time": "09:00"}).json()
    assert st["records_seen"] > 0 and st["now"].endswith("09:00:00")


def test_template_waits_for_approval_then_sends(client):
    pid = val_pid(client)
    client.post(f"/api/ws/{pid}/load", json={"day": 25})
    to = client.get("/api/personas").json()
    to = next(p for p in to if p["id"] == pid)["contacts"][0]
    res = client.post(f"/api/ws/{pid}/goal", json={"generator": "template:weekly_recap", "params": {"to": to}}).json()
    assert res["status"] == "waiting_approval"
    pending = client.get(f"/api/ws/{pid}/approvals?status=pending").json()
    assert pending and pending[0]["tool"] == "human.approval"
    done = client.post(f"/api/ws/{pid}/approvals/{res['approval_id']}", json={"decision": "approve"}).json()
    assert done["status"] in ("waiting_approval", "succeeded")
    if done["status"] == "waiting_approval":  # mail.send may itself need an OK for this persona
        done = client.post(f"/api/ws/{pid}/approvals/{done['approval_id']}", json={"decision": "approve"}).json()
    assert done["status"] == "succeeded"
    ex = client.get(f"/api/ws/{pid}/executions/{done['execution_id']}").json()
    assert ex["sent"] and ex["sent"][0]["body"] == "A calm week with a few dinners."


def test_pending_models_return_409(client):
    pid = val_pid(client)
    r = client.post(f"/api/ws/{pid}/goal", json={"text": "hi", "generator": "model_a"})
    assert r.status_code == 409 and "not trained" in r.json()["detail"]


def test_delete_then_storage_audit_is_empty(client):
    pid = val_pid(client)
    client.post(f"/api/ws/{pid}/load", json={"day": 25})
    rec = next(r for r in client.get(f"/api/ws/{pid}/records?channel=message").json() if any(ch.isdigit() for ch in r["text"]))
    client.post(f"/api/ws/{pid}/delete", json={"record_ids": [rec["id"]], "request": "test"})
    hits = client.post(f"/api/ws/{pid}/storage-audit", json={"value": rec["text"][:40]}).json()["found_in"]
    assert hits == []
    assert client.get(f"/api/ws/{pid}/privacy").json()["receipts"][0]["record_ids"] == [rec["id"]]


def test_policy_update_and_privacy_view(client):
    pid = val_pid(client)
    client.put(f"/api/ws/{pid}/policy", json={"tool": "calendar.create", "value": "deny"})
    assert client.get(f"/api/ws/{pid}/policy").json()["tools"]["calendar.create"] == "deny"
    assert client.put(f"/api/ws/{pid}/policy", json={"tool": "x", "value": "maybe"}).status_code == 400
    priv = client.get(f"/api/ws/{pid}/privacy").json()
    assert set(priv["classes"]) <= {"personal", "sensitive", "public"} and priv["egress_summary"]["nonlocal_bytes"] == 0


def test_catalog_suggestions_and_benchmark(client):
    assert {g["status"] for g in client.get("/api/generators").json()} >= {"ready", "pending training"}
    pid = val_pid(client)
    s = client.get(f"/api/ws/{pid}/suggestions").json()
    assert s["models"]["model_b"] == "pending training" and "routines" in s
    assert client.get("/api/benchmark").json()["runs"] == []
