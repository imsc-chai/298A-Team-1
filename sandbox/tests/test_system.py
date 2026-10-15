from collections import defaultdict
from datetime import datetime

import pytest

from p17sim.agents.baseline import BaselineAgent, truncate_if_needed
from p17sim.agents.model_a import ModelA, ModelAConfig, Trained, parse_date, preference_window
from p17sim.agents.reference import NullAgent, OracleAgent
from p17sim.agents.train import train_all
from p17sim.gen2.build import export, generate, load
from p17sim.gen2.render import render_cohort
from p17sim.io import write_jsonl
from p17sim.render import FakeBackend
from p17sim.system.llm import ScriptedChat
from p17sim.system.runner import Dataset, run_persona
from p17sim.system.shell import Shell
from p17sim.system.text import contains_value


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    root = tmp_path_factory.mktemp("sys")
    generate(root / "v2", 2026, 4, 2, 4, 2)
    personas, extras, events, splits = load(root / "v2")
    fake = {r: FakeBackend() for r in ("ollama:llama3.2", "ollama:phi3:mini")}
    write_jsonl(root / "v2" / "events_rendered.jsonl",
                render_cohort(events, personas, extras, root / "cache", 2, backends=fake))
    res = export(root / "v2", root / "trace")
    assert res["violations"] == [] and res["leakage"] == []
    d = Dataset.load(root / "trace", root / "v2")
    d.root = root
    return d


def _rates(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r["passed"])
    return {k: sum(v) / len(v) for k, v in by.items()}


def test_oracle_passes_everything(ds):
    rows = []
    for pid in ds.personas:
        rows += run_persona(ds, pid, lambda s, p, x: OracleAgent(s, p, x, ds), ("val", "test"))
    rates = _rates(rows)
    assert set(rates) >= {"recall", "privacy", "scheduling", "deletion_audit"}
    assert all(v == 1.0 for v in rates.values()), rates


def test_null_fails_utility_tasks(ds):
    rows = []
    for pid in ds.personas:
        rows += run_persona(ds, pid, lambda s, p, x: NullAgent(), ("val", "test"))
    rates = _rates(rows)
    for kind in ("recall", "update", "privacy", "scheduling"):
        if kind in rates:
            assert rates[kind] == 0.0, kind
    assert rates.get("deletion", 1.0) == 1.0  # answering nothing never reveals a deleted value
    assert rates["deletion_audit"] == 0.0  # but the records are still there


def test_shell_permissions_and_rollback(ds):
    pid = next(iter(ds.personas))
    p = ds.personas[pid]
    sh = Shell(persona=p, permissions={"mail.send": "draft"}, now=datetime(2026, 9, 20, 9))
    out = sh.execute("send_email", {"to": p.contacts[0].name, "body": "hi"})
    assert "approved" in out and sh.confirmations == 1
    out = sh.execute("create_event", {"date": "2026-09-21", "start": "10:00", "duration_minutes": 30,
                                      "with_contact": p.contacts[0].name})
    assert out.startswith("Created")
    assert sh.execute("cancel_event", {"event_id": "new-1"}).startswith("Cancelled")
    assert sh.calendar_on(datetime(2026, 9, 21).date()) == []
    assert [entry["tool"] for entry in sh.log] == ["send_email", "create_event", "cancel_event"]


def test_value_matching_ignores_spacing_but_not_digits():
    assert contains_value("It's P3 217 on level 3", "P3-217")
    assert contains_value("level 3, Spot 217", "level 3 spot 217")
    assert not contains_value("code 14821", "4821")


def test_baseline_loop_calls_tools_then_answers(ds):
    pid = next(iter(ds.personas))
    sh = Shell(persona=ds.personas[pid], permissions={}, now=datetime(2026, 9, 30, 9))
    llm = ScriptedChat([{"tool_calls": [{"id": "1", "name": "search_records", "arguments": {"query": "gate code"}}]},
                        {"content": "<think>x</think>Your gate code is 1234."}])
    res = BaselineAgent(sh, ds.personas[pid], llm).run("What is my gate code?")
    assert res.answer == "Your gate code is 1234." and res.turns == 2
    assert sh.log[0]["tool"] == "search_records"
    assert llm.seen[1][-1]["role"] == "tool"


def test_openjarvis_truncation_rule():
    msgs = [{"role": "system", "content": "s" * 100}, {"role": "user", "content": "u" * 20000}]
    out = truncate_if_needed(msgs)
    assert out[1]["content"].endswith("[Input truncated to fit context window]")
    assert sum(len(m["content"]) for m in out) // 4 <= 3100


def test_model_a_trains_and_runs_every_plan(ds, tmp_path):
    report = train_all(ds, tmp_path / "model_a", use_embeddings=False)
    assert report["reranker"]["grid"] and report["intent"]["grid"] and report["preference"]["chosen"]
    trained = Trained.load(tmp_path / "model_a")
    cfg = ModelAConfig.trained(tmp_path / "model_a")
    replies = [{"content": "It is 1234."}] * 200
    test_pids = [p for p, s in ds.splits.items() if s == "test"]
    rows = []
    for pid in test_pids:
        rows += run_persona(ds, pid, lambda s, p, x: ModelA(s, p, ScriptedChat(replies), cfg, trained,
                                                            use_embeddings=False), ("val", "test"))
    assert rows and all(r["error"] is None for r in rows), [r["error"] for r in rows if r["error"]]
    assert {r["agent_trace"][0]["intent"] for r in rows if r["agent_trace"]} >= {"lookup", "send", "schedule"}
    audits = [r for r in rows if r["kind"] == "deletion_audit"]
    assert audits and all(r["passed"] for r in audits)


def test_deletion_unaware_ablation_fails_the_storage_audit(ds, tmp_path):
    train_all(ds, tmp_path / "m", use_embeddings=False)
    trained = Trained.load(tmp_path / "m")
    cfg = ModelAConfig.trained(tmp_path / "m", deletion_aware=False)
    rows = []
    for pid in ds.personas:
        rows += run_persona(ds, pid, lambda s, p, x: ModelA(s, p, ScriptedChat([]), cfg, trained,
                                                            use_embeddings=False), ("val",), kinds={"recall"})
    audits = [r for r in rows if r["kind"] == "deletion_audit"]
    assert audits and not any(r["passed"] for r in audits)
    assert all(r["left_in_agent_memory"] for r in audits)


def test_preference_window_weights_recent_feedback():
    now = datetime(2026, 9, 30, 9)
    recs = [{"channel": "preference", "timestamp": "2026-09-10T10:00:00",
             "text": "Suggested a morning meeting slot: accepted"}] * 5
    recs += [{"channel": "preference", "timestamp": "2026-09-29T10:00:00",
              "text": "I'd rather have meetings in the evening."}]
    assert preference_window(recs, now, half_life=3, w_feedback=3, w_decline=0.5) == "evening"
    assert preference_window(recs, now, half_life=1000, w_feedback=1, w_decline=0.5) == "morning"


def test_parse_date_relative_forms():
    now = datetime(2026, 9, 30, 9)  # Wednesday
    assert str(parse_date("this Friday", now)) == "2026-10-02"
    assert str(parse_date("on the 3rd", now)) == "2026-10-03"
    assert parse_date("sometime soon", now) is None


def test_report_builds_from_run_files(ds, tmp_path):
    from p17sim.analysis.report import build
    from p17sim.system.runner import evaluate

    test_pids = sorted(p for p, s in ds.splits.items() if s == "test")
    ood_pids = sorted(p for p, s in ds.splits.items() if s == "ood_test")
    runs = tmp_path / "runs"
    oracle = lambda s, p, x: OracleAgent(s, p, x, ds)  # noqa: E731
    for seed in (11, 22):
        evaluate(ds, oracle, test_pids, ("test",), runs / f"model_a_s{seed}_test_test.jsonl", {})
        evaluate(ds, lambda s, p, x: NullAgent(), test_pids, ("test",), runs / f"baseline_s{seed}_test_test.jsonl", {})
        evaluate(ds, oracle, ood_pids, ("test",), runs / f"model_a_s{seed}_ood_test_test.jsonl", {})
    evaluate(ds, lambda s, p, x: NullAgent(), test_pids, ("test",), runs / "model_a-no_reranker_s11_test_test.jsonl", {})
    text = build(tmp_path)
    assert "Verdict: **superior**" in text
    assert "model_a-no_reranker" in text and "Fairness slices: model_a" in text
    assert (tmp_path / "figures" / "success_by_kind.png").exists()


def test_demo_replays_the_month_and_answers(ds):
    from p17sim.system.app import Demo

    demo = Demo(ds, "unused")
    pid = next(p["id"] for p in demo.personas() if p["split"] == "test")
    lab = next(ds.labels[r["event_id"]] for r in ds.records[pid]
               if (ds.labels[r["event_id"]].get("checker") or {}).get("kind") == "recall")
    req = next(r["text"] for r in ds.records[pid] if r["event_id"] == lab["event_id"])
    hhmm = f"{lab['minute'] // 60:02d}:{lab['minute'] % 60:02d}"
    out = demo.run({"persona_id": pid, "day": lab["day"], "time": hhmm, "system": "oracle", "request": req})
    assert out["answer"] == lab["checker"]["args"]["expected"]
    assert out["replayed"] > 0 and out["profile"]["raw_export_bytes"] == 0
