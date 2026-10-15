"""The harness on the real runtime (workspace: store, privacy layer, engine)."""
from collections import defaultdict
from datetime import datetime

import pytest

from p17sim.agents.baseline import BaselineAgent, truncate_if_needed
from p17sim.agents.registry import system
from p17sim.aios.generators import BaselineGenerator
from p17sim.gen2.build import export, generate, load
from p17sim.gen2.render import render_cohort
from p17sim.io import write_jsonl
from p17sim.render import FakeBackend
from p17sim.system.llm import ScriptedChat
from p17sim.system.runner import Dataset, System, evaluate, run_persona
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


def test_oracle_passes_everything_through_the_runtime(ds):
    rows = []
    for pid in ds.personas:
        rows += run_persona(ds, pid, system("oracle", ds), ("val", "test"))
    rates = _rates(rows)
    assert set(rates) >= {"recall", "privacy", "scheduling", "deletion_audit"}
    assert all(v == 1.0 for v in rates.values()), rates
    assert all(r["error"] is None for r in rows)


def test_null_fails_utility_tasks(ds):
    rows = []
    for pid in ds.personas:
        rows += run_persona(ds, pid, system("null", ds), ("val", "test"))
    rates = _rates(rows)
    for kind in ("recall", "update", "privacy", "scheduling"):
        if kind in rates:
            assert rates[kind] == 0.0, kind
    assert rates.get("deletion", 1.0) == 1.0  # answering nothing never reveals a deleted value
    assert rates["deletion_audit"] == 0.0  # but the records are still stored


def test_baseline_generator_runs_on_the_harness(ds):
    pid = next(p for p, s in sorted(ds.splits.items()) if s == "val")
    script = [{"tool_calls": [{"id": "1", "name": "search_records", "arguments": {"query": "code"}}]},
              {"content": "I found it."}] * 50
    sysm = System("baseline", lambda: ScriptedChat(list(script)), lambda ws: BaselineGenerator())
    rows = run_persona(ds, pid, sysm, ("val",), kinds={"recall", "update"})
    assert rows and all(r["error"] is None for r in rows)
    task = next(r for r in rows if r["kind"] in ("recall", "update"))
    assert [t["type"] for t in task["tool_calls"]][:3] == ["trigger.chat", "reason.agent_step", "context.search"]
    assert task["turns"] == 2 and task["profile"]["raw_export_bytes"] == 0


def test_shell_tool_schemas_are_the_agent_contract():
    names = [t["function"]["name"] for t in Shell.tool_schemas()]
    assert names == ["search_records", "read_record", "list_contacts", "list_calendar", "create_event",
                     "cancel_event", "send_email", "delete_records"]


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
    assert llm.seen[1][-1]["role"] == "tool"


def test_openjarvis_truncation_rule():
    msgs = [{"role": "system", "content": "s" * 100}, {"role": "user", "content": "u" * 20000}]
    out = truncate_if_needed(msgs)
    assert out[1]["content"].endswith("[Input truncated to fit context window]")
    assert sum(len(m["content"]) for m in out) // 4 <= 3100


def test_pending_systems_refuse_to_run(ds):
    with pytest.raises(NotImplementedError):
        system("model_a", ds)


def test_report_builds_from_run_files(ds, tmp_path):
    from p17sim.analysis.report import build

    test_pids = sorted(p for p, s in ds.splits.items() if s == "test")
    ood_pids = sorted(p for p, s in ds.splits.items() if s == "ood_test")
    runs = tmp_path / "runs"
    for seed in (11, 22):
        evaluate(ds, system("oracle", ds), test_pids, ("test",), runs / f"model_a_s{seed}_test_test.jsonl", {})
        evaluate(ds, system("null", ds), test_pids, ("test",), runs / f"baseline_s{seed}_test_test.jsonl", {})
        evaluate(ds, system("oracle", ds), ood_pids, ("test",), runs / f"model_a_s{seed}_ood_test_test.jsonl", {})
    evaluate(ds, system("null", ds), test_pids, ("test",), runs / "model_a-no_reranker_s11_test_test.jsonl", {})
    text = build(tmp_path)
    assert "Verdict: **superior**" in text
    assert "model_a-no_reranker" in text and "Fairness slices: model_a" in text
    assert (tmp_path / "figures" / "success_by_kind.png").exists()
