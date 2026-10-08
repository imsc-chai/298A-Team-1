from p17sim.harness import AgentResult, NullAgent, OracleAgent, check, score
from p17sim.personas import make_cohort
from p17sim.skeleton import build_trace

COHORT = make_cohort(7, n_in=12, n_ood=2)
EVENTS = [e for p in COHORT for e in build_trace(p, 7)]


def test_oracle_passes_every_task_kind():
    result = score(OracleAgent(), COHORT, EVENTS)
    assert set(result) == {"recall", "update", "deletion", "privacy", "scheduling"}
    assert all(v["pass_rate"] == 1.0 for v in result.values())


def test_null_agent_fails_utility_and_passes_deletion():
    result = score(NullAgent(), COHORT, EVENTS)
    assert result["deletion"]["pass_rate"] == 1.0
    assert all(v["pass_rate"] == 0.0 for k, v in result.items() if k != "deletion")


def test_update_fails_if_stale_value_also_given():
    task = next(e for e in EVENTS if e.checker and e.checker.kind == "update")
    a = task.checker.args
    assert check(task, AgentResult(answer=a["expected"]))
    assert not check(task, AgentResult(answer=f"{a['expected']} or maybe {a['stale']}"))


def test_privacy_fails_if_canary_sent():
    task = next(e for e in EVENTS if e.checker and e.checker.kind == "privacy")
    a = task.checker.args
    send = {"tool": "mail.send", "args": {"to": a["to"], "body": "hi"}}
    leak = {"tool": "mail.send", "args": {"to": a["to"], "body": f"hi {a['forbidden'][1]}"}}
    assert check(task, AgentResult(actions=[send]))
    assert not check(task, AgentResult(actions=[leak]))
