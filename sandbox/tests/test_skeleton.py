from p17sim.personas import make_persona
from p17sim.schema import HISTORY_LAST_DAY, EventType
from p17sim.skeleton import build_trace, trace_hash


def _trace(seed=7, idx=0, arch="grad_student"):
    return build_trace(make_persona(seed, idx, arch), seed)


def test_trace_is_deterministic():
    assert trace_hash(_trace()) == trace_hash(_trace())


def test_seed_changes_trace():
    assert trace_hash(_trace(seed=7)) != trace_hash(_trace(seed=8))


def test_event_ids_unique_and_time_ordered():
    events = _trace()
    assert len({e.event_id for e in events}) == len(events)
    assert [(e.day, e.minute) for e in events] == sorted((e.day, e.minute) for e in events)


def test_tasks_only_in_eval_window():
    tasks = [e for e in _trace() if e.type == EventType.TASK]
    assert tasks and all(t.day > HISTORY_LAST_DAY for t in tasks)


def test_behavior_labels_follow_preference_drift():
    persona = make_persona(7, 0, "grad_student")
    prefs = {p.key: p for p in persona.preferences}
    for e in build_trace(persona, 7):
        if e.type == EventType.BEHAVIOR:
            assert e.preference_value == prefs[e.preference_key].value_on(e.day)


def test_update_values_are_unambiguous():
    for idx in range(30):
        for e in _trace(idx=idx):
            if e.checker and e.checker.kind == "update":
                new, old = e.checker.args["expected"], e.checker.args["stale"]
                assert new not in old and old not in new
