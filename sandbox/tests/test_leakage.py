from p17sim.leakage import canary_leak, eval_window_leak, history_view, near_duplicates, persona_overlap
from p17sim.personas import make_persona
from p17sim.skeleton import build_trace

A = build_trace(make_persona(7, 0, "retiree"), 7)
B = build_trace(make_persona(7, 1, "parent"), 7)
LONG = "the quarterly planning review moved to the large room on the third floor next to the kitchen"


def test_persona_in_two_splits_is_found():
    assert persona_overlap({"train": A, "test": A[:1]})
    assert persona_overlap({"train": A, "test": B}) == []


def test_history_view_hides_eval_window():
    assert eval_window_leak(history_view(A)) == []
    assert eval_window_leak(A)  # raw trace contains eval-day tasks


def test_canary_in_other_persona_is_found():
    canary = next(e.canary for e in A if e.canary)
    planted = B[0].model_copy(update={"text": f"see {canary}"})
    assert canary_leak(A + [planted])
    assert canary_leak(A + B) == []


def test_near_duplicate_label_text_across_splits():
    a = next(e for e in A if e.fact_id).model_copy(update={"text": LONG})
    b = next(e for e in B if e.fact_id).model_copy(update={"text": LONG})
    assert near_duplicates({"train": [a], "test": [b]})
    assert near_duplicates({"train": [a, b]}) == []
