from collections import Counter
from pathlib import Path

import pytest

from p17sim.gen2 import audit, phrasing
from p17sim.gen2.build import generate, load
from p17sim.gen2.personas import make_cohort, pref_value
from p17sim.gen2.render import build_prompt, render_cohort
from p17sim.gen2.skeleton import _free_slot, build_trace, trace_hash
from p17sim.gen2.trace import build, clean_text_v2, leakage, validate
from p17sim.render import FakeBackend
from p17sim.schema import EventType

SEED = 2026


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    out = tmp_path_factory.mktemp("v2")
    generate(out, SEED, 4, 2, 4, 2)
    personas, extras, events, splits = load(out)
    fake = {r: FakeBackend() for r in ("ollama:llama3.2", "ollama:phi3:mini")}
    rendered = render_cohort(events, personas, extras, out / "cache", workers=2, backends=fake)
    return personas, extras, rendered, splits


def test_same_seed_same_trace():
    a, xa, sa = make_cohort(SEED, 4, 2, 4, 2)
    b, xb, sb = make_cohort(SEED, 4, 2, 4, 2)
    ta = [e for p, x in zip(a, xa) for e in build_trace(p, x, SEED, sa[p.persona_id])]
    tb = [e for p, x in zip(b, xb) for e in build_trace(p, x, SEED, sb[p.persona_id])]
    assert trace_hash(ta) == trace_hash(tb)


def test_each_split_is_balanced_by_region_and_name_list():
    personas, extras, splits = make_cohort(SEED)
    for split in ("train", "test"):
        xs = [x for p, x in zip(personas, extras) if splits[p.persona_id] == split]
        assert max(Counter(x.region for x in xs).values()) - min(Counter(x.region for x in xs).values()) <= 1
        assert abs(Counter(x.name_list for x in xs)["female"] - Counter(x.name_list for x in xs)["male"]) <= 2


def test_second_renderer_only_in_held_out_splits():
    personas, extras, splits = make_cohort(SEED)
    for p, x in zip(personas, extras):
        if "phi3" in x.renderer:
            assert splits[p.persona_id] in ("test", "ood_test")


def test_pref_value_follows_every_change():
    personas, extras, _ = make_cohort(SEED)
    x = next(x for x in extras if len(x.pref_timeline["meeting_window"]) == 3)
    (d0, v0), (d1, v1), (d2, v2) = x.pref_timeline["meeting_window"]
    assert pref_value(x, "meeting_window", d1 - 1) == v0
    assert pref_value(x, "meeting_window", d1) == v1
    assert pref_value(x, "meeting_window", 30 if d2 <= 30 else d2) == v2


def test_scheduling_tasks_have_a_free_slot():
    personas, extras, splits = make_cohort(SEED, 4, 2, 4, 2)
    for p, x in zip(personas, extras):
        for e in build_trace(p, x, SEED, splits[p.persona_id]):
            if e.type == EventType.TASK and e.checker.kind == "scheduling":
                a = e.checker.args
                assert _free_slot(a["earliest"], a["window"][1], a["duration_minutes"],
                                  [tuple(b) for b in a["busy"]]) is not None


def test_prompt_names_owner_for_distractors(small):
    personas, extras, rendered, _ = small
    p_by, x_by = {p.persona_id: p for p in personas}, {x.persona_id: x for x in extras}
    d = next(e for e in rendered if e.provenance.startswith("distractor:"))
    prompt = build_prompt(d, p_by[d.persona_id], x_by[d.persona_id])
    assert d.payload["owner"] in prompt and "not" in prompt and d.fact_value in prompt


def test_full_pipeline_has_no_violations_or_leaks(small):
    personas, extras, rendered, splits = small
    built = build(personas, extras, rendered, splits)
    assert validate(built["records"], built["labels"], extras) == []
    assert leakage(built["records"], built["labels"], splits) == []
    assert built["log"]["pii_remaining"] == 0


def test_validator_names_a_corrupted_timestamp(small):
    personas, extras, rendered, splits = small
    built = build(personas, extras, rendered, splits)
    built["records"][5]["timestamp"] = "2031-01-01T00:00:00"
    assert any(v.startswith("timestamp_outside_window") for v in validate(built["records"], built["labels"], extras))


def test_heldout_wordings_stay_in_the_test_window(small):
    personas, extras, rendered, splits = small
    for e in rendered:
        if e.type == EventType.TASK and e.payload.get("wording") == "heldout":
            assert e.day > 24 and splits[e.persona_id] in ("test", "ood_test")


@pytest.mark.parametrize("raw,protected,fix", [
    ("Hi there.\n\n(12 words, casual tone)", [], "meta_commentary"),
    ("Locker E-26 is yours.\n\nBest,\n[Your Name]", ["E-26"], "placeholder"),
    ("Meet at Lantern Cafe.\n\n(15 words: includes the exact phrase", ["Lantern Cafe"], "meta_commentary"),
])
def test_v2_cleaner_strips_renderer_artifacts(raw, protected, fix):
    out, fixes = clean_text_v2(raw, protected)
    assert fix in fixes
    assert all(p in out for p in protected)
    assert "(" not in out and "[" not in out


def test_cleaner_keeps_real_parentheses():
    out, fixes = clean_text_v2("Dinner at 7 (at Maple Bistro).", ["Maple Bistro"])
    assert out == "Dinner at 7 (at Maple Bistro)." and fixes == []


def test_when_phrase_names_the_right_date():
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(50):
        phrase = phrasing.when_phrase(3, 22, 24, rng)
        d = phrasing.on_date(3, 24)
        assert any(s in phrase for s in (str(d.day), "day after tomorrow", phrasing.WEEKDAYS[d.weekday()]))


def test_audit_finds_v1_shortcuts_absent_in_v2(small):
    personas, extras, rendered, splits = small
    v1 = audit.view_v1(Path("data/v1")) if Path("data/v1/events.jsonl").exists() else None
    v2 = audit.View("v2", [], [], [], [x.sensitive_canaries[k] for x in extras for k in x.sensitive_canaries])
    assert audit.shortcuts(v2)["canary_caught_by_v1_regex"] == 0.0
    if v1:
        assert audit.shortcuts(v1)["canary_caught_by_v1_regex"] == 1.0


def test_package_locker_value_is_the_bare_code():
    import re

    import numpy as np

    from p17sim.gen2.skeleton import _value

    rng = np.random.default_rng(0)
    for _ in range(20):
        assert re.fullmatch(r"[A-F]-\d{1,2}", _value("package_locker", rng))
