from p17sim.cli import main
from p17sim.personas import make_cohort
from p17sim.render import FakeBackend, render_events
from p17sim.schema import split_for_day
from p17sim.skeleton import build_trace as build_skeleton
from p17sim.trace import build_trace, clean_text, leakage_checks, pii_findings, validate_trace

COHORT = make_cohort(7, n_in=3, n_ood=0)


def _raw(tmp_path):
    events = [e for p in COHORT for e in build_skeleton(p, 7)]
    return render_events(events, {p.persona_id: p for p in COHORT}, FakeBackend(), tmp_path, workers=2)


def test_split_windows_match_workbook():
    assert [split_for_day(d) for d in (1, 20, 21, 24, 25, 30)] == ["train", "train", "val", "val", "test", "test"]


def test_clean_removes_llm_artifacts():
    text, fixes = clean_text('Here is a short message:\n"Hi Sam, the  gate code is **4821**."', ["4821"])
    assert text == "Hi Sam, the gate code is 4821."
    assert {"llm_preamble", "wrapping_quotes", "markdown", "whitespace"} <= set(fixes)


def test_pii_allows_synthetic_forms_and_flags_real_looking_ones():
    assert pii_findings("mail p0001@example.com or call 555-0142") == []
    assert {k for k, _ in pii_findings("mail bob@gmail.com or call 408-555-7788")} == {"email", "phone"}


def test_clean_redacts_pii_but_keeps_protected_values():
    text, fixes = clean_text("Call 408-555-7788 about 555-0142", ["555-0142"])
    assert "[PHONE]" in text and "555-0142" in text and "redacted_phone" in fixes


def test_missing_text_is_imputed_with_its_labels(tmp_path):
    raw = _raw(tmp_path)
    i = next(i for i, e in enumerate(raw) if e.canary)
    raw[i] = raw[i].model_copy(update={"text": None, "render_status": "render_failed"})
    built = build_trace(COHORT, raw, 7)
    rec = next(r for r in built["records"] if r["event_id"] == raw[i].event_id)
    assert raw[i].canary in rec["text"] and built["log"]["missing_text_imputed"] == 1


def test_clean_trace_validates_and_has_no_leakage(tmp_path):
    built = build_trace(COHORT, _raw(tmp_path), 7)
    assert validate_trace(built["records"], built["labels"]) == []
    assert leakage_checks(built["records"], built["labels"]) == []


def test_validation_catches_corruption(tmp_path):
    built = build_trace(COHORT, _raw(tmp_path), 7)
    records = [dict(r) for r in built["records"]]
    records[5]["timestamp"] = "2026-11-30T09:00:00"
    records[6]["event_type"] = "tweet"
    records[7]["text"] = "reach me at bob@gmail.com"
    kinds = {v.split(":")[0] for v in validate_trace(records, built["labels"])}
    assert {"timestamp_outside_window", "schema", "pii"} <= kinds


def test_build_trace_cli_writes_versioned_outputs(tmp_path):
    data, out = tmp_path / "v1", tmp_path / "trace"
    main(["generate", "--n-in", "3", "--n-ood", "0", "--seed", "7", "--out", str(data)])
    raw = _raw(tmp_path / "cache")
    from p17sim.io import write_jsonl
    write_jsonl(data / "events_rendered.jsonl", raw)
    assert main(["build-trace", "--data", str(data), "--renderer", "fake", "--out", str(out)]) == 0
    for name in ("trace.jsonl", "labels.jsonl", "raw_events.jsonl", "splits.json", "manifest.json", "eda/report.md"):
        assert (out / name).exists()
    assert main(["validate-trace", str(out / "trace.jsonl"), "--labels", str(out / "labels.jsonl")]) == 0
