from p17sim.personas import make_persona
from p17sim.skeleton import build_trace
from p17sim.validate import validate_rows


def _rows():
    return [e.model_dump(mode="json") for e in build_trace(make_persona(7, 0, "retiree"), 7)]


def _kinds(violations):
    return {v.split(":")[0] for v in violations}


def test_clean_trace_passes_validation():
    assert validate_rows(_rows()) == []


def test_out_of_window_day_is_a_schema_violation():
    rows = _rows()
    rows[0]["day"] = 45
    assert "schema" in _kinds(validate_rows(rows))


def test_deletion_of_unknown_fact_is_caught():
    rows = _rows()
    row = next(r for r in rows if r["type"] == "deletion_request" or r["type"] == "message")
    rows.append({**row, "event_id": "bad-1", "type": "deletion_request", "fact_id": "no-such-fact", "day": 20})
    assert "deletion_request_without_prior_fact" in _kinds(validate_rows(rows))


def test_task_without_checker_is_caught():
    rows = _rows()
    task = next(r for r in rows if r["type"] == "user_task")
    task["checker"] = None
    assert "task_without_checker" in _kinds(validate_rows(rows))


def test_sensitive_without_canary_is_caught():
    rows = _rows()
    next(r for r in rows if r["sensitivity"] != "none")["canary"] = None
    assert "sensitive_without_canary" in _kinds(validate_rows(rows))


def test_rendered_text_missing_fact_is_caught():
    rows = _rows()
    row = next(r for r in rows if r["fact_value"] and r["type"] == "message")
    row.update(text="no fact here", render_status="ok")
    assert "rendered_text_missing_slot" in _kinds(validate_rows(rows))


def test_duplicate_event_id_is_caught():
    rows = _rows()
    rows.append(dict(rows[0]))
    assert "duplicate_event_id" in _kinds(validate_rows(rows))
