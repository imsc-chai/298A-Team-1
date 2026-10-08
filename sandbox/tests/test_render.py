from p17sim.personas import make_persona
from p17sim.render import FakeBackend, build_prompt, render_event, render_events
from p17sim.skeleton import build_trace

PERSONA = make_persona(7, 0, "retiree")
EVENTS = build_trace(PERSONA, 7)
SENSITIVE = next(e for e in EVENTS if e.canary)


def test_prompt_lists_required_strings():
    prompt = build_prompt(SENSITIVE, PERSONA)
    assert SENSITIVE.fact_value in prompt and SENSITIVE.canary in prompt


def test_render_ok_contains_slots(tmp_path):
    out = render_event(SENSITIVE, PERSONA, FakeBackend(), tmp_path)
    assert out.render_status == "ok" and SENSITIVE.canary in out.text and SENSITIVE.fact_value in out.text


def test_render_retries_then_succeeds(tmp_path):
    assert render_event(SENSITIVE, PERSONA, FakeBackend(fail_first=2), tmp_path).render_status == "ok"


def test_render_fails_after_max_attempts(tmp_path):
    out = render_event(SENSITIVE, PERSONA, FakeBackend(fail_first=5), tmp_path)
    assert out.render_status == "render_failed" and out.text is None


def test_cache_avoids_second_call(tmp_path):
    backend = FakeBackend()
    first = render_event(SENSITIVE, PERSONA, backend, tmp_path)
    calls = sum(backend.calls.values())
    assert render_event(SENSITIVE, PERSONA, backend, tmp_path) == first
    assert sum(backend.calls.values()) == calls


def test_render_events_skips_templated_types(tmp_path):
    out = render_events(EVENTS, {PERSONA.persona_id: PERSONA}, FakeBackend(), tmp_path, workers=2)
    assert {e.render_status for e in out} == {"ok", "not_needed"}
