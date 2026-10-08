import pytest
from pydantic import ValidationError

from p17sim.schema import Event, EventType, Preference


def test_preference_drifts_on_drift_day():
    p = Preference(key="meeting_window", options=["morning", "evening"], initial="morning",
                   drift_day=10, drift_to="evening")
    assert p.value_on(9) == "morning" and p.value_on(10) == "evening"


def test_event_day_outside_window_rejected():
    with pytest.raises(ValidationError):
        Event(event_id="x", persona_id="p", day=31, minute=0, type=EventType.MESSAGE, provenance="t")
