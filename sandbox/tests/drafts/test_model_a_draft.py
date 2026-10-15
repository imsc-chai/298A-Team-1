"""Model A draft (pending): its pure helpers still have tests; the end-to-end parts wait for Model A."""
from datetime import datetime

from p17sim.agents.drafts.model_a import parse_date, preference_window


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
