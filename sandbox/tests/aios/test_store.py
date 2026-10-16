from datetime import date

from p17sim.aios.store import Store


def rec(i, text, ts="2026-09-10T09:00:00", channel="message", perm="read", pid="q1", end=None):
    return {"event_id": f"{pid}-e{i}", "persona_id": pid, "timestamp": ts, "end": end, "channel": channel,
            "event_type": channel, "sender": "Ana", "text": text, "permission_label": perm, "deleted": False}


def test_search_ranks_the_matching_record_first():
    s = Store()
    s.add_record(rec(1, "dinner on friday was great"), "personal")
    s.add_record(rec(2, "your building gate code is 4821"), "personal")
    s.add_record(rec(3, "the weather is nice"), "personal")
    hits = s.search("q1", "what is my gate code")
    assert hits[0]["id"] == "q1-e2"


def test_deleted_and_future_records_are_not_visible():
    s = Store()
    s.add_record(rec(1, "gate code 4821"), "personal")
    s.add_record(rec(2, "gate code 9999", ts="2026-09-20T09:00:00"), "personal")
    s.mark_deleted(["q1-e1"])
    assert [r["id"] for r in s.search("q1", "gate code", before="2026-09-30T00:00:00")] == ["q1-e2"]
    assert s.search("q1", "gate code", before="2026-09-15T00:00:00") == []
    assert [r["id"] for r in s.visible("q1")] == ["q1-e2"]


def test_search_respects_persona_and_channel():
    s = Store()
    s.add_record(rec(1, "gate code 4821", pid="q1"), "personal")
    s.add_record(rec(1, "gate code 1111", pid="q2"), "personal")
    s.add_record(rec(2, "gate code note", channel="note"), "personal")
    assert {r["id"] for r in s.search("q1", "gate code", channels={"message"})} == {"q1-e1"}


def test_calendar_merges_routines_and_created_events():
    s = Store()
    s.add_record(rec(1, "Gym", ts="2026-09-21T18:00:00", channel="calendar", end="2026-09-21T19:00:00"), "personal")
    s.add_event({"id": "new-1", "persona_id": "q1", "date": "2026-09-21", "start_min": 600, "duration": 30,
                 "with_id": "q1-c0", "with_name": "Ana", "title": "Meeting with Ana", "execution_id": "x",
                 "ts": "2026-09-20T09:00:00"})
    cal = s.calendar_on("q1", date(2026, 9, 21))
    assert [(e["start"], e["end"]) for e in cal] == [("10:00", "10:30"), ("18:00", "19:00")]
    s.cancel_event("new-1")
    assert len(s.calendar_on("q1", date(2026, 9, 21))) == 1


def test_file_store_persists(tmp_path):
    path = tmp_path / "w.db"
    s = Store(path)
    s.add_record(rec(1, "gate code 4821"), "personal")
    s.close()
    assert Store(path).record("q1-e1")["text"] == "gate code 4821"


def test_text_columns_cover_every_place_user_text_can_land():
    cols = set(Store().text_columns())
    for need in [("records", "text"), ("node_runs", "input"), ("node_runs", "output"), ("outbox", "body"),
                 ("approvals", "request"), ("executions", "answer"), ("audit", "detail"), ("events_created", "title")]:
        assert need in cols
