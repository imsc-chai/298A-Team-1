import copy
import json

from test_guardplan import Fixture, FakeModel
from guardplan.core import MemoryRecord
from guardplan.planner import GuardPlan
from guardplan.privacy import Privacy
from guardplan.time_claims import validate_time_text


class ControllerContracts(Fixture):
    def test_explicit_preference_update_is_saved(self):
        adapter = self.adapter()
        model = FakeModel([{"subgoals": ["Save preference"]},
            {"tool": "update_profile", "args": {"field": "preferences.earliest_meeting_time", "value": "10:30"},
             "evidence_ids": ["user_request"]}, {"done": True}])
        result = GuardPlan(adapter, model).run("Don't schedule meetings before 10:30 AM. Please update my profile.")
        self.assertEqual(result["status"], "COMPLETE", result)
        self.assertEqual(adapter.get_profile()["preferences"]["earliest_meeting_time"], "10:30")

    def test_requested_update_cannot_save_a_different_value(self):
        adapter = self.adapter()
        before = copy.deepcopy(adapter.get_profile())
        model = FakeModel([{"subgoals": ["Save preference"]},
            {"tool": "update_profile", "args": {"field": "preferences.earliest_meeting_time", "value": "07:00"},
             "evidence_ids": ["user_request"]}])
        result = GuardPlan(adapter, model).run("Update my profile: no meetings before 10:30 AM.")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(adapter.get_profile(), before)

    def test_explicit_calendar_creation_is_still_allowed(self):
        adapter = self.adapter()
        args = {"title": "Sync", "date": "2026-11-10", "start_time": "15:00", "end_time": "15:30", "busy": True}
        model = FakeModel([{"subgoals": ["Create event"]},
            {"tool": "create_calendar_event", "args": args, "evidence_ids": ["user_request"]}, {"done": True}])
        result = GuardPlan(adapter, model).run("Please create a calendar event on Tuesday Nov 10 from 3 PM to 3:30 PM for a meeting.")
        self.assertEqual(result["status"], "COMPLETE", result)
        self.assertTrue(result["steps"][0]["result_id"])
        self.assertEqual(result["steps"][0]["validation"]["schedule_checks"][0]["violations"], [])

    def test_calendar_changes_need_user_authorization(self):
        adapter = self.adapter()
        before = adapter.search_events()
        model = FakeModel([{"subgoals": ["Check calendar"]},
            {"tool": "create_calendar_event", "args": {"title": "Unrequested", "date": "2026-11-10",
                "start_time": "15:00", "end_time": "15:30", "busy": True}, "evidence_ids": ["user_request"]}])
        result = GuardPlan(adapter, model).run("Check my calendar for Tuesday Nov 10.")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(adapter.search_events(), before)

    def test_valid_complete_dated_range_and_duration(self):
        proposals = [{"date": "2026-11-10", "start_time": "15:00", "end_time": "15:30"}]
        good = "Would Tuesday Nov 10, 2026 from 3 PM to 3:30 PM work for a 30-minute meeting?"
        self.assertEqual(validate_time_text(good, proposals, set()), [])
        self.assertTrue(validate_time_text(good.replace("30-minute", "45-minute"), proposals, set()))

    def test_factual_permission_does_not_authorize_a_meeting_offer(self):
        proposals = [{"date": "2026-11-10", "start_time": "15:00", "end_time": "15:30"}]
        self.assertEqual(validate_time_text("My flight departs at 8 AM.", proposals, set(), factual_times={"08:00"}), [])
        self.assertTrue(validate_time_text("My flight departs at 8 AM. Would 8 AM work?", proposals, set(), factual_times={"08:00"}))

    def test_nested_sensitive_fields_are_opaque_and_structure_remains(self):
        privacy = Privacy(self.policy, self.profile)
        record = MemoryRecord("health", "p", "s", "calendar_event", "Private record", sensitivity="health_and_medical",
            data={"is_sensitive": True, "diagnosis": {"private_detail": "unlisted disease"}, "busy": True,
                  "start_time": "08:30", "end_time": "09:30", "notes": "", "attendees": []})
        typed = privacy.record(record)
        self.assertNotIn("unlisted disease", json.dumps(typed))
        self.assertEqual(typed["data"]["start_time"], "08:30")
        self.assertTrue(typed["data"]["busy"])
        self.assertEqual(privacy.release("Generic reply", "friend"), "Generic reply")
