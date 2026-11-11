"""Channel preference and source-grounded private-conflict regressions."""
import json

from test_guardplan import Fixture, FakeModel
from guardplan.core import GuardPlanError, write_json
from guardplan.planner import GuardPlan
from guardplan.recipients import RecipientBinding
from guardplan.time_claims import validate_time_text


class DraftGroundingTests(Fixture):
    def setUp(self):
        super().setUp()
        self.profile["contacts"][0]["preferred_channel"] = "sms"
        write_json(self.persona / "agent_memory/user_profile.json", self.profile)

    def binding(self, query):
        binding = RecipientBinding(self.profile["contacts"])
        binding.bind(query)
        return binding

    def test_generic_reply_uses_sms_preference(self):
        binding = self.binding("Friend asked for a meeting. Draft a reply.")
        self.assertEqual(binding.channel_tools(), ("draft_sms",))
        with self.assertRaisesRegex(GuardPlanError, "channel mismatch"):
            binding.validate("friend@example.test", "draft_email")
        self.assertEqual(binding.validate("+1555001", "draft_sms")["contact_id"], "friend")

    def test_explicit_email_overrides_sms_preference(self):
        for query in ("Draft an email to Friend.", "Draft a reply email to Friend.",
                      "Reply to Friend via email."):
            with self.subTest(query=query):
                binding = self.binding(query)
                self.assertEqual(binding.channel_tools(), ("draft_email",))
                self.assertEqual(binding.validate("friend@example.test", "draft_email")["contact_id"], "friend")

    def test_explicit_address_overrides_preference(self):
        binding = self.binding("Draft a reply to friend@example.test.")
        self.assertEqual(binding.channel_tools(), ("draft_email",))
        binding.validate("friend@example.test", "draft_email")

    def test_incoming_address_does_not_override_explicit_outgoing_channel(self):
        self.profile["contacts"][0]["phone"] = "+15551234567"
        binding = self.binding("Friend texted from +15551234567. Draft an email to Friend.")
        self.assertEqual(binding.channel_tools(), ("draft_email",))
        binding.validate("friend@example.test", "draft_email")

    def test_incoming_email_does_not_pin_generic_reply_channel(self):
        binding = self.binding("Friend emailed from friend@example.test. Draft a reply to Friend.")
        self.assertEqual(binding.channel_tools(), ("draft_sms",))

    def test_quoted_channel_instructions_do_not_override_preference(self):
        binding = self.binding("Friend asked: 'Draft an email to me.' Draft a reply.")
        self.assertEqual(binding.channel_tools(), ("draft_sms",))

    def test_preference_snapshot_cannot_be_changed_after_binding(self):
        binding = RecipientBinding(self.profile["contacts"])
        self.profile["contacts"][0]["preferred_channel"] = "email"
        binding.bind("Draft a reply to Friend.")
        self.assertEqual(binding.channel_tools(), ("draft_sms",))

    def test_missing_preferred_address_falls_back_to_available_address(self):
        self.profile["contacts"][0].pop("phone")
        binding = self.binding("Draft a reply to Friend.")
        self.assertEqual(binding.channel_tools(), ("draft_email",))

    def test_explicit_unavailable_channel_cannot_complete_without_draft(self):
        self.profile["contacts"][0].pop("email")
        write_json(self.persona / "agent_memory/user_profile.json", self.profile)
        model = FakeModel()
        result = GuardPlan(self.adapter(), model).run("Draft an email to Friend.")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("channel", result["error"])
        self.assertEqual(model.calls, [])

    def test_wrong_channel_is_repaired_without_saving_wrong_draft(self):
        model = FakeModel([{"subgoals": ["Reply"]},
            {"tool": "draft_email", "args": {"to": "friend@example.test", "subject": "Hello", "body": "Hello"},
             "evidence_ids": ["user_request"]},
            {"tool": "draft_sms", "args": {"to": "+1555001", "body": "Hello"}, "evidence_ids": ["user_request"]},
            {"done": True}])
        adapter = self.adapter()
        result = GuardPlan(adapter, model).run("Draft a reply to Friend.")
        self.assertEqual(result["status"], "COMPLETE", result)
        self.assertEqual([s["tool"] for s in result["steps"]], ["draft_sms"])
        self.assertEqual(len(list((adapter.state_dir / "drafts").glob("*.json"))), 1)
        self.assertEqual(model.seen[1]["intended_recipient"]["allowed_draft_tools"], ["draft_sms"])

    def test_explicit_email_still_completes_in_controller(self):
        model = FakeModel([{"subgoals": ["Reply by email"]},
            {"tool": "draft_email", "args": {"to": "friend@example.test", "subject": "Hello", "body": "Hello"},
             "evidence_ids": ["user_request"]}, {"done": True}])
        result = GuardPlan(self.adapter(), model).run("Draft an email to Friend.")
        self.assertEqual(result["status"], "COMPLETE", result)
        self.assertEqual(result["steps"][0]["tool"], "draft_email")

    def assessed_plan(self, recipient="Friend"):
        planner = GuardPlan(self.adapter(), FakeModel())
        planner.recipient_binding.bind("Draft a reply to " + recipient + ".")
        planner.requested_times = {"13:15"}
        planner.assess_requested_slots("2026-11-10")
        return planner

    def test_private_reason_uses_verified_policy_and_records_evidence(self):
        plan = self.assessed_plan()
        args = {"to": "+1555001", "body": "I have a sensitive early meeting then. Would 3 PM work?"}
        result, evidence = plan.ground_private_reason(args)
        self.assertEqual(result["body"], "I have a personal appointment then. Would 3 PM work?")
        self.assertEqual(evidence, ["cal_001"])
        self.assertIn("sensitive early meeting", args["body"])  # input is not mutated

    def test_reason_rewrite_is_idempotent(self):
        plan = self.assessed_plan()
        result, _ = plan.ground_private_reason({"body": "I have a sensitive early meeting then."})
        again, evidence = plan.ground_private_reason(result)
        self.assertEqual(again, result)
        self.assertEqual(evidence, [])

    def test_reason_is_not_invented_without_verified_overlap(self):
        plan = self.assessed_plan()
        plan.schedule_assessments = []
        args = {"body": "I have a sensitive early meeting then."}
        self.assertEqual(plan.ground_private_reason(args), (args, []))

    def test_authorized_recipient_and_unrelated_private_meeting_are_preserved(self):
        args = {"body": "I have a private appointment then."}
        self.assertEqual(self.assessed_plan("Doctor").ground_private_reason(args), (args, []))
        args = {"body": "Would a private meeting about the project at 3 PM work?"}
        self.assertEqual(self.assessed_plan().ground_private_reason(args), (args, []))

    def test_fixed_reason_and_workflow_match_saved_draft(self):
        model = FakeModel([{"subgoals": ["Check calendar and reply"]},
            {"tool": "search_calendar", "args": {"date": "2026-11-10"}, "evidence_ids": ["user_request"]},
            {"tool": "draft_sms", "args": {"to": "+1555001", "body":
                "I can't meet at 1:15 PM. I have a sensitive early meeting then. Would 3 PM work?"},
             "evidence_ids": ["user_request"], "schedule_proposals": [
                 {"date": "2026-11-10", "start_time": "15:00", "end_time": "15:30"}]}, {"done": True}])
        adapter = self.adapter()
        result = GuardPlan(adapter, model).run("Friend asked: 'Can we meet Tuesday Nov 10 at 1:15 PM?' Check my calendar and draft a text reply.")
        self.assertEqual(result["status"], "COMPLETE", result)
        draft = result["steps"][-1]
        saved = json.loads((adapter.state_dir / "drafts" / (draft["draft_id"] + ".json")).read_text())
        self.assertEqual(saved["parameters"]["body"], draft["args"]["body"])
        self.assertIn("personal appointment", saved["parameters"]["body"])
        self.assertNotIn("sensitive early meeting", saved["parameters"]["body"])
        self.assertIn("cal_001", draft["evidence_ids"])
        self.assertLessEqual(result["telemetry"]["executor_turns"], 8)

    def test_live_mixed_refusal_and_rule_are_formatted_without_weakening_checks(self):
        proposals = [{"date": "2026-11-10", "start_time": "12:00", "end_time": "12:30"}]
        args = {"body": "Hi Marcus, I'm not free at 10:00 AM on Tuesday Nov 10 because my earliest meeting time is 10:30 AM. However, I can sync with you at 12:00 PM that day."}
        result = GuardPlan.separate_refusal_rule(args)
        self.assertIn("Nov 10. My earliest meeting time is 10:30 AM", result["body"])
        self.assertEqual(validate_time_text(result["body"], proposals, {"10:00"}, "10:30"), [])
        for old, new in (("10:30 AM", "9:30 AM"), ("12:00 PM that day", "10:00 AM that day")):
            unsafe = GuardPlan.separate_refusal_rule({"body": args["body"].replace(old, new)})
            self.assertTrue(validate_time_text(unsafe["body"], proposals, {"10:00"}, "10:30"))
        self.assertEqual(GuardPlan.separate_refusal_rule(result), result)

    def test_mixed_refusal_rule_completes_without_consuming_repair_budget(self):
        self.profile["preferences"]["earliest_meeting_time"] = "10:30"
        write_json(self.persona / "agent_memory/user_profile.json", self.profile)
        model = FakeModel([{"subgoals": ["Check calendar and reply"]},
            {"tool": "search_calendar", "args": {"date": "2026-11-10"}, "evidence_ids": ["user_request"]},
            {"tool": "draft_sms", "args": {"to": "+1555001", "body":
                "I'm not free at 10 AM on Tuesday Nov 10 because my earliest meeting time is 10:30 AM. However, would 3 PM work?"},
             "evidence_ids": ["user_request"], "schedule_proposals": [
                 {"date": "2026-11-10", "start_time": "15:00", "end_time": "15:30"}]}, {"done": True}])
        result = GuardPlan(self.adapter(), model).run("Friend wants to meet Tuesday Nov 10 at 10 AM. Check my calendar and draft a reply.")
        self.assertEqual(result["status"], "COMPLETE", result)
        self.assertEqual(result["telemetry"]["executor_turns"], 3)
        self.assertFalse(any(t.get("event") == "repair" for t in result["audit"]["trace"]))
