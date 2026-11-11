"""Synthetic behavior tests; test doubles are never benchmark model substitutes."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
import tempfile
import unittest
import urllib.error
from unittest.mock import patch
from pathlib import Path

from guardplan.adapter import DatasetAdapter
from guardplan.constraints import ConstraintChecker, draft_time_violations, mentioned_times, normalize_slot
from guardplan.core import BackendError, GuardPlanError, MemoryRecord, ModelOutputError, Settings, fingerprint, write_json
from guardplan.planner import GuardPlan
from guardplan.llm import CloudPlanner, LocalModel, parse_planner_response
from guardplan.privacy import Privacy
from guardplan.retrieval import Retriever
from guardplan.evaluation import score
from guardplan.recipients import RecipientBinding


SOURCE = Path(__file__).resolve().parents[3] / "Project Data"


class FakeModel:
    def __init__(self, responses=()):
        self.responses, self.seen, self.calls, self.embedded = list(responses), [], [], []
        self.systems = []

    def chat(self, system, payload, component="planner"):
        self.seen.append(copy.deepcopy(payload))
        self.systems.append(system)
        self.calls.append({})
        if not self.responses:
            raise BackendError("No scripted model responses remain")
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def embed(self, texts):
        self.embedded.extend(texts)
        return [[float(hashlib.sha256(t.encode()).digest()[i]) for i in range(8)] for t in texts]

    def telemetry(self, since=0):
        return {"llm_calls": len(self.calls)-since, "input_tokens": 0, "output_tokens": 0}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.workspace = Path(self.temp.name) / "derived"
        self.persona = self.root / "persona_test"
        self.root.mkdir()
        shutil.copyfile(SOURCE / "connectors.py", self.root / "connectors.py")
        shutil.copyfile(SOURCE / "tool_registry.json", self.root / "tool_registry.json")
        self.profile = {"persona_id": "persona_test", "contacts": [
            {"id": "friend", "name": "Friend", "email": "friend@example.test", "phone": "+1555001"},
            {"id": "doctor", "name": "Doctor", "email": "doctor@example.test"}],
            "preferences": {"meeting_window": "afternoon", "buffer_between_meetings_minutes": 15,
                            "max_meetings_per_day": 4,
                            "deep_work_focus_slots": [{"weekday": 5, "start": "13:00", "end": "17:00"}]},
            "routines": [], "action_execution_policy": {
                "allowed_autonomous_tools": ["read_local_file", "search_calendar", "read_messages", "lookup_contact",
                                             "create_calendar_event", "update_profile", "draft_email", "draft_sms"],
                "restricted_confirmation_tools": ["send_email", "send_sms", "delete_calendar_event"]}}
        self.policy = {"sensitive_categories": [{"category": "health_and_medical", "forbidden_tokens": ["secret diagnosis"],
                        "allowed_contacts": ["doctor"], "redaction_substitute": "personal appointment"},
                       {"category": "access_and_security_codes", "forbidden_tokens": ["gate code #4829"],
                        "allowed_contacts": ["doctor"], "redaction_substitute": "[code omitted]"}]}
        self.events = [{"id": "cal_001", "date": "2026-11-10", "weekday": "Tuesday",
                        "start_time": "13:00", "end_time": "14:00", "title": "secret diagnosis",
                        "is_sensitive": True, "busy": True}]
        write_json(self.persona / "agent_memory/user_profile.json", self.profile)
        write_json(self.persona / "agent_memory/privacy_policy.json", self.policy)
        write_json(self.persona / "streams/calendar.json", self.events)
        write_json(self.persona / "streams/messages.json", [
            {"id": "past", "timestamp": "2026-11-01T09:00:00", "sender": "a", "recipient": "b", "body": "old"},
            {"id": "future", "timestamp": "2026-11-30T09:00:00", "sender": "a", "recipient": "b", "body": "future secret"}])
        write_json(self.persona / "test_suite/eval_tasks.json", [])
        (self.persona / "local_files").mkdir()
        (self.persona / "local_files/notes.md").write_text("Private: secret diagnosis. Entry gate code #4829.")

    def adapter(self, reference_time="2026-11-07T10:00:00"):
        return DatasetAdapter(self.root, "persona_test", self.workspace, reference_time)


class AdapterTests(Fixture):
    def test_sparse_source_calendar_ids_do_not_collide_with_created_events(self):
        self.events[0]["id"] = "cal_002"
        write_json(self.persona / "streams/calendar.json", self.events)
        adapter = self.adapter()
        created = adapter.mutate("create_calendar_event", {"title": "New meeting", "date": "2026-11-11",
            "start_time": "15:00", "end_time": "15:30", "busy": True})
        self.assertNotEqual(created["id"], "cal_002")
        events = adapter.search_events()
        self.assertEqual(len({e["id"] for e in events}), len(events))
        self.assertEqual(self.adapter().search_events(date="2026-11-11")[0]["id"], created["id"])

    def test_preference_update_rejects_uncompilable_time_rules(self):
        adapter = self.adapter()
        with self.assertRaises(GuardPlanError):
            adapter.mutate("update_profile", {"field": "preferences.meeting_window", "value": "after 10:30 AM"})
        with self.assertRaises(GuardPlanError):
            adapter.mutate("update_profile", {"field": "preferences.earliest_meeting_time", "value": "after 10:30"})
        self.assertEqual(adapter.get_profile()["preferences"]["meeting_window"], "afternoon")
        self.assertFalse(adapter.journal)

    def test_connector_state_is_persistent_but_source_is_unchanged(self):
        before = fingerprint(self.root)
        adapter = self.adapter()
        self.assertEqual(type(adapter.connector).__name__, "LocalBenchmarkConnector")
        adapter.mutate("update_profile", {"field": "preferences.earliest_meeting_time", "value": "10:30"})
        adapter.mutate("create_calendar_event", {"title": "Meeting", "date": "2026-11-11", "start_time": "15:00", "end_time": "15:30", "busy": True})
        adapter.create_draft({"to": "friend@example.test", "body": "Hello"}, "sms")
        restored = self.adapter()
        self.assertEqual(restored.get_profile()["preferences"]["earliest_meeting_time"], "10:30")
        self.assertEqual(len(restored.search_events()), 2)
        self.assertNotIn("earliest_meeting_time", self.adapter("2026-11-06T10:00:00").get_profile()["preferences"])
        self.assertEqual(before, fingerprint(self.root))

    def test_temporal_messages_and_future_calendar(self):
        self.assertEqual([m["id"] for m in self.adapter().search_messages()], ["past"])
        self.assertEqual(len(self.adapter().search_events(date="2026-11-10")), 1)

    def test_paths_and_derived_workspace_are_confined(self):
        with self.assertRaises(GuardPlanError):
            DatasetAdapter(self.root, "persona_test", self.root / "generated", "2026-11-07T10:00:00")
        with self.assertRaises(GuardPlanError):
            self.adapter().read_file("../../notes.md")
        link = self.persona / "local_files/link.md"
        link.symlink_to(self.root / "tool_registry.json")
        with self.assertRaises(GuardPlanError):
            self.adapter().read_file("local_files/link.md")


class PrivacyTests(Fixture):
    def test_policy_provider_name_abbreviations_remain_sensitive(self):
        policy = copy.deepcopy(self.policy)
        policy["sensitive_categories"][0]["forbidden_tokens"].append("Dr. Robert Keller")
        privacy = Privacy(policy, self.profile)
        for name in ["Dr. Keller", "Dr Keller", "Doctor Keller", "Robert Keller"]:
            with self.subTest(name=name):
                with self.assertRaises(GuardPlanError):
                    privacy.check(name, "friend")
                self.assertEqual(privacy.release(privacy.type_text(name), "friend"), "personal appointment")
                self.assertEqual(privacy.release(privacy.type_text(name), "doctor"), name)

    def test_generated_health_phrases_use_policy_substitute_per_recipient(self):
        privacy = Privacy(self.policy, self.profile)
        for phrase in ["medical appointment", "health-related appointment", "doctor's appointment",
                       "doctor’s appointment", "healthcare visit", "therapy session", "rehabilitation session",
                       "MEDICAL APPOINTMENT", "medical reasons"]:
            with self.subTest(phrase=phrase):
                text = "I have a " + phrase + " then."
                safe = privacy.type_text(text)
                self.assertNotIn(phrase, safe)
                for recipient in ["friend", None]:
                    self.assertEqual(privacy.release(safe, recipient), "I have a personal appointment then.")
                    self.assertEqual(privacy.release(text, recipient), "I have a personal appointment then.")
                    with self.assertRaises(GuardPlanError):
                        privacy.check(text, recipient)
                self.assertEqual(privacy.release(safe, "doctor"), text)
                self.assertEqual(privacy.release(text, "doctor"), text)
                privacy.check(text, "doctor")

    def test_generic_health_phrase_does_not_union_different_category_acls(self):
        policy = {"sensitive_categories": [
            {"category": "child_health", "forbidden_tokens": ["child diagnosis"], "allowed_contacts": ["family"],
             "redaction_substitute": "personal appointment"},
            {"category": "patient_medical", "forbidden_tokens": ["patient diagnosis"], "allowed_contacts": ["clinician"],
             "redaction_substitute": "personal appointment"}]}
        privacy = Privacy(policy, {})
        for recipient in ["family", "clinician", None]:
            self.assertEqual(privacy.release("medical appointment", recipient), "personal appointment")
        self.assertEqual(privacy.release(privacy.type_text("child diagnosis"), "family"), "child diagnosis")

    def test_unrelated_text_and_privacy_ablations_keep_their_behavior(self):
        privacy = Privacy(self.policy, self.profile)
        args = {"to": "friend@example.test", "subject": "Meet tomorrow", "body": "I have a personal appointment at 15:00."}
        self.assertEqual(privacy.release(args, "friend"), args)
        self.assertEqual(Privacy({"sensitive_categories": []}, {}).release("medical appointment", None), "medical appointment")
        for options in [{"typing_enabled": False}, {"release_enabled": False}]:
            ablation = Privacy(self.policy, self.profile, **options)
            self.assertEqual(ablation.release("medical appointment", "friend"), "medical appointment")

    def test_evaluator_separates_category_disclosure_from_exact_tokens(self):
        task = {"task_id": "privacy", "reference_time": "2026-11-07T10:00:00"}
        workflow = {"status": "COMPLETE", "steps": [{"tool": "draft_sms", "status": "DRAFT",
                     "args": {"to": "+1555001", "body": "I have a medical appointment."}}],
                    "telemetry": {"latency_ms": 0, "input_tokens": 0, "output_tokens": 0}}
        result = score(task, workflow, self.profile, self.events,
                       Privacy(self.policy, self.profile), self.adapter().contact_id)
        self.assertEqual(result["leak_rate_exact"], 0)
        self.assertEqual(result["category_disclosure_rate"], 1)
        self.assertIsNone(result["semantic_leak_rate"])

    def test_evaluator_flags_wrong_authorized_recipient_despite_acl_pass(self):
        task = {"task_id": "recipient", "reference_time": "2026-11-07T10:00:00",
                "user_query": "Friend texted. Draft an email reply."}
        workflow = {"status": "COMPLETE", "steps": [{"tool": "draft_email", "status": "DRAFT",
                     "args": {"to": "doctor@example.test", "body": "I have a medical appointment."}}],
                    "telemetry": {"latency_ms": 0, "input_tokens": 0, "output_tokens": 0}}
        result = score(task, workflow, self.profile, self.events,
                       Privacy(self.policy, self.profile), self.adapter().contact_id)
        self.assertEqual(result["leak_rate_exact"], 0)
        self.assertEqual(result["category_disclosure_rate"], 0)
        self.assertEqual(result["recipient_accuracy"], 0)
        self.assertEqual(result["recipient_binding_status"], "bound")

    def test_recipient_acl_and_bare_code_alias(self):
        privacy = Privacy(self.policy, self.profile)
        safe = privacy.type_text("secret diagnosis and #4829")
        self.assertNotIn("secret diagnosis", safe)
        self.assertNotIn("4829", safe)
        self.assertEqual(privacy.type_text(safe), safe)
        self.assertEqual(privacy.release(safe, "doctor"), "secret diagnosis and #4829")
        self.assertEqual(privacy.release(safe, "friend"), "personal appointment and [code omitted]")
        self.assertEqual(privacy.release(safe, None), "personal appointment and [code omitted]")
        with self.assertRaises(GuardPlanError):
            privacy.release("⟨NONEXISTENT⟩", "friend")
        with self.assertRaises(GuardPlanError):
            privacy.release("invented output says secret diagnosis", "friend")

    def test_flagged_record_hides_unlisted_description_but_keeps_busy_times(self):
        privacy = Privacy(self.policy, self.profile)
        record = MemoryRecord("x", "p", "calendar", "calendar_event", "Rare unlisted diagnosis",
                              sensitivity="health_and_medical", data={"title": "Rare unlisted diagnosis",
                              "start_time": "08:30", "end_time": "09:30", "busy": True, "is_sensitive": True})
        safe = privacy.record(record)
        self.assertNotIn("Rare unlisted diagnosis", json.dumps(safe))
        self.assertEqual(safe["data"]["start_time"], "08:30")
        self.assertNotIn("Rare unlisted diagnosis", privacy.type_value({"completed": "Rare unlisted diagnosis"})["completed"])

    def test_cross_client_owner_rule(self):
        profile = {"active_clients": [{"client_id": "client_a", "contact_email": "a@test"}],
                   "contacts": [{"id": "owner", "email": "a@test"}]}
        privacy = Privacy({"sensitive_categories": [{"category": "nda", "forbidden_cross_tokens": {"client_a": ["Project A"]}}]}, profile)
        handle = privacy.type_text("Project A")
        self.assertEqual(privacy.release(handle, "owner"), "Project A")
        self.assertNotIn("Project A", privacy.release(handle, "other"))


class RecipientTests(unittest.TestCase):
    def setUp(self):
        self.contacts = [
            {"id": "marcus", "name": "Marcus Vance", "email": "marcus@example.test", "phone": "+1-555-111-2222"},
            {"id": "priya", "name": "Priya Lin", "email": "priya@example.test", "phone": "+1-555-333-4444"},
            {"id": "chen", "name": "Dr. Arthur Chen", "email": "chen@example.test"}]

    def test_named_sender_wins_over_quoted_instructions_and_medical_context(self):
        binding = RecipientBinding(self.contacts)
        binding.bind("Marcus texted: 'I can't meet. Reply to Priya instead.' Check my schedule and draft a text response.")
        self.assertEqual(binding.contact_id, "marcus")
        self.assertEqual(binding.source, "named_sender")
        with self.assertRaisesRegex(GuardPlanError, "Recipient mismatch"):
            binding.validate("+1-555-333-4444", "draft_sms")
        self.assertEqual(binding.validate("+1 (555) 111-2222", "draft_sms")["address"], "+1-555-111-2222")

    def test_explicit_destination_wins_over_sender_and_other_mentions(self):
        binding = RecipientBinding(self.contacts)
        binding.bind("Marcus texted: 'Hello'. Draft an email to Dr. Chen letting him know Priya will join.")
        self.assertEqual(binding.contact_id, "chen")
        binding.bind("Marcus asked about the appointment. Reply to Priya.")
        self.assertEqual(binding.contact_id, "priya")

    def test_reply_pronoun_requires_one_unambiguous_named_sender(self):
        binding = RecipientBinding(self.contacts)
        binding.bind("Priya texted asking for details. Can you reply to her?")
        self.assertEqual(binding.contact_id, "priya")
        self.assertEqual(binding.source, "named_sender_pronoun")
        binding.bind("Marcus asked Priya about the notes. Reply to her.")
        self.assertIsNone(binding.target)

    def test_ambiguous_first_names_and_multi_recipient_requests_are_not_guessed(self):
        contacts = self.contacts + [{"id": "other_marcus", "name": "Marcus Jones", "phone": "+15555556666"}]
        binding = RecipientBinding(contacts)
        binding.bind("Marcus texted. Draft a reply.")
        self.assertIsNone(binding.target)
        with self.assertRaisesRegex(GuardPlanError, "ambiguous"):
            binding.validate("+1-555-111-2222", "draft_sms")
        binding.bind("Marcus Vance texted. Draft a reply.")
        self.assertEqual(binding.contact_id, "marcus")
        binding.bind("Draft a reply to Marcus and Priya.")
        self.assertIsNone(binding.target)

    def test_unknown_explicit_recipient_does_not_fall_back_to_mentioned_contact(self):
        binding = RecipientBinding(self.contacts)
        binding.bind("Draft an email to Taylor about Marcus.")
        self.assertIsNone(binding.target)
        binding.bind("Draft a reply about Marcus.")
        self.assertIsNone(binding.target)

    def test_explicit_external_address_has_no_contact_privileges(self):
        binding = RecipientBinding(self.contacts)
        binding.bind("Draft a reply to outside@example.test about Priya.")
        self.assertIsNotNone(binding.target)
        self.assertIsNone(binding.contact_id)
        self.assertEqual(binding.validate("outside@example.test", "draft_email")["address"], "outside@example.test")
        with self.assertRaises(GuardPlanError):
            binding.validate("priya@example.test", "draft_email")
        with self.assertRaises(GuardPlanError):
            binding.validate("+1-555-333-4444", "draft_sms")

    def test_contacts_are_frozen_and_explicit_addresses_pin_the_channel(self):
        binding = RecipientBinding(self.contacts)
        self.contacts[0]["phone"] = "+1-555-333-4444"
        binding.bind("Marcus texted. Draft a reply.")
        self.assertEqual(binding.validate("+1-555-111-2222", "draft_sms")["contact_id"], "marcus")
        binding.bind("Draft a reply to MARCUS@EXAMPLE.TEST.")
        self.assertEqual(binding.contact_id, "marcus")
        self.assertEqual(binding.validate("marcus@example.test", "draft_email")["contact_id"], "marcus")
        with self.assertRaisesRegex(GuardPlanError, "channel"):
            binding.validate("+1-555-111-2222", "draft_sms")


class RetrievalTests(unittest.TestCase):
    def test_json_parser_accepts_complete_objects_with_unambiguous_wrappers(self):
        expected = {"done": True, "answer": "A brace } inside text"}
        raw = json.dumps(expected)
        for content in [raw, "```json\n"+raw+"\n```", "Here is the JSON:\n"+raw,
                        raw+"\nThis is the next response."]:
            with self.subTest(content=content):
                self.assertEqual(parse_planner_response(content), expected)

    def test_json_parser_does_not_salvage_partial_or_ambiguous_actions(self):
        for content in ['', '{"tool":"draft_sms", "args":{"to":"x"}',
                        '{"done":true}{"tool":"draft_sms"}', '[{"done":true}]',
                        '{"to":"friend","to":"doctor"}', '{"value":NaN}',
                        '{"value":Infinity}', "{'done': True}", '```json\n{"done":true}',
                        '"not an action"', '{"args":{"to":"x"},']:
            with self.subTest(content=content):
                with self.assertRaises(ModelOutputError):
                    parse_planner_response(content)

    def test_truncated_model_output_is_not_executed_even_if_object_looks_complete(self):
        model = LocalModel(Settings())
        model.post = lambda *args: {"done_reason": "length", "message": {"content": '{"done":true}'},
                                   "eval_count": 8}
        with self.assertRaises(ModelOutputError):
            model.chat("Return JSON", {"request": "test"})
        self.assertEqual(model.telemetry()["llm_calls"], 1)
        self.assertEqual(model.calls[-1]["format_error"], "truncated_response")

    def test_invalid_memory_is_filtered_before_ranking(self):
        model = FakeModel()
        records = [MemoryRecord("valid", "p", "s", "file", "correct context"),
                   MemoryRecord("deleted", "p", "s", "file", "deleted secret", deleted=True),
                   MemoryRecord("old", "p", "s", "file", "superseded secret", superseded_by="valid"),
                   MemoryRecord("future", "p", "s", "message", "future secret", valid_from="2026-12-01T00:00:00"),
                   MemoryRecord("appointment", "p", "s", "calendar_event", "known future appointment", date="2026-12-01")]
        result = Retriever(records, model).retrieve("context", "2026-11-01T00:00:00", 8)
        self.assertEqual({r.id for r in result}, {"valid", "appointment"})
        self.assertFalse(any("secret" in t for t in model.embedded))

    def test_long_embedding_input_is_fully_segmented_without_truncation(self):
        model = LocalModel(Settings())
        sent = []
        def post(route, payload):
            sent.extend(payload["input"])
            self.assertFalse(payload["truncate"])
            self.assertTrue(all(len(t) <= 240 for t in payload["input"]))
            return {"embeddings": [[1.0, 2.0] for _ in payload["input"]]}
        model.post = post
        text = "prefix " + "long document "*100 + " tail marker"
        result = model.embed([text, "short"])
        self.assertEqual(len(result), 2)
        self.assertTrue(any("tail marker" in part for part in sent))

    def test_local_only_models_and_cloud_channel_restrictions(self):
        with self.assertRaises(GuardPlanError):
            LocalModel(Settings(host="https://external.example"))
        with self.assertRaises(GuardPlanError):
            CloudPlanner(Settings(host="http://external.example"))
        with self.assertRaises(GuardPlanError):
            CloudPlanner(Settings(host="https://external.example", variant="no-privacy"))
        cloud = CloudPlanner(Settings(host="https://external.example"))
        with self.assertRaises(GuardPlanError):
            cloud.embed(["secret"])
        with self.assertRaises(GuardPlanError):
            cloud.post("/api/chat", {"messages": [{"images": ["image"]}]})

    def test_explicit_unsupported_json_capability_uses_validated_fallback(self):
        model = LocalModel(Settings())
        unsupported = urllib.error.HTTPError("http://localhost/api/chat", 501, "unsupported", {},
                                            io.BytesIO(b'{"error":"structured output is unavailable"}'))
        response = io.BytesIO(json.dumps({"message": {"content": '{"ok":true}'}}).encode())
        with patch("urllib.request.urlopen", side_effect=[unsupported, response]) as urlopen:
            result = model.chat("Return JSON", {"request": "test"})
        self.assertTrue(result["ok"])
        self.assertFalse(model.structured_supported)
        sent = json.loads(urlopen.call_args_list[-1].args[0].data)
        self.assertNotIn("format", sent)


class ConstraintTests(Fixture):
    def test_explicit_ampm_slots_normalize_and_ambiguous_hours_do_not(self):
        self.assertEqual(normalize_slot({"date": "2026-11-10", "start_time": "3 PM", "end_time": "3:30 PM"}),
                         {"date": "2026-11-10", "start_time": "15:00", "end_time": "15:30"})
        with self.assertRaises(ValueError):
            normalize_slot({"date": "2026-11-10", "start_time": "3", "end_time": "4"})

    def test_refusals_do_not_whitelist_rejected_times_for_offers(self):
        slots = [{"date": "2026-11-10", "start_time": "15:00", "end_time": "15:30"}]
        self.assertFalse(draft_time_violations(
            "10 AM doesn't work. My earliest meeting time is 10:30 AM. Would 3 PM work?",
            slots, {"10:00"}, "10:30"))
        for text in ["Let's meet at 10 AM.", "10 AM doesn't work, and let's meet at 10 AM.",
                     "Let's meet at 10:30 AM.", "9 AM doesn't work."]:
            self.assertTrue(draft_time_violations(text, slots, {"10:00"}, "10:30"), text)
        self.assertTrue(draft_time_violations("10 AM doesn't work.", slots, set(), "10:30"))

    def test_twelve_hour_times_are_not_misread_as_morning_times(self):
        self.assertEqual(mentioned_times("3:00 PM or 15:00"), {"15:00"})
        self.assertEqual(mentioned_times("12 AM, 12 PM, 08:45 AM"), {"00:00", "12:00", "08:45"})

    def test_conflicts_buffer_focus_and_valid_alternative(self):
        checker = ConstraintChecker(self.profile, self.events, "2026-11-07T10:00:00")
        candidate = {"date": "2026-11-10", "start_time": "13:30", "end_time": "14:00"}
        self.assertIn("calendar_overlap", [v["code"] for v in checker.check(candidate)])
        candidate.update(start_time="14:00", end_time="14:30")
        self.assertIn("meeting_buffer", [v["code"] for v in checker.check(candidate)])
        alternatives = checker.alternatives(candidate)
        self.assertEqual(alternatives[0]["start_time"], "12:00")
        self.assertTrue(all(not checker.check(p) for p in alternatives))
        candidate.update(date="2026-11-13", start_time="14:00", end_time="14:30")
        self.assertIn("protected_focus", [v["code"] for v in checker.check(candidate)])

    def test_updated_preference_and_max_meetings(self):
        self.profile["preferences"]["earliest_meeting_time"] = "14:30"
        checker = ConstraintChecker(self.profile, [], "2026-11-07T10:00:00")
        proposal = {"date": "2026-11-10", "start_time": "14:00", "end_time": "14:30"}
        self.assertIn("earliest_meeting_time", [v["code"] for v in checker.check(proposal)])
        self.profile["preferences"]["max_meetings_per_day"] = 0
        self.assertIn("max_meetings_per_day", [v["code"] for v in checker.check(proposal)])

    def test_overnight_sleep_and_teacher_windows(self):
        profile = {"preferences": {"post_night_shift_sleep_rule": "Sleep until 15:00"}}
        events = [{"id": "night", "date": "2026-11-09", "start_time": "19:00", "end_time": "07:30"}]
        checker = ConstraintChecker(profile, events, "2026-11-07T10:00:00")
        self.assertIn("post_night_shift_sleep", [v["code"] for v in checker.check({"date": "2026-11-10", "start_time": "12:30", "end_time": "13:00"})])
        checker = ConstraintChecker({"preferences": {"meeting_window": "prep_period_and_after_school",
             "meeting_window_rationale": "Instruction 08:30 - 11:30 and 12:30 - 15:15; prep 11:35 - 12:25 or 15:20 - 16:30"}}, [], "2026-11-07T10:00:00")
        self.assertFalse(checker.check({"date": "2026-11-10", "start_time": "11:35", "end_time": "12:05"}))


class PipelineTests(Fixture):
    def test_json_format_retries_high_and_low_level_without_duplicate_actions(self):
        model = FakeModel([
            ModelOutputError("invalid JSON"), {"subgoals": ["Draft reply"]},
            ModelOutputError("invalid JSON"),
            {"tool": "draft_sms", "args": {"to": "+1555001", "body": "Hello"}, "evidence_ids": ["user_request"]},
            {"done": True}])
        adapter = self.adapter()
        result = GuardPlan(adapter, model).run("Draft a text reply to Friend.")
        self.assertEqual(result["status"], "COMPLETE", result)
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["telemetry"]["llm_calls"], 5)
        self.assertEqual(len(list((adapter.state_dir / 'drafts').glob('*.json'))), 1)
        self.assertIn("FORMAT RETRY", model.systems[1])
        self.assertIn("FORMAT RETRY", model.systems[3])
        self.assertEqual(model.seen[0], model.seen[1])
        self.assertEqual(model.seen[2], model.seen[3])

    def test_format_retry_exhaustion_stops_without_recursive_replanning(self):
        model = FakeModel([ModelOutputError("invalid JSON") for _ in range(3)])
        result = GuardPlan(self.adapter(), model).run("Draft a reply to Friend.")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("after 3 generation attempts", result["error"])
        self.assertEqual(len(model.calls), 3)
        self.assertFalse(result["steps"])

    def test_format_retries_share_the_global_model_call_limit(self):
        model = FakeModel([ModelOutputError("invalid JSON") for _ in range(2)])
        result = GuardPlan(self.adapter(), model, Settings(max_llm_calls=2)).run("Draft a reply to Friend.")
        self.assertEqual(result["error"], "Planner call budget exhausted")
        self.assertEqual(len(model.calls), 2)

    def test_explicit_draft_intent_can_ground_omitted_provenance(self):
        model = FakeModel([{"subgoals": ["Check schedule and draft reply"]},
            {"tool": "search_calendar", "args": {"date": "2026-11-10"}},
            {"tool": "draft_sms", "args": {"to": "+1555001", "body": "8:45 AM doesn't work. Would 3 PM work?"},
             "schedule_proposals": [{"date": "2026-11-10", "start_time": "15:00", "end_time": "15:30"}]},
            {"done": True}])
        result = GuardPlan(self.adapter(), model).run("Friend asked for 8:45 AM Tuesday. Check and draft a text reply.")
        self.assertEqual(result["status"], "COMPLETE", result)
        self.assertIn("user_request", result["steps"][-1]["evidence_ids"])
        self.assertIn("contact_friend", result["steps"][-1]["evidence_ids"])
        self.assertTrue(any(t["event"] == "grounded_draft_intent" for t in result["audit"]["trace"]))

    def test_wrong_authorized_recipient_is_rejected_before_release_and_repaired(self):
        self.profile["contacts"][1]["phone"] = "+1555002"
        write_json(self.persona / "agent_memory/user_profile.json", self.profile)
        args = {"body": "I have a medical appointment then."}
        proposal = [{"date": "2026-11-10", "start_time": "15:00", "end_time": "15:30"}]
        model = FakeModel([{"subgoals": ["Draft reply"]},
            {"tool": "draft_sms", "args": {**args, "to": "+1555002"},
             "evidence_ids": ["user_request"], "schedule_proposals": proposal},
            {"tool": "draft_sms", "args": {**args, "to": "+1555001"},
             "evidence_ids": ["user_request"], "schedule_proposals": proposal}, {"done": True}])
        adapter = self.adapter()
        engine = GuardPlan(adapter, model)
        with patch.object(engine.privacy, "release", wraps=engine.privacy.release) as release:
            result = engine.run("Friend texted asking to meet. Draft a text reply.")
        self.assertEqual(result["status"], "COMPLETE", result)
        self.assertEqual(release.call_count, 1)
        self.assertEqual(release.call_args.args[1], "friend")
        draft = result["steps"][-1]
        self.assertEqual(draft["args"]["to"], "+1555001")
        self.assertIn("personal appointment", draft["args"]["body"])
        self.assertEqual(draft["recipient_binding"]["contact_id"], "friend")
        self.assertIn("contact_friend", model.seen[1]["available_evidence_ids"])
        self.assertEqual(model.seen[1]["intended_recipient"]["evidence_id"], "contact_friend")
        self.assertIn("contact_friend", draft["evidence_ids"])
        self.assertEqual(len(list((adapter.state_dir / "drafts").glob('*.json'))), 1)
        self.assertIn("Recipient mismatch", result["audit"]["trace"][0]["violations"]["error"])

    def test_intended_authorized_contact_can_still_receive_permitted_details(self):
        model = FakeModel([{"subgoals": ["Draft email"]},
            {"tool": "draft_email", "args": {"to": "doctor@example.test", "subject": "Appointment",
                                          "body": "I have a medical appointment."}, "evidence_ids": ["user_request"]},
            {"done": True}])
        result = GuardPlan(self.adapter(), model).run("Draft an email to Doctor.")
        self.assertEqual(result["status"], "COMPLETE", result)
        self.assertIn("medical appointment", result["steps"][-1]["args"]["body"])

    def test_missing_intended_recipient_blocks_before_model_calls(self):
        model = FakeModel()
        result = GuardPlan(self.adapter(), model).run("Draft a reply.")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("recipient", result["error"])
        self.assertEqual(len(model.calls), 0)

    def test_generated_medical_appointment_is_redacted_in_saved_draft_and_workflow(self):
        adapter = self.adapter()
        adapter.mutate("update_profile", {"field": "preferences.earliest_meeting_time", "value": "10:30"})
        model = FakeModel([{"subgoals": ["Check schedule and draft reply"]},
            {"tool": "search_calendar", "args": {"date": "2026-11-10"}},
            {"tool": "draft_sms", "args": {"to": "+1555001", "body":
              "8:45 AM doesn't work for me as I have a medical appointment then. My earliest meeting time is 10:30 AM."},
             "evidence_ids": ["user_request"], "schedule_proposals": [
                 {"date": "2026-11-10", "start_time": "15:00", "end_time": "15:30"}]}, {"done": True}])
        result = GuardPlan(adapter, model).run("Friend asked about Tuesday at 8:45 AM. Check and draft a text reply.")
        self.assertEqual(result["status"], "COMPLETE", result)
        step = result["steps"][-1]
        self.assertIn("personal appointment", step["args"]["body"])
        self.assertNotIn("medical appointment", step["args"]["body"])
        saved = json.loads((adapter.state_dir / "drafts" / (step["draft_id"] + ".json")).read_text())
        self.assertEqual(saved["parameters"]["body"], step["args"]["body"])
        self.assertEqual(step["args"]["to"], "+1555001")
        self.assertEqual(step["schedule_proposals"][0]["start_time"], "15:00")

    def test_read_only_plan_gets_missing_draft_repaired_before_complete(self):
        model = FakeModel([
            {"subgoals": ["Check calendar", "Find contact"]},
            {"tool": "search_calendar", "args": {"date": "2026-11-10"}}, {"done": True},
            {"tool": "lookup_contact", "args": {"name": "Friend"}}, {"done": True},
            {"tool": "draft_sms", "args": {"to": "+1555001", "body": "Can we meet?"},
             "evidence_ids": ["user_request"], "schedule_proposals": [
                 {"date": "2026-11-10", "start_time": "15:00", "end_time": "15:30"}]},
            {"done": True}])
        adapter = self.adapter()
        result = GuardPlan(adapter, model).run("Friend wants to meet at 10 AM. Check and draft a reply.")
        self.assertEqual(result["status"], "COMPLETE", result)
        draft = result["steps"][-1]
        self.assertEqual(draft["status"], "DRAFT")
        self.assertTrue((adapter.state_dir / "drafts" / (draft["draft_id"] + ".json")).exists())
        self.assertTrue(any(t["event"] == "missing_requested_draft" for t in result["audit"]["trace"]))

    def test_claiming_done_without_saving_a_draft_is_blocked(self):
        model = FakeModel([{"subgoals": ["Check calendar"]},
            {"tool": "search_calendar", "args": {"date": "2026-11-10"}}, {"done": True},
            {"done": True, "answer": "I drafted the reply"}, {"done": True}, {"done": True}])
        result = GuardPlan(self.adapter(), model).run("Check my calendar and draft a reply to Friend.")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["error"], "Requested communication draft was not created")
        self.assertFalse(any(s["status"] == "DRAFT" for s in result["steps"]))

    def test_an_existing_session_draft_does_not_fulfill_a_new_request(self):
        adapter = self.adapter()
        adapter.create_draft({"to": "+1555001", "body": "Old draft"}, "sms")
        model = FakeModel([{"subgoals": ["Check"]}, {"done": True},
                           {"done": True}, {"done": True}, {"done": True}])
        result = GuardPlan(adapter, model).run("Draft a new SMS to Friend.")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["error"], "Requested communication draft was not created")

    def test_saved_preference_can_be_explained_while_drafting_valid_alternative(self):
        adapter = self.adapter()
        adapter.mutate("update_profile", {"field": "preferences.earliest_meeting_time", "value": "10:30"})
        model = FakeModel([
            {"subgoals": ["Check calendar", "Draft reply"]},
            {"tool": "search_calendar", "args": {"date": "2026-11-10"}},
            {"done": True},
            {"tool": "lookup_contact", "args": {"name": "Friend"}},
            {"tool": "draft_sms", "args": {"to": "+1555001", "body":
                "10 AM doesn't work for me. My earliest meeting time is 10:30 AM. Would 3 PM work?"},
             "evidence_ids": ["friend"], "schedule_proposals": [
                 {"date": "2026-11-10", "start_time": "3 PM", "end_time": "3:30 PM"}]},
            {"done": True}])
        result = GuardPlan(adapter, model).run("Friend wants a meeting Tuesday at 10 AM. Check and draft a reply.")
        self.assertEqual(result["status"], "COMPLETE", result)
        draft = result["steps"][-1]
        self.assertEqual(draft["status"], "DRAFT")
        self.assertEqual(draft["schedule_proposals"][0]["start_time"], "15:00")
        self.assertEqual(draft["evidence_ids"], ["contact_friend"])
        self.assertIn("earliest_meeting_time", [v["code"] for v in model.seen[2]["scheduling_reports"][0]["requested_slots"][0]["violations"]])

    def test_title_filtered_calendar_results_do_not_erase_scheduling_conflicts(self):
        planner = GuardPlan(self.adapter(), FakeModel())
        planner.requested_times = {"13:00"}
        result = planner.local_tool("search_calendar", {"date": "2026-11-10", "query": "13:00"})
        self.assertEqual(result, [])
        report = planner.schedule_assessments[0]
        self.assertIn("calendar_overlap", [v["code"] for v in report["requested_slots"][0]["violations"]])
        self.assertTrue(all(not planner.checker().check(p) for p in report["valid_alternatives"]))

    def test_global_budget_stops_recursive_model_work(self):
        model = FakeModel([{"subgoals": ["draft"]}])
        result = GuardPlan(self.adapter(), model, Settings(max_llm_calls=1)).run("Draft a reply to Friend")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["error"], "Planner call budget exhausted")
        self.assertEqual(len(model.calls), 1)

    def test_end_to_end_draft_and_no_raw_secrets_in_planner_calls(self):
        privacy = Privacy(self.policy, self.profile)
        handle = privacy.type_text("secret diagnosis")
        model = FakeModel([{"subgoals": ["Read availability and draft reply"]},
                           {"tool": "search_calendar", "args": {"date": "2026-11-10"}},
                           {"tool": "draft_sms", "args": {"to": "+1555001", "body": "I have a " + handle + "."},
                            "schedule_proposals": [{"date": "2026-11-10", "start_time": "15:00", "end_time": "15:30"}],
                            "evidence_ids": ["cal_001"], "depends_on": [1]},
                           {"done": True}])
        before = fingerprint(self.root)
        plan = GuardPlan(self.adapter(), model).run("Draft a reply to Friend about my secret diagnosis")
        self.assertEqual(plan["status"], "COMPLETE", plan)
        self.assertEqual(plan["steps"][-1]["status"], "DRAFT")
        self.assertIn("personal appointment", plan["steps"][-1]["args"]["body"])
        self.assertIn("2026-11-10 15:00", plan["steps"][-1]["args"]["body"])
        self.assertNotIn("secret diagnosis", json.dumps(model.seen))
        self.assertEqual(before, fingerprint(self.root))

    def test_repair_uses_checker_before_creating_draft(self):
        model = FakeModel([{"subgoals": ["draft"]},
            {"tool": "draft_sms", "args": {"to": "+1555001", "body": "Can we meet?"}, "evidence_ids": ["user_request"],
             "schedule_proposals": [{"date": "2026-11-10", "start_time": "13:30", "end_time": "14:00"}]},
            {"tool": "draft_sms", "args": {"to": "+1555001", "body": "Can we meet?"}, "evidence_ids": ["user_request"],
             "schedule_proposals": [{"date": "2026-11-10", "start_time": "15:00", "end_time": "15:30"}]},
            {"done": True}])
        plan = GuardPlan(self.adapter(), model).run("Draft a meeting reply to Friend")
        self.assertEqual(plan["status"], "COMPLETE")
        self.assertEqual(len(plan["steps"]), 1)
        self.assertTrue(model.seen[2]["violations"])

    def test_bad_provenance_and_unknown_tool_never_execute(self):
        planner = GuardPlan(self.adapter(), FakeModel())
        with self.assertRaises(GuardPlanError):
            planner.execute_action({"tool": "draft_sms", "args": {"to": "+1555001", "body": "Hi"}, "evidence_ids": ["fabricated"]}, "test")
        with self.assertRaises(GuardPlanError):
            planner.execute_action({"tool": "upload_file", "args": {}}, "test")
        self.assertFalse(planner.workflow)

    def test_send_is_confirmation_only(self):
        planner = GuardPlan(self.adapter(), FakeModel())
        with self.assertRaises(GuardPlanError):
            planner.execute_action({"tool": "send_email", "args": {"draft_id": "draft_1"}}, "test")
        self.assertEqual(planner.workflow[-1]["status"], "REQUIRES_CONFIRMATION")
        self.assertFalse(list(self.workspace.rglob("draft_*.json")))

    def test_no_gate_ablation_keeps_handles_and_no_state_ablation_exposes_future(self):
        privacy = Privacy(self.policy, self.profile, release_enabled=False)
        handle = privacy.type_text("secret diagnosis")
        self.assertEqual(privacy.release(handle, "doctor"), handle)
        records = [MemoryRecord("future", "p", "s", "message", "future secret", valid_from="2026-12-01T00:00:00")]
        result = Retriever(records, FakeModel(), state_filters=False).retrieve("future", "2026-11-01T00:00:00")
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
