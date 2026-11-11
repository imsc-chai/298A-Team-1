"""Read-only implementation audit with isolated synthetic counterexamples.

Run from model6: .venv/bin/python audit_pipeline.py
No Ollama requests are made. Scripted outputs probe runtime guarantees, not
model accuracy. Production code, original data and existing sessions are frozen.
Reports are written to a new UUID-named directory under artifacts/audits/.
"""
from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "tests"))
from test_guardplan import Fixture, FakeModel
from guardplan.adapter import DatasetAdapter
from guardplan.constraints import ConstraintChecker
from guardplan.core import GuardPlanError, MemoryRecord, fingerprint, write_json
from guardplan.evaluation import score
from guardplan.planner import GuardPlan
from guardplan.privacy import Privacy
from guardplan.recipients import RecipientBinding
from guardplan.llm import parse_planner_response


def draft(body, start="15:00", end="15:30", day="2026-11-10", email=False, subject="Sync"):
    args = {"to": "friend@example.test" if email else "+1555001", "body": body}
    if email:
        args["subject"] = subject
    return {"tool": "draft_email" if email else "draft_sms", "args": args,
            "evidence_ids": ["user_request"], "schedule_proposals": [
                {"date": day, "start_time": start, "end_time": end}]}


def run_actions(fixture, query, actions, adapter=None):
    adapter = adapter or fixture.adapter()
    model = FakeModel([{"subgoals": ["Handle request"]}, *actions, {"done": True}])
    engine = GuardPlan(adapter, model)
    return engine.run(query), engine


def prepare_direct(fixture, query):
    engine = GuardPlan(fixture.adapter(), FakeModel())
    engine.available.add("user_request")
    engine.requested_output_tools = engine.requested_draft_tools(query)
    engine.recipient_binding.bind(query)
    return engine


def unauthorized_preference_change(f):
    adapter = f.adapter()
    adapter.mutate("update_profile", {"field": "preferences.earliest_meeting_time", "value": "10:30"})
    original = adapter.get_profile()["preferences"]
    result, engine = run_actions(f, "Friend wants to meet at 10 AM Tuesday. Check and draft a text reply.", [
        {"tool": "update_profile", "args": {"field": "preferences.earliest_meeting_time", "value": "07:00"}, "evidence_ids": ["user_request"]},
        {"tool": "update_profile", "args": {"field": "preferences.meeting_window", "value": {"start": "08:00", "end": "18:00"}}, "evidence_ids": ["user_request"]},
        draft("10 AM works for me.", "10:00", "10:30")], adapter)
    changed = adapter.get_profile()["preferences"] != original
    return not changed, {"status": result["status"], "profile_changed_without_request": changed,
        "original_earliest": original["earliest_meeting_time"],
        "final_earliest": adapter.get_profile()["preferences"]["earliest_meeting_time"],
        "tools": [s["tool"] for s in result["steps"]]}


def update_false_complete(f):
    model = FakeModel([{"subgoals": ["Review preference"]}, {"done": True, "answer": "Updated your profile."}])
    adapter = f.adapter()
    result = GuardPlan(adapter, model).run("Please update my profile: no meetings before 10:30 AM.")
    saved = adapter.get_profile()["preferences"].get("earliest_meeting_time") == "10:30"
    return result["status"] != "COMPLETE" or saved, {"status": result["status"], "steps": result["steps"], "preference_saved": saved}


def reply_false_complete(f):
    result, _ = run_actions(f, "Friend texted asking for the gate code. Can you reply to Friend with the details?", [
        {"tool": "lookup_contact", "args": {"name": "Friend"}}])
    saved = any(s["status"] == "DRAFT" for s in result["steps"])
    return result["status"] != "COMPLETE" or saved, {"status": result["status"], "draft_created": saved}


def end_time_offer(f):
    result, engine = run_actions(f, "Draft a text to Friend proposing a meeting.", [
        draft("Would 12:30 PM work for a 30-minute meeting?", "12:00", "12:30")])
    advertised = {"date": "2026-11-10", "start_time": "12:30", "end_time": "13:00"}
    violations = engine.checker().check(advertised)
    saved = any(s["status"] == "DRAFT" for s in result["steps"])
    return not saved or not violations, {"status": result["status"], "draft_created": saved,
        "structured_slot": result["steps"][-1].get("schedule_proposals") if result["steps"] else None,
        "body_advertised_slot": advertised, "body_slot_violations": violations}


def wrong_date_offer(f):
    f.events.append({"id": "wed_busy", "date": "2026-11-11", "weekday": "Wednesday",
        "start_time": "15:00", "end_time": "16:00", "title": "Busy meeting", "busy": True})
    write_json(f.persona / "streams/calendar.json", f.events)
    result, engine = run_actions(f, "Draft an email to Friend proposing a meeting.", [
        draft("Would Wednesday November 11 at 3 PM work?", email=True)])
    advertised = {"date": "2026-11-11", "start_time": "15:00", "end_time": "15:30"}
    violations = engine.checker().check(advertised)
    saved = any(s["status"] == "DRAFT" for s in result["steps"])
    return not saved or not violations, {"status": result["status"], "draft_created": saved,
        "structured_date": "2026-11-10", "body_date": "2026-11-11", "body_slot_violations": violations}


def unchecked_subject(f):
    result, engine = run_actions(f, "Draft an email to Friend proposing a meeting.", [
        draft("Would 3 PM work?", email=True, subject="Meeting confirmed Tuesday at 10 AM")])
    invalid = engine.checker().check({"date": "2026-11-10", "start_time": "10:00", "end_time": "10:30"})
    saved = any(s["status"] == "DRAFT" for s in result["steps"])
    return not saved or not invalid, {"status": result["status"], "draft_created": saved,
        "subject": "Meeting confirmed Tuesday at 10 AM", "subject_slot_violations": invalid}


def factual_flight_time(f):
    query = "Draft an email to Friend explaining that my flight departs at 8 AM."
    engine = prepare_direct(f, query)
    action = {"tool": "draft_email", "args": {"to": "friend@example.test", "subject": "Travel",
              "body": "My flight departs at 8 AM."}, "evidence_ids": ["user_request"]}
    try:
        engine.execute_action(action, "Explain travel fact supplied by user")
    except GuardPlanError as exc:
        return False, {"legitimate_factual_message_rejected": True, "error": str(exc)}
    return True, {"legitimate_factual_message_rejected": False}


def privacy_paraphrase(f):
    privacy = Privacy(f.policy, f.profile)
    args = {"to": "+1555001", "body": "I am seeing my doctor then."}
    try:
        released = privacy.release(privacy.type_value(args), "friend")
    except GuardPlanError as exc:
        return True, {"blocked": str(exc)}
    return "seeing my doctor" not in released["body"], {"released_body": released["body"]}


def structured_sensitive_field(f):
    privacy = Privacy(f.policy, f.profile)
    record = MemoryRecord("health", "p", "calendar", "calendar_event", "Private event",
        sensitivity="health_and_medical", data={"is_sensitive": True,
        "diagnosis": "unlisted private diagnosis", "busy": True, "start_time": "08:30", "end_time": "09:30"})
    typed = privacy.record(record)
    leaked = "unlisted private diagnosis" in json.dumps(typed)
    return not leaked, {"raw_sensitive_value_in_typed_data": leaked, "typed_data": typed["data"]}


def redaction_grammar(f):
    policy = json.loads(json.dumps(f.policy))
    policy["sensitive_categories"][0]["forbidden_tokens"].append("Dr. Robert Keller")
    privacy = Privacy(policy, f.profile)
    original = "I have a medical appointment with Dr. Keller."
    result = privacy.release(privacy.type_text(original), "friend")
    broken = "personal appointment with personal appointment" in result
    return not broken, {"original": original, "released": result, "duplicate_substitution": broken}


def evaluator_self_overlap(f):
    adapter = f.adapter()
    slot = {"date": "2026-11-10", "start_time": "15:00", "end_time": "15:30"}
    pre = ConstraintChecker(adapter.get_profile(), adapter.search_events(), adapter.reference_time).check(slot)
    assert not pre
    args = {**slot, "title": "Meeting with Friend", "busy": True}
    created = adapter.mutate("create_calendar_event", args)
    workflow = {"status": "COMPLETE", "steps": [{"tool": "create_calendar_event", "args": args,
        "status": "EXECUTED", "schedule_proposals": [slot]}],
        "telemetry": {"latency_ms": 0, "input_tokens": 0, "output_tokens": 0}}
    row = score({"task_id": "self_overlap", "reference_time": adapter.reference_time}, workflow,
        adapter.get_profile(), adapter.search_events(), Privacy(f.policy, f.profile), adapter.contact_id)
    return row["preference_adherence"] == 1, {"pre_execution_violations": pre,
        "created_event_id": created["id"], "reported_preference_adherence": row["preference_adherence"]}


def complete_fenced_json(f):
    text = '```json\n{"done": true}\n```\nThis completes the subgoal.'
    try:
        parsed = parse_planner_response(text)
    except GuardPlanError as exc:
        return False, {"complete_unambiguous_object_rejected": True, "error": str(exc)}
    return parsed == {"done": True}, {"parsed": parsed}


def sender_modifier(f):
    binding = RecipientBinding(f.profile['contacts'])
    query = "Friend from the lab asked for an update. Draft a reply to her."
    binding.bind(query)
    return binding.contact_id == 'friend', {"query": query, "expected_contact_id": 'friend',
                                           "binding": binding.summary()}


CASES = [
    ("P1", "Unauthorized preference changes can make a forbidden meeting pass", unauthorized_preference_change),
    ("P1", "Profile update can be COMPLETE without an update", update_false_complete),
    ("P1", "Reply request without literal draft can be COMPLETE without a draft", reply_false_complete),
    ("P1", "A structured end time can be offered as an unchecked start time", end_time_offer),
    ("P1", "A different date in message text is not checked", wrong_date_offer),
    ("P1", "Email subject can offer an unchecked time", unchecked_subject),
    ("P1", "Factual flight time is incorrectly treated as a meeting offer", factual_flight_time),
    ("P1", "Health paraphrase passes the bounded phrase filter", privacy_paraphrase),
    ("P1", "Sensitive structured field reaches planner context unmasked", structured_sensitive_field),
    ("P2", "Separate substitutions produce an incoherent draft", redaction_grammar),
    ("P2", "Scorer checks a created event against itself", evaluator_self_overlap),
    ("P2", "Complete fenced JSON with trailing prose is rejected", complete_fenced_json),
    ("P2", "Unambiguous named sender with a company/group modifier is unresolved", sender_modifier),
]


def main():
    source = BASE.parent.parent / "Project Data"
    before = fingerprint(source)
    code_before = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((BASE / "guardplan").glob("*.py"))}
    results = []
    for priority, title, probe in CASES:
        fixture = Fixture()
        fixture.setUp()
        try:
            passed, details = probe(fixture)
            row = {"priority": priority, "title": title, "status": "PASS" if passed else "FAIL", "details": details}
        except Exception as exc:
            row = {"priority": priority, "title": title, "status": "PROBE_ERROR", "error": type(exc).__name__ + ": " + str(exc)}
        finally:
            fixture.doCleanups()
        results.append(row)
        print(f"{row['status']}: {title}", flush=True)
    code_after = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((BASE / "guardplan").glob("*.py"))}
    report = {"kind": "deterministic_runtime_counterexamples_not_model_benchmark",
              "results": results, "confirmed_failures": sum(r["status"] == "FAIL" for r in results),
              "probe_errors": sum(r["status"] == "PROBE_ERROR" for r in results),
              "production_code_unchanged": code_before == code_after,
              "original_dataset_unchanged": before == fingerprint(source), "production_code_hashes": code_before}
    path = BASE / "artifacts" / "audits" / uuid.uuid4().hex / "counterexamples.json"
    write_json(path, report)
    print("Report:", path)
    return int(report["confirmed_failures"] > 0 or report["probe_errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
