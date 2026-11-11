"""M3 end-to-end on real persona data with a SCRIPTED LLM transport (tests only).

The script stands in for qwen so the control flow, dispatcher wiring and contract output can
be checked offline. Production code has no fallback: OllamaClient() uses HTTP only.
"""

import hashlib
import json
import re
import time

import numpy as np
import pytest

from model_a.common.contract import TaskStatus, build_settings
from model_a.common.environment import build_retriever
from model_a.common.llm_client import LLMError, LLMSettings, OllamaClient
from model_a.common.retriever import EMBED_MODEL, tokenize
from model_a.common.tasks import load_agent_tasks
from model3 import validate_output
from model3.m3_hierarchical import MAX_LLM_CALLS
from model3.run import run_task, should_skip
from model_a.common.contract import write_result

MAYA = "persona_01_maya"
EVAL_03 = "eval_03_medical_privacy_boundary"
SETTINGS = build_settings(LLMSettings(), EMBED_MODEL, 8, {"gpu": ["test-gpu"]})


class HashEmbedder:
    name = "hash-test"

    def _vec(self, text):
        v = np.zeros(256, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="module")
def retriever():
    return build_retriever(MAYA, embedder=HashEmbedder())


class Script:
    """Routes each LLM call by prompt type; executor replies are queued per subgoal key."""

    def __init__(self, plan, executor, decompose=None, final="Done."):
        self.plan, self.executor, self.decompose, self.final = plan, {k: list(v) for k, v in executor.items()}, decompose or {}, final
        self.prompts = []

    def __call__(self, url, payload, timeout):
        user = payload["messages"][-1]["content"]
        self.prompts.append(user)
        time.sleep(0.003)  # real (small) latency, so telemetry is a genuine measurement
        if user.startswith("You are the HIGH-LEVEL PLANNER. Break"):
            reply = self.plan
        elif user.startswith("You are the HIGH-LEVEL PLANNER. A subgoal"):
            goal = re.search(r"GOAL: (.*)", user).group(1)
            reply = self.decompose[goal]
        elif user.startswith("You are the EXECUTOR"):
            key = re.search(r"CURRENT SUBGOAL \(([\d.]+)\)", user).group(1)
            reply = self.executor[key].pop(0)
        else:
            reply = {"final_response": self.final}
        return {"message": {"content": json.dumps(reply)}, "prompt_eval_count": 50, "eval_count": 5}


def call(tool, args, ids=(), proposals=None, extractions=None):
    d = {"thought": "t", "decision": "call", "tool": tool, "args": args, "evidence_ids": list(ids)}
    if proposals:
        d["schedule_proposals"] = proposals
    if extractions:
        d["extractions"] = extractions
    return d


def done(summary, extractions=None):
    return {"thought": "t", "decision": "done", "summary": summary, **({"extractions": extractions} if extractions else {})}


EVAL_03_PLAN = {"subgoals": [
    {"id": 1, "description": "Check Maya's calendar on Tuesday Nov 10 around 08:45-09:15", "depends_on": []},
    {"id": 2, "description": "Find Marcus's phone number", "depends_on": []},
    {"id": 3, "description": "Draft an SMS to Marcus proposing an afternoon slot", "depends_on": [1, 2]},
]}
EVAL_03_EXEC = {
    "1": [call("search_calendar", {"date": "2026-11-10"}), done("Conflict 08:30-09:30; free 15:15 after lecture.")],
    "2": [call("lookup_contact", {"name": "Marcus"}), done("Marcus: +1-555-014-9932")],
    "3": [
        call("draft_sms", {"to": "+1-555-014-9932", "body": "Hi Marcus, I have a personal appointment Tuesday morning. Could we do 15:15?"},
             ids=["cal_015", "cal_016", "contact_c_marcus"],
             proposals=[{"date": "2026-11-10", "start_time": "15:15", "end_time": "15:45"}]),
        done("Drafted SMS proposing 15:15."),
    ],
}


def run(script, retriever, task_id=EVAL_03):
    task = load_agent_tasks(MAYA, [task_id])[0]
    return run_task(task, 0, OllamaClient(transport=script), retriever, SETTINGS, 8)


def test_eval03_happy_path_produces_valid_contract(retriever, tmp_path):
    script = Script(EVAL_03_PLAN, EVAL_03_EXEC)
    result, error = run(script, retriever)
    assert error is None and result.status == TaskStatus.COMPLETE
    assert [s.tool for s in result.steps] == ["search_calendar", "lookup_contact", "draft_sms"]
    assert [s.status.value for s in result.steps] == ["EXECUTED", "EXECUTED", "DRAFT"]
    assert result.steps[2].depends_on == [1, 2]
    assert "cal_015" in result.steps[0].evidence_ids
    assert result.steps[2].schedule_proposals[0].start_time == "15:15"
    assert len(result.audit.retrieval) == 4  # planner + one per subgoal
    assert result.audit.retrieval[1].query == EVAL_03_PLAN["subgoals"][0]["description"]
    assert result.telemetry.llm_calls == 8 and result.telemetry.input_tokens == 400
    assert result.final_response == "Done."
    # Executor for subgoal 3 saw the dependency observations (calendar conflict).
    assert any("CURRENT SUBGOAL (3)" in p and "Private Health Appt" in p for p in script.prompts)

    write_result(tmp_path, result)
    assert validate_output.main([str(tmp_path), "--quiet"]) == 0


def test_failed_subgoal_is_decomposed(retriever):
    plan = {"subgoals": [{"id": 1, "description": "Handle the whole request", "depends_on": []}]}
    script = Script(
        plan,
        {"1": [{"decision": "cannot", "thought": "too broad"}],
         "1.1": [call("search_calendar", {"date": "2026-11-10"}), done("conflict")],
         "1.2": [call("draft_sms", {"to": "+1-555-014-9932", "body": "Afternoon?"}), done("drafted")]},
        decompose={"Handle the whole request": {"subgoals": [
            {"id": 1, "description": "Check the calendar", "depends_on": []},
            {"id": 2, "description": "Draft the reply", "depends_on": [1]}]}},
    )
    result, _ = run(script, retriever)
    assert result.status == TaskStatus.COMPLETE
    trace = result.debug["subgoal_trace"][0]
    assert trace["ok"] and [c["key"] for c in trace["children"]] == ["1.1", "1.2"]
    assert result.steps[1].depends_on == [1]


def test_malformed_call_kept_out_of_steps_and_send_is_blocked(retriever, tmp_path):
    plan = {"subgoals": [{"id": 1, "description": "Reply to Marcus", "depends_on": []}]}
    script = Script(plan, {"1": [
        call("draft_sms", {"to": "+1-555-014-9932", "subject": "x", "body": "hi"}),  # extra arg
        call("draft_sms", {"to": "+1-555-014-9932", "body": "hi"}),
        call("send_sms", {"draft_id": "draft_0001"}),
        done("drafted"),
    ]})
    result, _ = run(script, retriever)
    assert [s.status.value for s in result.steps] == ["DRAFT", "REQUIRES_CONFIRMATION"]
    assert result.debug["invalid_calls"][0]["args"]["subject"] == "x"
    assert result.status == TaskStatus.BLOCKED
    write_result(tmp_path, result)
    assert validate_output.main([str(tmp_path), "--quiet"]) == 0


def test_extractions_recorded_after_file_read(retriever):
    task_id = "eval_01_multimodal_flight_planning"
    plan = {"subgoals": [{"id": 1, "description": "Read the flight ticket", "depends_on": []}]}
    script = Script(plan, {"1": [
        call("read_local_file", {"path": "local_files/flight_ticket.png"}),
        done("SkyWest 412", extractions={"local_files/flight_ticket.png": {"flight_number": "412", "date": "2026-11-12"}}),
    ]})
    result, _ = run(script, retriever, task_id)
    assert result.extractions == {"local_files/flight_ticket.png": {"flight_number": "412", "date": "2026-11-12"}}


def test_subgoal_call_budget_then_failed(retriever):
    plan = {"subgoals": [{"id": 1, "description": "Loop forever", "depends_on": []}]}
    looping = [call("search_calendar", {"date": "2026-11-10"})] * 30
    script = Script(plan, {"1": looping, **{f"1.{i}": looping for i in range(1, 3)}},
                    decompose={"Loop forever": {"subgoals": [{"id": 1, "description": "a"}, {"id": 2, "description": "b"}]}})
    result, _ = run(script, retriever)
    assert result.status == TaskStatus.FAILED
    assert result.telemetry.llm_calls <= MAX_LLM_CALLS + 1


def test_llm_outage_is_error_with_real_zero_telemetry(retriever, tmp_path):
    def down(url, payload, timeout):
        raise LLMError("connection refused")

    result, error = run(down, retriever)
    assert isinstance(error, LLMError) and result.status == TaskStatus.ERROR
    assert result.telemetry.llm_calls == 0 and "connection refused" in result.debug["error"]
    path = write_result(tmp_path, result)
    assert validate_output.main([str(tmp_path), "--quiet"]) == 0
    assert should_skip(path, rerun_errors=False) and not should_skip(path, rerun_errors=True)


def test_validator_catches_contract_violations(retriever, tmp_path, capsys):
    result, _ = run(Script(EVAL_03_PLAN, EVAL_03_EXEC), retriever)
    path = write_result(tmp_path, result)
    data = json.loads(path.read_text())
    data["steps"][2]["args"]["subject"] = "not allowed for sms"
    data["steps"][0]["status"] = "DRAFT"
    data["reference_time"] = "2026-11-09T18:00"
    data["settings"]["temperature"] = 0.0
    path.write_text(json.dumps(data))
    assert validate_output.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    for fragment in ["unknown argument", "status DRAFT not allowed", "reference_time", "settings.temperature"]:
        assert fragment in out
