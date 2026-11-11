import json

import pytest
from pydantic import ValidationError

from common.contract import ResultBuilder, TaskResult, TaskStatus, result_path, write_result
from common.schemas import StepStatus, ToolResult

SETTINGS = {"model": "qwen3.5:9b", "temperature": 0.6, "max_tokens": 8192,
            "embedding_model": "BAAI/bge-small-en-v1.5", "k": 8, "seed": 42, "hardware": {"gpu": ["A100"]}}
TELEMETRY = {"llm_calls": 2, "input_tokens": 100, "output_tokens": 10, "latency_ms": 500}


def builder():
    return ResultBuilder("M3", "persona_01_maya", "eval_x", 0, "2026-11-09T18:00:00", SETTINGS, {"cal_015", "contact_c_marcus"})


def tr(tool, args, status=StepStatus.EXECUTED, ids=(), well_formed=True):
    return ToolResult(tool=tool, args=args, status=status, observation="obs", evidence_ids=list(ids), well_formed=well_formed)


def test_malformed_calls_go_to_debug_not_steps():
    b = builder()
    assert b.add_tool_result(tr("draft_sms", {"to": "1", "subject": "x"}, StepStatus.FAILED, well_formed=False), "sg", [], []) is None
    assert b.steps == [] and b.debug["invalid_calls"]


def test_evidence_read_tools_keep_returned_ids_write_tools_only_known_cited():
    b = builder()
    n1 = b.add_tool_result(tr("search_calendar", {"date": "2026-11-10"}, ids=["cal_015", "cal_016"]), "sg", [], ["made_up"])
    n2 = b.add_tool_result(tr("draft_sms", {"to": "+1-555-014-9932", "body": "hi"}, StepStatus.DRAFT, ids=["draft_0001"]),
                           "sg", [n1], ["cal_015", "contact_c_marcus", "nope"])
    assert b.steps[0].evidence_ids == ["cal_015", "cal_016"]
    assert b.steps[1].evidence_ids == ["cal_015", "contact_c_marcus"]  # draft id is output, not evidence
    assert b.steps[1].depends_on == [1] and n2 == 2
    assert any("nope" in d.get("unknown_evidence_ids", []) for d in b.debug["dropped"])


def test_schedule_proposals_validated_and_omitted_when_empty(tmp_path):
    b = builder()
    b.add_tool_result(tr("draft_sms", {"to": "+1-555-014-9932", "body": "3:15pm?"}, StepStatus.DRAFT), "sg", [], [],
                      [{"date": "2026-11-10", "start_time": "15:15", "end_time": "15:45"}, {"date": "Nov 10", "start_time": "3pm", "end_time": "x"}])
    b.add_tool_result(tr("search_calendar", {}), "sg", [], [])
    result = b.finish(TaskStatus.COMPLETE, TELEMETRY, "ok")
    data = json.loads(write_result(tmp_path, result).read_text())
    assert data["steps"][0]["schedule_proposals"] == [{"date": "2026-11-10", "start_time": "15:15", "end_time": "15:45"}]
    assert "schedule_proposals" not in data["steps"][1]
    assert result_path(tmp_path, 0, "persona_01_maya", "eval_x").is_file()
    assert not list(tmp_path.rglob("*.tmp"))


def test_extractions_only_for_files_read():
    b = builder()
    b.add_extractions({"local_files/flight_ticket.png": {"flight_number": "412"}}, "sg")
    assert b.extractions == {}
    b.add_tool_result(tr("read_local_file", {"path": "flight_ticket.png"}), "sg", [], [])
    b.add_extractions({"flight_ticket.png": {"flight_number": 412}}, "sg")
    assert b.extractions == {"local_files/flight_ticket.png": {"flight_number": 412}}


def test_none_args_are_stripped():
    b = builder()
    b.add_tool_result(tr("search_calendar", {"date": "2026-11-10", "query": None}), "sg", [], [])
    assert b.steps[0].args == {"date": "2026-11-10"}


def test_task_result_rejects_extra_step_keys_and_forward_deps():
    base = builder().finish(TaskStatus.COMPLETE, TELEMETRY, None).to_json_dict()
    step = {"step": 1, "subgoal": "s", "tool": "search_calendar", "args": {}, "depends_on": [], "evidence_ids": [], "status": "EXECUTED"}
    with pytest.raises(ValidationError):
        TaskResult.model_validate({**base, "steps": [{**step, "observation": "x"}]})
    with pytest.raises(ValidationError):
        TaskResult.model_validate({**base, "steps": [{**step, "depends_on": [2]}]})
    with pytest.raises(ValidationError):
        TaskResult.model_validate({**base, "manifest": {}})
