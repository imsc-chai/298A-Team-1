"""The team Output contract (CLAUDE.md "Output contract") as pydantic models, plus a builder
and an atomic writer shared by every model.

One JSON file per task per run: <results_root>/run_XX/<persona_id>/<task_id>.json.
Debug material (observations, final_response, errors) lives only in top-level
"final_response" / "debug", never in steps[] or args.
"""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from model_a.common.schemas import StepStatus, ToolResult

DATE_RE = r"^\d{4}-\d{2}-\d{2}$"
TIME_RE = r"^([01]\d|2[0-3]):[0-5]\d$"
DRAFT_TOOLS = {"draft_email", "draft_sms"}
READ_TOOLS = {"read_local_file", "search_calendar", "read_messages", "lookup_contact"}
REQUIRED_SETTINGS = ("model", "temperature", "max_tokens", "embedding_model", "k", "seed")

# Ids a tool returns that are outputs, not records the step used.
_NON_RECORD_ID = re.compile(r"^(draft_\d+|profile_updated)$")


class TaskStatus(str, Enum):
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    ERROR = "ERROR"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScheduleProposal(_Strict):
    date: str = Field(pattern=DATE_RE)
    start_time: str = Field(pattern=TIME_RE)
    end_time: str = Field(pattern=TIME_RE)

    @model_validator(mode="after")
    def _ordered(self) -> ScheduleProposal:
        if self.end_time <= self.start_time:
            raise ValueError(f"end_time {self.end_time} must be after start_time {self.start_time}")
        return self


class ContractStep(_Strict):
    step: int = Field(gt=0)
    subgoal: str
    tool: str
    args: dict[str, Any]
    depends_on: list[int]
    evidence_ids: list[str]
    status: StepStatus
    schedule_proposals: list[ScheduleProposal] | None = None


class RetrievalEntry(_Strict):
    query: str
    evidence_ids: list[str]


class Audit(_Strict):
    retrieval: list[RetrievalEntry]


class Telemetry(_Strict):
    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class Settings(BaseModel):
    """Required keys from the contract; extra keys (num_ctx, hardware) are allowed for traceability."""

    model_config = ConfigDict(extra="allow")
    model: str
    temperature: float
    max_tokens: int
    embedding_model: str
    k: int
    seed: int


class TaskResult(_Strict):
    model: str
    persona_id: str
    task_id: str
    run: int = Field(ge=0)
    reference_time: str
    status: TaskStatus
    steps: list[ContractStep]
    audit: Audit
    extractions: dict[str, dict[str, Any]]
    telemetry: Telemetry
    settings: Settings
    final_response: str | None = None
    debug: dict[str, Any] | None = None

    @field_validator("steps")
    @classmethod
    def _step_numbers(cls, steps: list[ContractStep]) -> list[ContractStep]:
        seen: set[int] = set()
        for s in steps:
            if s.step in seen:
                raise ValueError(f"duplicate step number {s.step}")
            bad = [d for d in s.depends_on if d >= s.step or d not in seen]
            if bad:
                raise ValueError(f"step {s.step} depends_on non-earlier steps {bad}")
            seen.add(s.step)
        return steps

    def to_json_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        for step in data["steps"]:
            if step.get("schedule_proposals") is None:  # contract: omit if none
                step.pop("schedule_proposals", None)
        for key in ("final_response", "debug"):
            if data.get(key) is None:
                data.pop(key)
        return data


def build_settings(llm_settings: Any, embedding_model: str, k: int, hardware: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": llm_settings.model,
        "temperature": llm_settings.temperature,
        "max_tokens": llm_settings.max_tokens,
        "embedding_model": embedding_model,
        "k": k,
        "seed": llm_settings.seed,
        "num_ctx": llm_settings.num_ctx,
        "hardware": hardware,
    }


def result_path(results_root: Path, run: int, persona_id: str, task_id: str) -> Path:
    return Path(results_root) / f"run_{run:02d}" / persona_id / f"{task_id}.json"


def write_result(results_root: Path, result: TaskResult) -> Path:
    """Atomic write (tmp + rename): a Colab disconnect mid-write must not leave a partial file
    that resume logic would mistake for a finished task."""
    path = result_path(results_root, result.run, result.persona_id, result.task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False))
    os.replace(tmp, path)
    return path


class ResultBuilder:
    """Accumulates one task's contract output while a model runs."""

    def __init__(self, model: str, persona_id: str, task_id: str, run: int, reference_time: str,
                 settings: dict[str, Any], known_record_ids: set[str]) -> None:
        self.header = {"model": model, "persona_id": persona_id, "task_id": task_id, "run": run,
                       "reference_time": reference_time}
        self.settings = settings
        self._known = set(known_record_ids)
        self._files_read: set[str] = set()
        self.steps: list[ContractStep] = []
        self.retrieval: list[RetrievalEntry] = []
        self.extractions: dict[str, dict[str, Any]] = {}
        self.debug: dict[str, Any] = {"invalid_calls": [], "dropped": [], "observations": {}}

    def log_retrieval(self, query: str, evidence_ids: list[str]) -> None:
        self.retrieval.append(RetrievalEntry(query=query, evidence_ids=list(evidence_ids)))

    def add_tool_result(self, result: ToolResult, subgoal: str, depends_on: list[int],
                        cited_ids: list[str], schedule_proposals: list[dict[str, Any]] | None = None) -> int | None:
        """Append a contract step; returns its number, or None if the call was malformed."""
        if not result.well_formed:
            self.debug["invalid_calls"].append(
                {"subgoal": subgoal, "tool": result.tool, "args": result.args, "error": result.observation}
            )
            return None

        returned = [i for i in result.evidence_ids if not _NON_RECORD_ID.match(i)]
        self._known.update(returned)
        used = (returned if result.tool in READ_TOOLS and result.status == StepStatus.EXECUTED else []) + [
            i for i in cited_ids if i in self._known
        ]
        unknown = [i for i in cited_ids if i not in self._known]
        if unknown:
            self.debug["dropped"].append({"subgoal": subgoal, "unknown_evidence_ids": unknown})

        proposals = self._clean_proposals(result.tool, schedule_proposals or [], subgoal)
        number = len(self.steps) + 1
        self.steps.append(
            ContractStep(
                step=number,
                subgoal=subgoal,
                tool=result.tool,
                args={k: v for k, v in result.args.items() if v is not None},
                depends_on=sorted({d for d in depends_on if 0 < d < number}),
                evidence_ids=list(dict.fromkeys(used)),
                status=result.status,
                schedule_proposals=proposals or None,
            )
        )
        self.debug["observations"][str(number)] = result.observation
        if result.tool == "read_local_file" and result.status == StepStatus.EXECUTED:
            self._files_read.add(_file_key(result.args["path"]))
        return number

    def add_extractions(self, extractions: dict[str, dict[str, Any]], subgoal: str) -> None:
        """Keep only fields for files actually read in this task (no claims about unread files)."""
        for raw_key, fields in extractions.items():
            key = _file_key(raw_key)
            if key not in self._files_read or not isinstance(fields, dict):
                self.debug["dropped"].append({"subgoal": subgoal, "extraction_for_unread_file": raw_key})
                continue
            self.extractions.setdefault(key, {}).update(
                {str(k): (v if isinstance(v, (str, int, float, bool)) else json.dumps(v)) for k, v in fields.items()}
            )

    def finish(self, status: TaskStatus, telemetry: dict[str, int], final_response: str | None,
               debug: dict[str, Any] | None = None) -> TaskResult:
        return TaskResult(
            **self.header,
            status=status,
            steps=self.steps,
            audit=Audit(retrieval=self.retrieval),
            extractions=self.extractions,
            telemetry=Telemetry(**telemetry),
            settings=Settings(**self.settings),
            final_response=final_response,
            debug={**self.debug, **(debug or {})},
        )

    def _clean_proposals(self, tool: str, raw: list[dict[str, Any]], subgoal: str) -> list[ScheduleProposal]:
        if tool not in DRAFT_TOOLS:
            if raw:
                self.debug["dropped"].append({"subgoal": subgoal, "proposals_on_non_draft": raw})
            return []
        good = []
        for p in raw:
            try:
                good.append(ScheduleProposal.model_validate(p))
            except ValueError as exc:
                self.debug["dropped"].append({"subgoal": subgoal, "bad_schedule_proposal": p, "error": str(exc)[:200]})
        return good


def _file_key(path: str) -> str:
    name = path.strip().lstrip("/")
    return name if name.startswith("local_files/") else f"local_files/{name}"
