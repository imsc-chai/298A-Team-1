"""M3 prompts and their JSON schemas (approved 2026-11-11).

No privacy instructions beyond the shared PersonaContext: M3 must not carry M6 machinery.
The executor's optional schedule_proposals / extractions fields exist only to fill the Output
contract; they report what the model did, they do not change what it may do.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

PLANNER = """You are the HIGH-LEVEL PLANNER. Break the user's request into an ordered list of subgoals.

USER REQUEST: {user_query}

AVAILABLE TOOLS (reference only; you do not call them):
{tools}

RETRIEVED CONTEXT (ids in brackets):
{context}

- Each subgoal is one concrete outcome in plain language, achievable with 1-3 tool calls.
- Cover every part of the request; put information-gathering before actions that depend on it.
- Name concrete people, dates and files when the request or context gives them.
- End with the actions the user asked for (drafts, calendar events, profile updates).
- Use 2-6 subgoals; list dependencies by subgoal id.

Return JSON: {{"subgoals": [{{"id": 1, "description": "...", "depends_on": []}}]}}"""

DECOMPOSE = """You are the HIGH-LEVEL PLANNER. A subgoal of the user's request could not be completed directly.

USER REQUEST: {user_query}
GOAL: {goal}
IT FAILED BECAUSE: {reason}

RESULTS SO FAR:
{results}

AVAILABLE TOOLS (reference only; you do not call them):
{tools}

RETRIEVED CONTEXT (ids in brackets):
{context}

Break the GOAL into 2-4 smaller subgoals, each achievable with 1-3 tool calls.
Name concrete people, dates and files when known. List dependencies by subgoal id (within this list).

Return JSON: {{"subgoals": [{{"id": 1, "description": "...", "depends_on": []}}]}}"""

EXECUTOR = """You are the EXECUTOR for one subgoal of a larger plan.

OVERALL REQUEST: {user_query}
CURRENT SUBGOAL ({subgoal_id}): {subgoal}

RESULTS FROM EARLIER SUBGOALS:
{results}

RETRIEVED CONTEXT FOR THIS SUBGOAL (ids in brackets):
{context}

TOOLS:
{tools}

TOOL CALLS ALREADY MADE FOR THIS SUBGOAL:
{calls}

Decide the next action for THIS subgoal only:
- "call": exactly one tool call that advances it.
- "done": achieved (by calls above, or the facts are already available).
- "cannot": not achievable as stated (needs breaking down or information is missing).
Formats: dates YYYY-MM-DD, times HH:MM (24h), paths local_files/<name>. Use exactly the argument names listed for the tool. Cite record ids you used.
If your call drafts a message that offers meeting times, list every offered time in "schedule_proposals".
If you read a local file, put the key facts you took from it in "extractions", keyed by its path.

Return JSON: {{"thought": "...", "decision": "call|done|cannot", "tool": "...|null",
  "args": {{}}, "evidence_ids": [], "summary": "facts/results this subgoal produced",
  "schedule_proposals": [{{"date": "YYYY-MM-DD", "start_time": "HH:MM", "end_time": "HH:MM"}}],
  "extractions": {{"local_files/<name>": {{"field": "value"}}}}}}"""

FINAL = """All planned subgoals have been worked through. Write the reply to the user.

USER REQUEST: {user_query}

WHAT WAS DONE:
{results}

Tell the user briefly what you found and did (including any drafts created for their review, and anything that could not be done).

Return JSON: {{"final_response": "..."}}"""


class PlannedSubgoal(BaseModel):
    id: int
    description: str = Field(min_length=1)
    depends_on: list[int] = Field(default_factory=list)


class Plan(BaseModel):
    subgoals: list[PlannedSubgoal] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _unique_ids(self) -> Plan:
        ids = [s.id for s in self.subgoals]
        if len(set(ids)) != len(ids):
            raise ValueError(f"subgoal ids must be unique, got {ids}")
        return self


class ExecutorDecision(BaseModel):
    thought: str = ""
    decision: Literal["call", "done", "cannot"]
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    # Validated leniently later (bad entries go to debug) so one malformed time does not
    # force a whole retry of an otherwise good decision.
    schedule_proposals: list[Any] = Field(default_factory=list)
    extractions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("args", "extractions", mode="before")
    @classmethod
    def _none_to_empty_dict(cls, v: Any) -> Any:
        return {} if v is None else v

    @field_validator("evidence_ids", "schedule_proposals", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v: Any) -> Any:
        return [] if v is None else v

    @field_validator("summary", "thought", mode="before")
    @classmethod
    def _none_to_empty_str(cls, v: Any) -> Any:
        return "" if v is None else v

    @model_validator(mode="after")
    def _call_needs_tool(self) -> ExecutorDecision:
        if self.decision == "call" and not self.tool:
            raise ValueError('decision "call" requires a tool name')
        return self


class FinalAnswer(BaseModel):
    final_response: str
