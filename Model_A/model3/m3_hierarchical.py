"""M3: hierarchical planner with per-subgoal retrieval (proposal §5, M3; ADaPT-style).

High level: one planner call turns the request into subgoals.
Low level: for each subgoal, retrieval re-runs on the subgoal text (this is how facts not
named in the request, e.g. the symposium in research_notes.md, get found), then the executor
issues up to MAX_CALLS_PER_SUBGOAL tool calls.
As-needed decomposition: a subgoal that fails is re-planned into smaller subgoals, but only
down to MAX_DEPTH, so cost grows only where the task is actually hard.

Everything else (LLM, dispatcher, retriever, persona context, output contract) is shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.contract import ResultBuilder, TaskStatus
from common.environment import TaskEnv
from common.llm_client import LLMParseError, OllamaClient
from common.retriever import DEFAULT_K, HybridRetriever, format_hits
from common.schemas import StepStatus
from common.tools import render_tools
from model3 import prompts

MODEL_NAME = "M3"
MAX_DEPTH = 2  # top-level subgoals are depth 1; one level of as-needed decomposition
MAX_CALLS_PER_SUBGOAL = 3
MAX_LLM_CALLS = 40  # hard cap per task so a looping model cannot run for hours on Colab
OBS_CHARS = 2500  # per observation, in prompts only (full text is kept in debug)


class LLMCallBudgetExceeded(RuntimeError):
    pass


@dataclass
class SubgoalRun:
    key: str  # "2", or "2.1" for a decomposed child
    description: str
    dep_keys: list[str]
    depth: int
    steps: list[int] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    ok: bool = False
    failure: str = ""
    children: list[SubgoalRun] = field(default_factory=list)

    def all_steps(self) -> list[int]:
        return self.steps + [s for c in self.children for s in c.all_steps()]

    def all_calls(self) -> list[dict[str, Any]]:
        return self.calls + [c for ch in self.children for c in ch.all_calls()]

    def trace(self) -> dict[str, Any]:
        return {
            "key": self.key, "description": self.description, "depends_on": self.dep_keys,
            "ok": self.ok, "failure": self.failure or None, "summary": self.summary,
            "steps": self.steps, "children": [c.trace() for c in self.children],
        }


class M3HierarchicalPlanner:
    name = MODEL_NAME

    def __init__(self, llm: OllamaClient, retriever: HybridRetriever, k: int = DEFAULT_K) -> None:
        self.llm = llm
        self.retriever = retriever
        self.k = k

    def run(self, env: TaskEnv, builder: ResultBuilder) -> tuple[TaskStatus, str | None, dict[str, Any]]:
        """Solve one task. LLM transport errors propagate (the runner records status ERROR)."""
        return _TaskRun(self, env, builder).solve()


class _TaskRun:
    """Per-task state, so the planner object itself holds nothing between tasks."""

    def __init__(self, model: M3HierarchicalPlanner, env: TaskEnv, builder: ResultBuilder) -> None:
        self.m = model
        self.env = env
        self.b = builder
        self.query = env.task.user_query
        self.system = env.context.system_text(env.task.reference_time)
        self.tools = render_tools(env.registry)
        self.finished: list[SubgoalRun] = []  # top-level subgoals in execution order
        self.by_key: dict[str, SubgoalRun] = {}

    # ---- top level ------------------------------------------------------------------------

    def solve(self) -> tuple[TaskStatus, str | None, dict[str, Any]]:
        debug: dict[str, Any] = {}
        try:
            plan = self._plan()
        except LLMParseError as exc:
            return TaskStatus.FAILED, None, {"failure": f"planner: {exc}"}
        debug["plan"] = plan.model_dump()

        try:
            for sg in self._to_runs(plan, parent=None, inherited_deps=[], depth=1):
                self._solve_subgoal(sg)
                self.finished.append(sg)
        except LLMCallBudgetExceeded as exc:
            debug.update(self._trace_debug(), failure=str(exc))
            return TaskStatus.FAILED, None, debug

        final_response = self._final_response(debug)
        debug.update(self._trace_debug())
        return self._status(), final_response, debug

    def _status(self) -> TaskStatus:
        if not all(sg.ok for sg in self.finished):
            return TaskStatus.FAILED
        if any(s.status in (StepStatus.REQUIRES_CONFIRMATION, StepStatus.BLOCKED) for s in self.b.steps):
            return TaskStatus.BLOCKED
        return TaskStatus.COMPLETE

    def _trace_debug(self) -> dict[str, Any]:
        out: dict[str, Any] = {"subgoal_trace": [sg.trace() for sg in self.finished]}
        if self.env.dispatcher.violations:
            out["violations"] = self.env.dispatcher.violations
        if any(s.tool == "update_profile" and s.status == StepStatus.EXECUTED for s in self.b.steps):
            out["final_profile"] = self.env.connector.get_profile()  # in-memory state after writes
        return out

    # ---- planning -------------------------------------------------------------------------

    def _plan(self) -> prompts.Plan:
        context = self._retrieve(self.query)
        user = prompts.PLANNER.format(user_query=self.query, tools=self.tools, context=context)
        return self._llm(user, prompts.Plan, "plan")

    def _decompose(self, sg: SubgoalRun) -> list[SubgoalRun]:
        context = self._retrieve(sg.description)
        user = prompts.DECOMPOSE.format(
            user_query=self.query, goal=sg.description, reason=sg.failure or "unknown",
            results=self._results_text(sg), tools=self.tools, context=context,
        )
        plan = self._llm(user, prompts.Plan, f"decompose:{sg.key}")
        return self._to_runs(plan, parent=sg, inherited_deps=sg.dep_keys, depth=sg.depth + 1)

    def _to_runs(self, plan: prompts.Plan, parent: SubgoalRun | None, inherited_deps: list[str], depth: int) -> list[SubgoalRun]:
        prefix = f"{parent.key}." if parent else ""
        valid = {s.id for s in plan.subgoals}
        runs = []
        for s in plan.subgoals:
            local = [f"{prefix}{d}" for d in s.depends_on if d in valid and d != s.id]
            run = SubgoalRun(key=f"{prefix}{s.id}", description=s.description,
                             dep_keys=list(dict.fromkeys(inherited_deps + local)), depth=depth)
            self.by_key[run.key] = run
            runs.append(run)
        return runs

    # ---- execution ------------------------------------------------------------------------

    def _solve_subgoal(self, sg: SubgoalRun) -> None:
        self._execute(sg)
        if sg.ok or sg.depth >= MAX_DEPTH:
            return
        try:
            sg.children = self._decompose(sg)
        except LLMParseError as exc:
            sg.failure += f" | decomposition failed: {exc}"
            return
        for child in sg.children:
            self._solve_subgoal(child)
        sg.ok = all(c.ok for c in sg.children)
        sg.summary = " ".join(c.summary for c in sg.children if c.summary)

    def _execute(self, sg: SubgoalRun) -> None:
        context = self._retrieve(sg.description)
        dep_steps = [s for k in sg.dep_keys if k in self.by_key for s in self.by_key[k].all_steps()]
        n_calls = 0
        while True:
            user = prompts.EXECUTOR.format(
                user_query=self.query, subgoal_id=sg.key, subgoal=sg.description,
                results=self._results_text(sg), context=context, tools=self.tools,
                calls=_calls_text(sg.calls),
            )
            try:
                decision = self._llm(user, prompts.ExecutorDecision, f"execute:{sg.key}")
            except LLMParseError as exc:
                sg.failure = f"executor returned invalid JSON twice: {exc}"
                return
            if decision.extractions:
                self.b.add_extractions(decision.extractions, sg.description)
            if decision.decision == "done":
                sg.ok, sg.summary = True, decision.summary or decision.thought
                return
            if decision.decision == "cannot":
                sg.failure = decision.thought or decision.summary or "executor reported it cannot proceed"
                return
            if n_calls >= MAX_CALLS_PER_SUBGOAL:
                sg.failure = f"used {MAX_CALLS_PER_SUBGOAL} tool calls without completing the subgoal"
                return

            result = self.env.dispatcher.dispatch(decision.tool or "", decision.args)
            n_calls += 1
            number = self.b.add_tool_result(
                result, subgoal=sg.description, depends_on=dep_steps + sg.steps,
                cited_ids=decision.evidence_ids, schedule_proposals=decision.schedule_proposals,
            )
            if number is not None:
                sg.steps.append(number)
            sg.calls.append({"tool": result.tool, "args": result.args, "status": result.status.value,
                             "observation": result.observation})

    def _final_response(self, debug: dict[str, Any]) -> str | None:
        user = prompts.FINAL.format(user_query=self.query, results=self._results_text(None))
        try:
            return self._llm(user, prompts.FinalAnswer, "final").final_response
        except LLMParseError as exc:
            debug["final_response_error"] = str(exc)
            return None

    # ---- helpers --------------------------------------------------------------------------

    def _retrieve(self, query: str) -> str:
        hits = self.m.retriever.search(query, self.env.task.reference_time, k=self.m.k)
        self.b.log_retrieval(query, [h.record.id for h in hits])
        return format_hits(hits)

    def _llm(self, user: str, schema: type, purpose: str):
        if len(self.m.llm.telemetry.calls) >= MAX_LLM_CALLS:
            raise LLMCallBudgetExceeded(f"stopped after {MAX_LLM_CALLS} LLM calls (at {purpose})")
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        return self.m.llm.chat_json(messages, schema, purpose)

    def _results_text(self, current: SubgoalRun | None) -> str:
        """Summaries of every finished subgoal; full observations only for direct dependencies."""
        deps = set(current.dep_keys) if current else {sg.key for sg in self.finished}
        lines = []
        for sg in self._finished_flat():
            if current is not None and sg.key == current.key:
                continue
            state = "DONE" if sg.ok else f"INCOMPLETE ({sg.failure})"
            lines.append(f"Subgoal {sg.key}: {sg.description} -> {state}. {sg.summary}".rstrip())
            if sg.key in deps:
                lines += [f"    {c['tool']}({c['args']}) [{c['status']}]: {_clip(c['observation'])}" for c in sg.all_calls()]
        return "\n".join(lines) or "(none yet)"

    def _finished_flat(self) -> list[SubgoalRun]:
        """Finished top-level subgoals, plus completed children of the one being decomposed."""
        done = list(self.finished)
        done_keys = {sg.key for sg in done}
        for key, sg in self.by_key.items():
            # Children of finished subgoals are already covered by their parent's summary/calls.
            if "." in key and key.split(".")[0] not in done_keys and (sg.ok or sg.failure):
                done.append(sg)
        return done


def _calls_text(calls: list[dict[str, Any]]) -> str:
    if not calls:
        return "(none)"
    return "\n".join(f"{i}. {c['tool']}({c['args']}) [{c['status']}]: {_clip(c['observation'])}" for i, c in enumerate(calls, 1))


def _clip(text: str) -> str:
    return text if len(text) <= OBS_CHARS else text[:OBS_CHARS] + " [...]"
