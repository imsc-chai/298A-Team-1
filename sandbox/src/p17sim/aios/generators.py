"""Workflow generators: what turns a goal (plus the person's context) into a workflow.

- baseline: the OpenJarvis native_openhands-style agent. It decides one step at a time; each model
  call and each tool call is recorded as a node, so its run is a workflow like any other.
- template:<id>: a hand-built workflow with parameters.
- model_a / model_b / model_c: slots for the learned models. They report "pending training" until
  their trained artifacts exist.
Two simple, non-learned references give the B and C slots something to show and a bar to beat:
weekly routines found by counting repeated calendar entries (B), and the most-accepted option per
preference over the last week (C).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from p17sim.aios import templates
from p17sim.aios.engine import Approver, ExecutionResult, auto_approve
from p17sim.aios.workspace import Workspace


class NotTrained(Exception):
    pass


class RecordingChat:
    """Wraps the workspace's model client so every agent model call is logged as a workflow node."""

    def __init__(self, inner, engine, execution_id: str):
        self.inner, self.engine, self.execution_id = inner, engine, execution_id

    @property
    def profile(self):
        return self.inner.profile

    def chat(self, messages, tools=None, component="llm", json_mode=False, max_tokens=None):
        r = self.inner.chat(messages, tools=tools, component=component, json_mode=json_mode, max_tokens=max_tokens)
        self.engine.record(self.execution_id, "reason.agent_step", {
            "content": (r["content"] or "")[:2000],
            "tool_calls": [{"name": tc["name"], "arguments": tc["arguments"]} for tc in r["tool_calls"]]},
            label=component)
        return r


@dataclass
class BaselineGenerator:
    name: str = "baseline"
    status: str = "ready"
    description: str = ("OpenJarvis native_openhands-style agent with qwen3.5:4b: decides one tool call at a time "
                        "(the reproduced baseline).")
    max_turns: int = 10

    def run(self, goal: str, ws: Workspace, approver: Approver = auto_approve, **_) -> ExecutionResult:
        from p17sim.agents.baseline import BaselineAgent

        eid = ws.engine.open(goal, self.name)
        agent = BaselineAgent(ws.tools(eid, approver), ws.persona, RecordingChat(ws.llm, ws.engine, eid),
                              max_turns=self.max_turns)
        try:
            res = agent.run(goal)
        except Exception as exc:  # the model server timed out or failed: keep the execution, mark it failed
            out = ws.engine.close(eid, "", status="failed")
            out.error = f"{type(exc).__name__}: {exc}"
            return out
        return ws.engine.close(eid, res.answer)


@dataclass
class TemplateGenerator:
    template_id: str
    status: str = "ready"

    @property
    def name(self) -> str:
        return f"template:{self.template_id}"

    @property
    def description(self) -> str:
        return templates.TEMPLATES[self.template_id][1]

    def run(self, goal: str, ws: Workspace, approver: Approver = auto_approve, params: dict | None = None,
            **_) -> ExecutionResult:
        wf = templates.build(self.template_id, params or {})
        return ws.engine.run(wf, self.name, trigger="chat", approver=approver)


@dataclass
class PendingGenerator:
    name: str
    description: str
    status: str = "pending training"

    def run(self, goal: str, ws: Workspace, approver: Approver = auto_approve, **_) -> ExecutionResult:
        raise NotTrained(f"{self.name} is not trained yet. Train it, then register its artifacts.")


PENDING = {
    "model_a": PendingGenerator("model_a", "Retrieval-augmented hierarchical planner: goal + context -> full workflow."),
    "model_b": PendingGenerator("model_b", "Routine and preference model: proposes recurring workflows and triggers."),
    "model_c": PendingGenerator("model_c", "Contextual bandit: which proactive workflow to suggest, and when."),
}


def generator(name: str):
    if name == "baseline":
        return BaselineGenerator()
    if name.startswith("template:"):
        return TemplateGenerator(name.split(":", 1)[1])
    if name in PENDING:
        return PENDING[name]
    raise KeyError(f"unknown generator {name}")


def catalog() -> list[dict[str, Any]]:
    out = [{"name": "baseline", "kind": "goal -> workflow", "status": "ready",
            "description": BaselineGenerator.description}]
    out += [{"name": f"template:{k}", "kind": "template", "status": "ready", "description": v[1], "params": v[2]}
            for k, v in templates.TEMPLATES.items()]
    out += [{"name": g.name, "kind": {"model_a": "goal -> workflow", "model_b": "routines -> triggers",
                                      "model_c": "context -> suggestion"}[g.name], "status": g.status,
             "description": g.description} for g in PENDING.values()]
    return out


# ------------------------------------------------------------------ references for the B and C slots

def _norm(title: str) -> str:
    return re.sub(r"\s*[(\-].*$", "", title.lower()).strip()


def routine_reference(ws: Workspace, min_count: int = 3, tolerance_min: int = 45) -> list[dict]:
    """Calendar entries with the same title on the same weekday at about the same time, at least
    `min_count` times: proposed as a weekly reminder trigger 15 minutes before."""
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for r in ws.store.visible(ws.pid, before=ws.now.isoformat(), channels={"calendar"}):
        t = datetime.fromisoformat(r["ts"])
        groups[(_norm(r["text"]), t.weekday())].append(t.hour * 60 + t.minute)
    out = []
    for (title, wd), mins in sorted(groups.items()):
        med = sorted(mins)[len(mins) // 2]
        close = [m for m in mins if abs(m - med) <= tolerance_min]
        if len(close) >= min_count:
            at = max(0, med - 15)
            out.append({"title": title, "weekday": wd, "typical_start": f"{med // 60:02d}:{med % 60:02d}",
                        "occurrences": len(close),
                        "proposed_trigger": {"kind": "time", "at": f"{at // 60:02d}:{at % 60:02d}", "weekdays": [wd]},
                        "source": "reference (counting), not Model B"})
    return out


def suggestion_reference(ws: Workspace, days: int = 7) -> list[dict]:
    """For each preference, the option the person accepted most often in the last `days` days."""
    since = (ws.now - timedelta(days=days)).isoformat()
    counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in ws.store.visible(ws.pid, before=ws.now.isoformat(), channels={"preference"}):
        if r["ts"] < since:
            continue
        t = r["text"].lower()
        for key, rx in (("meeting_window", r"suggested an? (morning|afternoon|evening) meeting slot: (accepted|declined)"),
                        ("channel", r"suggested reaching out by (email|sms): (accepted|declined)"),
                        ("reminder_lead_minutes", r"suggested a reminder (\d+) minutes before: (accepted|declined)")):
            m = re.search(rx, t)
            if m:
                counts[key][m.group(1)][1] += 1
                counts[key][m.group(1)][0] += m.group(2) == "accepted"
    out = []
    for key, opts in sorted(counts.items()):
        best = max(opts, key=lambda o: (opts[o][0], -opts[o][1]))
        out.append({"key": key, "suggest": best, "accepted": opts[best][0], "offered": opts[best][1],
                    "source": "reference (most accepted), not Model C"})
    return out
