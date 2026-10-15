"""Unified harness: replays each persona's 30 days through the shell and scores an agent.

Protocol, the same for every agent:
1. Records arrive in time order. Each one is shown to the shell and to the agent (`observe`).
2. A deletion request is a user instruction: the agent gets it as a turn (`run`) when it happens,
   whether or not the window is scored. Right after, a storage audit checks that none of the fact's
   values is left in the shell's records or in the agent's own memory (`memory_holds`).
3. A task is run only if it is in the scored set (persona split and day window). Its checker sees
   the answer and only the emails and events the agent made during that task.
Nothing about a task's answer is given to the agent; labels stay with the harness.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from p17sim.gen2.personas import PersonaExtra
from p17sim.schema import Persona
from p17sim.system.checkers import check
from p17sim.system.llm import Profile
from p17sim.system.shell import Shell


@dataclass
class AgentResult:
    answer: str = ""
    turns: int = 0
    error: str | None = None
    trace: list[dict] = field(default_factory=list)  # the agent's own step log (plan, evidence, ...)


class Agent(Protocol):
    name: str
    profile: Profile

    def observe(self, record: dict) -> None: ...

    def run(self, request: str) -> AgentResult: ...

    def memory_holds(self, value: str) -> list[str]: ...


AgentFactory = Callable[[Shell, Persona, PersonaExtra], Agent]


@dataclass
class Dataset:
    personas: dict[str, Persona]
    extras: dict[str, PersonaExtra]
    records: dict[str, list[dict]]  # persona_id -> trace records in time order
    labels: dict[str, dict]  # event_id -> label
    splits: dict[str, str]

    @classmethod
    def load(cls, trace_dir: Path, data_dir: Path) -> "Dataset":
        from p17sim.gen2.build import load
        from p17sim.io import read_jsonl

        personas, extras, _, splits = load(data_dir)
        records: dict[str, list[dict]] = {}
        for r in read_jsonl(trace_dir / "trace.jsonl"):
            records.setdefault(r["persona_id"], []).append(r)
        for rs in records.values():
            rs.sort(key=lambda r: (r["timestamp"], r["event_id"]))
        labels = {lab["event_id"]: lab for lab in read_jsonl(trace_dir / "labels.jsonl")}
        return cls({p.persona_id: p for p in personas}, {x.persona_id: x for x in extras}, records, labels, splits)

    def tasks(self, persona_ids: list[str], windows: tuple[str, ...]) -> list[dict]:
        return [self.labels[r["event_id"]] for pid in persona_ids for r in self.records[pid]
                if self.labels[r["event_id"]].get("checker") and self.labels[r["event_id"]]["split"] in windows]


def _permissions(records: list[dict]) -> dict[str, str]:
    return {r["text"].split(":")[0]: r["permission_label"] for r in records if r["channel"] == "permission"}


def run_persona(ds: Dataset, pid: str, make_agent: AgentFactory, windows: tuple[str, ...],
                kinds: set[str] | None = None, on_row: Callable[[dict], None] | None = None) -> list[dict]:
    persona, extra = ds.personas[pid], ds.extras[pid]
    records = ds.records[pid]
    shell = Shell(persona=persona, permissions=_permissions(records), now=datetime.fromisoformat(records[0]["timestamp"]))
    agent = make_agent(shell, persona, extra)
    rows = []
    fact_values: dict[str, list[str]] = {}
    for lab in (ds.labels[r["event_id"]] for r in records):
        if lab.get("fact_id") and lab.get("fact_value") and not lab.get("provenance", "").startswith("distractor"):
            fact_values.setdefault(lab["fact_id"], []).append(lab["fact_value"])
    for r in records:
        lab = ds.labels[r["event_id"]]
        shell.now = datetime.fromisoformat(r["timestamp"])
        is_delete = lab.get("provenance", "").startswith("delete:")
        if lab.get("checker"):
            ck = lab["checker"]
            if lab["split"] not in windows or (kinds and ck["kind"] not in kinds):
                continue
            row = _run_turn(agent, shell, r["text"], "task")
            row.update(check(ck["kind"], ck["args"], row["answer"], row.pop("_sent"), row.pop("_created"),
                             extra.start_weekday))
            row.update({"kind": ck["kind"], "wording": lab.get("wording"), "fact_kind": lab.get("fact_kind")})
        elif is_delete:
            shell.observe(r)
            agent.observe(r)
            row = _run_turn(agent, shell, r["text"], "deletion_request")
            row.pop("_sent"), row.pop("_created")
            left_shell = sorted({rid for v in fact_values.get(lab["fact_id"], []) for rid in shell.holds(v)})
            left_agent = sorted({k for v in fact_values.get(lab["fact_id"], []) for k in agent.memory_holds(v)})
            row.update({"kind": "deletion_audit", "passed": not left_shell and not left_agent,
                        "left_in_shell": left_shell, "left_in_agent_memory": left_agent})
        else:
            shell.observe(r)
            agent.observe(r)
            continue
        row.update({"persona_id": pid, "persona_split": ds.splits[pid], "task_id": r["event_id"],
                    "window": lab["split"], "day": lab["day"], "archetype": persona.archetype,
                    "region": extra.region, "name_list": extra.name_list, "renderer": extra.renderer,
                    "request": r["text"]})
        rows.append(row)
        if on_row:
            on_row(row)
    return rows


def _run_turn(agent: Agent, shell: Shell, text: str, what: str) -> dict[str, Any]:
    n_sent, n_created, n_log, n_conf = len(shell.sent), len(shell.created), len(shell.log), shell.confirmations
    calls_before, blocks_before = len(agent.profile.calls), dict(agent.profile.blocks)
    t = time.perf_counter()
    try:
        res = agent.run(text)
    except Exception as exc:  # a crash is a failed task, recorded with its error
        res = AgentResult(answer="", error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")
    wall = time.perf_counter() - t
    sub = Profile(calls=agent.profile.calls[calls_before:],
                  blocks={k: v - blocks_before.get(k, 0.0) for k, v in agent.profile.blocks.items()})
    return {"what": what, "answer": res.answer, "turns": res.turns, "error": res.error, "latency_s": round(wall, 3),
            "interventions": shell.confirmations - n_conf, "tool_calls": shell.log[n_log:],
            "agent_trace": res.trace, "profile": sub.summary(),
            "_sent": shell.sent[n_sent:], "_created": shell.created[n_created:]}


def evaluate(ds: Dataset, make_agent: AgentFactory, persona_ids: list[str], windows: tuple[str, ...],
             out: Path, meta: dict[str, Any], kinds: set[str] | None = None) -> list[dict]:
    """Run and write one JSON line per scored task or deletion audit, plus a meta file.
    Rows are appended as they finish, so a stopped run keeps its work."""
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=1, default=str) + "\n")
    done: set[str] = set()
    if out.exists():  # resume: keep finished personas, drop the rows of a persona cut off mid-way
        old = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        done = {row["persona_id"] for row in old if row.get("persona_done")}
        out.write_text("".join(json.dumps(row, default=str) + "\n" for row in old if row["persona_id"] in done))
    rows = []
    with out.open("a") as f:
        for pid in persona_ids:
            if pid in done:
                continue
            def write(row):
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()
            rows += run_persona(ds, pid, make_agent, windows, kinds, on_row=write)
            write({"persona_id": pid, "persona_done": True})
    return rows
