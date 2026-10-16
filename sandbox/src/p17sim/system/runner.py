"""Unified harness: replays each persona's 30 days through the Personal AI OS and scores a generator.

Protocol, the same for every system (it runs the real workspace: SQLite store, privacy layer, engine):
1. Records arrive in time order (`Workspace.advance_to`). Benchmark questions are hidden from the
   workspace; they are only ever asked, never stored as history.
2. A deletion request is a user instruction: the generator gets it as a goal when it happens,
   whether or not the window is scored. Right after, a storage audit searches every store
   (records, node runs, outbox, approvals, audit log, ...) for every value of that fact.
3. A task is run only if it is in the scored set (persona split and day window). Its checker sees
   the answer and only the emails and events created by that task's execution.
Nothing about a task's answer reaches the generator; labels stay with the harness.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from p17sim.gen2.personas import PersonaExtra
from p17sim.schema import Persona
from p17sim.system.checkers import check
from p17sim.system.llm import Chat, Profile


@dataclass
class AgentResult:
    answer: str = ""
    turns: int = 0
    error: str | None = None
    trace: list[dict] = field(default_factory=list)


@dataclass
class Dataset:
    personas: dict[str, Persona]
    extras: dict[str, PersonaExtra]
    records: dict[str, list[dict]]  # persona_id -> trace records in time order
    labels: dict[str, dict]  # event_id -> label
    splits: dict[str, str]

    @classmethod
    def load(cls, trace_dir: Path, data_dir: Path) -> "Dataset":
        from p17sim.io import read_jsonl

        personas = [Persona.model_validate(r) for r in read_jsonl(data_dir / "personas.jsonl")]
        extras = [PersonaExtra.model_validate(r) for r in read_jsonl(data_dir / "extras.jsonl")]
        splits = json.loads((data_dir / "split_manifest.json").read_text())["persona_splits"]
        records: dict[str, list[dict]] = {}
        for r in read_jsonl(trace_dir / "trace.jsonl"):
            records.setdefault(r["persona_id"], []).append(r)
        for rs in records.values():
            rs.sort(key=lambda r: (r["timestamp"], r["event_id"]))
        labels = {lab["event_id"]: lab for lab in read_jsonl(trace_dir / "labels.jsonl")}
        return cls({p.persona_id: p for p in personas}, {x.persona_id: x for x in extras}, records, labels, splits)

    def is_task(self, record: dict) -> bool:
        return bool(self.labels[record["event_id"]].get("checker"))


@dataclass
class System:
    """What is evaluated: a model client per persona and a generator built on the workspace."""
    name: str
    llm: Callable[[], Chat]
    generator: Callable[[Any], Any]  # Workspace -> generator with .run(goal, ws, approver)


def run_persona(ds: Dataset, pid: str, system: System, windows: tuple[str, ...], kinds: set[str] | None = None,
                on_row: Callable[[dict], None] | None = None) -> list[dict]:
    from p17sim.aios.store import Store
    from p17sim.aios.workspace import Workspace

    persona, extra = ds.personas[pid], ds.extras[pid]
    ws = Workspace(Store(), persona, extra, ds.records[pid], llm=system.llm())
    gen = system.generator(ws)
    fact_values: dict[str, list[str]] = {}
    for r in ds.records[pid]:
        lab = ds.labels[r["event_id"]]
        if lab.get("fact_id") and lab.get("fact_value") and not lab.get("provenance", "").startswith("distractor"):
            fact_values.setdefault(lab["fact_id"], []).append(lab["fact_value"])
    rows: list[dict] = []

    def meta(r: dict) -> dict:
        lab = ds.labels[r["event_id"]]
        return {"persona_id": pid, "persona_split": ds.splits[pid], "task_id": r["event_id"], "window": lab["split"],
                "day": lab["day"], "archetype": persona.archetype, "region": extra.region,
                "name_list": extra.name_list, "renderer": extra.renderer, "request": r["text"]}

    def emit(row: dict) -> None:
        rows.append(row)
        if on_row:
            on_row(row)

    def on_instruction(r: dict) -> None:
        lab = ds.labels[r["event_id"]]
        if not lab.get("provenance", "").startswith("delete:"):
            return
        row, _ = _turn(ws, gen, r["text"])
        hits = sorted({f"{t}.{c}" for v in fact_values.get(lab["fact_id"], [])
                       for t, c, _k in ws.deletion.storage_audit(pid, v)})
        row.update({"kind": "deletion_audit", "passed": not hits, "left_in": hits, **meta(r)})
        emit(row)

    for r in ds.records[pid]:
        lab = ds.labels[r["event_id"]]
        if not lab.get("checker"):
            continue
        ws.advance_to(datetime.fromisoformat(r["timestamp"]), on_instruction=on_instruction, skip=ds.is_task)
        ck = lab["checker"]
        if lab["split"] not in windows or (kinds and ck["kind"] not in kinds):
            continue
        row, res = _turn(ws, gen, r["text"])
        sent = [{"to_id": m["to_id"], "subject": m["subject"] or "", "body": m["body"] or ""}
                for m in ws.store.outbox(pid, res.execution_id)] if res else []
        created = [{"with_id": e["with_id"], "date": e["date"], "start_minute": e["start_min"],
                    "duration": e["duration"], "cancelled": bool(e["cancelled"])}
                   for e in ws.store.events(pid, res.execution_id)] if res else []
        row.update(check(ck["kind"], ck["args"], row["answer"], sent, created, extra.start_weekday))
        row.update({"kind": ck["kind"], "wording": lab.get("wording"), "fact_kind": lab.get("fact_kind"), **meta(r)})
        emit(row)
    ws.advance_to(datetime(2100, 1, 1), on_instruction=on_instruction, skip=ds.is_task)
    return rows


def _turn(ws: Any, gen: Any, text: str) -> tuple[dict, Any]:
    from p17sim.aios.engine import auto_approve

    calls_before, blocks_before = len(ws.llm.profile.calls), dict(ws.llm.profile.blocks)
    t = time.perf_counter()
    res, error = None, None
    try:
        res = gen.run(text, ws, approver=auto_approve)
        error = res.error if res.status != "succeeded" else None
    except Exception as exc:  # a crash is a failed task, recorded with its error
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
    wall = time.perf_counter() - t
    sub = Profile(calls=ws.llm.profile.calls[calls_before:],
                  blocks={k: v - blocks_before.get(k, 0.0) for k, v in ws.llm.profile.blocks.items()})
    nodes = ws.store.node_runs(res.execution_id) if res else []
    row = {"answer": res.answer if res else "", "status": res.status if res else "crashed", "error": error,
           "latency_s": round(wall, 3), "interventions": res.interventions if res else 0,
           "turns": sum(n["type"] == "reason.agent_step" for n in nodes),
           "execution_id": res.execution_id if res else None,
           "tool_calls": [{"node": n["node_id"], "type": n["type"], "status": n["status"], "location": n["location"],
                           "data_class": n["data_class"]} for n in nodes],
           "profile": sub.summary()}
    return row, res


def evaluate(ds: Dataset, system: System, persona_ids: list[str], windows: tuple[str, ...], out: Path,
             meta: dict[str, Any], kinds: set[str] | None = None) -> list[dict]:
    """Run and write one JSON line per scored task or deletion audit, plus a meta file.
    Rows are appended as they finish, so a stopped run keeps its work."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    done: set[str] = set()
    if out.exists():  # resume: keep finished personas, drop the rows of a persona cut off mid-way
        old = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        done = {row["persona_id"] for row in old if row.get("persona_done")}
        out.write_text("".join(json.dumps(row, default=str) + "\n" for row in old if row["persona_id"] in done))
    rows = []
    with out.open("a") as f:
        def write(row):
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
        for pid in persona_ids:
            if pid in done:
                continue
            rows += run_persona(ds, pid, system, windows, kinds, on_row=write)
            write({"persona_id": pid, "persona_done": True})
    return rows
