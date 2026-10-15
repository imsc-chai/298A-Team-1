"""Training data for the pending models, from train and val personas only (test and OOD never leave
the harness).

- Model A (goal + context -> workflow): one row per request (tasks and "forget" instructions) with the
  goal, the time, the ids of the records visible then, and a **gold workflow** built from ground truth.
  Every gold workflow is executed in the runtime and checked by the task's checker before export.
- Model B (routines): each persona's time-ordered events with the routine each calendar entry belongs to.
- Model C (proactive suggestions): every suggestion shown, its context, the option offered, whether it was
  accepted, and the true preference at that time (for regret).
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from p17sim.aios.engine import Approver, ExecutionResult, auto_approve
from p17sim.aios.workflow import Edge, Node, Workflow
from p17sim.gen2 import phrasing as ph
from p17sim.gen2.personas import pref_value
from p17sim.system.llm import ScriptedChat
from p17sim.system.runner import Dataset, System, run_persona

WINDOWS = {"morning": (480, 720), "afternoon": (720, 1020), "evening": (1020, 1260)}
TRAINING_SPLITS = ("train", "val")


def _chain(goal: str, steps: list[tuple[str, str, dict]]) -> Workflow:
    nodes = [Node(id="trigger", type="trigger.chat", params={"goal": goal})] + \
            [Node(id=i, type=t, params=p) for i, t, p in steps]
    return Workflow(id="gold", name=goal[:60], source="gold", goal=goal, nodes=nodes,
                    edges=[Edge(source=nodes[k].id, target=nodes[k + 1].id) for k in range(len(nodes) - 1)])


def _fact_state(ds: Dataset, pid: str, fact_id: str, before: str) -> tuple[str | None, list[str]]:
    """(current value or None if deleted/unknown, ids of the records carrying the current value)."""
    value, ids = None, []
    for r in ds.records[pid]:
        if r["timestamp"] >= before:
            break
        lab = ds.labels[r["event_id"]]
        if lab.get("fact_id") != fact_id or lab.get("checker"):
            continue
        prov = lab.get("provenance", "")
        if prov.startswith("delete:"):
            value, ids = None, []
        elif lab.get("fact_value") and not prov.startswith("distractor"):
            if lab["fact_value"] != value:
                ids = []
            value = lab["fact_value"]
            ids.append(r["event_id"])
    return value, ids


def gold_workflow(ds: Dataset, pid: str, record: dict) -> tuple[Workflow, dict]:
    lab = ds.labels[record["event_id"]]
    goal, ts = record["text"], record["timestamp"]
    persona, extra = ds.personas[pid], ds.extras[pid]
    if lab.get("provenance", "").startswith("delete:"):
        ids = [r["event_id"] for r in ds.records[pid] if r["timestamp"] < ts and r["channel"] != "instruction"
               and ds.labels[r["event_id"]].get("fact_id") == lab["fact_id"] and not ds.labels[r["event_id"]].get("checker")]
        return _chain(goal, [("find", "context.search", {"query": goal}),
                             ("delete", "action.delete_records", {"ids": ids, "request": goal}),
                             ("answer", "output.respond", {"text": "Done. I deleted it everywhere."})]), \
            {"kind": "forget", "evidence_ids": ids}
    ck = lab["checker"]
    kind, args = ck["kind"], ck["args"]
    if kind in ("recall", "update", "deletion"):
        value, ids = _fact_state(ds, pid, lab["fact_id"], ts)
        name = ph.FACT_NAMES[lab["fact_kind"]][0][0] if lab.get("fact_kind") else "that"
        text = f"Your {name} is {value}." if value else "You asked me to forget that, so I don't have it."
        return _chain(goal, [("find", "context.search", {"query": goal}),
                             ("answer", "output.respond", {"text": text})]), {"kind": kind, "evidence_ids": ids}
    contact = next(c.name for c in persona.contacts if c.contact_id == args["to"])
    if kind == "privacy":
        return _chain(goal, [
            ("week", "context.recent", {"days": 7, "exclude_private": True, "limit": 12}),
            ("draft", "reason.llm", {"prompt": "Write a short, friendly email to " + contact + " about the user's week "
                                               "from these records, without private details.\n{{week.items}}",
                                     "max_tokens": 300}),
            ("send", "action.send_email", {"to": contact, "subject": "My week", "body": "{{draft.text}}"}),
            ("answer", "output.respond", {"text": "Sent " + contact + " a short summary of your week."})]), \
            {"kind": kind, "evidence_ids": []}
    lo, hi = WINDOWS[pref_value(extra, "meeting_window", lab["day"])]
    start = -(-max(lo, args["earliest"]) // 15) * 15
    while any(start < b1 and start + args["duration_minutes"] > b0 for b0, b1 in args["busy"]):
        start += 15
    day = ph.on_date(extra.start_weekday, args["day"]).isoformat()
    return _chain(goal, [
        ("calendar", "context.calendar", {"date": day}),
        ("prefs", "context.preferences", {"key": "meeting_window", "days": 14}),
        ("book", "action.create_event", {"date": day, "start": f"{start // 60:02d}:{start % 60:02d}",
                                         "duration_minutes": args["duration_minutes"], "with_contact": contact}),
        ("answer", "output.respond", {"text": "Booked {{book.title}} on {{book.date}} at {{book.start}}."})]), \
        {"kind": kind, "evidence_ids": []}


class GoldGenerator:
    """Runs the gold workflow for each request: used to prove every gold example passes its checker."""
    name = "gold"

    def __init__(self, ds: Dataset, pid: str):
        self.ds, self.pid = ds, pid
        self.by_text = {r["text"]: r for r in ds.records[pid] if ds.labels[r["event_id"]].get("checker")
                        or ds.labels[r["event_id"]].get("provenance", "").startswith("delete:")}

    def run(self, goal: str, ws: Any, approver: Approver = auto_approve, **_) -> ExecutionResult:
        cands = [r for r in self.ds.records[self.pid] if r["text"] == goal and r["timestamp"] <= ws.now.isoformat()
                 and r["event_id"] in {x["event_id"] for x in self.by_text.values()}]
        wf, _ = gold_workflow(self.ds, self.pid, cands[-1] if cands else self.by_text[goal])
        return ws.engine.run(wf, "gold", approver=approver)


def model_a_rows(ds: Dataset, pids: list[str]) -> list[dict]:
    rows = []
    for pid in pids:
        deleted_at: dict[str, str] = {}
        for r in ds.records[pid]:
            lab = ds.labels[r["event_id"]]
            if lab.get("provenance", "").startswith("delete:"):
                deleted_at[lab["fact_id"]] = r["timestamp"]
            if not (lab.get("checker") or lab.get("provenance", "").startswith("delete:")):
                continue
            wf, info = gold_workflow(ds, pid, r)
            visible = [x["event_id"] for x in ds.records[pid] if x["timestamp"] < r["timestamp"]
                       and not ds.labels[x["event_id"]].get("checker")
                       and not (ds.labels[x["event_id"]].get("fact_id") in deleted_at
                                and deleted_at[ds.labels[x["event_id"]]["fact_id"]] < r["timestamp"]
                                and x["channel"] != "instruction")]
            rows.append({"persona_id": pid, "persona_split": ds.splits[pid], "window": lab.get("split"),
                         "ts": r["timestamp"], "goal": r["text"], "wording": lab.get("wording"), **info,
                         "context_ids": visible, "gold_workflow": wf.model_dump(), "checker": lab.get("checker")})
    return rows


def model_b_rows(ds: Dataset, pids: list[str]) -> list[dict]:
    rows = []
    for pid in pids:
        for r in ds.records[pid]:
            lab = ds.labels[r["event_id"]]
            if lab.get("checker") or r["channel"] in ("permission",):
                continue
            t = datetime.fromisoformat(r["timestamp"])
            prov = lab.get("provenance", "")
            rows.append({"persona_id": pid, "persona_split": ds.splits[pid], "ts": r["timestamp"],
                         "weekday": t.weekday(), "minute": t.hour * 60 + t.minute, "channel": r["channel"],
                         "sender": r.get("sender"), "text": r["text"] if r["channel"] == "calendar" else None,
                         "routine_label": prov.split(":", 1)[1] if prov.startswith("routine:") else None,
                         "archetype": ds.personas[pid].archetype})
    return rows


_OFFER = {"meeting_window": re.compile(r"suggested an? (morning|afternoon|evening) meeting slot: (accepted|declined)", re.I),
          "channel": re.compile(r"suggested reaching out by (email|sms): (accepted|declined)", re.I),
          "reminder_lead_minutes": re.compile(r"suggested a reminder (\d+) minutes before: (accepted|declined)", re.I)}


def model_c_rows(ds: Dataset, pids: list[str]) -> list[dict]:
    rows = []
    for pid in pids:
        for r in ds.records[pid]:
            lab = ds.labels[r["event_id"]]
            key = lab.get("preference_key")
            if not key or not lab.get("provenance", "").startswith("pref:"):
                continue
            m = _OFFER[key].search(r["text"])
            if not m:
                continue
            t = datetime.fromisoformat(r["timestamp"])
            rows.append({"persona_id": pid, "persona_split": ds.splits[pid], "ts": r["timestamp"], "day": lab["day"],
                         "weekday": t.weekday(), "hour": t.hour, "key": key, "offered": m.group(1),
                         "accepted": m.group(2).lower() == "accepted", "true_value": lab["preference_value"],
                         "archetype": ds.personas[pid].archetype})
    return rows


def verify_gold(ds: Dataset, pids: list[str]) -> dict[str, Any]:
    """Run every gold workflow through the runtime; all must pass their checkers and deletion audits."""
    rows = []
    for pid in pids:
        sysm = System("gold", lambda: ScriptedChat([{"content": "It was a busy, good week. Talk soon!"}] * 500),
                      lambda ws, pid=pid: GoldGenerator(ds, pid))
        rows += run_persona(ds, pid, sysm, ("train", "val", "test"))
    failed = [r for r in rows if not r["passed"]]
    return {"checked": len(rows), "failed": len(failed), "failures": [
        {k: r[k] for k in ("task_id", "kind", "request", "error")} for r in failed[:20]]}


def _write(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_all(ds: Dataset, out: Path, verify: bool = True) -> dict[str, Any]:
    pids = sorted(p for p, s in ds.splits.items() if s in TRAINING_SPLITS)
    manifest: dict[str, Any] = {"personas": pids, "splits": TRAINING_SPLITS, "files": {}}
    for name, fn in (("model_a_goal_to_workflow", model_a_rows), ("model_b_event_sequences", model_b_rows),
                     ("model_c_suggestion_log", model_c_rows)):
        rows = fn(ds, pids)
        for split in TRAINING_SPLITS:
            part = [r for r in rows if r["persona_split"] == split]
            f = out / f"{name}.{split}.jsonl"
            manifest["files"][f.name] = {"rows": len(part), "sha256": _write(f, part)}
    if verify:
        manifest["gold_check"] = verify_gold(ds, pids)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
