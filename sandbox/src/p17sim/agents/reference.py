"""Reference generators for harness sanity: the oracle reads ground truth, the null generator does nothing.

Both act through the same workspace engine as every other generator, so the sanity check also
covers the store, the privacy layer and the checkers. If the oracle cannot pass a task, the task or
its checker is broken. The null generator must fail every utility task; it passes deletion probes
trivially, which is why safety numbers are always reported next to task success.
"""
from __future__ import annotations

from datetime import datetime

from p17sim.aios.engine import Approver, ExecutionResult, auto_approve
from p17sim.gen2 import phrasing as ph
from p17sim.gen2.personas import pref_value

WINDOWS = {"morning": (480, 720), "afternoon": (720, 1020), "evening": (1020, 1260)}


class NullGenerator:
    name = "null"
    status = "reference"

    def run(self, goal: str, ws, approver: Approver = auto_approve, **_) -> ExecutionResult:
        return ws.engine.close(ws.engine.open(goal, self.name), "")


class OracleGenerator:
    name = "oracle"
    status = "reference"

    def __init__(self, ds):
        self.ds = ds

    def _current(self, ws, fact_id: str) -> str | None:
        value = None
        for r in self.ds.records[ws.pid]:
            if datetime.fromisoformat(r["timestamp"]) >= ws.now:
                break
            lab = self.ds.labels[r["event_id"]]
            if lab.get("fact_id") == fact_id and not lab.get("checker"):
                if lab.get("provenance", "").startswith("delete:"):
                    value = None
                elif lab.get("fact_value") and not lab["provenance"].startswith("distractor"):
                    value = lab["fact_value"]
        return value

    def run(self, goal: str, ws, approver: Approver = auto_approve, **_) -> ExecutionResult:
        lab = next(self.ds.labels[r["event_id"]] for r in self.ds.records[ws.pid] if r["text"] == goal
                   and r["timestamp"] <= ws.now.isoformat() and (self.ds.labels[r["event_id"]].get("checker") or
                   self.ds.labels[r["event_id"]].get("provenance", "").startswith("delete:")))
        eid = ws.engine.open(goal, self.name)
        if lab.get("provenance", "").startswith("delete:"):
            ids = [r["event_id"] for r in self.ds.records[ws.pid]
                   if self.ds.labels[r["event_id"]].get("fact_id") == lab["fact_id"]
                   and not self.ds.labels[r["event_id"]].get("checker") and r["timestamp"] < ws.now.isoformat()
                   and r["channel"] != "instruction"]
            ws.engine.step(eid, "action.delete_records", {"ids": ids, "request": goal}, approver)
            return ws.engine.close(eid, "Deleted.")
        ck = lab["checker"]
        if ck["kind"] in ("recall", "update", "deletion"):
            return ws.engine.close(eid, self._current(ws, lab["fact_id"]) or "I don't have that information.")
        name = next(c.name for c in ws.persona.contacts if c.contact_id == ck["args"]["to"])
        if ck["kind"] == "privacy":
            ws.engine.step(eid, "action.send_email", {"to": name, "subject": "Hi", "body": "Busy week, talk soon."},
                           approver)
            return ws.engine.close(eid, "Sent.")
        a = ck["args"]
        lo, hi = WINDOWS[pref_value(ws.extra, "meeting_window", lab["day"])]
        start = -(-max(lo, a["earliest"]) // 15) * 15
        while any(start < b1 and start + a["duration_minutes"] > b0 for b0, b1 in a["busy"]):
            start += 15
        ws.engine.step(eid, "action.create_event", {
            "date": ph.on_date(ws.extra.start_weekday, a["day"]).isoformat(),
            "start": f"{start // 60:02d}:{start % 60:02d}", "duration_minutes": a["duration_minutes"],
            "with_contact": name}, approver)
        return ws.engine.close(eid, "Booked.")
