"""Reference agents for harness sanity (v2): the oracle reads ground truth, the null agent does nothing.

If the oracle cannot pass a task, the task or its checker is broken. The null agent must fail every
utility task; it passes deletion probes trivially, which is why safety numbers are always reported
next to task success.
"""
from __future__ import annotations

import re
from datetime import datetime

from p17sim.gen2.personas import PersonaExtra, pref_value
from p17sim.schema import Persona
from p17sim.system.llm import Profile
from p17sim.system.runner import AgentResult, Dataset
from p17sim.system.shell import Shell

WINDOWS = {"morning": (480, 720), "afternoon": (720, 1020), "evening": (1020, 1260)}


class NullAgent:
    name = "null"

    def __init__(self, *_):
        self.profile = Profile()

    def observe(self, record: dict) -> None:
        pass

    def run(self, request: str) -> AgentResult:
        return AgentResult()

    def memory_holds(self, value: str) -> list[str]:
        return []


class OracleAgent:
    name = "oracle"

    def __init__(self, shell: Shell, persona: Persona, extra: PersonaExtra, ds: Dataset):
        self.shell, self.persona, self.extra, self.ds = shell, persona, extra, ds
        self.profile = Profile()
        self.by_text = {r["text"]: ds.labels[r["event_id"]] for r in ds.records[persona.persona_id]
                        if ds.labels[r["event_id"]].get("checker") or
                        ds.labels[r["event_id"]].get("provenance", "").startswith("delete:")}

    def observe(self, record: dict) -> None:
        pass

    def memory_holds(self, value: str) -> list[str]:
        return []

    def _current(self, fact_id: str) -> str | None:
        value = None
        for r in self.ds.records[self.persona.persona_id]:
            if datetime.fromisoformat(r["timestamp"]) >= self.shell.now:
                break
            lab = self.ds.labels[r["event_id"]]
            if lab.get("fact_id") == fact_id and not lab.get("checker"):
                if lab.get("provenance", "").startswith("delete:"):
                    value = None
                elif lab.get("fact_value") and not lab["provenance"].startswith("distractor"):
                    value = lab["fact_value"]
        return value

    def run(self, request: str) -> AgentResult:
        lab = self.by_text[request]
        if lab.get("provenance", "").startswith("delete:"):
            ids = [r["event_id"] for r in self.ds.records[self.persona.persona_id]
                   if self.ds.labels[r["event_id"]].get("fact_id") == lab["fact_id"]
                   and not self.ds.labels[r["event_id"]].get("checker") and r["event_id"] in self.shell.records]
            self.shell.execute("delete_records", {"ids": ids})
            return AgentResult(answer="Deleted.")
        ck = lab["checker"]
        if ck["kind"] in ("recall", "update", "deletion"):
            return AgentResult(answer=self._current(lab["fact_id"]) or "I don't have that information.")
        if ck["kind"] == "privacy":
            name = next(c.name for c in self.persona.contacts if c.contact_id == ck["args"]["to"])
            self.shell.execute("send_email", {"to": name, "subject": "Hi", "body": "Busy week, talk soon."})
            return AgentResult(answer="Sent.")
        a = ck["args"]
        lo, hi = WINDOWS[pref_value(self.extra, "meeting_window", lab["day"])]
        start = -(-max(lo, a["earliest"]) // 15) * 15
        while any(start < b1 and start + a["duration_minutes"] > b0 for b0, b1 in a["busy"]):
            start += 15
        from p17sim.gen2 import phrasing as ph

        name = next(c.name for c in self.persona.contacts if c.contact_id == a["to"])
        self.shell.execute("create_event", {"date": ph.on_date(self.extra.start_weekday, a["day"]).isoformat(),
                                            "start": f"{start // 60:02d}:{start % 60:02d}",
                                            "duration_minutes": a["duration_minutes"], "with_contact": name})
        return AgentResult(answer="Booked.")
