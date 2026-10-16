"""A persona's workspace: their store, privacy layer, engine, clock and triggers.

The clock is simulated: `advance_to(t)` delivers every record whose time is before t (messages arrive,
notes are saved, preference signals happen), fires event triggers as records arrive and time
triggers as the clock passes them. User instructions in the stream (for example "please forget my
gate code") are handed to `on_instruction`, so the chosen generator acts on them when they happen.

`tools(execution_id, approver)` gives an agent the classic tool interface (`tool_schemas`,
`tool_descriptions`, `execute`), but every call runs as an engine node: policies, logging, data
classes and rollback apply to the baseline exactly as to any generated workflow.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable

from p17sim.aios.engine import Approver, Engine, ExecutionResult, auto_approve
from p17sim.aios.nodes import NODES
from p17sim.aios.privacy import DeletionManager, EgressGateway, classify_record
from p17sim.aios.store import Store
from p17sim.aios.workflow import Workflow
from p17sim.gen2.personas import PersonaExtra
from p17sim.schema import Persona
from p17sim.system.llm import Chat, PersonaText
from p17sim.system.shell import Shell

# Agent tool name -> (node type, how agent arguments map to node parameters)
TOOL_NODES: dict[str, tuple[str, Callable[[dict], dict]]] = {
    "search_records": ("context.search", lambda a: {"query": a.get("query", ""), "kind": a.get("kind", "all"),
                                                     "limit": a.get("limit", 8)}),
    "read_record": ("context.read", lambda a: {"id": a.get("id", "")}),
    "list_contacts": ("context.contacts", lambda a: {}),
    "list_calendar": ("context.calendar", lambda a: {"date": a.get("date", "")}),
    "create_event": ("action.create_event", lambda a: {k: a.get(k) for k in ("date", "start", "duration_minutes",
                                                                             "with_contact", "title")}),
    "cancel_event": ("action.cancel_event", lambda a: {"event_id": a.get("event_id", "")}),
    "send_email": ("action.send_email", lambda a: {"to": a.get("to", ""), "subject": a.get("subject", ""),
                                                   "body": a.get("body", "")}),
    "delete_records": ("action.delete_records", lambda a: {"ids": a.get("ids", []), "request": a.get("request", "")}),
}


class EngineTools:
    """The shell-style tool interface an agent expects, backed by engine nodes."""

    def __init__(self, ws: "Workspace", execution_id: str, approver: Approver):
        self.ws, self.execution_id, self.approver = ws, execution_id, approver
        self.persona = ws.persona

    @property
    def now(self) -> datetime:
        return self.ws.now

    def tool_schemas(self) -> list[dict]:
        return Shell.tool_schemas()

    def tool_descriptions(self) -> str:
        return Shell.tool_descriptions()

    def execute(self, name: str, args: dict[str, Any]) -> str:
        if name not in TOOL_NODES:
            return f"Error: unknown tool '{name}'."
        node_type, mapping = TOOL_NODES[name]
        try:
            params = mapping(args or {})
        except (TypeError, AttributeError) as exc:
            return f"Error: {exc}"
        status, out = self.ws.engine.step(self.execution_id, node_type, params, self.approver, label=name)
        if status == "ok":
            return NODES[node_type].display(out)
        if status == "waiting_approval":
            return "This action is waiting for the user's approval."
        return f"Error: {out}"


class Workspace:
    def __init__(self, store: Store, persona: Persona, extra: PersonaExtra, records: list[dict],
                 llm: Chat | None = None, model: str = "qwen3.5:4b", seed: int | None = None,
                 allow_hosts: set[str] | None = None):
        self.store, self.persona, self.extra, self.pid = store, persona, extra, persona.persona_id
        self.stream = sorted(records, key=lambda r: (r["timestamp"], r["event_id"]))
        self.cursor = 0
        self.now = datetime.fromisoformat(self.stream[0]["timestamp"]) if self.stream else datetime(2026, 9, 7)
        self.gateway = EgressGateway(store, allow_hosts)
        self.deletion = DeletionManager(store)
        self.llm = llm or Chat(model=model, seed=seed)
        self.llm.gateway, self.llm.persona_id = self.gateway, self.pid
        self.llm.persona_text = PersonaText([r["text"] for r in records])
        store.put_persona(persona.model_dump())
        self.engine = Engine(self)
        self.fired: list[dict] = []

    # ---------------------------------------------------------------- data in
    def ingest(self, record: dict) -> list[ExecutionResult]:
        cls = "public" if record["channel"] == "permission" else classify_record(record)
        self.store.add_record(record, cls)
        return self._event_triggers(record)

    def advance_to(self, until: datetime, on_instruction: Callable[[dict], Any] | None = None,
                   skip: Callable[[dict], bool] | None = None) -> list[ExecutionResult]:
        """Deliver every record before `until`, in time order. `skip` hides records (e.g. benchmark
        questions); `on_instruction` acts on user instructions as they arrive."""
        fired = []
        while self.cursor < len(self.stream) and datetime.fromisoformat(self.stream[self.cursor]["timestamp"]) < until:
            r = self.stream[self.cursor]
            self.cursor += 1
            t = datetime.fromisoformat(r["timestamp"])
            fired += self._time_triggers(self.now, t)
            self.now = t
            if skip and skip(r):
                continue
            fired += self.ingest(r)
            if r["channel"] == "instruction" and on_instruction:
                on_instruction(r)
        fired += self._time_triggers(self.now, until)
        self.now = until
        return fired

    # ---------------------------------------------------------------- triggers
    def add_trigger(self, wf: Workflow, kind: str, spec: dict) -> str:
        tid = f"tr-{uuid.uuid4().hex[:8]}"
        self.store.save_workflow(wf.model_dump(), self.pid, self.now.isoformat())
        self.store.add_trigger({"id": tid, "persona_id": self.pid, "workflow_id": wf.id, "kind": kind, "spec": spec})
        return tid

    def _run_triggered(self, trig: dict, when: datetime, payload: dict) -> ExecutionResult:
        wf = Workflow.model_validate(self.store.workflow(trig["workflow_id"]))
        wf = wf.model_copy(update={"id": f"{wf.id}-{when.strftime('%m%d%H%M')}"})
        if wf.nodes and wf.nodes[0].type.startswith("trigger."):
            wf.nodes[0].params = {**wf.nodes[0].params, **payload}
        saved = self.now
        self.now = when
        res = self.engine.run(wf, generator=wf.source, trigger=trig["kind"], approver=self.trigger_approver)
        self.now = saved
        self.store.trigger_fired(trig["id"], when.isoformat())
        self.fired.append({"trigger": trig["id"], "at": when.isoformat(), "execution": res.execution_id})
        return res

    trigger_approver: Approver = staticmethod(auto_approve)

    def _time_triggers(self, start: datetime, end: datetime) -> list[ExecutionResult]:
        out = []
        for trig in (t for t in self.store.triggers(self.pid) if t["kind"] == "time"):
            hh, mm = (int(x) for x in trig["spec"]["at"].split(":"))
            day = start.replace(hour=0, minute=0, second=0, microsecond=0)
            while day <= end:
                at = day.replace(hour=hh, minute=mm)
                weekdays = trig["spec"].get("weekdays") or list(range(7))
                if start <= at < end and at.weekday() in weekdays:
                    out.append(self._run_triggered(trig, at, {"time": at.isoformat()}))
                day += timedelta(days=1)
        return out

    def _event_triggers(self, record: dict) -> list[ExecutionResult]:
        out = []
        for trig in (t for t in self.store.triggers(self.pid) if t["kind"] == "event"):
            spec = trig["spec"]
            if spec.get("channel") and record["channel"] != spec["channel"]:
                continue
            if spec.get("sender") and spec["sender"].lower() not in (record.get("sender") or "").lower():
                continue
            out.append(self._run_triggered(trig, self.now, {"record": {k: record[k] for k in (
                "event_id", "timestamp", "sender", "channel", "text")}}))
        return out

    # ---------------------------------------------------------------- tools and goals
    def tools(self, execution_id: str, approver: Approver = auto_approve) -> EngineTools:
        return EngineTools(self, execution_id, approver)

    def state(self) -> dict:
        return {"persona_id": self.pid, "name": self.persona.name, "now": self.now.isoformat(),
                "records_seen": self.cursor, "records_total": len(self.stream),
                "pending_approvals": len(self.store.approvals(self.pid, "pending")),
                "triggers": [{"id": t["id"], "kind": t["kind"], "spec": t["spec"], "workflow_id": t["workflow_id"]}
                             for t in self.store.triggers(self.pid)]}

    def permissions_text(self) -> str:
        return json.dumps(self.engine.policy.tools())
