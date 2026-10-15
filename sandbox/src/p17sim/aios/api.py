"""Backend API (FastAPI) for the Personal AI OS, and the static web front end.

One workspace per persona, kept in a SQLite file under `data/aios/`. Loading a persona at a moment
replays their records up to that moment. One model request runs at a time, because only one model
fits in memory on the 8 GB laptop. Actions that need the user's say wait in Approvals.
Run with `p17sim serve` and open http://localhost:8017.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from p17sim.aios.engine import queue_approval
from p17sim.aios.generators import NotTrained, catalog, generator, routine_reference, suggestion_reference
from p17sim.aios.store import Store
from p17sim.aios.templates import TEMPLATES
from p17sim.aios.workflow import Workflow
from p17sim.aios.workspace import Workspace
from p17sim.system.llm import Chat
from p17sim.system.runner import Dataset

STATIC = Path(__file__).with_name("static")


class LoadBody(BaseModel):
    day: int = 25
    time: str = "09:00"
    handle_instructions: bool = False  # act on past "forget" instructions with the chosen generator while replaying
    generator: str = "baseline"


class AdvanceBody(BaseModel):
    minutes: int = 60


class GoalBody(BaseModel):
    text: str = ""
    generator: str = "baseline"
    params: dict[str, Any] = {}


class DecisionBody(BaseModel):
    decision: str  # approve | reject


class PolicyBody(BaseModel):
    tool: str
    value: str  # allow | ask | deny


class DeleteBody(BaseModel):
    record_ids: list[str]
    request: str = "Deleted from the Data tab"


class AuditBody(BaseModel):
    value: str


class SaveWorkflowBody(BaseModel):
    workflow: dict[str, Any]
    trigger: dict[str, Any] | None = None  # {"kind": "time", "at": "08:00"} or {"kind": "event", ...}


class Hub:
    def __init__(self, ds: Dataset, model: str, store_dir: Path | None, results: Path, llm_factory=None):
        self.ds, self.model, self.store_dir, self.results = ds, model, store_dir, results
        self.llm_factory = llm_factory or (lambda: Chat(model=model, seed=11))
        self.workspaces: dict[str, Workspace] = {}
        self.lock = threading.Lock()

    def load(self, pid: str, body: LoadBody) -> Workspace:
        if pid not in self.ds.personas:
            raise HTTPException(404, f"unknown persona {pid}")
        if self.store_dir:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            path = self.store_dir / f"{pid}.db"
            if pid in self.workspaces:
                self.workspaces[pid].store.close()
            path.unlink(missing_ok=True)
            store = Store(path)
        else:
            store = Store()
        ws = Workspace(store, self.ds.personas[pid], self.ds.extras[pid], self.ds.records[pid], llm=self.llm_factory())
        start = datetime.fromisoformat(self.ds.records[pid][0]["timestamp"]).replace(hour=0, minute=0)
        hh, mm = (int(x) for x in body.time.split(":"))
        target = start + timedelta(days=max(1, min(30, body.day)) - 1, hours=hh, minutes=mm)
        gen = generator(body.generator) if body.handle_instructions else None
        ws.advance_to(target, skip=self.ds.is_task,
                      on_instruction=(lambda r: gen.run(r["text"], ws)) if gen else None)
        self.workspaces[pid] = ws
        return ws

    def ws(self, pid: str) -> Workspace:
        if pid not in self.workspaces:
            return self.load(pid, LoadBody())
        return self.workspaces[pid]


def _result(res) -> dict:
    return {"execution_id": res.execution_id, "status": res.status, "answer": res.answer, "workflow": res.workflow,
            "interventions": res.interventions, "error": res.error, "approval_id": res.approval_id,
            "node_runs": res.node_runs}


def create_app(ds: Dataset, model: str = "qwen3.5:4b", store_dir: Path | None = None,
               results: Path = Path("research/results"), llm_factory=None) -> FastAPI:
    hub = Hub(ds, model, store_dir, results, llm_factory)
    app = FastAPI(title="Personal AI OS", version="0.1", description="Local-first personal assistant (Topic 17)")
    app.state.hub = hub

    @app.get("/api/personas")
    def personas():
        order = {"test": 0, "ood_test": 1, "val": 2, "train": 3}
        out = [{"id": pid, "name": p.name, "archetype": p.archetype, "split": ds.splits[pid],
                "region": ds.extras[pid].region, "contacts": [c.name for c in p.contacts]}
               for pid, p in ds.personas.items()]
        return sorted(out, key=lambda x: (order[x["split"]], x["id"]))

    @app.post("/api/ws/{pid}/load")
    def load(pid: str, body: LoadBody):
        with hub.lock:
            return hub.load(pid, body).state()

    @app.get("/api/ws/{pid}/state")
    def state(pid: str):
        return hub.ws(pid).state()

    @app.post("/api/ws/{pid}/advance")
    def advance(pid: str, body: AdvanceBody):
        ws = hub.ws(pid)
        with hub.lock:
            fired = ws.advance_to(ws.now + timedelta(minutes=body.minutes), skip=ds.is_task)
        return {"state": ws.state(), "fired": [_result(r) for r in fired]}

    @app.get("/api/ws/{pid}/records")
    def records(pid: str, channel: str = "", q: str = "", limit: int = 100):
        ws = hub.ws(pid)
        chans = {channel} if channel else None
        rows = ws.store.search(pid, q, chans, limit, before=ws.now.isoformat()) if q else \
            ws.store.visible(pid, before=ws.now.isoformat(), channels=chans)[-limit:][::-1]
        return [{k: r[k] for k in ("id", "ts", "end_ts", "channel", "sender", "text", "permission", "data_class")}
                for r in rows]

    @app.get("/api/ws/{pid}/calendar")
    def calendar(pid: str, date: str = ""):
        ws = hub.ws(pid)
        d = datetime.strptime(date, "%Y-%m-%d").date() if date else ws.now.date()
        return {"date": d.isoformat(), "entries": ws.store.calendar_on(pid, d)}

    @app.post("/api/ws/{pid}/goal")
    def goal(pid: str, body: GoalBody):
        ws = hub.ws(pid)
        try:
            gen = generator(body.generator)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        with hub.lock:
            t = time.perf_counter()
            try:
                res = gen.run(body.text, ws, approver=queue_approval, params=body.params)
            except NotTrained as exc:
                raise HTTPException(409, str(exc))
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(400, str(exc))
            wall = time.perf_counter() - t
        return {**_result(res), "latency_s": round(wall, 2)}

    @app.get("/api/templates")
    def templates():
        return [{"id": k, "description": v[1], "params": v[2]} for k, v in TEMPLATES.items()]

    @app.get("/api/ws/{pid}/workflows")
    def workflows(pid: str):
        return hub.ws(pid).store.workflows(pid)

    @app.get("/api/ws/{pid}/workflows/{wid}")
    def workflow(pid: str, wid: str):
        wf = hub.ws(pid).store.workflow(wid)
        if wf is None:
            raise HTTPException(404, "no such workflow")
        return wf

    @app.post("/api/ws/{pid}/workflows")
    def save_workflow(pid: str, body: SaveWorkflowBody):
        ws = hub.ws(pid)
        try:
            wf = Workflow.model_validate(body.workflow)
            wf.order()
        except ValueError as exc:
            raise HTTPException(400, f"invalid workflow: {exc}")
        if body.trigger:
            return {"trigger_id": ws.add_trigger(wf, body.trigger["kind"], body.trigger), "workflow_id": wf.id}
        ws.store.save_workflow(wf.model_dump(), pid, ws.now.isoformat())
        return {"workflow_id": wf.id}

    @app.post("/api/ws/{pid}/workflows/{wid}/run")
    def run_workflow(pid: str, wid: str):
        ws = hub.ws(pid)
        raw = ws.store.workflow(wid)
        if raw is None:
            raise HTTPException(404, "no such workflow")
        with hub.lock:
            return _result(ws.engine.run(Workflow.model_validate(raw), raw["source"], approver=queue_approval))

    @app.get("/api/ws/{pid}/executions")
    def executions(pid: str):
        return hub.ws(pid).store.executions(pid)

    @app.get("/api/ws/{pid}/executions/{eid}")
    def execution(pid: str, eid: str):
        ws = hub.ws(pid)
        e = ws.store.execution(eid)
        if e is None:
            raise HTTPException(404, "no such execution")
        return {**e, "workflow": ws.store.workflow(e["workflow_id"]), "node_runs": ws.store.node_runs(eid),
                "sent": ws.store.outbox(pid, eid), "created": ws.store.events(pid, eid)}

    @app.post("/api/ws/{pid}/executions/{eid}/undo")
    def undo(pid: str, eid: str):
        ws = hub.ws(pid)
        cancelled = [e["id"] for e in ws.store.events(pid, eid) if not e["cancelled"] and ws.store.cancel_event(e["id"])]
        ws.store.audit_log(pid, ws.now.isoformat(), "user", "undo_execution", eid, {"cancelled": cancelled})
        return {"cancelled_events": cancelled, "note": "Sent emails cannot be recalled."}

    @app.get("/api/ws/{pid}/approvals")
    def approvals(pid: str, status: str = ""):
        return hub.ws(pid).store.approvals(pid, status or None)

    @app.post("/api/ws/{pid}/approvals/{aid}")
    def decide(pid: str, aid: str, body: DecisionBody):
        ws = hub.ws(pid)
        with hub.lock:
            try:
                return _result(ws.engine.resume(aid, body.decision))
            except ValueError as exc:
                raise HTTPException(409, str(exc))

    @app.get("/api/ws/{pid}/policy")
    def policy(pid: str):
        ws = hub.ws(pid)
        return {"tools": ws.engine.policy.tools(),
                "locations": {"local": "may process public, personal and sensitive data",
                              "cloud": "may process public data only"},
                "egress_allowlist": sorted(ws.gateway.allow_hosts)}

    @app.put("/api/ws/{pid}/policy")
    def set_policy(pid: str, body: PolicyBody):
        ws = hub.ws(pid)
        try:
            ws.engine.policy.set_tool(body.tool, body.value)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        ws.store.audit_log(pid, ws.now.isoformat(), "user", "set_policy", body.tool, {"value": body.value})
        return ws.engine.policy.tools()

    @app.get("/api/ws/{pid}/privacy")
    def privacy(pid: str):
        ws = hub.ws(pid)
        egress = ws.store.egress_log(pid)
        return {"egress": egress[:100],
                "egress_summary": {"calls": len(egress), "blocked": sum(not e["allowed"] for e in egress),
                                   "nonlocal_bytes": sum(e["bytes"] for e in egress if e["host"] not in
                                                         ("localhost", "127.0.0.1", "::1"))},
                "receipts": ws.store.receipts(pid), "audit": ws.store.audits(pid, 100),
                "classes": {r["data_class"]: r["n"] for r in ws.store._rows(
                    "SELECT data_class, COUNT(*) AS n FROM records WHERE persona_id = ? AND deleted = 0 "
                    "GROUP BY data_class", (pid,))}}

    @app.post("/api/ws/{pid}/storage-audit")
    def storage_audit(pid: str, body: AuditBody):
        hits = hub.ws(pid).deletion.storage_audit(pid, body.value)
        return {"value": body.value, "found_in": [{"table": t, "column": c, "row": k} for t, c, k in hits]}

    @app.post("/api/ws/{pid}/delete")
    def delete(pid: str, body: DeleteBody):
        ws = hub.ws(pid)
        return ws.deletion.delete(pid, body.record_ids, body.request, actor="user", ts=ws.now.isoformat())

    @app.get("/api/generators")
    def generators():
        return catalog()

    @app.get("/api/ws/{pid}/suggestions")
    def suggestions(pid: str):
        ws = hub.ws(pid)
        return {"routines": routine_reference(ws), "suggestions": suggestion_reference(ws),
                "models": {g["name"]: g["status"] for g in catalog() if g["name"].startswith("model_")}}

    @app.get("/api/benchmark")
    def benchmark():
        from p17sim.analysis.metrics import load_run, summarize

        runs = []
        for f in sorted((results / "runs").glob("*.jsonl")) if (results / "runs").exists() else []:
            rows, meta = load_run(f)
            s = summarize(rows)
            runs.append({"file": f.name, "system": meta.get("system"), "seed": meta.get("seed"),
                         **{k: s[k] for k in ("tasks", "task_success", "success_by_kind", "deletion_compliance",
                                              "privacy_leakage", "latency_p95_s", "raw_export_bytes")}})
        md = results / "RESULTS.md"
        return {"runs": runs, "results_md": md.read_text() if md.exists() else None}

    @app.get("/api/results/figures/{name}")
    def figure(name: str):
        path = (results / "figures" / name).resolve()
        if path.parent != (results / "figures").resolve() or not path.exists() or path.suffix != ".png":
            raise HTTPException(404, "no such figure")
        return FileResponse(path)

    @app.get("/api/profile")
    def profile():
        p = results / "model_profile.json"
        return json.loads(p.read_text()) if p.exists() else []

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        page = STATIC / "index.html"
        if not page.exists():
            raise HTTPException(404, "front end not built")
        return FileResponse(page)

    return app


def serve(trace: Path, data: Path, model: str, port: int, store_dir: Path) -> None:
    import uvicorn

    app = create_app(Dataset.load(trace, data), model, store_dir)
    print(f"Personal AI OS on http://localhost:{port}  (API docs: /docs)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
