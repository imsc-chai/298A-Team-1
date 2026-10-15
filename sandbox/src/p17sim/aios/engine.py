"""The workflow engine: runs a workflow node by node under the privacy layer and logs everything.

For each node, in dependency order:
1. skip it if it sits on a branch its `logic.if` parent did not take;
2. resolve its parameters from upstream outputs; its input data class is the highest class among
   the nodes it reads from (references and parent edges);
3. location policy: a cloud node may only receive public data, otherwise the execution is blocked;
4. tool policy: deny fails the node; ask goes to the approver. The harness approver approves and
   counts an intervention. The app approver parks the node as a pending approval, and `resume` runs it
   once the user decides;
5. run it, timed, and write a node run (input, output, data class, location) and an audit entry;
6. on an error, undo the reversible actions this execution already took (rollback).
`open` / `step` / `close` run single nodes inside an open execution: the baseline agent's tool calls
go through `step`, so its run becomes a workflow too.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from p17sim.aios.nodes import NODES, NodeSpec, RunContext
from p17sim.aios.privacy import Policy, PolicyError, max_class
from p17sim.aios.workflow import Edge, Node, Workflow, refs, resolve

Approver = Callable[[dict], str]  # request -> "approved" | "rejected" | "pending"
LOG_CHARS = 4000


def auto_approve(request: dict) -> str:
    """The harness's simulated user: reads the request and approves it."""
    return "approved"


def queue_approval(request: dict) -> str:
    """The app's approver: park the action until the user decides in the Approvals tab."""
    return "pending"


@dataclass
class ExecState:
    execution_id: str
    workflow: Workflow
    generator: str
    outputs: dict[str, Any] = field(default_factory=dict)
    classes: dict[str, str] = field(default_factory=dict)
    undo: list[tuple[NodeSpec, dict]] = field(default_factory=list)
    interventions: int = 0
    status: str = "running"
    error: str | None = None
    approval_id: str | None = None
    answer: str = ""
    llm_calls_at_start: int = 0
    stepwise: bool = False  # an agent loop: a failed step is reported back, nothing is rolled back
    approver: Approver = auto_approve  # kept so a resumed workflow asks again for later "ask" actions


@dataclass
class ExecutionResult:
    execution_id: str
    status: str
    answer: str
    workflow: dict
    interventions: int
    error: str | None
    approval_id: str | None
    node_runs: list[dict]


class Engine:
    def __init__(self, ws: Any):
        self.ws = ws
        self.policy = Policy(ws.store, ws.pid)
        self.paused: dict[str, ExecState] = {}
        self.open_states: dict[str, ExecState] = {}

    # ---------------------------------------------------------------- whole workflows
    def run(self, wf: Workflow, generator: str, trigger: str = "chat", approver: Approver = auto_approve,
            start_at: int = 0, state: ExecState | None = None) -> ExecutionResult:
        wf.order()  # validate before anything runs
        st = state or self._open(wf, generator, trigger)
        if state is None:
            st.approver = approver
        for node in wf.order()[start_at:]:
            if not self._branch_taken(st, node):
                self._log(st, node, "skipped", "local", "public", {}, None, None, 0.0)
                continue
            stop = self._run_node(st, node, approver)
            if stop:
                break
        if st.status == "running":
            st.status = "succeeded"
        return self._close(st)

    def resume(self, approval_id: str, decision: str) -> ExecutionResult:
        """Finish a parked execution after the user decides on its pending approval."""
        appr = self.ws.store.approval(approval_id)
        if appr is None or appr["status"] != "pending" or appr["execution_id"] not in self.paused:
            raise ValueError(f"approval {approval_id} is not pending in this session")
        st = self.paused.pop(appr["execution_id"])
        self.ws.store.decide(approval_id, "approved" if decision == "approve" else "rejected", self.ws.now.isoformat())
        order = st.workflow.order()
        idx = next(i for i, n in enumerate(order) if n.id == appr["node_id"])
        st.status, st.approval_id = "running", None
        if decision != "approve":
            self._log(st, order[idx], "rejected", NODES[order[idx].type].location, "public", appr["request"], None,
                      "the user rejected this action", 0.0)
            st.status = "rejected"
            return self._close(st)
        st.interventions += 1
        if self._run_node(st, order[idx], lambda r: "approved", already_approved=True):
            return self._close(st)
        return self.run(st.workflow, st.generator, approver=st.approver, start_at=idx + 1, state=st)

    # ---------------------------------------------------------------- single steps (agent loops)
    def open(self, goal: str, generator: str, trigger: str = "chat") -> str:
        wf = Workflow(id=f"wf-{uuid.uuid4().hex[:10]}", name=goal[:60] or generator, source=generator, goal=goal,
                      nodes=[Node(id="trigger", type="trigger.chat", params={"goal": goal})], trigger={"kind": trigger})
        st = self._open(wf, generator, trigger)
        st.stepwise = True
        self._run_node(st, wf.nodes[0], auto_approve)
        self.open_states[st.execution_id] = st
        return st.execution_id

    def step(self, execution_id: str, node_type: str, params: dict, approver: Approver = auto_approve,
             label: str = "") -> tuple[str, Any]:
        """Append one node to an open execution and run it. Returns (status, output or error text)."""
        st = self.open_states[execution_id]
        prev = st.workflow.nodes[-1].id
        nid = f"n{len(st.workflow.nodes)}"
        node = Node(id=nid, type=node_type, params=params, label=label)
        st.workflow.nodes.append(node)
        st.workflow.edges.append(Edge(source=prev, target=nid))
        st.status = "running"
        self._run_node(st, node, approver)
        if nid in st.outputs:
            return "ok", st.outputs[nid]
        status = st.status
        st.status = "running"  # an agent may carry on after a refused or failed tool call
        return status, st.error

    def record(self, execution_id: str, node_type: str, output: dict, label: str = "") -> None:
        """Log a step that already happened (an agent's model call) as a node of the workflow."""
        st = self.open_states[execution_id]
        prev = st.workflow.nodes[-1].id
        nid = f"n{len(st.workflow.nodes)}"
        st.workflow.nodes.append(Node(id=nid, type=node_type, params={}, label=label))
        st.workflow.edges.append(Edge(source=prev, target=nid))
        cls = max_class(*st.classes.values())
        st.outputs[nid], st.classes[nid] = output, cls
        self._log(st, st.workflow.nodes[-1], "ok", "local", cls, {}, output, None, 0.0)

    def close(self, execution_id: str, answer: str, status: str = "succeeded") -> ExecutionResult:
        st = self.open_states.pop(execution_id)
        nid = f"n{len(st.workflow.nodes)}"
        prev = st.workflow.nodes[-1].id
        node = Node(id=nid, type="output.respond", params={"text": answer})
        st.workflow.nodes.append(node)
        st.workflow.edges.append(Edge(source=prev, target=nid))
        self._run_node(st, node, auto_approve)
        st.status = status if st.status in ("running", "succeeded") else st.status
        return self._close(st)

    # ---------------------------------------------------------------- internals
    def _open(self, wf: Workflow, generator: str, trigger: str) -> ExecState:
        eid = f"ex-{uuid.uuid4().hex[:10]}"
        self.ws.store.start_execution(eid, wf.id, self.ws.pid, generator, trigger, wf.goal, self.ws.now.isoformat())
        st = ExecState(eid, wf, generator)
        st.llm_calls_at_start = len(self.ws.llm.profile.calls)
        return st

    def _close(self, st: ExecState) -> ExecutionResult:
        self.ws.store.save_workflow(st.workflow.model_dump(), self.ws.pid, self.ws.now.isoformat())
        if st.status == "waiting_approval":
            self.paused[st.execution_id] = st
        from p17sim.system.llm import Profile

        prof = Profile(calls=self.ws.llm.profile.calls[st.llm_calls_at_start:]).summary()
        answers = [st.outputs[n.id]["text"] for n in st.workflow.nodes
                   if n.type == "output.respond" and n.id in st.outputs]
        st.answer = answers[-1] if answers else st.answer
        self.ws.store.finish_execution(st.execution_id, st.status, st.answer, {**prof, "interventions": st.interventions},
                                       self.ws.now.isoformat())
        return ExecutionResult(st.execution_id, st.status, st.answer, st.workflow.model_dump(), st.interventions,
                               st.error, st.approval_id, self.ws.store.node_runs(st.execution_id))

    def _branch_taken(self, st: ExecState, node: Node) -> bool:
        for e in st.workflow.parents(node.id):
            if e.source not in st.outputs:
                return False  # its parent was skipped or stopped
            if e.branch is not None and str(st.outputs[e.source].get("result")).lower() != e.branch.lower():
                return False
        return True

    def _input_class(self, st: ExecState, node: Node) -> str:
        sources = refs(node.params) | {e.source for e in st.workflow.parents(node.id)}
        return max_class("public", *(st.classes.get(s, "public") for s in sources))

    def _log(self, st: ExecState, node: Node, status: str, location: str, cls: str, params: Any, output: Any,
             error: str | None, wall: float) -> None:
        self.ws.store.add_node_run({
            "execution_id": st.execution_id, "persona_id": self.ws.pid, "node_id": node.id, "type": node.type,
            "status": status, "location": location, "data_class": cls,
            "input": json.dumps(params, default=str)[:LOG_CHARS],
            "output": None if output is None else json.dumps(output, default=str)[:LOG_CHARS],
            "error": error, "started": self.ws.now.isoformat(), "wall_s": round(wall, 4)})

    def _stop(self, st: ExecState, status: str, error: str) -> bool:
        st.status, st.error = status, error
        if status in ("failed", "blocked") and not st.stepwise:
            for spec, out in reversed(st.undo):  # rollback of what this execution did
                try:
                    spec.undo(RunContext(self.ws, st.execution_id), out)
                    self.ws.store.audit_log(self.ws.pid, self.ws.now.isoformat(), "engine", "rollback", spec.type, out)
                except Exception as exc:  # keep rolling back the rest
                    self.ws.store.audit_log(self.ws.pid, self.ws.now.isoformat(), "engine", "rollback_failed",
                                            spec.type, str(exc))
            st.undo.clear()
        return True

    def _run_node(self, st: ExecState, node: Node, approver: Approver, already_approved: bool = False) -> bool:
        """Run one node. Returns True if the execution has to stop here."""
        spec = NODES.get(node.type)
        if spec is None:
            self._log(st, node, "failed", "local", "public", node.params, None, f"unknown node type {node.type}", 0.0)
            return self._stop(st, "failed", f"unknown node type {node.type}")
        try:
            params = resolve(node.params, st.outputs)
        except (KeyError, IndexError, ValueError) as exc:
            self._log(st, node, "failed", spec.location, "public", node.params, None, f"bad reference: {exc}", 0.0)
            return self._stop(st, "failed", f"bad reference in {node.id}: {exc}")
        cls = self._input_class(st, node)
        ok, why = self.policy.may_process(spec.location, cls)
        if not ok:
            self._log(st, node, "blocked", spec.location, cls, {"withheld": f"{cls} input"}, None, why, 0.0)
            self.ws.store.audit_log(self.ws.pid, self.ws.now.isoformat(), "privacy", "blocked_location", node.type,
                                    {"node": node.id, "reason": why})
            return self._stop(st, "blocked", why)
        if spec.tool and not already_approved:
            decision = "ask" if spec.tool == "human.approval" else self.policy.tool(spec.tool)
            self.ws.store.audit_log(self.ws.pid, self.ws.now.isoformat(), "privacy", f"tool_{decision}", spec.tool,
                                    {"node": node.id})
            if decision == "deny":
                self._log(st, node, "denied", spec.location, cls, params, None, f"policy denies {spec.tool}", 0.0)
                return self._stop(st, "denied", f"the policy does not allow {spec.tool}")
            if decision == "ask" and spec.category in ("action", "human"):
                request = {"tool": spec.tool, "node": node.id, "type": node.type, "params": params}
                answer = approver(request)
                if answer == "approved":
                    st.interventions += 1
                elif answer == "rejected":
                    self._log(st, node, "rejected", spec.location, cls, params, None, "the user rejected this action", 0.0)
                    return self._stop(st, "rejected", "the user rejected this action")
                else:
                    aid = f"ap-{uuid.uuid4().hex[:8]}"
                    self.ws.store.add_approval({"id": aid, "persona_id": self.ws.pid, "execution_id": st.execution_id,
                                                "node_id": node.id, "tool": spec.tool, "request": request,
                                                "ts": self.ws.now.isoformat()})
                    self._log(st, node, "waiting", spec.location, cls, params, None, None, 0.0)
                    st.approval_id = aid
                    st.status = "waiting_approval"
                    st.error = f"waiting for the user's approval ({aid})"
                    return True
        if spec.category == "action":
            params = {**params, "_approved": True}
        ctx = RunContext(self.ws, st.execution_id, node.id, cls)
        t = time.perf_counter()
        try:
            out, out_cls = spec.run(ctx, params)
        except (PolicyError, ValueError, KeyError, TypeError, OSError) as exc:
            self._log(st, node, "failed", spec.location, cls, params, None, str(exc), time.perf_counter() - t)
            return self._stop(st, "failed", f"{node.type}: {exc}")
        wall = time.perf_counter() - t
        out_cls = max_class(cls, out_cls) if spec.category not in ("context", "trigger") else out_cls
        st.outputs[node.id], st.classes[node.id] = out, out_cls
        logged = {k: v for k, v in params.items() if k != "_approved"}
        if node.type == "action.delete_records":
            # This log row is written after the deletion ran: keep only the ids, never what the records said.
            logged = {"ids": out["deleted"], "request": logged.get("request", "")}
        self._log(st, node, "ok", spec.location, out_cls, logged, out, None, wall)
        if spec.category == "action":
            self.ws.store.audit_log(self.ws.pid, self.ws.now.isoformat(), st.generator, node.type, node.id, out)
        if spec.reversible and spec.undo:
            st.undo.append((spec, out))
        return False

