"""Workflow graphs: typed nodes, edges and n8n-style references between node outputs.

A parameter can refer to an upstream node's output with `{{ node_id.path }}`, for example
`{{ search.items.0.text }}`. A parameter that is exactly one reference gets the raw value (a list or
dict stays a list or dict); references inside longer text are written in as text.
An edge may carry a branch ("true"/"false"): its target runs only if the source `logic.if` gave that result.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

REF = re.compile(r"\{\{\s*([A-Za-z0-9_\-]+)((?:\.[A-Za-z0-9_\-]+)*)\s*\}\}")


class Node(BaseModel):
    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    label: str = ""


class Edge(BaseModel):
    source: str
    target: str
    branch: str | None = None


class Workflow(BaseModel):
    id: str
    name: str
    source: str  # which generator made it: baseline, template:<id>, model_a, ...
    goal: str = ""
    nodes: list[Node]
    edges: list[Edge] = Field(default_factory=list)
    trigger: dict[str, Any] | None = None

    def node(self, node_id: str) -> Node:
        return next(n for n in self.nodes if n.id == node_id)

    def order(self) -> list[Node]:
        """Nodes in dependency order (Kahn). Raises ValueError on a cycle or an edge to an unknown node."""
        ids = [n.id for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate node id")
        indeg = {i: 0 for i in ids}
        for e in self.edges:
            if e.source not in indeg or e.target not in indeg:
                raise ValueError(f"edge {e.source} -> {e.target} names an unknown node")
            indeg[e.target] += 1
        ready = [i for i in ids if indeg[i] == 0]
        out = []
        while ready:
            i = ready.pop(0)
            out.append(i)
            for e in self.edges:
                if e.source == i:
                    indeg[e.target] -= 1
                    if indeg[e.target] == 0:
                        ready.append(e.target)
        if len(out) != len(ids):
            raise ValueError("the workflow has a cycle")
        return [self.node(i) for i in out]

    def parents(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.target == node_id]


def _walk(value: Any, path: list[str]) -> Any:
    for part in path:
        if isinstance(value, list):
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise KeyError(part)
    return value


def refs(value: Any) -> set[str]:
    """Node ids a parameter value refers to."""
    if isinstance(value, str):
        return {m.group(1) for m in REF.finditer(value)}
    if isinstance(value, dict):
        return set().union(*(refs(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(refs(v) for v in value)) if value else set()
    return set()


def resolve(value: Any, outputs: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: resolve(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, outputs) for v in value]
    if not isinstance(value, str):
        return value
    whole = REF.fullmatch(value.strip())
    if whole:
        return _walk(outputs[whole.group(1)], [p for p in whole.group(2).split(".") if p])

    def text(m: re.Match) -> str:
        v = _walk(outputs[m.group(1)], [p for p in m.group(2).split(".") if p])
        return v if isinstance(v, str) else json.dumps(v, default=str)
    return REF.sub(text, value)
