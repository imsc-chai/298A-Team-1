"""Every system the harness can evaluate, by name. All share the workspace, data and model settings."""
from __future__ import annotations

import json
import subprocess
import urllib.request
from typing import Any

from p17sim.aios.generators import PENDING, BaselineGenerator
from p17sim.agents.reference import NullGenerator, OracleGenerator
from p17sim.system.llm import Chat
from p17sim.system.runner import Dataset, System

# Model A ablations, pre-registered (H4). They run once Model A is trained.
ABLATIONS = {
    "model_a-no_reranker": ({"use_reranker": False}, {"recall", "update"}),
    "model_a-no_recency": ({"use_recency": False}, {"update"}),
    "model_a-no_intent": ({"use_intent_model": False}, {"scheduling", "privacy"}),
    "model_a-no_deletion_memory": ({"deletion_aware": False}, {"deletion"}),
}
READY = ["baseline", "oracle", "null"]
SYSTEMS = READY + list(PENDING) + list(ABLATIONS)


def system(name: str, ds: Dataset, model: str = "qwen3.5:4b", seed: int | None = 11) -> System:
    def llm() -> Chat:
        return Chat(model=model, seed=seed)

    if name == "baseline":
        return System(name, llm, lambda ws: BaselineGenerator())
    if name == "oracle":
        return System(name, llm, lambda ws: OracleGenerator(ds))
    if name == "null":
        return System(name, llm, lambda ws: NullGenerator())
    if name in PENDING or name in ABLATIONS:
        raise NotImplementedError(f"{name} is pending training; see aios/generators.py PENDING")
    raise KeyError(f"unknown system {name}")


def run_meta(name: str, model: str, seed: int, personas: list[str], windows: tuple[str, ...]) -> dict[str, Any]:
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None

    digest = None
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            digest = next((m["digest"] for m in json.loads(r.read())["models"] if m["name"] == model), None)
    except Exception:
        pass
    return {"system": name, "model": model, "model_digest": digest, "seed": seed, "personas": personas,
            "windows": list(windows), "git_commit": sh("git", "rev-parse", "HEAD"),
            "git_dirty": bool(sh("git", "status", "--porcelain", "--", "src")),
            "llm_settings": Chat(model=model, seed=seed).settings(),
            "hardware": sh("sysctl", "-n", "machdep.cpu.brand_string"), "memory_bytes": sh("sysctl", "-n", "hw.memsize"),
            "runtime": "aios workspace (SQLite store, privacy layer, engine)"}
