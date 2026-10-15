"""Every system we evaluate, by name. All share the shell, the data and the model settings."""
from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from p17sim.agents.baseline import BaselineAgent
from p17sim.agents.model_a import MODEL_DIR, ModelA, ModelAConfig, Trained
from p17sim.agents.reference import NullAgent, OracleAgent
from p17sim.system.llm import Chat, PersonaText
from p17sim.system.runner import Dataset

ABLATIONS = {  # name -> (config override, task kinds it targets) as pre-registered (H4)
    "model_a-no_reranker": ({"use_reranker": False}, {"recall", "update"}),
    "model_a-no_recency": ({"use_recency": False}, {"update"}),
    "model_a-no_intent": ({"use_intent_model": False}, {"scheduling", "privacy"}),
    "model_a-no_deletion_memory": ({"deletion_aware": False}, {"deletion"}),
}
SYSTEMS = ["baseline", "model_a", *ABLATIONS, "oracle", "null"]


def factory(name: str, ds: Dataset, model: str, seed: int, model_dir: Path = MODEL_DIR):
    trained = Trained.load(model_dir) if name.startswith("model_a") else None

    def make(shell, persona, extra):
        llm = Chat(model=model, seed=seed)
        llm.persona_text = PersonaText([r["text"] for r in ds.records[persona.persona_id]])
        if name == "baseline":
            return BaselineAgent(shell, persona, llm)
        if name == "oracle":
            return OracleAgent(shell, persona, extra, ds)
        if name == "null":
            return NullAgent()
        overrides = ABLATIONS[name][0] if name in ABLATIONS else {}
        cfg = ModelAConfig.trained(model_dir, name=name, **overrides)
        return ModelA(shell, persona, llm, cfg, trained)
    return make


def run_meta(name: str, model: str, seed: int, personas: list[str], windows: tuple[str, ...],
             model_dir: Path = MODEL_DIR) -> dict[str, Any]:
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
    meta = {"system": name, "model": model, "model_digest": digest, "seed": seed, "personas": personas,
            "windows": list(windows), "git_commit": sh("git", "rev-parse", "HEAD"),
            "git_dirty": bool(sh("git", "status", "--porcelain", "--", "src")),
            "llm_settings": Chat(model=model, seed=seed).settings(), "hardware": sh("sysctl", "-n",
                                                                                     "machdep.cpu.brand_string"),
            "memory_bytes": sh("sysctl", "-n", "hw.memsize")}
    if name.startswith("model_a") and (model_dir / "params.json").exists():
        meta["model_a_params"] = json.loads((model_dir / "params.json").read_text())
        meta["ablation"] = ABLATIONS.get(name, ({}, None))[0]
    return meta
