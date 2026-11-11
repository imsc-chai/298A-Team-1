"""Locations of shared inputs.

DATA_ROOT must come from the environment (CLAUDE.md: never hard-code paths) so the same code
runs on a laptop and on a Colab clone without edits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """Return DATA_ROOT, failing loudly if unset so no run silently reads the wrong dataset."""
    value = os.environ.get("DATA_ROOT")
    if not value:
        raise RuntimeError("DATA_ROOT is not set. Point it at the dataset root (folder holding tool_registry.json).")
    root = Path(value).expanduser().resolve()
    if not (root / "tool_registry.json").is_file():
        raise RuntimeError(f"DATA_ROOT={root} does not contain tool_registry.json")
    return root


def cache_dir() -> Path:
    """Cache for parsed files and embeddings. Derived artifacts only; never ground truth."""
    path = Path(os.environ.get("MODEL_A_CACHE", REPO_ROOT / ".cache")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def persona_dir(persona_id: str) -> Path:
    path = data_root() / persona_id
    if not path.is_dir():
        raise FileNotFoundError(f"Persona not found: {path}")
    return path


def load_tool_registry() -> dict[str, Any]:
    return json.loads((data_root() / "tool_registry.json").read_text())


def load_splits() -> dict[str, Any]:
    return json.loads((data_root() / "splits.json").read_text())
