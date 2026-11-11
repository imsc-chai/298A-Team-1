"""Adapter over the dataset's own connectors.py (shared with M1 and the team evaluator).

The module is loaded from DATA_ROOT, the same way Model_A/evaluation does, so every model and
the scorer use the one connector that ships with the data, wherever that folder lives. We reuse
it rather than fork it; the dispatcher in tools.py adds what it lacks (contact-id resolution,
multi-word queries).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from common.config import data_root

_MODULE_NAME = "dataset_connectors"


def load_dataset_connectors() -> ModuleType:
    """Import DATA_ROOT/connectors.py once and cache it in sys.modules."""
    cached = sys.modules.get(_MODULE_NAME)
    path = data_root() / "connectors.py"
    if cached is not None and Path(getattr(cached, "__file__", "")) == path:
        return cached
    if not path.is_file():
        raise FileNotFoundError(f"No connectors.py in DATA_ROOT ({path.parent})")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def open_connector(sandbox_persona_dir: Path) -> Any:
    """LocalBenchmarkConnector bound to a task sandbox (never to DATA_ROOT itself)."""
    return load_dataset_connectors().LocalBenchmarkConnector(str(sandbox_persona_dir))
