"""Adapter over the team's connectors.py (repo root, shared with M1).

We reuse it rather than fork it so M1 and M3 hit the same data-access code. The dispatcher in
tools.py adds what the connector lacks (contact-id resolution, day_range, multi-word queries).
"""

from __future__ import annotations

import json
from pathlib import Path

from connectors import LocalBenchmarkConnector  # noqa: F401  (re-exported)


def open_connector(sandbox_persona_dir: Path) -> LocalBenchmarkConnector:
    """Connector bound to a task sandbox (never to DATA_ROOT itself)."""
    return LocalBenchmarkConnector(str(sandbox_persona_dir))


def persist_profile(connector: LocalBenchmarkConnector) -> None:
    """The connector keeps profile edits in memory; write them to the sandbox copy so the
    end-of-task state can be exported for scoring (e.g. eval_02a)."""
    Path(connector.profile_file).write_text(json.dumps(connector.get_profile(), indent=2))
