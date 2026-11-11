"""Read-only benchmark access through the dataset's own connectors.py."""
from __future__ import annotations

import copy
import hashlib
import sys
import types
from dataclasses import asdict
from pathlib import Path

from .core import GuardPlanError, read_json, timestamp, outside_source
from .recipients import address_key


class DatasetAdapter:
    """Expose reads only. Never replay candidate journals or execute their tools."""

    def __init__(self, data_root: Path, persona_id: str, workspace: Path,
                 reference_time: str, session: str = "unused"):
        self.data_root = Path(data_root).resolve()
        if persona_id not in {p.name for p in self.data_root.glob("persona_*") if p.is_dir()}:
            raise ValueError("Unknown persona")
        timestamp(reference_time)
        outside_source(self.data_root, workspace)
        self.persona_id, self.reference_time = persona_id, reference_time
        path = self.data_root / "connectors.py"
        key = "_common_evaluation_connectors_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
        if key not in sys.modules:
            module = types.ModuleType(key)
            module.__file__ = str(path)
            sys.modules[key] = module
            try:
                # Avoid writing __pycache__ into the original dataset.
                exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
            except Exception:
                del sys.modules[key]
                raise
        self.connector = sys.modules[key].get_connector(persona_id, mode="benchmark", data_root=str(self.data_root))

    def get_profile(self):
        return copy.deepcopy(self.connector.get_profile())

    def search_events(self):
        return [asdict(e) for e in self.connector.search_events(reference_time=self.reference_time)]

    def privacy_policy(self):
        return read_json(self.data_root / self.persona_id / "agent_memory/privacy_policy.json")

    def tasks(self):
        return read_json(self.data_root / self.persona_id / "test_suite/eval_tasks.json")

    def registry(self):
        return {t["name"]: t for t in read_json(self.data_root / "tool_registry.json")["tools"]}

    def contact_id(self, destination):
        if not destination:
            return None
        matches = []
        for contact in self.get_profile().get("contacts", []):
            direct = destination.casefold() == str(contact.get("id", "")).casefold()
            email = address_key(destination, "email")
            phone = address_key(destination, "sms")
            if (direct or (email and email == address_key(str(contact.get("email", "")), "email")) or
                    (phone and phone == address_key(str(contact.get("phone", "")), "sms"))):
                matches.append(contact["id"])
        return matches[0] if len(matches) == 1 else None
