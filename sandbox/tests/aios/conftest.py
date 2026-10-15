from dataclasses import dataclass, field
from datetime import datetime

import pytest

from p17sim.aios.privacy import DeletionManager, EgressGateway
from p17sim.aios.store import Store
from p17sim.system.llm import ScriptedChat


@dataclass
class FakeWS:
    """The parts of a Workspace that nodes use."""
    store: Store
    pid: str = "q1"
    now: datetime = datetime(2026, 9, 25, 9, 0)
    llm: ScriptedChat = field(default_factory=lambda: ScriptedChat([]))

    def __post_init__(self):
        self.deletion = DeletionManager(self.store)
        self.gateway = EgressGateway(self.store)


def make_rec(i, text, ts="2026-09-10T09:00:00", channel="message", perm="read", pid="q1", end=None, sender="Ana"):
    return {"event_id": f"{pid}-e{i}", "persona_id": pid, "timestamp": ts, "end": end, "channel": channel,
            "event_type": channel, "sender": sender, "text": text, "permission_label": perm, "deleted": False}


@pytest.fixture
def ws():
    s = Store()
    s.put_persona({"persona_id": "q1", "name": "Mia Lopez", "contacts": [
        {"contact_id": "q1-c0", "name": "Ana Silva", "relation": "friend", "email": "q1.c0@example.org"},
        {"contact_id": "q1-c1", "name": "Ben Okafor", "relation": "coworker", "email": "q1.c1@example.org"}]})
    s.add_record(make_rec(1, "your building gate code is 4821"), "personal")
    s.add_record(make_rec(2, "therapy at Sampaio Clinic, chart P81108-V", perm="deny"), "sensitive")
    s.add_record(make_rec(3, "Gym", ts="2026-09-26T18:00:00", channel="calendar", end="2026-09-26T19:00:00"), "personal")
    s.add_record(make_rec(4, "Suggested a morning meeting slot: accepted", ts="2026-09-24T10:00:00",
                          channel="preference"), "personal")
    return FakeWS(s)
