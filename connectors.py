from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os, json, datetime, re

@dataclass
class CalendarEvent:
    id: str
    date: str
    weekday: str
    start_time: str
    end_time: str
    duration_minutes: int
    title: str
    category: str
    location: str
    busy: bool = True
    is_sensitive: bool = False
    notes: str = ""
    attendees: List[str] = field(default_factory=list)

@dataclass
class Message:
    id: str
    timestamp: str
    channel: str
    sender: str
    recipient: str
    subject: str = ""
    body: str = ""

@dataclass
class Contact:
    id: str
    name: str
    email: str
    role: str
    relation: str
    preferred_channel: str
    phone: Optional[str] = None

@dataclass
class DraftAction:
    draft_id: str
    tool: str
    status: str = "DRAFT"
    recipient: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

class BaseCalendarConnector(ABC):
    @abstractmethod
    def search_events(self, date=None, query=None, reference_time=None): pass
    @abstractmethod
    def create_event(self, title, date, start_time, end_time, busy=True, notes=""): pass
    @abstractmethod
    def delete_event(self, event_id): pass

class BaseMailConnector(ABC):
    @abstractmethod
    def search_messages(self, contact=None, query=None, reference_time=None): pass
    @abstractmethod
    def create_draft(self, to, subject, body, channel="email"): pass

class BaseFilesConnector(ABC):
    @abstractmethod
    def read_file(self, relative_path): pass
    @abstractmethod
    def list_files(self): pass

class BaseProfileConnector(ABC):
    @abstractmethod
    def get_profile(self): pass
    @abstractmethod
    def lookup_contact(self, name_or_query): pass
    @abstractmethod
    def update_profile(self, field_path, value): pass

class LocalBenchmarkConnector(BaseCalendarConnector, BaseMailConnector, BaseFilesConnector, BaseProfileConnector):
    def __init__(self, persona_dir: str):
        self.persona_dir = os.path.abspath(persona_dir)
        self.persona_id = os.path.basename(self.persona_dir)
        self.calendar_file = os.path.join(self.persona_dir, "streams", "calendar.json")
        self.messages_file = os.path.join(self.persona_dir, "streams", "messages.json")
        self.profile_file = os.path.join(self.persona_dir, "agent_memory", "user_profile.json")
        self.privacy_file = os.path.join(self.persona_dir, "agent_memory", "privacy_policy.json")
        self.local_files_dir = os.path.join(self.persona_dir, "local_files")
        self.reset()

    def reset(self):
        with open(self.calendar_file, "r") as f: self._events = json.load(f)
        with open(self.profile_file, "r") as f: self._profile = json.load(f)
        self._drafts = {}
        self._draft_counter = 1

    def search_events(self, date=None, query=None, reference_time=None):
        results = []
        for e in self._events:
            if date and e["date"] != date: continue
            if query and (query.lower() not in e["title"].lower() and query.lower() not in e.get("notes", "").lower()): continue
            results.append(CalendarEvent(
                id=e["id"], date=e["date"], weekday=e["weekday"],
                start_time=e["start_time"], end_time=e["end_time"],
                duration_minutes=e.get("duration_minutes", 0),
                title=e["title"], category=e.get("category", "general"),
                location=e.get("location", ""), busy=e.get("busy", True),
                is_sensitive=e.get("is_sensitive", False), notes=e.get("notes", ""),
                attendees=e.get("attendees", [])
            ))
        return results

    def create_event(self, title, date, start_time, end_time, busy=True, notes=""):
        sh, sm = map(int, start_time.split(":"))
        eh, em = map(int, end_time.split(":"))
        dur = (eh * 60 + em) - (sh * 60 + sm)
        dt = datetime.datetime.strptime(date, "%Y-%m-%d")
        new_event = {
            "id": f"cal_{len(self._events) + 1:03d}",
            "day": dt.day, "date": date, "weekday": dt.strftime("%A"),
            "start_time": start_time, "end_time": end_time, "duration_minutes": dur,
            "title": title, "category": "personal", "location": "Local",
            "busy": busy, "notes": notes
        }
        self._events.append(new_event)
        return CalendarEvent(
            id=new_event["id"], date=new_event["date"], weekday=new_event["weekday"],
            start_time=new_event["start_time"], end_time=new_event["end_time"],
            duration_minutes=dur, title=new_event["title"], category=new_event["category"],
            location=new_event["location"], busy=busy, notes=notes
        )

    def delete_event(self, event_id: str) -> bool:
        initial_len = len(self._events)
        self._events = [e for e in self._events if e["id"] != event_id]
        return len(self._events) < initial_len

    def search_messages(self, contact=None, query=None, reference_time=None):
        with open(self.messages_file, "r") as f: raw_msgs = json.load(f)
        results = []
        for m in raw_msgs:
            if reference_time and m.get("timestamp", "") > reference_time: continue
            if contact:
                c_low = contact.lower()
                if c_low not in m.get("sender", "").lower() and c_low not in m.get("recipient", "").lower(): continue
            if query:
                if query.lower() not in f"{m.get('subject', '')} {m.get('body', '')}".lower(): continue
            results.append(Message(
                id=m["id"], timestamp=m["timestamp"], channel=m.get("channel", "email"),
                sender=m["sender"], recipient=m["recipient"], subject=m.get("subject", ""),
                body=m.get("body", "")
            ))
        return results

    def create_draft(self, to, subject, body, channel="email"):
        draft_id = f"draft_{self._draft_counter:04d}"
        self._draft_counter += 1
        draft = DraftAction(
            draft_id=draft_id,
            tool="draft_email" if channel == "email" else "draft_sms",
            status="DRAFT", recipient=to,
            parameters={"to": to, "subject": subject, "body": body, "channel": channel}
        )
        self._drafts[draft_id] = draft
        return draft

    def read_file(self, relative_path: str) -> bytes:
        clean_name = os.path.basename(relative_path)
        full_path = os.path.join(self.local_files_dir, clean_name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found in local_files: {relative_path}")
        with open(full_path, "rb") as f: return f.read()

    def list_files(self) -> List[str]:
        return sorted([os.path.join("local_files", f) for f in os.listdir(self.local_files_dir) if not f.startswith(".")])

    def get_profile(self) -> Dict[str, Any]:
        return self._profile

    def lookup_contact(self, name_or_query: str) -> Optional[Contact]:
        query_terms = [t.lower() for t in name_or_query.replace(".", "").split()]
        for c in self._profile.get("contacts", []):
            c_text = f"{c['id']} {c['name']} {c['email']} {c.get('role', '')}".lower().replace(".", "")
            if all(term in c_text for term in query_terms) or (c.get("phone") and name_or_query in c["phone"]):
                return Contact(id=c["id"], name=c["name"], email=c["email"], role=c.get("role", ""), relation=c.get("relation", ""), preferred_channel=c.get("preferred_channel", "email"), phone=c.get("phone"))
        return None

    def update_profile(self, field_path: str, value: Any) -> bool:
        keys = field_path.split(".")
        target = self._profile
        for k in keys[:-1]:
            if k not in target: target[k] = {}
            target = target[k]
        target[keys[-1]] = value
        return True

class GoogleWorkspaceLiveAdapter(BaseCalendarConnector, BaseMailConnector):
    def search_events(self, date=None, query=None, reference_time=None): return []
    def create_event(self, title, date, start_time, end_time, busy=True, notes=""): pass
    def delete_event(self, event_id): return True
    def search_messages(self, contact=None, query=None, reference_time=None): return []
    def create_draft(self, to, subject, body, channel="email"): pass

def get_connector(persona_id_or_path: str, mode: str = "benchmark", data_root: Optional[str] = None):
    if data_root is None:
        data_root = os.path.dirname(os.path.abspath(__file__))
    if mode == "benchmark":
        if os.path.isabs(persona_id_or_path):
            p_dir = persona_id_or_path
        else:
            p_dir = os.path.join(data_root, persona_id_or_path)
        if not os.path.exists(p_dir):
            raise FileNotFoundError(f"Persona directory not found: {p_dir}")
        return LocalBenchmarkConnector(p_dir)
    elif mode == "live":
        return GoogleWorkspaceLiveAdapter()
    else:
        raise ValueError(f"Unknown mode: {mode}")
