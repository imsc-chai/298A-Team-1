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
    ends_next_day: bool = False  # overnight event, e.g. 19:00-07:30 night shift


def is_overnight(start_time: str, end_time: str) -> bool:
    """An end time at or before the start time means the event runs past midnight."""
    return end_time <= start_time


def duration_minutes(start_time: str, end_time: str) -> int:
    sh, sm = map(int, start_time.split(":"))
    eh, em = map(int, end_time.split(":"))
    minutes = (eh * 60 + em) - (sh * 60 + sm)
    return minutes + 24 * 60 if minutes <= 0 else minutes

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
        """reference_time is accepted but ignored: future events are known plans.
        A date search also returns overnight events from the previous day, because they
        still occupy that date's early morning (e.g. a 19:00-07:30 shift)."""
        previous_day = None
        if date:
            previous_day = (datetime.date.fromisoformat(date) - datetime.timedelta(days=1)).isoformat()
        results = []
        for e in self._events:
            overnight = is_overnight(e["start_time"], e["end_time"])
            if date and e["date"] != date and not (overnight and e["date"] == previous_day): continue
            if query and (query.lower() not in e["title"].lower() and query.lower() not in e.get("notes", "").lower()): continue
            results.append(self._to_event(e))
        return results

    def _to_event(self, e):
        return CalendarEvent(
            id=e["id"], date=e["date"], weekday=e["weekday"],
            start_time=e["start_time"], end_time=e["end_time"],
            duration_minutes=e.get("duration_minutes") or duration_minutes(e["start_time"], e["end_time"]),
            title=e["title"], category=e.get("category", "general"),
            location=e.get("location", ""), busy=e.get("busy", True),
            is_sensitive=e.get("is_sensitive", False), notes=e.get("notes", ""),
            attendees=e.get("attendees", []),
            ends_next_day=is_overnight(e["start_time"], e["end_time"]),
        )

    def _next_event_id(self) -> str:
        """Unique id in the persona's own scheme (cal_048, cal_p3_034). len+1 is NOT safe:
        Maya has 47 events but ids up to cal_048, and deletions leave gaps."""
        prefix, width, highest = "cal_", 3, 0
        for e in self._events:
            m = re.match(r"^(.*?_)(\d+)$", e["id"])
            if m and int(m.group(2)) >= highest:
                prefix, width, highest = m.group(1), len(m.group(2)), int(m.group(2))
        existing = {e["id"] for e in self._events} | getattr(self, "_issued_ids", set())
        n = highest + 1
        while f"{prefix}{n:0{width}d}" in existing:
            n += 1
        new_id = f"{prefix}{n:0{width}d}"
        self._issued_ids = existing | {new_id}  # never reuse an id, even after delete
        return new_id

    def create_event(self, title, date, start_time, end_time, busy=True, notes=""):
        """end_time <= start_time is an overnight event ending the next day."""
        if start_time == end_time:
            raise ValueError("start_time and end_time are equal (zero-length event)")
        dur = duration_minutes(start_time, end_time)
        dt = datetime.datetime.strptime(date, "%Y-%m-%d")
        new_event = {
            "id": self._next_event_id(),
            "day": dt.day, "date": date, "weekday": dt.strftime("%A"),
            "start_time": start_time, "end_time": end_time, "duration_minutes": dur,
            "title": title, "category": "personal", "location": "Local",
            "busy": busy, "notes": notes
        }
        self._events.append(new_event)
        return self._to_event(new_event)

    def delete_event(self, event_id: str) -> bool:
        initial_len = len(self._events)
        self._events = [e for e in self._events if e["id"] != event_id]
        return len(self._events) < initial_len

    def search_messages(self, contact=None, query=None, reference_time=None, day_range=None):
        """day_range = [start_day, end_day] inclusive, matching tool_registry.json read_messages."""
        if day_range is not None and (len(day_range) != 2 or day_range[0] > day_range[1]):
            raise ValueError(f"day_range must be [start_day, end_day] with start <= end, got {day_range}")
        with open(self.messages_file, "r") as f: raw_msgs = json.load(f)
        results = []
        for m in raw_msgs:
            if reference_time and m.get("timestamp", "") > reference_time: continue
            if day_range is not None:
                day = m.get("day") or int(m["timestamp"][8:10])
                if not (day_range[0] <= day <= day_range[1]): continue
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
    # Precedence: explicit argument > DATA_ROOT env var > this file's folder (legacy default,
    # kept so existing M1 calls still work when the dataset sits next to connectors.py).
    if data_root is None:
        data_root = os.environ.get("DATA_ROOT") or os.path.dirname(os.path.abspath(__file__))
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
