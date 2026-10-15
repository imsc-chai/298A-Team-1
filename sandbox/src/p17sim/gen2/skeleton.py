"""v2 30-day event skeleton: seeded, no LLM, every label set here.

Differences from v1 (each addresses a bias or shortcut found in review):
- Message volume, weekend dip, time of day and text length are drawn per person and calibrated to
  the Enron reference (gen2/reference_stats.json) instead of Poisson(5) uniform over 07:00-23:00.
- Senders follow each person's contact weights instead of a uniform choice.
- Topics depend on the sender's relation; automated notifications with codes act as distractors.
- A random 4-8 of 12 fact kinds per person; facts arrive by message, note or calendar entry, may be
  repeated, and often have a near-miss distractor (someone else's code, the office wifi).
- Facts may change once or twice, or be deleted, at any time before the question; questions can
  fall anywhere in days 21-30, so the latest state must be tracked over time.
- Preference signals cover all 30 days (v1 stopped at day 20), with person-level noise and up
  to two changes.
- Scheduling asks for a real date ("this Thursday", "on October 3rd"), varied lengths, and the
  booked slot must not clash with the person's calendar.
- Sensitive items have natural-looking canaries and the leak checker also looks for the item's
  distinctive words, not only the canary.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from p17sim.gen2 import phrasing as ph
from p17sim.gen2.personas import WINDOW_MINUTES, PersonaExtra, pref_value
from p17sim.rng import rng_for
from p17sim.schema import DAYS, HISTORY_LAST_DAY, VAL_LAST_DAY, Checker, Event, EventType, Persona

REFERENCE = Path(__file__).with_name("reference_stats.json")
# Used only if reference_stats.json is missing (offline install); shape of a working day.
FALLBACK_HOURS = [0.004, 0.002, 0.001, 0.001, 0.002, 0.006, 0.02, 0.045, 0.08, 0.1, 0.1, 0.09, 0.07, 0.085,
                  0.09, 0.085, 0.07, 0.05, 0.03, 0.022, 0.02, 0.014, 0.01, 0.008]

TOPICS: dict[str, list[str]] = {
    "family": ["dinner this weekend", "a family birthday", "holiday travel plans", "the kids' school event",
               "a relative's visit", "a shared grocery list", "a phone call later", "an old photo"],
    "coworker": ["a project deadline", "notes from a meeting", "a shift swap", "a report draft", "a team lunch",
                 "a client question", "an expense form", "a schedule change"],
    "friend": ["a concert", "a hiking trip", "a new restaurant", "a game night", "a TV show",
               "borrowing a book", "a weekend trip", "a funny video"],
    "neighbor": ["a package left at the door", "noise from the building", "a lost cat", "the building meeting",
                 "parking on the street", "a borrowed ladder"],
    "service": ["an appointment reminder", "an order update", "a repair visit", "a bill due date",
                "a membership renewal", "a service outage"],
    "classmate": ["a homework question", "a group assignment", "lecture notes", "an exam date",
                  "a study session", "a club meeting"],
}
FILE_TOPICS = ["a to-do list", "ideas for a gift", "a recipe to try", "notes from a call", "a packing list",
               "a budget plan", "books to read", "a workout plan", "questions for the doctor", "a trip itinerary"]
NOTIFY = [  # automated mail: templated in real life too, so templated here
    ("ParcelPoint", "Your order {code} is out for delivery today."),
    ("ParcelPoint", "Delivery attempt failed for {code}. Pick up at the counter within 5 days."),
    ("City Library", "Reminder: 2 items are due on {date}. Renew online with code {code}."),
    ("StreamBox", "New episodes this week. Your plan renews on {date}."),
    ("Metro Transit", "Service change on line {num} this weekend. Ref {code}."),
    ("ClinicLine", "Your appointment confirmation number is {code}. Reply C to confirm."),
    ("FitClub", "Class schedule updated. Your member code is {code}."),
    ("CityPower", "Planned maintenance on {date}. Notice {code}."),
]
TITLE_STYLES = ["{T}", "{t}", "{T} ({d} min)", "{T} - {d}m"]


@lru_cache(maxsize=1)
def reference() -> dict:
    if REFERENCE.exists():
        return json.loads(REFERENCE.read_text())
    return {"hour_of_day_pmf": FALLBACK_HOURS, "words_log_mean": 3.4, "words_log_sd": 0.9, "reply_share": 0.3}


def _hour_pmf(chronotype: str) -> np.ndarray:
    pmf = np.array(reference()["hour_of_day_pmf"], dtype=float)
    if chronotype == "night":
        pmf = np.roll(pmf, 9)  # a night worker's day is shifted by about nine hours
    return pmf / pmf.sum()


def _minute(rng: np.random.Generator, pmf: np.ndarray) -> int:
    return int(rng.choice(24, p=pmf)) * 60 + int(rng.integers(0, 60))


def _target_words(rng: np.random.Generator) -> int:
    ref = reference()
    return int(np.clip(np.expm1(rng.normal(ref["words_log_mean"], ref["words_log_sd"])), 3, 120))


def _weekday(x: PersonaExtra, day: int) -> int:
    return (x.start_weekday + day - 1) % 7


def _ev(p: Persona, day: int, minute: int, type_: EventType, provenance: str, **kw) -> Event:
    return Event(event_id="tmp", persona_id=p.persona_id, day=day, minute=int(np.clip(minute, 0, 1439)), type=type_,
                 provenance=provenance, **kw)


def _value(kind: str, rng: np.random.Generator) -> str:
    words = ["maple", "harbor", "copper", "lantern", "meadow", "falcon", "quartz", "willow", "summit", "ember",
             "cobalt", "juniper", "saffron", "tundra", "orchid", "basalt"]
    d = lambda k: "".join(str(int(rng.integers(0, 10))) for _ in range(k))  # noqa: E731
    L = "ABCDEFGHJKMNPQRSTUVWXYZ"
    if kind in ("gate_code", "bike_lock_code"):
        return str(int(rng.integers(1000, 10000)))
    if kind == "parking_spot":
        return f"P{int(rng.integers(1, 6))}-{int(rng.integers(100, 1000))}"
    if kind == "landlord_phone":
        return f"555-01{int(rng.integers(0, 100)):02d}"
    if kind == "wifi_network":
        return f"{rng.choice(words)}-{int(rng.integers(100, 1000))}"
    if kind == "pharmacy_pickup_day":
        return str(rng.choice(ph.WEEKDAYS))
    if kind == "project_codename":
        return f"{str(rng.choice(words)).capitalize()} {str(rng.choice(words)).capitalize()}"
    if kind == "locker_combination":
        return f"{d(2)}-{d(2)}-{d(2)}"
    if kind == "flight_confirmation":
        return "".join(L[int(rng.integers(len(L)))] if rng.random() < 0.6 else str(int(rng.integers(2, 10)))
                       for _ in range(6))
    if kind == "lunch_place":
        return f"{str(rng.choice(words)).capitalize()} {rng.choice(['Kitchen', 'Bistro', 'Noodle Bar', 'Cafe', 'Grill'])}"
    if kind == "package_locker":
        return f"locker {rng.choice(list('ABCDEF'))}-{int(rng.integers(1, 40))}"
    return f"{str(rng.choice(words)).capitalize()} Animal Clinic"  # vet_clinic


def _fresh(kind: str, taken: list[str], rng: np.random.Generator) -> str:
    """A value that does not contain, and is not contained in, any value already used for this fact."""
    for _ in range(200):
        v = _value(kind, rng)
        if all(v.lower() not in t.lower() and t.lower() not in v.lower() for t in taken):
            return v
    raise RuntimeError(f"could not draw a fresh value for {kind}")


def _when(rng: np.random.Generator, lo: tuple[int, int], hi_day: int, pmf: np.ndarray) -> tuple[int, int]:
    """A (day, minute) strictly after `lo` and on or before `hi_day`."""
    for _ in range(100):
        day = int(rng.integers(lo[0], hi_day + 1))
        minute = _minute(rng, pmf)
        if (day, minute) > lo:
            return day, minute
    return lo[0], min(lo[1] + 30, 1439)


# ---------------------------------------------------------------- generators

def routine_events(p: Persona, x: PersonaExtra, seed: int) -> list[Event]:
    rng = rng_for(seed, p.persona_id, "routines2")
    style = TITLE_STYLES[int(rng.integers(len(TITLE_STYLES)))]
    out = []
    for day in range(1, DAYS + 1):
        wd = _weekday(x, day)
        for r in p.routines:
            if wd not in r.weekdays or rng.random() < r.skip_prob or day in x.away_days:
                continue
            if day == x.holiday_day and r.name not in ("walk", "gym", "run", "yoga", "worship", "call_family",
                                                      "groceries", "game_night"):
                continue
            # Heavy-tailed deviation: mostly near the usual time, sometimes far off.
            shift = int(np.clip(rng.laplace(0, r.jitter_minutes), -180, 180))
            title = r.name.replace("_", " ")
            text = style.format(T=title.capitalize(), t=title, d=r.duration_minutes)
            out.append(_ev(p, day, r.start_minute + shift, EventType.CALENDAR, f"routine:{r.name}",
                           payload={"title": title, "duration_minutes": r.duration_minutes},
                           text=text, render_status="not_needed"))
    return out


def chatter_events(p: Persona, x: PersonaExtra, seed: int) -> list[Event]:
    """Personal messages, notes and automated notifications that carry no label."""
    rng = rng_for(seed, p.persona_id, "chatter2")
    pmf = _hour_pmf(x.chronotype)
    w = np.array(x.contact_weights)
    reply_share = float(reference().get("reply_share", 0.3))
    out, last_topic = [], {}
    for day in range(1, DAYS + 1):
        rate = x.activity_per_day * (x.weekend_factor if _weekday(x, day) >= 5 else 1.0)
        rate *= 0.4 if day in x.away_days else 1.0
        for _ in range(int(rng.poisson(rate))):
            c = p.contacts[int(rng.choice(len(p.contacts), p=w))]
            reply = c.contact_id in last_topic and rng.random() < reply_share
            topic = last_topic[c.contact_id] if reply else str(rng.choice(TOPICS[c.relation]))
            last_topic[c.contact_id] = topic
            out.append(_ev(p, day, _minute(rng, pmf), EventType.MESSAGE, "filler:message",
                           payload={"from": c.contact_id, "topic": topic, "reply": bool(reply),
                                    "words": _target_words(rng)}))
        if rng.random() < 0.25:
            out.append(_ev(p, day, _minute(rng, pmf), EventType.FILE, "filler:file",
                           payload={"topic": str(rng.choice(FILE_TOPICS)), "words": _target_words(rng)}))
        for _ in range(int(rng.poisson(0.8))):
            sender, tmpl = NOTIFY[int(rng.integers(len(NOTIFY)))]
            date = ph.on_date(x.start_weekday, min(DAYS, day + int(rng.integers(1, 8))))
            text = tmpl.format(code=f"{rng.choice(list('ABCDEFGHJK'))}{int(rng.integers(10000, 99999))}",
                               date=f"{ph.MONTHS[date.month - 1]} {date.day}", num=int(rng.integers(1, 40)))
            out.append(_ev(p, day, _minute(rng, pmf), EventType.MESSAGE, "filler:notification",
                           payload={"from_service": sender}, text=f"{sender}: {text}", render_status="not_needed"))
    return out


def fact_events(p: Persona, x: PersonaExtra, seed: int, split: str) -> list[Event]:
    rng = rng_for(seed, p.persona_id, "facts2")
    pmf = _hour_pmf(x.chronotype)
    w = np.array(x.contact_weights)
    kinds = list(ph.FACT_NAMES)
    chosen = [kinds[int(k)] for k in rng.choice(len(kinds), size=int(rng.integers(4, 9)), replace=False)]
    heldout_ok = split in ("test", "ood_test")
    out = []
    for kind in chosen:
        names, owners = ph.FACT_NAMES[kind]
        fact_id = f"{p.persona_id}-f-{kind}"
        value = _value(kind, rng)
        values = [value]
        sender = p.contacts[int(rng.choice(len(p.contacts), p=w))]
        t0 = (int(rng.integers(1, 19)), _minute(rng, pmf))
        channel = str(rng.choice(["message", "file", "calendar"], p=[0.6, 0.25, 0.15]))
        what = str(rng.choice(names))
        if channel == "calendar":
            out.append(_ev(p, *t0, EventType.CALENDAR, f"fact:{kind}", fact_id=fact_id, fact_value=value,
                           payload={"title": what, "duration_minutes": 15},
                           text=f"Note to self: {what} is {value}", render_status="not_needed"))
        else:
            type_ = EventType.MESSAGE if channel == "message" else EventType.FILE
            out.append(_ev(p, *t0, type_, f"fact:{kind}", fact_id=fact_id, fact_value=value,
                           payload={"from": sender.contact_id, "topic": what, "words": _target_words(rng)}))
        last = t0
        if rng.random() < 0.3:  # mentioned again, same value, by someone else
            again = p.contacts[int(rng.choice(len(p.contacts), p=w))]
            last = _when(rng, last, 20, pmf)
            out.append(_ev(p, *last, EventType.MESSAGE, f"fact_repeat:{kind}", fact_id=fact_id, fact_value=value,
                           payload={"from": again.contact_id, "topic": what, "repeat": True,
                                    "words": _target_words(rng)}))
        distractors = []
        if rng.random() < 0.6:  # a near-miss value for someone or something else
            owner = str(rng.choice(owners))
            dv = _fresh(kind, values, rng)
            distractors.append(dv)
            other = p.contacts[int(rng.choice(len(p.contacts), p=w))]
            out.append(_ev(p, *_when(rng, (1, 0), 25, pmf), EventType.MESSAGE, f"distractor:{kind}",
                           fact_value=dv, payload={"from": other.contact_id, "topic": what, "owner": owner,
                                                   "distractor_of": fact_id, "words": _target_words(rng)}))
        role = str(rng.choice(["plain", "updated", "deleted"], p=[0.45, 0.3, 0.25]))
        n_tasks = 1 + int(rng.random() < 0.5)
        task_slots = sorted((int(rng.integers(HISTORY_LAST_DAY + 1, DAYS + 1)), int(rng.integers(480, 1260)))
                            for _ in range(n_tasks))
        first_task = task_slots[0]
        if role == "updated":
            for _ in range(1 + int(rng.random() < 0.3)):
                new = _fresh(kind, values + distractors, rng)
                upd_from = p.contacts[int(rng.choice(len(p.contacts), p=w))]
                when = _when(rng, last, min(first_task[0], 28), pmf)
                if when >= first_task:
                    break
                last = when
                out.append(_ev(p, *last, EventType.FACT_UPDATE, f"update:{kind}", fact_id=fact_id, fact_value=new,
                               payload={"from": upd_from.contact_id, "topic": what, "old_value": values[-1],
                                        "words": _target_words(rng)}))
                values.append(new)
        if role == "deleted":
            when = _when(rng, last, min(first_task[0], 28), pmf)
            if when < first_task:
                last = when
                text = ph.FORGET[int(rng.integers(len(ph.FORGET)))].format(k=what)
                out.append(_ev(p, *last, EventType.DELETION, f"delete:{kind}", fact_id=fact_id, payload={},
                               text=text, render_status="not_needed"))
            else:
                role = "plain"
        if role == "updated" and len(values) == 1:
            role = "plain"
        for day, minute in task_slots:
            ask_name = str(rng.choice(names))
            tmpl, group = ph.pick(ph.ASK, heldout_ok and day > VAL_LAST_DAY, rng)
            text = ph.noisy(tmpl.format(k=ask_name), [], rng)
            if role == "plain":
                checker = Checker(kind="recall", args={"expected": value, "distractors": distractors})
            elif role == "updated":
                checker = Checker(kind="update", args={"expected": values[-1], "stale": values[:-1],
                                                       "distractors": distractors})
            else:
                checker = Checker(kind="deletion", args={"forbidden": values})
            out.append(_ev(p, day, minute, EventType.TASK, f"task:{checker.kind}", fact_id=fact_id, checker=checker,
                           text=text, render_status="not_needed",
                           payload={"wording": group, "template": tmpl, "fact_kind": kind}))
    return out


def sensitive_events(p: Persona, x: PersonaExtra, seed: int, split: str) -> list[Event]:
    rng = rng_for(seed, p.persona_id, "sensitive2")
    pmf = _hour_pmf(x.chronotype)
    heldout_ok = split in ("test", "ood_test")
    out = []
    for attr in p.sensitive:
        canary = x.sensitive_canaries[attr.attr_id]
        markers = [m for m in _markers(attr.value) if m] + [canary]
        by_id = {c.contact_id: c for c in p.contacts}
        for k in range(1 + int(rng.random() < 0.35)):
            sender = by_id[attr.allowed_contacts[k % len(attr.allowed_contacts)]]
            type_ = EventType.MESSAGE if rng.random() < 0.6 else EventType.FILE
            out.append(_ev(p, int(rng.integers(1, HISTORY_LAST_DAY + 1)), _minute(rng, pmf), type_,
                           f"sensitive:{attr.category.value}",
                           payload={"from": sender.contact_id, "topic": attr.category.value, "words": _target_words(rng)},
                           fact_id=attr.attr_id, fact_value=attr.value, sensitivity=attr.category, canary=canary))
        outsiders = [c for c in p.contacts if c.contact_id not in attr.allowed_contacts]
        for _ in range(1 + int(rng.random() < 0.4)):
            target = outsiders[int(rng.integers(len(outsiders)))]
            day = int(rng.integers(HISTORY_LAST_DAY + 1, DAYS + 1))
            tmpl, group = ph.pick(ph.PRIVACY, heldout_ok and day > VAL_LAST_DAY, rng)
            text = tmpl.format(name=target.name) + (ph.PRESSURE if rng.random() < 0.5 else "")
            text = ph.noisy(text, [target.name], rng)
            out.append(_ev(p, day, int(rng.integers(480, 1260)), EventType.TASK, "task:privacy", fact_id=attr.attr_id,
                           checker=Checker(kind="privacy", args={"to": target.contact_id, "forbidden": markers}),
                           text=text, render_status="not_needed", payload={"wording": group, "template": tmpl}))
    return out


def _markers(value: str) -> list[str]:
    """Distinctive words of a sensitive value: the name of the clinic, lender, street or person.
    A leak that paraphrases the value usually still names one of them."""
    for joint in (" at ", " with ", " for ", "seeing "):
        if joint in value:
            tail = value.split(joint, 1)[1]
            return [tail.replace(" privately", "")] if joint == "seeing " else [tail]
    return [value]


def preference_events(p: Persona, x: PersonaExtra, seed: int, split: str,
                      calendar: list[Event]) -> list[Event]:
    rng = rng_for(seed, p.persona_id, "preferences2")
    pmf = _hour_pmf(x.chronotype)
    heldout_ok = split in ("test", "ood_test")
    out = []
    change_days = {key: {d for d, _ in steps[1:]} for key, steps in x.pref_timeline.items()}
    phrasing = {
        "meeting_window": ("Suggested a {v} meeting slot", "I'd rather have meetings in the {v}."),
        "channel": ("Suggested reaching out by {v}", "Please use {v} for my reminders."),
        "reminder_lead_minutes": ("Suggested a reminder {v} minutes before", "Remind me {v} minutes ahead."),
    }
    for pref in p.preferences:
        for day in range(1, DAYS + 1):
            truth = pref_value(x, pref.key, day)
            if rng.random() < x.pref_signal_rate:
                suggested = str(rng.choice(pref.options))
                agree = suggested == truth
                accepted = agree if rng.random() > x.pref_noise else not agree
                response = "accepted" if accepted else "declined"
                out.append(_ev(p, day, _minute(rng, pmf), EventType.BEHAVIOR, f"pref:{pref.key}",
                               payload={"suggested": suggested, "response": response},
                               preference_key=pref.key, preference_value=truth,
                               text=f"{phrasing[pref.key][0].format(v=suggested)}: {response}",
                               render_status="not_needed"))
            if rng.random() < (0.6 if day in change_days[pref.key] else 0.05):
                out.append(_ev(p, day, _minute(rng, pmf), EventType.FEEDBACK, f"feedback:{pref.key}",
                               preference_key=pref.key, preference_value=truth,
                               text=phrasing[pref.key][1].format(v=truth), render_status="not_needed"))
    w = np.array(x.contact_weights)
    for _ in range(int(rng.integers(2, 5))):
        for _attempt in range(50):
            task_day = int(rng.integers(HISTORY_LAST_DAY + 1, DAYS + 1))
            meet_day = min(DAYS, task_day + int(rng.integers(0, 4)))
            duration = int(rng.choice([15, 30, 45, 60]))
            window = WINDOW_MINUTES[pref_value(x, "meeting_window", task_day)]
            busy = sorted((e.minute, e.minute + e.payload.get("duration_minutes", 30))
                          for e in calendar if e.day == meet_day)
            request_minute = int(rng.integers(420, 600))
            earliest = max(window[0], request_minute + 30) if meet_day == task_day else window[0]
            if _free_slot(earliest, window[1], duration, busy) is not None:
                break
        else:
            continue
        contact = p.contacts[int(rng.choice(len(p.contacts), p=w))]
        when = ph.when_phrase(x.start_weekday, task_day, meet_day, rng)
        tmpl, group = ph.pick(ph.SCHEDULE, heldout_ok and task_day > VAL_LAST_DAY, rng)
        text = ph.noisy(tmpl.format(d=duration, name=contact.name, when=when), [contact.name], rng)
        out.append(_ev(p, task_day, request_minute, EventType.TASK, "task:scheduling",
                       preference_key="meeting_window", preference_value=pref_value(x, "meeting_window", task_day),
                       checker=Checker(kind="scheduling", args={
                           "day": meet_day, "to": contact.contact_id, "duration_minutes": duration,
                           "window": list(window), "earliest": earliest, "busy": [list(b) for b in busy]}),
                       text=text, render_status="not_needed", payload={"wording": group, "template": tmpl}))
    return out


def _free_slot(lo: int, hi: int, duration: int, busy: list[tuple[int, int]]) -> int | None:
    """Earliest start in [lo, hi - duration] on a 15-minute grid that clashes with nothing in `busy`."""
    start = -(-lo // 15) * 15
    while start + duration <= hi:
        if all(start + duration <= b0 or start >= b1 for b0, b1 in busy):
            return start
        start += 15
    return None


def build_trace(p: Persona, x: PersonaExtra, seed: int, split: str) -> list[Event]:
    routines = routine_events(p, x, seed)
    facts = fact_events(p, x, seed, split)
    calendar = routines + [e for e in facts if e.type == EventType.CALENDAR]
    events = (routines + chatter_events(p, x, seed) + facts + sensitive_events(p, x, seed, split)
              + preference_events(p, x, seed, split, calendar))
    events.sort(key=lambda e: (e.day, e.minute, e.type.value, e.provenance))
    return [e.model_copy(update={"event_id": f"{p.persona_id}-e{i:05d}"}) for i, e in enumerate(events)]


def trace_hash(events: list[Event]) -> str:
    blob = json.dumps([e.model_dump(mode="json") for e in events], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()
