"""v2 persona specs: sampled per person, not fixed per archetype.

What changed from v1, and the bias each change removes:
- 16 Faker locales in 6 regions instead of en_US only (names, places).
- Routines drawn from a per-archetype library with person-level start times, weekdays, jitter
  scale and skip rates, instead of one schedule shared by every member of an archetype.
- 4 to 14 contacts with person-level relation mix and Zipf-like contact activity, instead of
  exactly 6 contacts with the same relation mix.
- 1 to 4 sensitive items of 5 categories, with varied natural-looking canaries, instead of exactly
  4 items whose canaries all match REF-[0-9a-f]{8}.
- Person-level message volume, weekend factor, chronotype and preference-signal noise.
- Each trace starts on a sampled weekday instead of always Monday.
Every email still uses a reserved example domain, and phone-like values stay in 555-01xx.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from faker import Faker
from pydantic import BaseModel

from p17sim.rng import derive_seed, rng_for
from p17sim.schema import Contact, Persona, Preference, Routine, SensitiveAttribute, Sensitivity

REGIONS: dict[str, list[str]] = {
    "north_america": ["en_US"],
    "latin_america": ["es_MX", "pt_BR"],
    "europe": ["en_GB", "fr_FR", "de_DE", "it_IT", "pl_PL", "nl_NL", "tr_TR"],
    "south_asia": ["en_IN"],
    "southeast_asia": ["vi_VN", "id_ID"],
    "africa": ["yo_NG", "zu_ZA", "sw"],
}
ALL_LOCALES = [loc for locs in REGIONS.values() for loc in locs]

# Routine library: name -> (probability a member has it, weekday patterns to choose from,
# start-minute mean, start-minute spread across people, duration mean, duration spread).
_R = tuple[float, list[list[int]], int, int, int, int]
LIBRARY: dict[str, dict[str, _R]] = {
    "grad_student": {
        "class": (0.95, [[0, 2], [1, 3], [0, 2, 4]], 600, 90, 80, 10),
        "lab": (0.8, [[1, 3], [0, 1, 3], [1, 2, 3, 4]], 780, 75, 180, 45),
        "office_hours": (0.4, [[2], [4]], 900, 60, 60, 0),
        "study_group": (0.5, [[3], [6], [1]], 1140, 60, 90, 30),
    },
    "shift_nurse": {
        "day_shift": (0.6, [[0, 1, 4], [2, 3, 6], [0, 4, 5]], 420, 30, 720, 0),
        "night_shift": (0.5, [[3, 4], [5, 6], [1, 2]], 1140, 30, 720, 0),
        "charting_catchup": (0.4, [[2], [5]], 600, 90, 60, 15),
    },
    "parent": {
        "school_dropoff": (0.9, [[0, 1, 2, 3, 4]], 465, 20, 30, 10),
        "school_pickup": (0.7, [[0, 1, 2, 3, 4], [0, 2, 4]], 915, 20, 30, 10),
        "kids_practice": (0.6, [[1, 3], [2], [5]], 1020, 45, 90, 15),
        "work_block": (0.7, [[0, 1, 2, 3, 4]], 570, 40, 400, 60),
    },
    "freelancer": {
        "client_call": (0.85, [[0, 3], [1], [2, 4]], 660, 120, 45, 15),
        "deep_work": (0.8, [[0, 1, 2, 3, 4], [0, 1, 2, 3]], 540, 120, 180, 60),
        "cowork": (0.4, [[2], [1, 3]], 780, 60, 240, 60),
        "invoicing": (0.5, [[4], [0]], 960, 60, 60, 0),
    },
    "retiree": {
        "walk": (0.8, [[0, 1, 2, 3, 4, 5, 6], [0, 2, 4, 6]], 450, 60, 45, 15),
        "book_club": (0.4, [[2], [3]], 840, 60, 90, 0),
        "volunteer": (0.5, [[4], [1, 4], [5]], 600, 60, 180, 60),
        "doctor_checkup": (0.3, [[1]], 660, 90, 45, 0),
    },
    "eng_manager": {
        "standup": (0.9, [[0, 1, 2, 3, 4]], 570, 30, 15, 0),
        "one_on_one": (0.85, [[1, 3], [0, 2, 4]], 840, 90, 30, 0),
        "planning": (0.7, [[0], [1]], 780, 60, 60, 15),
        "hiring_loop": (0.4, [[2], [3]], 660, 90, 60, 0),
    },
    "retail_worker": {
        "opening_shift": (0.6, [[0, 1, 5], [2, 3, 6], [4, 5, 6]], 480, 30, 480, 60),
        "closing_shift": (0.6, [[3, 4], [5, 6], [0, 6]], 840, 45, 480, 60),
        "inventory": (0.3, [[1]], 360, 30, 180, 0),
    },
    "teacher": {
        "teaching": (0.95, [[0, 1, 2, 3, 4]], 480, 20, 420, 30),
        "grading": (0.7, [[0, 2], [1, 3], [6]], 1170, 60, 90, 30),
        "staff_meeting": (0.6, [[2], [3]], 930, 30, 60, 0),
        "tutoring": (0.3, [[1, 3], [5]], 1020, 60, 60, 0),
    },
    "caregiver": {
        "care_visit": (0.95, [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]], 540, 90, 240, 60),
        "pharmacy_run": (0.6, [[2], [4], [1]], 1020, 60, 30, 0),
        "respite_break": (0.4, [[5], [6]], 780, 120, 180, 60),
    },
    "line_cook": {
        "dinner_service": (0.95, [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [0, 3, 4, 5, 6]], 960, 30, 360, 60),
        "prep_shift": (0.6, [[4, 5], [5, 6]], 600, 45, 180, 30),
        "supplier_pickup": (0.3, [[1], [2]], 540, 30, 45, 0),
    },
    # Out-of-distribution archetypes: test only.
    "small_business_owner": {
        "open_shop": (0.95, [[1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], 480, 30, 540, 60),
        "supplier_call": (0.8, [[0], [2]], 600, 60, 30, 0),
        "bookkeeping": (0.7, [[6], [0]], 1140, 60, 90, 30),
    },
    "rideshare_driver": {
        "evening_shift": (0.8, [[3, 4, 5], [4, 5, 6]], 1020, 60, 300, 60),
        "morning_shift": (0.7, [[0, 1], [0, 1, 2]], 360, 30, 240, 60),
        "car_service": (0.5, [[6], [2]], 600, 60, 60, 0),
    },
    "night_security_guard": {
        "night_patrol": (0.95, [[0, 1, 2, 3], [3, 4, 5, 6], [0, 2, 4, 6]], 1320, 30, 480, 0),
        "day_sleep": (0.6, [[0, 1, 2, 3, 4]], 540, 60, 360, 60),
        "training_course": (0.3, [[5]], 900, 60, 120, 0),
    },
}
PERSONAL: dict[str, _R] = {
    "gym": (0.4, [[0, 2, 4], [1, 3], [5, 6]], 1080, 150, 60, 15),
    "run": (0.25, [[1, 3, 5], [0, 6]], 400, 60, 40, 10),
    "yoga": (0.2, [[2], [6]], 1110, 60, 60, 0),
    "worship": (0.25, [[4], [5], [6]], 600, 120, 90, 30),
    "language_class": (0.15, [[1], [3]], 1140, 30, 90, 0),
    "call_family": (0.35, [[6], [0], [3]], 1200, 60, 45, 15),
    "groceries": (0.5, [[5], [6], [2]], 660, 180, 45, 15),
    "game_night": (0.15, [[4], [5]], 1230, 30, 150, 30),
}
IN_DIST = ["grad_student", "shift_nurse", "parent", "freelancer", "retiree", "eng_manager",
           "retail_worker", "teacher", "caregiver", "line_cook"]
OOD = ["small_business_owner", "rideshare_driver", "night_security_guard"]
NIGHT = {"shift_nurse", "line_cook", "rideshare_driver", "night_security_guard"}

PREFERENCES: list[tuple[str, list[str]]] = [
    ("meeting_window", ["morning", "afternoon", "evening"]),
    ("channel", ["email", "sms"]),
    ("reminder_lead_minutes", ["10", "30", "60"]),
]
WINDOW_MINUTES = {"morning": (480, 720), "afternoon": (720, 1020), "evening": (1020, 1260)}

RELATIONS = ["family", "coworker", "friend", "neighbor", "service", "classmate"]
# Relation mix by archetype: weights over RELATIONS.
RELATION_MIX: dict[str, list[float]] = {
    "grad_student": [2, 2, 3, 1, 1, 4], "retiree": [5, 0.2, 3, 3, 2, 0.1], "parent": [4, 2, 2, 2, 2, 0.5],
    "eng_manager": [2, 6, 2, 1, 1, 0.2], "freelancer": [2, 4, 3, 1, 2, 0.2], "teacher": [3, 4, 2, 1, 1, 0.5],
}
DEFAULT_MIX = [3, 3, 3, 1, 1.5, 0.3]

SENSITIVE_CATEGORIES = [Sensitivity.HEALTH, Sensitivity.FINANCE, Sensitivity.LOCATION, Sensitivity.RELATIONSHIP]


class PersonaExtra(BaseModel):
    """v2 attributes that v1's Persona has no field for. Stored next to the persona, not inside it,
    so v1 records keep their schema."""
    persona_id: str
    locale: str
    region: str
    name_list: str  # which Faker list the name came from (female/male); not a gender identity
    start_weekday: int  # weekday of day 1 (0 = Monday)
    activity_per_day: float  # mean personal messages per weekday
    weekend_factor: float
    chronotype: str  # day | night
    contact_weights: list[float]
    pref_timeline: dict[str, list[tuple[int, str]]]  # key -> [(first day, value), ...], day 1 first
    pref_noise: float  # share of suggestion responses that go against the true preference
    pref_signal_rate: float  # chance of a preference signal on a given day
    away_days: list[int]
    holiday_day: int | None
    renderer: str  # which LLM renders this persona's text
    styles: dict[str, dict[str, str]]  # contact_id -> writing style
    sensitive_canaries: dict[str, str]  # attr_id -> canary string


def _name(fake: Faker, name_list: str) -> str:
    raw = fake.name_female() if name_list == "female" else fake.name_male()
    parts = [p for p in raw.split() if not p.endswith(".") and p not in ("Mr", "Mrs", "Ms", "Dr", "Miss")]
    return " ".join(parts) or raw


def _faker(locale: str, seed: int) -> Faker:
    f = Faker(locale)
    f.seed_instance(seed)
    return f


def _canary(category: str, rng: np.random.Generator) -> str:
    """Unique strings in several realistic shapes, so no single pattern finds them all."""
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    n = lambda k: "".join(str(int(rng.integers(0, 10))) for _ in range(k))  # noqa: E731
    a = lambda k: "".join(letters[int(rng.integers(0, len(letters)))] for _ in range(k))  # noqa: E731
    shapes = {
        "health": [f"chart {a(1)}{n(5)}-{a(1)}", f"patient no. {n(4)}-{n(3)}", f"Rx {a(2)}{n(6)}"],
        "finance": [f"loan file {a(2)}-{n(6)}", f"claim {n(4)}{a(2)}{n(2)}", f"case {a(3)}{n(4)}"],
        "location": [f"unit {n(3)}{a(1)} keybox {n(4)}", f"lockbox {a(1)}{n(4)}", f"door code {n(3)}#{n(2)}"],
        "relationship": [f"booking {a(2)}{n(5)}", f"table ref {a(1)}{n(4)}{a(1)}", f"room {n(3)} hold {a(3)}"],
    }[category]
    return str(shapes[int(rng.integers(len(shapes)))])


def _sensitive_value(category: str, fake: Faker, other: str, rng: np.random.Generator) -> str:
    if category == "health":
        what = rng.choice(["therapy sessions", "fertility appointments", "physical therapy", "a dermatology follow-up",
                           "an addiction support group"])
        return f"{what} at {fake.last_name()} Clinic"
    if category == "finance":
        what = rng.choice(["a debt consolidation loan", "a missed rent payment plan", "a tax installment plan",
                           "a credit card dispute"])
        return f"{what} with {fake.last_name()} Credit"
    if category == "location":
        return f"spare key for {fake.street_name()}"
    return f"seeing {other} privately"


def _style(rng: np.random.Generator, locale: str) -> dict[str, str]:
    if locale == "en_GB":
        variety = "British English"
    elif locale == "en_IN":
        variety = "Indian English"
    elif locale == "en_US":
        variety = str(rng.choice(["American English", "casual texting English"]))
    else:
        variety = str(rng.choice(["a non-native English speaker's English", "American English",
                                  "casual texting English"]))
    return {
        "formality": str(rng.choice(["formal", "neutral", "casual", "casual"])),
        "length": str(rng.choice(["one short line", "two or three sentences", "a chatty paragraph"],
                                 p=[0.35, 0.45, 0.2])),
        "variety": variety,
        "quirk": str(rng.choice(["none", "none", "emoji", "all lowercase", "abbreviations like 'u' and 'thx'",
                                 "a small typo or two"])),
    }


def _routines(lib: dict[str, _R], rng: np.random.Generator, jitter_scale: float, skip_mean: float,
              at_least_one: bool, at_most: int = 99) -> list[Routine]:
    """Each library routine is kept with its own probability. Start time, weekdays, duration, jitter
    and skip rate are drawn per person. If `at_least_one` and none is kept, the likeliest is kept."""
    names = list(lib)
    rng.shuffle(names)
    kept = [n for n in names if rng.random() < lib[n][0]][:at_most]
    if at_least_one and not kept:
        kept = [max(lib, key=lambda n: lib[n][0])]
    out = []
    for name in kept:
        _, patterns, mu, sd, dur, dur_sd = lib[name]
        start = int(np.clip(round((mu + rng.normal(0, sd)) / 15) * 15, 0, 1425))
        out.append(Routine(
            name=name, weekdays=patterns[int(rng.integers(len(patterns)))], start_minute=start,
            jitter_minutes=max(1, int(rng.lognormal(np.log(jitter_scale), 0.4))),
            duration_minutes=max(15, int(round((dur + rng.normal(0, dur_sd)) / 5) * 5)),
            skip_prob=float(np.clip(rng.beta(2, 2 / skip_mean - 2), 0.01, 0.6))))
    return out


def make_persona(seed: int, index: int, archetype: str, renderer: str, region: str,
                 name_list: str) -> tuple[Persona, PersonaExtra]:
    pid = f"q{index:04d}"  # v2 ids start with q so they never collide with v1's p-ids
    rng = rng_for(seed, pid, "persona2")
    locale = REGIONS[region][int(rng.integers(len(REGIONS[region])))]
    fake = _faker(locale, derive_seed(seed, pid, "faker2"))

    n_contacts = int(np.clip(4 + rng.negative_binomial(3, 0.4), 4, 14))
    mix = np.array(RELATION_MIX.get(archetype, DEFAULT_MIX), dtype=float)
    relations = [RELATIONS[int(k)] for k in rng.choice(len(RELATIONS), size=n_contacts, p=mix / mix.sum())]
    contacts, used = [], set()
    for i, rel in enumerate(relations):
        cfake = fake if rng.random() < 0.7 else _faker(ALL_LOCALES[int(rng.integers(len(ALL_LOCALES)))],
                                                       derive_seed(seed, pid, "cfaker", i))
        name = _name(cfake, str(rng.choice(["female", "male"])))
        # Some people share a first name with another contact: realistic ambiguity.
        if name in used:
            name = f"{name} {cfake.last_name()[0]}."
        used.add(name)
        contacts.append(Contact(contact_id=f"{pid}-c{i}", name=name, email=f"{pid}.c{i}@example.org", relation=rel))
    weights = 1.0 / np.arange(1, n_contacts + 1) ** rng.uniform(0.6, 1.3)
    rng.shuffle(weights)

    night = archetype in NIGHT and rng.random() < 0.7
    routines = _routines(LIBRARY[archetype], rng, jitter_scale=float(rng.uniform(4, 25)),
                         skip_mean=float(rng.uniform(0.05, 0.2)), at_least_one=True)
    routines += _routines(PERSONAL, rng, jitter_scale=float(rng.uniform(8, 40)),
                          skip_mean=float(rng.uniform(0.1, 0.35)), at_least_one=False, at_most=2)

    preferences, timeline = [], {}
    for key, options in PREFERENCES:
        value = str(rng.choice(options))
        steps = [(1, value)]
        n_changes = int(rng.choice([0, 1, 2], p=[0.4, 0.45, 0.15]))
        for day in sorted(int(d) for d in rng.choice(np.arange(6, 29), size=n_changes, replace=False)):
            value = str(rng.choice([o for o in options if o != value]))
            steps.append((day, value))
        timeline[key] = steps
        # v1's Preference holds one change; the full timeline lives in PersonaExtra.pref_timeline.
        first = steps[1] if len(steps) > 1 else (None, None)
        preferences.append(Preference(key=key, options=options, initial=steps[0][1],
                                      drift_day=first[0], drift_to=first[1]))

    sensitive, canaries = [], {}
    n_sens = int(rng.integers(1, 5))
    cats = rng.choice(len(SENSITIVE_CATEGORIES), size=n_sens, replace=False)
    for j, k in enumerate(sorted(int(c) for c in cats)):
        cat = SENSITIVE_CATEGORIES[k]
        n_allowed = int(rng.integers(1, min(3, n_contacts - 1) + 1))
        allowed = sorted(contacts[int(x)].contact_id for x in rng.choice(n_contacts, size=n_allowed, replace=False))
        value = _sensitive_value(cat.value, fake, _name(fake, str(rng.choice(["female", "male"]))).split()[0], rng)
        attr_id = f"{pid}-s{j}"
        sensitive.append(SensitiveAttribute(attr_id=attr_id, category=cat, value=value, allowed_contacts=allowed))
        canaries[attr_id] = _canary(cat.value, rng)

    persona = Persona(persona_id=pid, archetype=archetype, name=_name(fake, name_list), email=f"{pid}@example.com",
                      contacts=contacts, routines=routines, preferences=preferences, sensitive=sensitive)
    away = sorted({int(d) for d in rng.choice(np.arange(1, 31), size=int(rng.integers(0, 3)), replace=False)})
    extra = PersonaExtra(
        persona_id=pid, locale=locale, region=region, name_list=name_list,
        start_weekday=int(rng.integers(0, 7)),
        activity_per_day=float(np.clip(rng.lognormal(np.log(4.0), 0.55), 1.0, 20.0)),
        weekend_factor=float(rng.uniform(0.85, 1.15) if archetype in ("retiree", "caregiver")
                             else rng.uniform(0.3, 0.9)),
        chronotype="night" if night else "day",
        contact_weights=[float(w) for w in weights / weights.sum()],
        pref_timeline=timeline,
        pref_noise=float(rng.uniform(0.05, 0.25)),
        pref_signal_rate=float(rng.uniform(0.35, 0.95)),
        away_days=away,
        holiday_day=int(rng.integers(1, 31)) if rng.random() < 0.5 else None,
        renderer=renderer,
        styles={c.contact_id: _style(rng, locale) for c in contacts},
        sensitive_canaries=canaries,
    )
    return persona, extra


def pref_value(extra: PersonaExtra, key: str, day: int) -> str:
    """The true preference on a given day, following every change in the timeline."""
    value = extra.pref_timeline[key][0][1]
    for start, v in extra.pref_timeline[key]:
        if start <= day:
            value = v
    return value


def cohort_plan(n_train: int, n_val: int, n_test: int, n_ood: int, seed: int) -> list[dict[str, Any]]:
    """Persona index -> archetype, region, name list, split and renderer. Archetypes, regions and
    name lists are each cycled inside every split, so no split leans on one group. The second
    renderer (phi3) is held out: it renders only half of the test and OOD personas."""
    plan: list[dict[str, Any]] = []
    rng = rng_for(seed, "cohort2")
    for split, n, archs in (("train", n_train, IN_DIST), ("val", n_val, IN_DIST),
                            ("test", n_test, IN_DIST), ("ood_test", n_ood, OOD)):
        order, regions = list(archs), list(REGIONS)
        rng.shuffle(order)
        rng.shuffle(regions)
        for k in range(n):
            renderer = "ollama:phi3:mini" if split in ("test", "ood_test") and k % 2 == 1 else "ollama:llama3.2"
            plan.append({"index": len(plan), "split": split, "archetype": order[k % len(order)],
                         "region": regions[k % len(regions)], "name_list": ("female", "male")[(k // 2) % 2],
                         "renderer": renderer})
    return plan


def make_cohort(seed: int, n_train: int = 12, n_val: int = 6, n_test: int = 12,
                n_ood: int = 6) -> tuple[list[Persona], list[PersonaExtra], dict[str, str]]:
    personas, extras, splits = [], [], {}
    for row in cohort_plan(n_train, n_val, n_test, n_ood, seed):
        p, x = make_persona(seed, row["index"], row["archetype"], row["renderer"], row["region"], row["name_list"])
        personas.append(p)
        extras.append(x)
        splits[p.persona_id] = row["split"]
    return personas, extras, splits
