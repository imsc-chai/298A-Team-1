"""Persona specs. Names come from Faker; every email uses a reserved example domain."""
from __future__ import annotations

from faker import Faker

from p17sim.rng import derive_seed, rng_for
from p17sim.schema import Contact, Persona, Preference, Routine, SensitiveAttribute, Sensitivity

# (routine name, weekdays, start minute, duration minutes)
ARCHETYPES: dict[str, list[tuple[str, list[int], int, int]]] = {
    "grad_student": [("class", [0, 2], 600, 75), ("lab", [1, 3], 780, 180), ("gym", [0, 2, 4], 1080, 60)],
    "shift_nurse": [("shift", [0, 1, 4], 420, 720), ("gym", [3, 5], 600, 60), ("groceries", [6], 660, 45)],
    "parent": [("school_dropoff", [0, 1, 2, 3, 4], 470, 30), ("soccer_practice", [1, 3], 1020, 90),
               ("work_block", [0, 1, 2, 3, 4], 570, 420)],
    "freelancer": [("client_call", [0, 3], 660, 45), ("deep_work", [0, 1, 2, 3, 4], 540, 180),
                   ("cowork", [2], 780, 240)],
    "retiree": [("walk", [0, 1, 2, 3, 4, 5, 6], 450, 45), ("book_club", [2], 840, 90), ("volunteer", [4], 600, 180)],
    "eng_manager": [("standup", [0, 1, 2, 3, 4], 570, 15), ("one_on_one", [1, 3], 840, 30), ("planning", [0], 780, 60)],
    # Out-of-distribution archetypes: test only, never in train or val.
    "small_business_owner": [("open_shop", [1, 2, 3, 4, 5], 480, 540), ("supplier_call", [0], 600, 30),
                             ("bookkeeping", [6], 1140, 90)],
    "rideshare_driver": [("evening_shift", [3, 4, 5], 1020, 300), ("morning_shift", [0, 1], 360, 240),
                         ("car_service", [6], 600, 60)],
}
IN_DIST = ["grad_student", "shift_nurse", "parent", "freelancer", "retiree", "eng_manager"]
OOD = ["small_business_owner", "rideshare_driver"]

PREFERENCES: list[tuple[str, list[str]]] = [
    ("meeting_window", ["morning", "afternoon", "evening"]),
    ("channel", ["email", "sms"]),
    ("reminder_lead_minutes", ["10", "30", "60"]),
]
# Minutes of the day covered by each meeting_window option.
WINDOW_MINUTES = {"morning": (480, 720), "afternoon": (720, 1020), "evening": (1020, 1260)}

RELATIONS = ["family", "family", "coworker", "coworker", "friend", "service"]

SENSITIVE: list[tuple[Sensitivity, str]] = [
    (Sensitivity.HEALTH, "therapy sessions at {clinic} Clinic"),
    (Sensitivity.FINANCE, "card ending {last4}"),
    (Sensitivity.LOCATION, "home at {address}"),
    (Sensitivity.RELATIONSHIP, "seeing {other} privately"),
]


def make_persona(seed: int, index: int, archetype: str) -> Persona:
    pid = f"p{index:04d}"
    rng = rng_for(seed, pid, "persona")
    fake = Faker("en_US")
    fake.seed_instance(derive_seed(seed, pid, "faker"))

    contacts = [
        Contact(contact_id=f"{pid}-c{i}", name=fake.unique.name(), email=f"{pid}.c{i}@example.org", relation=rel)
        for i, rel in enumerate(RELATIONS)
    ]
    routines = [
        Routine(name=name, weekdays=days, start_minute=start, jitter_minutes=int(rng.integers(5, 31)),
                duration_minutes=dur, skip_prob=float(rng.uniform(0.05, 0.25)))
        for name, days, start, dur in ARCHETYPES[archetype]
    ]
    preferences = []
    for key, options in PREFERENCES:
        initial = str(rng.choice(options))
        drifts = bool(rng.random() < 0.5)
        preferences.append(Preference(
            key=key, options=options, initial=initial,
            drift_day=int(rng.integers(8, 27)) if drifts else None,
            drift_to=str(rng.choice([o for o in options if o != initial])) if drifts else None,
        ))
    sensitive = []
    for j, (category, template) in enumerate(SENSITIVE):
        value = template.format(clinic=fake.last_name(), last4=f"{int(rng.integers(0, 10000)):04d}",
                                address=fake.street_address(), other=fake.first_name())
        allowed = sorted(contacts[int(k)].contact_id for k in rng.choice(len(contacts), size=2, replace=False))
        sensitive.append(SensitiveAttribute(attr_id=f"{pid}-s{j}", category=category, value=value,
                                            allowed_contacts=allowed))
    return Persona(persona_id=pid, archetype=archetype, name=fake.unique.name(), email=f"{pid}@example.com",
                   contacts=contacts, routines=routines, preferences=preferences, sensitive=sensitive)


def make_cohort(seed: int, n_in: int = 200, n_ood: int = 20) -> list[Persona]:
    personas = [make_persona(seed, i, IN_DIST[i % len(IN_DIST)]) for i in range(n_in)]
    personas += [make_persona(seed, n_in + i, OOD[i % len(OOD)]) for i in range(n_ood)]
    return personas
