from p17sim.personas import IN_DIST, OOD, make_cohort, make_persona


def test_same_seed_same_persona():
    assert make_persona(7, 0, "retiree") == make_persona(7, 0, "retiree")


def test_different_seed_different_persona():
    assert make_persona(7, 0, "retiree") != make_persona(8, 0, "retiree")


def test_emails_use_reserved_domains_only():
    p = make_persona(7, 3, "parent")
    emails = [p.email] + [c.email for c in p.contacts]
    assert all(e.endswith(("@example.com", "@example.org")) for e in emails)


def test_contact_names_unique_within_persona():
    p = make_persona(7, 3, "parent")
    names = [p.name] + [c.name for c in p.contacts]
    assert len(names) == len(set(names))


def test_sensitive_attributes_allow_two_known_contacts():
    p = make_persona(7, 3, "parent")
    ids = {c.contact_id for c in p.contacts}
    for attr in p.sensitive:
        assert len(attr.allowed_contacts) == 2 and set(attr.allowed_contacts) <= ids


def test_cohort_sizes_and_ood_at_end():
    cohort = make_cohort(7, n_in=12, n_ood=4)
    assert len(cohort) == 16
    assert {p.archetype for p in cohort[:12]} == set(IN_DIST)
    assert {p.archetype for p in cohort[12:]} == set(OOD)
