from collections import Counter

from p17sim.personas import IN_DIST, make_cohort
from p17sim.splits import assign_splits, build_manifest

COHORT = make_cohort(2026, n_in=200, n_ood=20)


def test_split_sizes():
    assert Counter(assign_splits(COHORT, 2026).values()) == {"train": 140, "val": 30, "test": 30, "ood_test": 20}


def test_splits_are_deterministic():
    assert assign_splits(COHORT, 2026) == assign_splits(COHORT, 2026)


def test_every_in_dist_archetype_in_val_and_test():
    splits = assign_splits(COHORT, 2026)
    for split in ("val", "test"):
        archs = {p.archetype for p in COHORT if splits[p.persona_id] == split}
        assert archs == set(IN_DIST)


def test_ood_archetypes_only_in_ood_test():
    splits = assign_splits(COHORT, 2026)
    assert all(splits[p.persona_id] == "ood_test" for p in COHORT if p.archetype not in IN_DIST)


def test_manifest_hash_depends_on_seed():
    assert build_manifest(COHORT, 2026)["sha256"] != build_manifest(COHORT, 2027)["sha256"]
