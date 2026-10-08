"""Persona-level splits, stratified by archetype. The manifest is committed to git."""
from __future__ import annotations

import hashlib
import json

from p17sim import GENERATOR_VERSION
from p17sim.personas import OOD
from p17sim.rng import rng_for
from p17sim.schema import HISTORY_LAST_DAY, Persona


def assign_splits(personas: list[Persona], seed: int, val_frac: float = 0.15,
                  test_frac: float = 0.15) -> dict[str, str]:
    by_arch: dict[str, list[str]] = {}
    for p in personas:
        by_arch.setdefault(p.archetype, []).append(p.persona_id)
    splits: dict[str, str] = {}
    for arch, ids in sorted(by_arch.items()):
        if arch in OOD:
            splits.update({pid: "ood_test" for pid in ids})
            continue
        ids = sorted(ids)
        rng_for(seed, "split", arch).shuffle(ids)
        n_val, n_test = round(len(ids) * val_frac), round(len(ids) * test_frac)
        for k, pid in enumerate(ids):
            splits[pid] = "val" if k < n_val else "test" if k < n_val + n_test else "train"
    return dict(sorted(splits.items()))


def build_manifest(personas: list[Persona], seed: int) -> dict:
    splits = assign_splits(personas, seed)
    body = {"generator_version": GENERATOR_VERSION, "seed": seed, "history_last_day": HISTORY_LAST_DAY,
            "splits": splits}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body
