"""Deterministic randomness: every component gets its own stream from (seed, names...)."""
from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def rng_for(*parts: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(*parts))
