"""Statistical tests from the pre-registration (Section 8)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import binomtest


def per_task_success(runs: list[list[dict]]) -> dict[str, tuple[str, float]]:
    """task_id -> (persona_id, mean success over the runs/seeds that scored it)."""
    acc: dict[str, list[float]] = defaultdict(list)
    owner = {}
    for rows in runs:
        for r in rows:
            acc[r["task_id"]].append(float(r["passed"]))
            owner[r["task_id"]] = r["persona_id"]
    return {k: (owner[k], float(np.mean(v))) for k, v in acc.items()}


def per_task_value(runs: list[list[dict]], key: str, kinds: set[str] | None = None) -> dict[str, tuple[str, float]]:
    """task_id -> (persona_id, mean of a numeric/bool field over seeds), optionally for some kinds only."""
    acc: dict[str, list[float]] = defaultdict(list)
    owner = {}
    for rows in runs:
        for r in rows:
            if kinds is None or r["kind"] in kinds:
                acc[r["task_id"]].append(float(r.get(key) or 0))
                owner[r["task_id"]] = r["persona_id"]
    return {k: (owner[k], float(np.mean(v))) for k, v in acc.items()}


def cluster_bootstrap_diff(a: dict[str, tuple[str, float]], b: dict[str, tuple[str, float]],
                           reps: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Mean(a - b) over tasks both systems scored, with a 95% CI from resampling personas."""
    common = sorted(set(a) & set(b))
    by_p: dict[str, list[float]] = defaultdict(list)
    for t in common:
        by_p[a[t][0]].append(a[t][1] - b[t][1])
    personas = sorted(by_p)
    sums = np.array([sum(by_p[p]) for p in personas])
    counts = np.array([len(by_p[p]) for p in personas])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(personas), size=(reps, len(personas)))
    boots = sums[idx].sum(1) / counts[idx].sum(1)
    return {"diff": round(float(sums.sum() / counts.sum()), 4), "ci_low": round(float(np.percentile(boots, 2.5)), 4),
            "ci_high": round(float(np.percentile(boots, 97.5)), 4), "tasks": len(common), "personas": len(personas)}


def cluster_bootstrap_mean(values: dict[str, tuple[str, float]], reps: int = 10_000, seed: int = 0) -> dict:
    by_p: dict[str, list[float]] = defaultdict(list)
    for p, v in values.values():
        by_p[p].append(v)
    personas = sorted(by_p)
    sums = np.array([sum(by_p[p]) for p in personas])
    counts = np.array([len(by_p[p]) for p in personas])
    idx = np.random.default_rng(seed).integers(0, len(personas), size=(reps, len(personas)))
    boots = sums[idx].sum(1) / counts[idx].sum(1)
    return {"mean": round(float(sums.sum() / max(1, counts.sum())), 4),
            "ci_low": round(float(np.percentile(boots, 2.5)), 4), "ci_high": round(float(np.percentile(boots, 97.5)), 4),
            "tasks": int(counts.sum()), "personas": len(personas)}


def group_gap(values_a: dict[str, tuple[str, float]], values_b: dict[str, tuple[str, float]],
              reps: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Mean(a) - mean(b) for two independent groups of personas, bootstrapping personas in each."""
    def prep(v):
        by_p: dict[str, list[float]] = defaultdict(list)
        for p, x in v.values():
            by_p[p].append(x)
        ps = sorted(by_p)
        return np.array([sum(by_p[p]) for p in ps]), np.array([len(by_p[p]) for p in ps])
    sa, ca = prep(values_a)
    sb, cb = prep(values_b)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(sa), size=(reps, len(sa)))
    ib = rng.integers(0, len(sb), size=(reps, len(sb)))
    boots = sa[ia].sum(1) / ca[ia].sum(1) - sb[ib].sum(1) / cb[ib].sum(1)
    return {"gap": round(float(sa.sum() / ca.sum() - sb.sum() / cb.sum()), 4),
            "ci_low": round(float(np.percentile(boots, 2.5)), 4), "ci_high": round(float(np.percentile(boots, 97.5)), 4),
            "n_a": int(ca.sum()), "n_b": int(cb.sum())}


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict[str, float]:
    """Exact McNemar on paired pass/fail. b10 = a passed and b failed; b01 = the reverse."""
    common = set(a) & set(b)
    b10 = sum(a[t] and not b[t] for t in common)
    b01 = sum(b[t] and not a[t] for t in common)
    p = binomtest(b10, b10 + b01, 0.5).pvalue if b10 + b01 else 1.0
    return {"n": len(common), "a_only": b10, "b_only": b01, "p": round(float(p), 5)}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = round(running, 5)
    return out
