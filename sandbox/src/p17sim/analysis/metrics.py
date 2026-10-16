"""The seven pre-registered metrics plus model profiling, computed from a run's result rows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SCORED = ("recall", "update", "deletion", "privacy", "scheduling")
WATTS, USD_PER_KWH = 20.0, 0.32  # pre-registered cost model (Section 7)


def load_run(path: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return [r for r in rows if not r.get("persona_done")], meta


def tasks(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("kind") in SCORED]


def summarize(rows: list[dict], scored_days_per_persona: int = 6) -> dict[str, Any]:
    t = tasks(rows)
    audits = [r for r in rows if r.get("kind") == "deletion_audit"]
    by_kind = {k: [r for r in t if r["kind"] == k] for k in SCORED}
    rate = lambda rs: round(float(np.mean([r["passed"] for r in rs])), 4) if rs else None  # noqa: E731
    sched = by_kind["scheduling"]
    lat = np.array([r["latency_s"] for r in t]) if t else np.array([0.0])
    compute_s = sum(r["profile"]["llm_wall_s"] for r in rows)
    personas = {r["persona_id"] for r in rows}
    user_days = max(1, len(personas) * scored_days_per_persona)
    probes = rate(by_kind["deletion"])
    audit = rate(audits)
    return {
        "tasks": len(t), "personas": len(personas),
        "task_success": rate(t),
        "success_by_kind": {k: rate(v) for k, v in by_kind.items()},
        "n_by_kind": {k: len(v) for k, v in by_kind.items()},
        "intervention_rate": round(float(np.mean([r["interventions"] for r in t])), 4) if t else None,
        "preference_regret": round(float(np.mean([1 - r.get("reward", 0.0) for r in sched])), 4) if sched else None,
        "deletion_probe_pass": probes, "deletion_storage_audit_pass": audit,
        "deletion_compliance": None if probes is None or audit is None else min(probes, audit),
        "privacy_leakage": round(float(np.mean([r.get("leak", False) for r in by_kind["privacy"]])), 4)
        if by_kind["privacy"] else None,
        "raw_export_bytes": int(sum(r["profile"]["raw_export_bytes"] for r in rows)),
        "nonlocal_calls": int(sum(r["profile"]["nonlocal_calls"] for r in rows)),
        "latency_p50_s": round(float(np.percentile(lat, 50)), 2),
        "latency_p95_s": round(float(np.percentile(lat, 95)), 2),
        "cost_per_user_day_usd": round(compute_s * WATTS / 3.6e6 * USD_PER_KWH / user_days, 6),
        "errors": sum(bool(r.get("error")) for r in rows),
        "profiling": profiling(rows),
    }


def profiling(rows: list[dict]) -> dict[str, Any]:
    t = tasks(rows)
    if not t:
        return {}
    p = [r["profile"] for r in t]
    prompt = sum(x["prompt_tokens"] for x in p)
    out = sum(x["output_tokens"] for x in p)
    prefill_s = sum(x["prefill_s"] for x in p)
    decode_s = sum(x["decode_s"] for x in p)
    ttft = [v for x in p for v in x["ttft_s"]]
    comp: dict[str, float] = {}
    for x in p:
        for k, v in x["time_by_component_s"].items():
            comp[k] = comp.get(k, 0.0) + v
    return {
        "llm_calls_per_task": round(float(np.mean([x["llm_calls"] for x in p])), 2),
        "prompt_tokens_per_task": round(prompt / len(t), 1),
        "output_tokens_per_task": round(out / len(t), 1),
        "prefill_tok_per_s": round(prompt / prefill_s, 1) if prefill_s else None,
        "decode_tok_per_s": round(out / decode_s, 1) if decode_s else None,
        "ttft_median_s": round(float(np.median(ttft)), 3) if ttft else None,
        "ttft_p95_s": round(float(np.percentile(ttft, 95)), 3) if ttft else None,
        "model_load_s_total": round(sum(x["load_s"] for x in p), 2),
        "turns_per_task": round(float(np.mean([r["turns"] for r in t])), 2),
        "seconds_by_component_per_task": {k: round(v / len(t), 3) for k, v in sorted(comp.items())},
    }


def per_task(rows: list[dict]) -> dict[str, dict]:
    """task_id -> row (for pairing across systems)."""
    return {r["task_id"]: r for r in tasks(rows)}
