"""v2 generate and render steps: write the skeleton, the split manifest and the rendered events."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from p17sim.gen2 import GENERATOR_VERSION
from p17sim.gen2.personas import PersonaExtra, make_cohort
from p17sim.gen2.skeleton import build_trace, trace_hash
from p17sim.io import read_jsonl, write_jsonl
from p17sim.schema import Event, HISTORY_LAST_DAY, Persona, VAL_LAST_DAY


def manifest(splits: dict[str, str], seed: int, sizes: dict[str, int]) -> dict:
    body = {"generator_version": GENERATOR_VERSION, "seed": seed, "sizes": sizes,
            "persona_splits": dict(sorted(splits.items())),
            "temporal_windows": {"train": [1, HISTORY_LAST_DAY], "val": [HISTORY_LAST_DAY + 1, VAL_LAST_DAY],
                                 "test": [VAL_LAST_DAY + 1, 30]},
            "scoring": "headline = test-window (days 25-30) tasks of test personas; OOD reported separately; "
                       "trained parts learn from train personas only; choices made on val personas"}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def generate(out: Path, seed: int, n_train: int, n_val: int, n_test: int, n_ood: int) -> dict:
    personas, extras, splits = make_cohort(seed, n_train, n_val, n_test, n_ood)
    events = [e for p, x in zip(personas, extras) for e in build_trace(p, x, seed, splits[p.persona_id])]
    write_jsonl(out / "personas.jsonl", personas)
    write_jsonl(out / "extras.jsonl", extras)
    write_jsonl(out / "events.jsonl", events)
    m = manifest(splits, seed, {"train": n_train, "val": n_val, "test": n_test, "ood_test": n_ood})
    (out / "split_manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    return {"personas": len(personas), "events": len(events), "trace_sha256": trace_hash(events),
            "manifest_sha256": m["sha256"]}


def load(data: Path, events_file: str = "events.jsonl"):
    personas = [Persona.model_validate(r) for r in read_jsonl(data / "personas.jsonl")]
    extras = [PersonaExtra.model_validate(r) for r in read_jsonl(data / "extras.jsonl")]
    events = [Event.model_validate(r) for r in read_jsonl(data / events_file)]
    splits = json.loads((data / "split_manifest.json").read_text())["persona_splits"]
    return personas, extras, events, splits


def render(data: Path, cache: Path, workers: int, out_file: str = "events_rendered.jsonl") -> dict:
    from p17sim.gen2.render import render_cohort

    personas, extras, events, _ = load(data)
    rendered = render_cohort(events, personas, extras, cache, workers)
    write_jsonl(data / out_file, rendered)
    status: dict[str, int] = {}
    for e in rendered:
        status[e.render_status] = status.get(e.render_status, 0) + 1
    return status


def export(data: Path, out: Path, raw_file: str = "events_rendered.jsonl") -> dict:
    """Rendered events -> cleaned v2 trace + labels, validated and leak-checked, with a hashed manifest."""
    from p17sim.gen2.trace import build, leakage, validate
    from p17sim.trace import sha256_file

    personas, extras, raw, splits = load(data, raw_file)
    built = build(personas, extras, raw, splits)
    violations = validate(built["records"], built["labels"], extras)
    leaks = leakage(built["records"], built["labels"], splits)
    write_jsonl(out / "trace.jsonl", built["records"])
    write_jsonl(out / "labels.jsonl", built["labels"])
    (out / "cleaning_log.json").write_text(json.dumps({"log": built["log"], "samples": built["samples"]},
                                                      indent=1, ensure_ascii=False) + "\n")
    (out / "checks.json").write_text(json.dumps({"violations": violations, "leakage": leaks}, indent=1) + "\n")
    files = ["trace.jsonl", "labels.jsonl", "cleaning_log.json"]
    m = {"generator_version": GENERATOR_VERSION, "records": len(built["records"]), "personas": len(personas),
         "split_manifest_sha256": json.loads((data / "split_manifest.json").read_text())["sha256"],
         "sha256": {f: sha256_file(out / f) for f in files}}
    (out / "manifest.json").write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    return {"log": built["log"], "violations": violations, "leakage": leaks, "manifest": m}
