"""Command line: p17sim generate | render | validate | leakage | sanity | eda."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from p17sim.harness import NullAgent, OracleAgent, score
from p17sim.io import read_jsonl, write_jsonl
from p17sim.leakage import (canary_leak, eval_window_leak, history_view, near_duplicates, persona_overlap,
                            value_overlap_count)
from p17sim.personas import make_cohort
from p17sim.schema import Event, Persona
from p17sim.skeleton import build_trace, trace_hash
from p17sim.splits import build_manifest
from p17sim.validate import validate_rows


def _load(data: Path, events_file: str):
    personas = [Persona.model_validate(r) for r in read_jsonl(data / "personas.jsonl")]
    events = [Event.model_validate(r) for r in read_jsonl(data / events_file)]
    manifest = json.loads((data / "split_manifest.json").read_text())
    return personas, events, manifest


def cmd_generate(a) -> int:
    personas = make_cohort(a.seed, a.n_in, a.n_ood)
    events = [e for p in personas for e in build_trace(p, a.seed)]
    write_jsonl(a.out / "personas.jsonl", personas)
    write_jsonl(a.out / "events.jsonl", events)
    manifest = build_manifest(personas, a.seed)
    (a.out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"personas={len(personas)} events={len(events)} trace_sha256={trace_hash(events)}")
    print(f"manifest_sha256={manifest['sha256']}")
    return 0


def cmd_render(a) -> int:
    from p17sim.render import FakeBackend, GeminiBackend, OllamaBackend, render_events

    personas, events, manifest = _load(a.data, "events.jsonl")
    by_split: dict[str, list[str]] = {}
    for pid, split in manifest["splits"].items():
        by_split.setdefault(split, []).append(pid)
    pilot = {pid for ids in by_split.values() for pid in sorted(ids)[:a.per_split]}
    chosen = [e for e in events if e.persona_id in pilot]
    backend = {"fake": FakeBackend, "ollama": OllamaBackend, "gemini": GeminiBackend}[a.backend]()
    rendered = render_events(chosen, {p.persona_id: p for p in personas}, backend, a.cache, a.workers)
    write_jsonl(a.data / a.out_file, rendered)
    failed = sum(e.render_status == "render_failed" for e in rendered)
    print(f"pilot_personas={len(pilot)} events={len(rendered)} render_failed={failed}")
    return 0


def cmd_validate(a) -> int:
    violations = validate_rows(read_jsonl(a.events))
    for v in violations[:50]:
        print(v)
    print(f"violations={len(violations)}")
    return 1 if violations else 0


def cmd_leakage(a) -> int:
    _, events, manifest = _load(a.data, a.events_file)
    by_split: dict[str, list[Event]] = {}
    for e in events:
        by_split.setdefault(manifest["splits"][e.persona_id], []).append(e)
    by_persona: dict[str, list[Event]] = {}
    for e in events:
        by_persona.setdefault(e.persona_id, []).append(e)
    found = persona_overlap(by_split) + canary_leak(events) + near_duplicates(by_split)
    for evs in by_persona.values():
        found += eval_window_leak(history_view(evs))
    for f in found[:50]:
        print(f)
    print(f"leakage_findings={len(found)} value_overlap_info={value_overlap_count(by_split)}")
    return 1 if found else 0


def cmd_sanity(a) -> int:
    personas, events, _ = _load(a.data, a.events_file)
    result = {"oracle": score(OracleAgent(), personas, events), "null": score(NullAgent(), personas, events)}
    print(json.dumps(result, indent=2))
    oracle_ok = all(v["pass_rate"] == 1.0 for v in result["oracle"].values())
    null_ok = all(v["pass_rate"] == 0.0 for k, v in result["null"].items() if k != "deletion")
    print(f"oracle_all_pass={oracle_ok} null_fails_utility={null_ok}")
    return 0 if oracle_ok and null_ok else 1


def cmd_build_trace(a) -> int:
    from p17sim.trace import build_trace, leakage_checks, sha256_file, splits_doc, validate_trace, write_jsonl
    from p17sim.trace_eda import write_report

    personas = [Persona.model_validate(r) for r in read_jsonl(a.data / "personas.jsonl")]
    raw = [Event.model_validate(r) for r in read_jsonl(a.data / a.raw_file)]
    seed = json.loads((a.data / "split_manifest.json").read_text())["seed"]
    print(f"[1/6] extract  {len(raw)} raw records, {len(personas)} personas, generator seed {seed}")
    built = build_trace(personas, raw, seed)
    records, labels, log = built["records"], built["labels"], built["log"]
    print("[2/6] clean    " + ", ".join(f"{k}={v}" for k, v in sorted(log.items())))
    violations = validate_trace(records, labels)
    print(f"[3/6] validate {len(violations)} violations")
    for v in violations[:20]:
        print("       " + v)
    splits = splits_doc(records)
    print(f"[4/6] split    {splits['counts']} (temporal: {splits['windows']})")
    leaks = leakage_checks(records, labels)
    print(f"[5/6] leakage  {len(leaks)} findings")
    for f in leaks[:20]:
        print("       " + f)
    out = a.out
    write_jsonl(out / "trace.jsonl", records)
    write_jsonl(out / "labels.jsonl", labels)
    write_jsonl(out / "raw_events.jsonl", [e.model_dump(mode="json") for e in raw])
    (out / "splits.json").write_text(json.dumps(splits, indent=1, sort_keys=True) + "\n")
    (out / "cleaning_log.json").write_text(json.dumps({"log": log, "samples": built["samples"]}, indent=1) + "\n")
    report = write_report(records, labels, log, built["samples"], out / "eda")
    print(f"[6/6] eda      {report}")
    files = ["trace.jsonl", "labels.jsonl", "raw_events.jsonl", "splits.json", "cleaning_log.json"]
    manifest = {"generator_seed": seed, "generator_version": __import__("p17sim").GENERATOR_VERSION,
                "renderer": a.renderer, "personas": len(personas), "records": len(records),
                "sha256": {f: sha256_file(out / f) for f in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"manifest  {out / 'manifest.json'}")
    return 1 if violations or leaks else 0


def cmd_validate_trace(a) -> int:
    from p17sim.trace import validate_trace

    violations = validate_trace(read_jsonl(a.trace), read_jsonl(a.labels))
    for v in violations[:50]:
        print(v)
    print(f"violations={len(violations)}")
    return 1 if violations else 0


def cmd_eda(a) -> int:
    from p17sim.eda import write_report

    personas, events, manifest = _load(a.data, a.events_file)
    print(write_report(personas, events, manifest["splits"], a.out))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p17sim")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--seed", type=int, default=2026)
    g.add_argument("--n-in", type=int, default=200)
    g.add_argument("--n-ood", type=int, default=20)
    g.add_argument("--out", type=Path, default=Path("data/v1"))
    g.set_defaults(fn=cmd_generate)

    r = sub.add_parser("render")
    r.add_argument("--data", type=Path, default=Path("data/v1"))
    r.add_argument("--per-split", type=int, default=8, help="pilot personas per split")
    r.add_argument("--backend", choices=["fake", "ollama", "gemini"], default="fake")
    r.add_argument("--cache", type=Path, default=Path("data/render_cache"))
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--out-file", default="events_rendered.jsonl")
    r.set_defaults(fn=cmd_render)

    v = sub.add_parser("validate")
    v.add_argument("events", type=Path)
    v.set_defaults(fn=cmd_validate)

    for name, fn in (("leakage", cmd_leakage), ("sanity", cmd_sanity), ("eda", cmd_eda)):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=Path("data/v1"))
        p.add_argument("--events-file", default="events.jsonl")
        if name == "eda":
            p.add_argument("--out", type=Path, default=Path("reports/eda"))
        p.set_defaults(fn=fn)

    b = sub.add_parser("build-trace", help="raw rendered events -> cleaned Workbook 1 trace, splits, EDA")
    b.add_argument("--data", type=Path, default=Path("data/v1"))
    b.add_argument("--raw-file", default="events_rendered.jsonl")
    b.add_argument("--renderer", default="ollama:llama3.2")
    b.add_argument("--out", type=Path, default=Path("../data/trace_v1"))
    b.set_defaults(fn=cmd_build_trace)

    t = sub.add_parser("validate-trace")
    t.add_argument("trace", type=Path)
    t.add_argument("--labels", type=Path, default=Path("../data/trace_v1/labels.jsonl"))
    t.set_defaults(fn=cmd_validate_trace)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
