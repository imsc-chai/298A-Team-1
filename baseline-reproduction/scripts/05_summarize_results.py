#!/usr/bin/env python3
"""Compare our latest PinchBench run to Table 1 (c) = 87.9%."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results" / "published" / "table1_hermes_agent.json"
RUNS = ROOT / "results" / "runs" / "table1_c"
OUT = ROOT / "results" / "runs" / "comparison.json"

TARGET = 87.9


def latest_run_dir() -> Path | None:
    if not RUNS.exists():
        return None
    dirs = [p for p in RUNS.iterdir() if p.is_dir()]
    return max(dirs, default=None, key=lambda p: p.name)


def extract_mean_from_text(text: str) -> float | None:
    # OpenJarvis reports mean pass rate like 0.879 or 87.9%
    pct = re.findall(r"Mean pass rate[^\d]*([0-9]*\.?[0-9]+)", text, flags=re.I)
    if pct:
        v = float(pct[-1])
        return v * 100.0 if v <= 1.0 else v
    return None


def main() -> int:
    published = json.loads(PUBLISHED.read_text())
    target = published["conditions"]["c_openjarvis_qwen3.5_9b"]["pinchbench_accuracy_percent"]

    run_dir = latest_run_dir()
    payload = {
        "published_percent": target,
        "our_percent": None,
        "gap_pp": None,
        "run_dir": str(run_dir) if run_dir else None,
        "status": "no_run_yet",
    }

    if run_dir:
        texts = []
        for p in run_dir.rglob("*"):
            if p.suffix.lower() in {".md", ".json", ".txt", ".log"}:
                try:
                    texts.append(p.read_text(errors="ignore"))
                except OSError:
                    pass
        mean = None
        for t in texts:
            mean = extract_mean_from_text(t)
            if mean is not None:
                break
        if mean is not None:
            payload["our_percent"] = round(mean, 3)
            payload["gap_pp"] = round(mean - target, 3)
            payload["status"] = "compared"
        else:
            payload["status"] = "run_found_but_unparsed"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["status"] == "no_run_yet":
        print("No full run yet. Use scripts/04_full_eval.sh", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
