"""Check a results folder against the team Output contract (CLAUDE.md).

    python model3/validate_output.py model3/results                 # every run_XX inside
    python model3/validate_output.py model3/results/run_00 --split dev --require-all

Structural/contract checks only; scoring against ground truth is the team evaluator's job.
It opens eval_tasks.json only to confirm task ids and reference_time.
Exit code 0 = no errors (warnings allowed), 1 = errors found.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):  # allow `python model3/validate_output.py` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from common.config import load_tool_registry, persona_dir
from common.contract import DRAFT_TOOLS, TaskResult
from common.ingest import build_records
from common.llm_client import LLMSettings
from common.retriever import EMBED_MODEL
from common.sandbox import task_sandbox
from common.schemas import StepStatus
from common.tools import ArgError, parse_registry, validate_args

RUN_DIR = re.compile(r"^run_(\d{2,})$")
EXPECTED = LLMSettings()
EXPECTED_SETTINGS = {"model": EXPECTED.model, "temperature": EXPECTED.temperature, "max_tokens": EXPECTED.max_tokens,
                     "seed": EXPECTED.seed, "embedding_model": EMBED_MODEL}
OK_STATUSES = {
    "draft": {StepStatus.DRAFT, StepStatus.FAILED, StepStatus.BLOCKED},
    "confirm": {StepStatus.REQUIRES_CONFIRMATION, StepStatus.BLOCKED},
    "other": {StepStatus.EXECUTED, StepStatus.FAILED, StepStatus.BLOCKED},
}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"{where}: {msg}")


class PersonaFacts:
    """Lazily loaded per-persona facts the checks need."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, str]] = {}
        self._records: dict[str, set[str]] = {}
        self._files: dict[str, set[str]] = {}

    def tasks(self, persona: str) -> dict[str, str]:
        if persona not in self._tasks:
            raw = json.loads((persona_dir(persona) / "test_suite" / "eval_tasks.json").read_text())
            self._tasks[persona] = {t["task_id"]: t.get("reference_time", "") for t in raw}
        return self._tasks[persona]

    def record_ids(self, persona: str) -> set[str]:
        if persona not in self._records:
            with task_sandbox(persona_dir(persona)) as sb:
                self._records[persona] = {r.id for r in build_records(sb, persona)}
        return self._records[persona]

    def files(self, persona: str) -> set[str]:
        if persona not in self._files:
            self._files[persona] = {f"local_files/{p.name}" for p in (persona_dir(persona) / "local_files").iterdir()}
        return self._files[persona]


def run_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise SystemExit(f"Results folder not found: {root}  (in a Colab `!` line, precompute the path; "
                         "`{RUN:02d}` is not expanded there)")
    if RUN_DIR.match(root.name):
        return [root]
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and RUN_DIR.match(p.name))
    if not dirs:
        raise SystemExit(f"No run_XX folders in {root}")
    return dirs


def check_file(path: Path, run_index: int, persona: str, model: str, facts: PersonaFacts, specs, rep: Report) -> None:
    where = str(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        rep.err(where, f"not valid JSON: {exc}")
        return
    try:
        result = TaskResult.model_validate(data)
    except ValidationError as exc:
        for e in exc.errors():
            rep.err(where, f"schema: {'.'.join(str(x) for x in e['loc'])}: {e['msg']}")
        return

    # header vs location and task definition
    if result.model != model:
        rep.err(where, f"model is {result.model!r}, expected {model!r}")
    if result.persona_id != persona:
        rep.err(where, f"persona_id {result.persona_id!r} != folder {persona!r}")
    if result.task_id != path.stem:
        rep.err(where, f"task_id {result.task_id!r} != file name {path.stem!r}")
    if result.run != run_index:
        rep.err(where, f"run {result.run} != folder index {run_index}")
    tasks = facts.tasks(persona)
    if result.task_id not in tasks:
        rep.err(where, f"unknown task_id for {persona}")
    elif result.reference_time != tasks[result.task_id]:
        rep.err(where, f"reference_time {result.reference_time!r} != task's {tasks[result.task_id]!r}")

    # steps
    known_ids = facts.record_ids(persona)
    for s in result.steps:
        sw = f"{where} step {s.step}"
        spec = specs.get(s.tool)
        if spec is None:
            rep.err(sw, f"tool {s.tool!r} not in tool_registry.json")
            continue
        try:
            validate_args(spec, s.args)
        except ArgError as exc:
            rep.err(sw, f"args: {exc}")
        if any(v is None for v in s.args.values()):
            rep.err(sw, "args contain null values")
        kind = "draft" if s.tool in DRAFT_TOOLS else "confirm" if spec.permission == "requires_confirmation" else "other"
        if s.status not in OK_STATUSES[kind]:
            rep.err(sw, f"status {s.status.value} not allowed for {s.tool}")
        if s.schedule_proposals is not None:
            if s.tool not in DRAFT_TOOLS:
                rep.err(sw, "schedule_proposals on a non-draft step")
            if not s.schedule_proposals:
                rep.err(sw, "schedule_proposals present but empty (omit it instead)")
        unknown = [i for i in s.evidence_ids if i not in known_ids]
        if unknown:
            rep.warn(sw, f"evidence ids not in the persona index (created this task?): {unknown}")

    if any(s.status in (StepStatus.REQUIRES_CONFIRMATION, StepStatus.BLOCKED) for s in result.steps) and result.status.value == "COMPLETE":
        rep.warn(where, "status COMPLETE but a step was refused/blocked")

    # audit, extractions
    if not result.audit.retrieval and result.status.value != "ERROR":
        rep.warn(where, "audit.retrieval is empty")
    for entry in result.audit.retrieval:
        bad = [i for i in entry.evidence_ids if i not in known_ids]
        if bad:
            rep.err(where, f"audit.retrieval ids not in the persona index: {bad}")
    for key, fields in result.extractions.items():
        if key not in facts.files(persona):
            rep.err(where, f"extractions key {key!r} is not a file in {persona}/local_files")
        if not fields:
            rep.warn(where, f"extractions[{key!r}] is empty")

    # telemetry must be real measurements
    t = result.telemetry
    if result.status.value != "ERROR" and t.llm_calls == 0:
        rep.err(where, "telemetry.llm_calls is 0 for a non-ERROR task")
    if t.llm_calls > 0 and (t.latency_ms == 0 or t.input_tokens == 0):
        rep.err(where, "telemetry has LLM calls but zero latency or input tokens (not a real measurement)")

    # settings fixed for fair comparison
    st = result.settings.model_dump()
    for key, expected in EXPECTED_SETTINGS.items():
        if st.get(key) != expected:
            rep.err(where, f"settings.{key} = {st.get(key)!r}, expected {expected!r}")
    if not (st.get("hardware") or {}).get("gpu"):
        rep.warn(where, "settings.hardware.gpu missing (run not on a GPU host?)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path, help="results root (with run_XX folders) or one run_XX folder")
    ap.add_argument("--model", default="M3")
    ap.add_argument("--split", choices=["dev", "test"], help="expected personas, for completeness checks")
    ap.add_argument("--personas", nargs="+", help="expected personas (overrides --split)")
    ap.add_argument("--require-all", action="store_true", help="missing tasks are errors, not warnings")
    ap.add_argument("--quiet", action="store_true", help="hide warnings")
    args = ap.parse_args(argv)

    from common.tasks import personas_for_split  # local: only needed with --split

    expected_personas = args.personas or (personas_for_split(args.split) if args.split else None)
    specs = parse_registry(load_tool_registry())
    facts = PersonaFacts()
    rep = Report()
    status_counts: dict[str, int] = defaultdict(int)
    n_files = 0

    for rdir in run_dirs(args.results_dir):
        run_index = int(RUN_DIR.match(rdir.name).group(1))
        for tmp in rdir.rglob("*.json.tmp"):
            rep.warn(str(tmp), "leftover partial write")
        persona_dirs = sorted(p for p in rdir.iterdir() if p.is_dir())
        for pdir in persona_dirs:
            if not _persona_known(pdir.name):
                rep.err(str(pdir), "folder is not a known persona id")
                continue
            for f in sorted(pdir.glob("*.json")):
                n_files += 1
                check_file(f, run_index, pdir.name, args.model, facts, specs, rep)
                try:
                    status_counts[json.loads(f.read_text()).get("status", "?")] += 1
                except json.JSONDecodeError:
                    pass
        if expected_personas:
            for persona in expected_personas:
                have = {f.stem for f in (rdir / persona).glob("*.json")} if (rdir / persona).is_dir() else set()
                for task_id in sorted(set(facts.tasks(persona)) - have):
                    (rep.err if args.require_all else rep.warn)(f"{rdir.name}/{persona}", f"missing result for {task_id}")

    if not args.quiet:
        for w in rep.warnings:
            print(f"WARN  {w}")
    for e in rep.errors:
        print(f"ERROR {e}")
    print(f"\nChecked {n_files} file(s): {len(rep.errors)} error(s), {len(rep.warnings)} warning(s). "
          f"Task status counts: {dict(status_counts)}")
    return 1 if rep.errors else 0


def _persona_known(name: str) -> bool:
    try:
        persona_dir(name)
        return True
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    sys.exit(main())
