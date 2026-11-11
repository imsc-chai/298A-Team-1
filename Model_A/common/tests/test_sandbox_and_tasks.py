import hashlib
import re
from pathlib import Path

from common.sandbox import task_sandbox
from common.config import MODEL_A_ROOT, persona_dir
from common.tasks import load_agent_tasks, personas_for_split


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode() + p.read_bytes())
    return h.hexdigest()


def test_sandbox_excludes_test_suite_and_protects_source():
    source = persona_dir("persona_01_maya")
    before = tree_digest(source)
    with task_sandbox(source) as sb:
        assert not (sb / "test_suite").exists()
        assert (sb / "local_files" / "flight_ticket.png").is_file()
        (sb / "streams" / "calendar.json").write_text("[]")
        kept = sb
    assert not kept.exists()
    assert tree_digest(source) == before


def test_agent_tasks_expose_only_query_and_time():
    tasks = load_agent_tasks("persona_01_maya", ["eval_03_medical_privacy_boundary"])
    assert len(tasks) == 1
    assert set(tasks[0].model_dump()) == {"persona_id", "task_id", "user_query", "reference_time"}


def test_every_dev_task_has_reference_time():
    for persona in personas_for_split("dev"):
        assert load_agent_tasks(persona)


def test_only_eval_code_mentions_ground_truth_files():
    """Guard rule 2: model/shared code must not open test_suite or eval_tasks.json."""
    offenders = []
    for py in list((MODEL_A_ROOT / "common").rglob("*.py")) + list((MODEL_A_ROOT / "model3").rglob("*.py")):
        rel = py.relative_to(MODEL_A_ROOT).as_posix()
        # tasks.py strips tasks to query+time; sandbox enforces exclusion; validator checks ids/times.
        allowed = {"common/tasks.py", "common/sandbox.py", "model3/validate_output.py"}
        if rel.startswith(("common/tests/", "model3/tests/")) or rel in allowed:
            continue
        # String literals only: docstrings may mention the folder, code must not path to it.
        if re.search(r"""["'](test_suite|eval_tasks(\.json)?)["'/]""", py.read_text()):
            offenders.append(rel)
    assert offenders == [], offenders
