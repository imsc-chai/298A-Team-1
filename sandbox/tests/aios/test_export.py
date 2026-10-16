import json

from p17sim.aios.export import export_all, model_c_rows


def test_exports_cover_only_training_personas_and_gold_passes(v2ds, tmp_path):
    m = export_all(v2ds, tmp_path / "train")
    allowed = {p for p, s in v2ds.splits.items() if s in ("train", "val")}
    for name in m["files"]:
        for line in (tmp_path / "train" / name).read_text().splitlines():
            assert json.loads(line)["persona_id"] in allowed
    assert m["gold_check"]["checked"] > 0 and m["gold_check"]["failed"] == 0, m["gold_check"]["failures"]


def test_gold_rows_name_their_evidence(v2ds, tmp_path):
    export_all(v2ds, tmp_path / "t", verify=False)
    rows = [json.loads(x) for x in (tmp_path / "t" / "model_a_goal_to_workflow.train.jsonl").read_text().splitlines()]
    recall = [r for r in rows if r["kind"] in ("recall", "update")]
    assert recall and all(r["evidence_ids"] and set(r["evidence_ids"]) <= set(r["context_ids"]) for r in recall)
    assert {r["kind"] for r in rows} >= {"recall", "privacy", "scheduling", "forget"}


def test_suggestion_log_matches_the_true_preference_up_to_noise(v2ds):
    pids = [p for p, s in v2ds.splits.items() if s == "train"]
    rows = model_c_rows(v2ds, pids)
    agree = sum((r["offered"] == r["true_value"]) == r["accepted"] for r in rows) / len(rows)
    noise = sum(v2ds.extras[p].pref_noise for p in pids) / len(pids)
    assert abs((1 - agree) - noise) < 0.1
