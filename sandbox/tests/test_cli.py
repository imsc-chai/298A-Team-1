from p17sim.cli import main


def test_end_to_end_offline(tmp_path):
    data = tmp_path / "v1"
    assert main(["generate", "--n-in", "12", "--n-ood", "2", "--out", str(data)]) == 0
    assert main(["validate", str(data / "events.jsonl")]) == 0
    assert main(["render", "--data", str(data), "--per-split", "1", "--cache", str(tmp_path / "cache")]) == 0
    assert main(["validate", str(data / "events_rendered.jsonl")]) == 0
    assert main(["leakage", "--data", str(data), "--events-file", "events_rendered.jsonl"]) == 0
    assert main(["sanity", "--data", str(data)]) == 0
    assert main(["eda", "--data", str(data), "--events-file", "events_rendered.jsonl",
                 "--out", str(tmp_path / "eda")]) == 0
    assert (tmp_path / "eda" / "report.md").exists()


def test_validate_fails_on_corrupt_file(tmp_path):
    data = tmp_path / "v1"
    main(["generate", "--n-in", "6", "--n-ood", "0", "--out", str(data)])
    lines = (data / "events.jsonl").read_text().splitlines()
    lines[0] = lines[0].replace('"day": 1,', '"day": 45,')
    (data / "bad.jsonl").write_text("\n".join(lines) + "\n")
    assert main(["validate", str(data / "bad.jsonl")]) == 1
