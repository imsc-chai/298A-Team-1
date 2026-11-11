"""Behavioral checks for the revised, task-wide comparison configuration."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from test_guardplan import Fixture, FakeModel
from guardplan import cli
from guardplan.core import GuardPlanError, MemoryRecord, ModelOutputError, Settings, write_json
from guardplan.llm import LocalModel
from guardplan.planner import GuardPlan
from guardplan.retrieval import LexicalRetriever


class ComparisonSettingsTests(Fixture):
    def test_lexical_pool_then_selection_with_stable_ties(self):
        records = [MemoryRecord(f"r{i:02}", "p", "s", "file", "alpha beta " + ("gamma" if i == 9 else ""))
                   for i in range(10)]
        records += [MemoryRecord("request", "p", "s", "request", "alpha beta gamma")]
        retriever = LexicalRetriever(records)
        result = retriever.retrieve("alpha beta gamma", "2026-11-01T00:00:00")
        self.assertEqual([r.id for r in result], ["r09", "r00", "r01", "r02", "r03", "r04"])
        self.assertEqual(len(retriever.history[0]["candidate_ids"]), 8)
        self.assertEqual(len(result), 6)

    def test_lexical_visibility_filter_precedes_candidate_pool(self):
        records = [MemoryRecord("deleted", "p", "s", "file", "alpha beta gamma", deleted=True),
                   MemoryRecord("future", "p", "s", "file", "alpha beta gamma", valid_from="2027-01-01"),
                   MemoryRecord("old", "p", "s", "file", "alpha beta gamma", superseded_by="live"),
                   MemoryRecord("live", "p", "s", "file", "alpha")]
        r = LexicalRetriever(records)
        self.assertEqual([x.id for x in r.retrieve("alpha beta gamma", "2026-11-01")], ["live"])
        self.assertEqual(r.history[0]["candidate_ids"], ["live"])

    def test_single_shot_context_is_reused_without_embeddings(self):
        model = FakeModel([{"subgoals": ["Check calendar", "Look up Friend"]},
            {"tool": "search_calendar", "args": {"date": "2026-11-10"}, "evidence_ids": ["user_request"]},
            {"done": True}, {"tool": "lookup_contact", "args": {"name": "Friend"}, "evidence_ids": ["user_request"]},
            {"done": True}])
        model.embed = lambda *_: self.fail("Single-shot lexical mode must not call embeddings")
        result = GuardPlan(self.adapter(), model).run("Check my calendar for Tuesday Nov 10 and look up Friend.")
        self.assertEqual(result["status"], "COMPLETE", result)
        self.assertEqual(len(result["audit"]["retrieval"]), 1)
        contexts = [p["context"] for p in model.seen]
        self.assertTrue(all(c == contexts[0] for c in contexts))

    def test_eight_turns_are_shared_across_subgoals(self):
        responses = [{"subgoals": ["Check " + str(i) for i in range(5)]}]
        for _ in range(5):
            responses += [{"tool": "search_calendar", "args": {"date": "2026-11-10"}, "evidence_ids": ["user_request"]}, {"done": True}]
        model = FakeModel(responses)
        result = GuardPlan(self.adapter(), model).run("Check the calendar.")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("turn budget", result["error"])
        self.assertEqual(result["telemetry"]["executor_turns"], 8)
        self.assertEqual(len(result["steps"]), 4)

    def test_format_retries_consume_executor_turns(self):
        model = FakeModel([{"subgoals": ["Check calendar"]}, ModelOutputError("bad JSON"),
                           {"tool": "search_calendar", "args": {}, "evidence_ids": ["user_request"]},
                           {"done": True}])
        result = GuardPlan(self.adapter(), model, Settings(max_turns=2)).run("Check the calendar.")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["telemetry"]["executor_turns"], 2)
        self.assertEqual(len(model.calls), 3)  # one decomposition, two executor attempts

    def test_request_payloads_use_fixed_sampling_for_text_and_vision(self):
        model = LocalModel(Settings())
        requests = []
        def post(route, payload):
            requests.append(payload)
            return {"message": {"content": '{"done":true}'}}
        model.post = post
        model.chat("Return JSON", {}, "execute")
        image = self.workspace / "image.png"
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(b"synthetic image bytes")
        model.extract_image(image)
        for payload in requests:
            self.assertEqual(payload["options"]["num_predict"], 1024)
            self.assertEqual(payload["options"]["temperature"], .6)
            self.assertEqual(payload["options"]["seed"], 42)

    def test_quantization_is_verified_and_mismatches_rejected(self):
        model = LocalModel(Settings())
        model.post = lambda *_: {"details": {"quantization_level": "nvfp4"}}
        with self.assertRaisesRegex(GuardPlanError, "quantization mismatch"):
            model.verify_model()
        model.post = lambda *_: {"details": {"quantization_level": "Q4_K_M", "parameter_size": "9B"}}
        self.assertEqual(model.verify_model()["quantization_level"], "Q4_K_M")

    def test_benchmark_repeats_keep_seed_42_and_export_run_indices(self):
        task = {"task_id": "settings_test", "reference_time": "2026-11-07T10:00:00", "user_query": "Check calendar."}
        write_json(self.persona / "test_suite/eval_tasks.json", [task])
        write_json(self.root / "splits.json", {"dev_personas": ["persona_test"], "test_personas": []})
        observed = []
        class Planner:
            def __init__(self, adapter, model, settings, remote):
                self.settings = settings
            def run(self, query, task_id):
                observed.append(self.settings.seed)
                return {"task_id": task_id, "steps": [], "status": "COMPLETE", "answer": "",
                        "telemetry": {"latency_ms": 1, "input_tokens": 0, "output_tokens": 0}}
        args = ["guardplan", "evaluate", "--data-root", str(self.root), "--workspace", str(self.workspace),
                "--persona", "persona_test", "--runs", "5"]
        with patch("sys.argv", args), patch.object(cli.LocalModel, "verify_model", return_value={}), \
                patch.object(cli, "GuardPlan", Planner):
            self.assertEqual(cli.main(), 0)
        self.assertEqual(observed, [42] * 5)
        workflows = [json.loads(p.read_text()) for p in (self.workspace / "evaluations").glob("*/persona_test/*/settings_test.json")]
        self.assertEqual(sorted(w["run"] for w in workflows), list(range(5)))
