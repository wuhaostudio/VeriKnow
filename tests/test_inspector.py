from pathlib import Path
import json
import unittest

from veriknow.modules.inspector import inspect_run, redact
from veriknow.schemas import RunRecord, TaskSpec


class InspectorTests(unittest.TestCase):
    def test_redact_masks_sensitive_keys_and_text(self) -> None:
        payload = {
            "api_key": "key-123",
            "nested": {
                "message": "Authorization: Bearer abc123 and token=raw-token",
                "safe": "visible",
            },
        }

        redacted = redact(payload)

        self.assertEqual(redacted["api_key"], "[REDACTED]")
        self.assertEqual(redacted["nested"]["safe"], "visible")
        self.assertNotIn("abc123", redacted["nested"]["message"])
        self.assertNotIn("raw-token", redacted["nested"]["message"])
        self.assertEqual(redact({"total_tokens": 42})["total_tokens"], 42)

    def test_inspect_run_summarizes_artifacts_with_redacted_preview(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            run_dir = tmp_path / "runs" / "run-test"
            run_dir.mkdir(parents=True)
            artifact_path = run_dir / "llm.json"
            artifact_path.write_text(
                json.dumps({"token": "secret-token", "message": "ok"}, ensure_ascii=False),
                encoding="utf-8",
            )
            record = RunRecord(
                run_id="run-test",
                raw_request="Research test",
                task=TaskSpec(raw_request="Research test", objective="Research", target="test"),
                artifacts={"llm": str(artifact_path)},
            )

            report = inspect_run(record, run_dir)

            self.assertEqual(report["artifact_count"], 1)
            self.assertEqual(report["artifacts"][0]["preview"]["token"], "[REDACTED]")
            self.assertEqual(report["artifacts"][0]["preview"]["message"], "ok")
            self.assertEqual(report["run_files"][0]["path"], "llm.json")

    def test_inspect_run_aggregates_llm_usage_without_exposing_prompts(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run-test"
            llm_dir = run_dir / "llm"
            llm_dir.mkdir(parents=True)
            (llm_dir / "planner.json").write_text(
                json.dumps(
                    {
                        "prompt": None,
                        "prompt_hash": "a" * 64,
                        "prompt_stored": False,
                        "call_metadata": {
                            "provider": "fake",
                            "model": "fake-model",
                            "status": "completed",
                            "error_code": None,
                            "latency_ms": 12.5,
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "total_tokens": 14,
                            "estimated_cost_usd": 0.001,
                            "attempts": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            record = RunRecord(
                run_id="run-test",
                raw_request="Research test",
                task=TaskSpec(
                    raw_request="Research test",
                    objective="Research",
                    target="test",
                ),
            )

            report = inspect_run(record, run_dir)

            self.assertEqual(report["llm_usage"]["call_count"], 1)
            self.assertEqual(report["llm_usage"]["total_tokens"], 14)
            self.assertEqual(report["llm_usage"]["total_latency_ms"], 12.5)
            self.assertEqual(report["llm_usage"]["estimated_cost_usd"], 0.001)
            self.assertEqual(report["llm_usage"]["prompt_suppressed_count"], 1)
