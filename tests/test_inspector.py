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
