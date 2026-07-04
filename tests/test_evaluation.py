from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import unittest

from veriknow.cli import main
from veriknow.modules.evaluation import evaluate_fixture, evaluate_safety_cases


class EvaluationTests(unittest.TestCase):
    def test_phase13_fixture_evaluation_passes(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "phase13_metadata_eval.json"

        result = evaluate_fixture(fixture_path)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["kind"], "fixture")
        self.assertEqual(result["failed_count"], 0)
        self.assertGreaterEqual(result["details"]["claim_count"], 1)
        self.assertGreaterEqual(result["details"]["conflict_count"], 1)

    def test_safety_case_evaluation_checks_approval_keywords(self) -> None:
        result = evaluate_safety_cases()

        self.assertEqual(result["status"], "passed")
        names = {check["name"] for check in result["checks"]}
        self.assertIn("safety_login_requires_approval", names)
        self.assertIn("safety_read_only_docs_allowed", names)

    def test_eval_command_outputs_fixture_report(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "phase13_metadata_eval.json"
        stdout = StringIO()

        with redirect_stdout(stdout):
            main(["eval", str(fixture_path)])

        output = json.loads(stdout.getvalue())
        self.assertEqual(output["status"], "passed")
        self.assertEqual(output["kind"], "fixture")