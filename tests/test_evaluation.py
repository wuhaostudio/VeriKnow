from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import unittest

from veriknow.cli import main
from veriknow.modules.evaluation import (
    evaluate_fixture,
    evaluate_run_artifacts,
    evaluate_safety_cases,
)
from veriknow.schemas import EvidenceClaim, KnowledgeMergeProposal


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

    def test_run_evaluation_validates_merge_and_llm_artifact_contracts(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run-test"
            llm_dir = run_dir / "llm"
            llm_dir.mkdir(parents=True)
            claims = [
                EvidenceClaim(
                    text="The feature is supported.",
                    source_url="https://example.com/docs",
                )
            ]
            (run_dir / "extracted_claims.json").write_text(
                json.dumps([claim.to_dict() for claim in claims]),
                encoding="utf-8",
            )
            proposal = KnowledgeMergeProposal(
                run_id="run-test",
                operation="update",
                target_path="data/knowledge/example.md",
                target_title="Example",
                rationale="Update with cited evidence.",
                evidence_urls=["https://example.com/docs"],
                diff="--- old\n+++ new\n",
                proposed_content="# Example\n\nSee https://example.com/docs.\n",
                base_content_hash="a" * 64,
            )
            (run_dir / "knowledge_merge_proposal.json").write_text(
                json.dumps(proposal.to_dict()),
                encoding="utf-8",
            )
            (llm_dir / "planner.json").write_text(
                json.dumps(
                    {
                        "prompt": None,
                        "prompt_hash": "b" * 64,
                        "prompt_stored": False,
                        "call_metadata": {
                            "provider": "fake",
                            "model": "fake-model",
                            "status": "completed",
                            "error_code": None,
                            "latency_ms": 4.0,
                            "input_tokens": None,
                            "output_tokens": None,
                            "total_tokens": None,
                            "estimated_cost_usd": None,
                            "attempts": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = evaluate_run_artifacts(run_dir)

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["details"]["llm"]["artifact_count"], 1)
            names = {check["name"] for check in result["checks"]}
            self.assertIn("merge_proposal_has_proposed_content", names)
            self.assertIn("llm_metadata_planner", names)
