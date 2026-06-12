from pathlib import Path
import json
import unittest

from veriknow.schemas import (
    EvidenceBundle,
    EvidenceItem,
    RunRecord,
    TaskSpec,
    VerificationPlan,
    VerificationResult,
    VerificationRun,
    VerificationStep,
)
from veriknow.tools.markdown import render_placeholder_report, render_report


class MarkdownTests(unittest.TestCase):
    def test_report_includes_evidence_sources_when_available(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            evidence = EvidenceBundle(
                task_id="run-test",
                summary="Collected public evidence.",
                items=[
                    EvidenceItem(
                        title="Official docs",
                        url="https://example.com/docs",
                        source_type="official_doc",
                        snippet="Official source.",
                        confidence="high",
                    )
                ],
            )
            evidence_path = tmp_path / "evidence.json"
            evidence_path.write_text(
                json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            record = RunRecord(
                run_id="run-test",
                raw_request="Research example",
                task=TaskSpec(
                    raw_request="Research example",
                    objective="Research",
                    target="example",
                ),
                artifacts={"evidence": str(evidence_path)},
            )

            report = render_placeholder_report(record)

            self.assertIn('status: "partial"', report)
            self.assertIn("Collected public evidence.", report)
            self.assertIn("[Official docs](https://example.com/docs)", report)
            self.assertIn("confidence: high", report)

    def test_report_includes_phase_five_operation_manual_sections(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            evidence = EvidenceBundle(
                task_id="run-test",
                summary="Use the official docs as the primary source.",
                items=[
                    EvidenceItem(
                        title="Official docs",
                        url="https://example.com/docs",
                        source_type="official_doc",
                        snippet="Official setup workflow.",
                        confidence="high",
                    )
                ],
            )
            plan = VerificationPlan(
                task_id="run-test",
                steps=[
                    VerificationStep(
                        description="Open source and verify setup workflow.",
                        expected_result="The source supports the workflow. URL: https://example.com/docs",
                        method="browser",
                        tools=["browser"],
                        screenshot_required=True,
                    ),
                    VerificationStep(
                        description="Compare collected sources for conflicts.",
                        expected_result="No conflicts remain.",
                        method="manual",
                        tools=["human_review"],
                    ),
                ],
            )
            verification = VerificationRun(
                task_id="run-test",
                status="partial",
                results=[
                    VerificationResult(
                        step_description="Open source and verify setup workflow.",
                        status="partial",
                        actual_result="Fallback browser trace was recorded.",
                        screenshot_path=str(run_dir / "screenshots" / "step-01.png"),
                        log_path=str(run_dir / "logs" / "step-01.log"),
                    ),
                    VerificationResult(
                        step_description="Compare collected sources for conflicts.",
                        status="manual",
                        actual_result="Manual checkpoint recorded.",
                    ),
                ],
                completed_at="2026-06-11T04:03:37+00:00",
            )
            evidence_path = run_dir / "evidence.json"
            plan_path = run_dir / "verification_plan.json"
            verification_path = run_dir / "verification.json"
            evidence_path.write_text(json.dumps(evidence.to_dict(), ensure_ascii=False), encoding="utf-8")
            plan_path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False), encoding="utf-8")
            verification_path.write_text(
                json.dumps(verification.to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
            record = RunRecord(
                run_id="run-test",
                raw_request="Research example",
                task=TaskSpec(
                    raw_request="Research example",
                    objective="Research",
                    target="example",
                ),
                artifacts={
                    "evidence": str(evidence_path),
                    "verification_plan": str(plan_path),
                    "verification": str(verification_path),
                },
            )

            report = render_report(record, run_dir)

            self.assertIn('verified_at: "2026-06-11T04:03:37+00:00"', report)
            self.assertIn('next_verify_at: "2026-07-11"', report)
            self.assertIn("## Step-by-Step Guide", report)
            self.assertIn("Verification status: partial", report)
            self.assertIn("![Step 1 screenshot](screenshots/step-01.png)", report)
            self.assertIn("- logs/step-01.log", report)
            self.assertIn("## Outdated or Unsupported Information", report)
            self.assertIn("Source date unknown", report)
            self.assertIn("## Manual Checkpoints", report)
            self.assertIn("Manual checkpoint recorded.", report)
