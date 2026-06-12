from pathlib import Path
import unittest

from veriknow.modules.verifier import Verifier
from veriknow.schemas import VerificationPlan, VerificationStep
from veriknow.tools.browser import BrowserObservation
from veriknow.tools.computer_use import ComputerUseSafetyConfig, ComputerUseVerifier


class FakeBrowser:
    def verify_url(
        self,
        url: str,
        *,
        expected_result: str,
        screenshot_path: Path,
        log_path: Path,
    ) -> BrowserObservation:
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_path.write_bytes(b"fake png")
        log_path.write_text(f"url={url}\nexpected={expected_result}\n", encoding="utf-8")
        return BrowserObservation(
            status="passed",
            actual_result=f"Verified {url}",
            screenshot_path=str(screenshot_path),
            log_path=str(log_path),
        )


class VerifierTests(unittest.TestCase):
    def test_verifier_executes_browser_steps_and_writes_artifacts(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            plan = VerificationPlan(
                task_id="run-test",
                steps=[
                    VerificationStep(
                        description="Open source",
                        expected_result="URL: https://example.com/docs",
                        method="browser",
                        tools=["browser"],
                        screenshot_required=True,
                    ),
                    VerificationStep(
                        description="Manual conflict check",
                        expected_result="No conflicts remain.",
                        method="manual",
                        tools=["human_review"],
                    ),
                ],
            )

            run = Verifier(FakeBrowser()).verify(plan, run_dir=run_dir)

            self.assertEqual(run.status, "completed")
            self.assertEqual(run.results[0].status, "passed")
            self.assertEqual(run.results[1].status, "manual")
            self.assertTrue((run_dir / "screenshots" / "step-01.png").exists())
            self.assertTrue((run_dir / "logs" / "step-01.log").exists())

    def test_verifier_skips_approval_required_steps_by_default(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            plan = VerificationPlan(
                task_id="run-test",
                steps=[
                    VerificationStep(
                        description="Open billing console",
                        expected_result="URL: https://example.com/billing",
                        method="browser",
                        requires_approval=True,
                    )
                ],
            )

            run = Verifier(FakeBrowser()).verify(plan, run_dir=Path(directory))

            self.assertEqual(run.status, "partial")
            self.assertEqual(run.results[0].status, "skipped")

    def test_verifier_executes_computer_use_mode_with_safety_trace(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            plan = VerificationPlan(
                task_id="run-test",
                steps=[
                    VerificationStep(
                        description="Open documentation in isolated browser",
                        expected_result="URL: https://example.com/docs",
                        method="browser",
                        tools=["browser"],
                        screenshot_required=True,
                    )
                ],
            )
            computer_use = ComputerUseVerifier(
                ComputerUseSafetyConfig(allowed_domains=("example.com",))
            )

            run = Verifier(FakeBrowser(), computer_use).verify(
                plan,
                run_dir=run_dir,
                mode="computer-use",
            )

            self.assertEqual(run.status, "partial")
            self.assertEqual(run.results[0].status, "partial")
            self.assertIn("open isolated computer-use browser", run.results[0].actions)
            self.assertTrue((run_dir / "screenshots" / "step-01.png").exists())
            self.assertTrue((run_dir / "logs" / "step-01.log").exists())

    def test_verifier_blocks_computer_use_when_domain_is_not_allowed(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            plan = VerificationPlan(
                task_id="run-test",
                steps=[
                    VerificationStep(
                        description="Open external documentation",
                        expected_result="URL: https://blocked.example/docs",
                        method="browser",
                    )
                ],
            )

            run = Verifier(FakeBrowser(), ComputerUseVerifier()).verify(
                plan,
                run_dir=Path(directory),
                mode="computer-use",
            )

            self.assertEqual(run.status, "blocked")
            self.assertEqual(run.results[0].status, "blocked")
            self.assertIn("allowlist", run.results[0].actual_result)
