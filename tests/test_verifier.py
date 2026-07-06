from pathlib import Path
import unittest

from veriknow.modules.verifier import Verifier
from veriknow.schemas import VerificationPlan, VerificationStep
from veriknow.tools.browser import BrowserObservation
from veriknow.tools.computer_runtime import (
    PlaywrightComputerRuntime,
    RuntimeObservation,
    create_computer_runtime,
)
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


class FakeComputerRuntime:
    name = "fake-runtime"

    def inspect_url(
        self,
        url: str,
        *,
        expected_result: str,
        screenshot_path: Path,
        log_path: Path,
    ) -> RuntimeObservation:
        screenshot_path.write_bytes(b"fake computer screenshot")
        return RuntimeObservation(
            status="passed",
            final_url=url,
            title="Fake Docs",
            http_status="200",
            screenshot_path=str(screenshot_path),
            log_path=str(log_path),
            observations=[f"expected={expected_result}"],
        )


class RiskyComputerRuntime:
    name = "risky-runtime"

    def inspect_url(
        self,
        url: str,
        *,
        expected_result: str,
        screenshot_path: Path,
        log_path: Path,
    ) -> RuntimeObservation:
        screenshot_path.write_bytes(b"fake risky screenshot")
        return RuntimeObservation(
            status="passed",
            final_url=url,
            title="Login",
            http_status="200",
            screenshot_path=str(screenshot_path),
            log_path=str(log_path),
            observations=["form_count=2", "body text includes login"],
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
            self.assertTrue(any("action_trace=" in item for item in run.results[0].observations))
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

    def test_verifier_uses_configured_computer_runtime(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            plan = VerificationPlan(
                task_id="run-test",
                steps=[
                    VerificationStep(
                        description="Open documentation in runtime browser",
                        expected_result="URL: https://example.com/docs",
                        method="computer-use",
                        screenshot_required=True,
                    )
                ],
            )
            computer_use = ComputerUseVerifier(
                ComputerUseSafetyConfig(allowed_domains=("example.com",)),
                FakeComputerRuntime(),
            )

            run = Verifier(FakeBrowser(), computer_use).verify(
                plan,
                run_dir=run_dir,
                mode="computer-use",
            )

            self.assertEqual(run.status, "completed")
            self.assertEqual(run.results[0].status, "passed")
            self.assertIn("runtime=fake-runtime", run.results[0].observations)
            self.assertIn("Fake Docs", run.results[0].actual_result)

    def test_create_computer_runtime_passes_safety_limits_to_playwright_runtime(self) -> None:
        runtime = create_computer_runtime(
            "playwright", max_steps=5, max_seconds=30, store_screenshots=False
        )

        self.assertIsInstance(runtime, PlaywrightComputerRuntime)
        self.assertEqual(runtime.max_steps, 5)
        self.assertEqual(runtime.max_seconds, 30)
        self.assertFalse(runtime.store_screenshots)

    def test_computer_use_blocks_runtime_observed_forms_by_default(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            plan = VerificationPlan(
                task_id="run-test",
                steps=[
                    VerificationStep(
                        description="Open documentation",
                        expected_result="URL: https://example.com/docs",
                        method="computer-use",
                    )
                ],
            )
            computer_use = ComputerUseVerifier(
                ComputerUseSafetyConfig(allowed_domains=("example.com",)),
                RiskyComputerRuntime(),
            )

            run = Verifier(FakeBrowser(), computer_use).verify(
                plan,
                run_dir=Path(directory),
                mode="computer-use",
            )

            self.assertEqual(run.status, "blocked")
            self.assertEqual(run.results[0].status, "blocked")
            self.assertIn("explicit approval", run.results[0].actual_result)
