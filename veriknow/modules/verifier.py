from __future__ import annotations

import re
from pathlib import Path

from veriknow.schemas import (
    VerificationPlan,
    VerificationResult,
    VerificationRun,
    VerificationStep,
    now_iso,
)
from veriknow.tools.browser import BrowserVerifier
from veriknow.tools.computer_use import ComputerUseVerifier


URL_PATTERN = re.compile(r"https?://[^\s)]+")


class Verifier:
    def __init__(
        self,
        browser: BrowserVerifier | None = None,
        computer_use: ComputerUseVerifier | None = None,
    ):
        self.browser = browser or BrowserVerifier()
        self.computer_use = computer_use or ComputerUseVerifier()

    def verify(
        self,
        plan: VerificationPlan,
        *,
        run_dir: Path,
        include_approval_required: bool = False,
        mode: str = "browser",
    ) -> VerificationRun:
        if mode not in {"browser", "computer-use"}:
            raise ValueError(f"unsupported verification mode: {mode}")

        results: list[VerificationResult] = []
        screenshots_dir = run_dir / "screenshots"
        logs_dir = run_dir / "logs"

        for index, step in enumerate(plan.steps, start=1):
            if step.requires_approval and not include_approval_required:
                results.append(self._skipped_result(step, "Step requires approval."))
                continue
            if mode == "computer-use" and step.method in {"browser", "computer-use"}:
                results.append(
                    self._verify_computer_use_step(
                        step,
                        index=index,
                        screenshots_dir=screenshots_dir,
                        logs_dir=logs_dir,
                        include_approval_required=include_approval_required,
                    )
                )
                continue
            if mode == "browser" and step.method == "browser":
                results.append(
                    self._verify_browser_step(
                        step,
                        index=index,
                        screenshots_dir=screenshots_dir,
                        logs_dir=logs_dir,
                    )
                )
                continue
            results.append(self._manual_result(step))

        status = self._status_for(results)
        return VerificationRun(
            task_id=plan.task_id,
            status=status,
            results=results,
            completed_at=now_iso(),
        )

    def _verify_browser_step(
        self,
        step: VerificationStep,
        *,
        index: int,
        screenshots_dir: Path,
        logs_dir: Path,
    ) -> VerificationResult:
        url = self._url_for(step)
        if url is None:
            return VerificationResult(
                step_description=step.description,
                status="failed",
                actual_result="No URL found in browser verification step.",
            )

        screenshot_path = screenshots_dir / f"step-{index:02d}.png"
        log_path = logs_dir / f"step-{index:02d}.log"
        observation = self.browser.verify_url(
            url,
            expected_result=step.expected_result,
            screenshot_path=screenshot_path,
            log_path=log_path,
        )
        return VerificationResult(
            step_description=step.description,
            status=observation.status,
            actual_result=observation.actual_result,
            screenshot_path=observation.screenshot_path,
            log_path=observation.log_path,
        )

    def _verify_computer_use_step(
        self,
        step: VerificationStep,
        *,
        index: int,
        screenshots_dir: Path,
        logs_dir: Path,
        include_approval_required: bool,
    ) -> VerificationResult:
        url = self._url_for(step)
        if url is None:
            return VerificationResult(
                step_description=step.description,
                status="failed",
                actual_result="No URL found in computer-use verification step.",
            )

        screenshot_path = screenshots_dir / f"step-{index:02d}.png"
        log_path = logs_dir / f"step-{index:02d}.log"
        observation = self.computer_use.verify_step(
            url,
            instruction=step.description,
            expected_result=step.expected_result,
            screenshot_path=screenshot_path,
            log_path=log_path,
            allow_approval_required=include_approval_required,
        )
        return VerificationResult(
            step_description=step.description,
            status=observation.status,
            actual_result=observation.actual_result,
            screenshot_path=observation.screenshot_path,
            log_path=observation.log_path,
            actions=observation.actions,
            observations=observation.observations,
        )

    def _manual_result(self, step: VerificationStep) -> VerificationResult:
        return VerificationResult(
            step_description=step.description,
            status="manual",
            actual_result="Manual checkpoint recorded; no automated action was executed.",
        )

    def _skipped_result(self, step: VerificationStep, reason: str) -> VerificationResult:
        return VerificationResult(
            step_description=step.description,
            status="skipped",
            actual_result=reason,
        )

    def _url_for(self, step: VerificationStep) -> str | None:
        haystack = f"{step.expected_result} {step.description}"
        match = URL_PATTERN.search(haystack)
        return match.group(0) if match else None

    def _status_for(self, results: list[VerificationResult]) -> str:
        statuses = {result.status for result in results}
        if "failed" in statuses:
            return "failed"
        if "blocked" in statuses:
            return "blocked"
        if statuses <= {"passed", "manual"}:
            return "completed"
        if statuses <= {"passed", "partial", "manual", "skipped"}:
            return "partial"
        return "pending"
