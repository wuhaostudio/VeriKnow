from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from veriknow.tools.browser import PLACEHOLDER_PNG


@dataclass(frozen=True)
class ComputerAction:
    action: str
    target: str = ""
    text: str = ""
    reason: str = ""
    requires_approval: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "target": self.target,
            "text": self.text,
            "reason": self.reason,
            "requires_approval": self.requires_approval,
        }

    def to_observation_line(self) -> str:
        return "action_proposal=" + json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass(frozen=True)
class RuntimeActionTrace:
    index: int
    action: str
    target: str = ""
    text: str = ""
    reason: str = ""
    safety_decision: str = "allowed"
    runtime_result: str = ""
    screenshot_path: str | None = None

    def to_observation_line(self) -> str:
        return "action_trace=" + json.dumps(
            {
                "index": self.index,
                "action": self.action,
                "target": self.target,
                "text": self.text,
                "reason": self.reason,
                "safety_decision": self.safety_decision,
                "runtime_result": self.runtime_result,
                "screenshot_path": self.screenshot_path,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass(frozen=True)
class RuntimeObservation:
    status: str
    final_url: str
    title: str = ""
    http_status: str = "unknown"
    screenshot_path: str | None = None
    log_path: str | None = None
    observations: list[str] | None = None
    action_traces: list[RuntimeActionTrace] | None = None


class ComputerRuntime(Protocol):
    name: str

    def inspect_url(
        self,
        url: str,
        *,
        expected_result: str,
        screenshot_path: Path,
        log_path: Path,
        action_plan: list[ComputerAction] | None = None,
    ) -> RuntimeObservation:
        ...


class TraceOnlyRuntime:
    name = "trace-only"

    def inspect_url(
        self,
        url: str,
        *,
        expected_result: str,
        screenshot_path: Path,
        log_path: Path,
        action_plan: list[ComputerAction] | None = None,
    ) -> RuntimeObservation:
        screenshot_path.write_bytes(PLACEHOLDER_PNG)
        traces = [
            RuntimeActionTrace(
                index=1,
                action="open",
                target=url,
                reason="record planned navigation without live runtime",
                safety_decision="allowed: domain and approval policy checked before runtime",
                runtime_result="not executed",
            ),
            RuntimeActionTrace(
                index=2,
                action="screenshot",
                target=str(screenshot_path),
                reason="store placeholder screenshot for traceability",
                safety_decision="allowed: local artifact write",
                runtime_result="placeholder written",
                screenshot_path=str(screenshot_path),
            ),
            RuntimeActionTrace(
                index=3,
                action="finish",
                reason="trace-only runtime cannot inspect page content",
                safety_decision="allowed",
                runtime_result="partial",
            ),
        ]
        return RuntimeObservation(
            status="partial",
            final_url=url,
            screenshot_path=str(screenshot_path),
            log_path=str(log_path),
            observations=[
                "computer-use runtime is not configured; traceable boundary recorded",
                f"expected_result={expected_result}",
                *[action.to_observation_line() for action in action_plan or []],
                *[trace.to_observation_line() for trace in traces],
            ],
            action_traces=traces,
        )


class PlaywrightComputerRuntime:
    name = "playwright"

    def __init__(
        self,
        *,
        headless: bool = True,
        timeout_ms: int = 15000,
        max_steps: int = 12,
        max_seconds: int = 180,
        store_screenshots: bool = True,
    ):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.max_steps = max(1, max_steps)
        self.max_seconds = max(1, max_seconds)
        self.store_screenshots = store_screenshots

    def inspect_url(
        self,
        url: str,
        *,
        expected_result: str,
        screenshot_path: Path,
        log_path: Path,
        action_plan: list[ComputerAction] | None = None,
    ) -> RuntimeObservation:
        from playwright.sync_api import sync_playwright

        started = time.monotonic()
        traces: list[RuntimeActionTrace] = []
        action_plan = action_plan or [
            ComputerAction("open", target=url, reason="navigate to verification source URL"),
            ComputerAction("screenshot", target=str(screenshot_path), reason="capture public page"),
            ComputerAction("finish", reason="finish read-only inspection"),
        ]

        def add_trace(
            action: str,
            *,
            target: str = "",
            text: str = "",
            reason: str = "",
            runtime_result: str = "",
            screenshot: Path | None = None,
        ) -> None:
            if len(traces) >= self.max_steps:
                return
            traces.append(
                RuntimeActionTrace(
                    index=len(traces) + 1,
                    action=action,
                    target=target,
                    text=text,
                    reason=reason,
                    safety_decision="allowed: read-only runtime action",
                    runtime_result=runtime_result,
                    screenshot_path=str(screenshot) if screenshot is not None else None,
                )
            )

        def screenshot_for(sequence: int, suffix: str) -> Path:
            return screenshot_path.with_name(
                f"{screenshot_path.stem}-{sequence:02d}-{suffix}{screenshot_path.suffix}"
            )

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self.headless)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            response = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            status_code = str(response.status if response else "unknown")
            add_trace(
                "open",
                target=url,
                reason="navigate to verification source URL",
                runtime_result=f"http_status={status_code}",
            )

            if self.store_screenshots and len(traces) < self.max_steps:
                open_screenshot = screenshot_for(len(traces) + 1, "open")
                page.screenshot(path=str(open_screenshot), full_page=True)
                add_trace(
                    "screenshot",
                    target=str(open_screenshot),
                    reason="capture page after navigation",
                    runtime_result="captured",
                    screenshot=open_screenshot,
                )

            title = page.title()
            final_url = page.url
            body_text = ""
            try:
                body_text = page.locator("body").inner_text(timeout=3000)
            except Exception as exc:
                add_trace(
                    "observe",
                    target="body",
                    reason="extract visible text for expected-result comparison",
                    runtime_result=f"{exc.__class__.__name__}: {exc}",
                )

            if body_text and len(traces) < self.max_steps:
                add_trace(
                    "observe",
                    target="body",
                    text=_compact_text(body_text),
                    reason="inspect visible page text",
                    runtime_result=f"characters={len(body_text)}",
                )

            form_count = _form_count(page)
            if form_count and len(traces) < self.max_steps:
                add_trace(
                    "observe",
                    target="forms",
                    reason="detect interactive forms without submitting data",
                    runtime_result=f"form_count={form_count}",
                )

            if self.store_screenshots and len(traces) + 1 < self.max_steps and _is_scrollable(page):
                page.mouse.wheel(0, 900)
                page.wait_for_timeout(500)
                scrolled_screenshot = screenshot_for(len(traces) + 1, "scroll")
                page.screenshot(path=str(scrolled_screenshot), full_page=True)
                add_trace(
                    "scroll",
                    target="page",
                    reason="read-only scroll to inspect additional public content",
                    runtime_result="scrolled down",
                    screenshot=scrolled_screenshot,
                )

            finish_screenshot = None
            if self.store_screenshots:
                page.screenshot(path=str(screenshot_path), full_page=True)
                finish_screenshot = screenshot_path
            add_trace(
                "finish",
                reason="finish read-only page inspection",
                runtime_result="completed",
                screenshot=finish_screenshot,
            )
            browser.close()

        elapsed = time.monotonic() - started
        action_budget_exhausted = len(traces) >= self.max_steps and traces[-1].action != "finish"
        status = "passed"
        if status_code.isdigit() and int(status_code) >= 400:
            status = "failed"
        elif elapsed > self.max_seconds or action_budget_exhausted:
            status = "partial"

        return RuntimeObservation(
            status=status,
            final_url=final_url,
            title=title,
            http_status=status_code,
            screenshot_path=str(screenshot_path) if self.store_screenshots else None,
            log_path=str(log_path),
            observations=[
                f"page title: {title}",
                f"http status: {status_code}",
                f"expected_result={expected_result}",
                f"elapsed_seconds={elapsed:.2f}",
                f"action_count={len(traces)}",
                f"form_count={form_count}",
                *[action.to_observation_line() for action in action_plan],
                *[trace.to_observation_line() for trace in traces],
            ],
            action_traces=traces,
        )


def _compact_text(text: str, *, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _is_scrollable(page) -> bool:
    try:
        return bool(page.evaluate("() => document.documentElement.scrollHeight > window.innerHeight"))
    except Exception:
        return False


def _form_count(page) -> int:
    try:
        return int(page.evaluate("() => document.querySelectorAll('form, input, textarea, select').length"))
    except Exception:
        return 0


def create_computer_runtime(
    name: str,
    *,
    max_steps: int = 12,
    max_seconds: int = 180,
    store_screenshots: bool = True,
) -> ComputerRuntime:
    normalized = name.strip().lower()
    if normalized in {"", "trace-only", "stub"}:
        return TraceOnlyRuntime()
    if normalized == "playwright":
        return PlaywrightComputerRuntime(
            max_steps=max_steps,
            max_seconds=max_seconds,
            store_screenshots=store_screenshots,
        )
    raise ValueError(f"unsupported computer-use runtime: {name}")
