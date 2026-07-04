from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from veriknow.tools.browser import PLACEHOLDER_PNG


@dataclass(frozen=True)
class RuntimeObservation:
    status: str
    final_url: str
    title: str = ""
    http_status: str = "unknown"
    screenshot_path: str | None = None
    log_path: str | None = None
    observations: list[str] | None = None


class ComputerRuntime(Protocol):
    name: str

    def inspect_url(
        self,
        url: str,
        *,
        expected_result: str,
        screenshot_path: Path,
        log_path: Path,
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
    ) -> RuntimeObservation:
        screenshot_path.write_bytes(PLACEHOLDER_PNG)
        return RuntimeObservation(
            status="partial",
            final_url=url,
            screenshot_path=str(screenshot_path),
            log_path=str(log_path),
            observations=[
                "computer-use runtime is not configured; traceable boundary recorded",
                f"expected_result={expected_result}",
            ],
        )


class PlaywrightComputerRuntime:
    name = "playwright"

    def __init__(self, *, headless: bool = True, timeout_ms: int = 15000):
        self.headless = headless
        self.timeout_ms = timeout_ms

    def inspect_url(
        self,
        url: str,
        *,
        expected_result: str,
        screenshot_path: Path,
        log_path: Path,
    ) -> RuntimeObservation:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self.headless)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            response = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            page.screenshot(path=str(screenshot_path), full_page=True)
            title = page.title()
            final_url = page.url
            status_code = str(response.status if response else "unknown")
            browser.close()

        return RuntimeObservation(
            status="passed",
            final_url=final_url,
            title=title,
            http_status=status_code,
            screenshot_path=str(screenshot_path),
            log_path=str(log_path),
            observations=[
                f"page title: {title}",
                f"http status: {status_code}",
                f"expected_result={expected_result}",
            ],
        )


def create_computer_runtime(name: str) -> ComputerRuntime:
    normalized = name.strip().lower()
    if normalized in {"", "trace-only", "stub"}:
        return TraceOnlyRuntime()
    if normalized == "playwright":
        return PlaywrightComputerRuntime()
    raise ValueError(f"unsupported computer-use runtime: {name}")
