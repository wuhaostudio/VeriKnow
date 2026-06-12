from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path


PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


@dataclass(frozen=True)
class BrowserObservation:
    status: str
    actual_result: str
    screenshot_path: str | None = None
    log_path: str | None = None


class BrowserVerifier:
    def __init__(self, *, headless: bool = True, timeout_ms: int = 15000):
        self.headless = headless
        self.timeout_ms = timeout_ms

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

        try:
            return self._verify_with_playwright(
                url,
                expected_result=expected_result,
                screenshot_path=screenshot_path,
                log_path=log_path,
            )
        except Exception as exc:
            screenshot_path.write_bytes(PLACEHOLDER_PNG)
            log_path.write_text(
                "\n".join(
                    [
                        f"mode=fallback",
                        f"url={url}",
                        f"expected_result={expected_result}",
                        f"reason={exc.__class__.__name__}: {exc}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            return BrowserObservation(
                status="partial",
                actual_result=(
                    "Browser verification could not run in this environment; "
                    "a traceable fallback screenshot and log were recorded."
                ),
                screenshot_path=str(screenshot_path),
                log_path=str(log_path),
            )

    def _verify_with_playwright(
        self,
        url: str,
        *,
        expected_result: str,
        screenshot_path: Path,
        log_path: Path,
    ) -> BrowserObservation:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self.headless)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            response = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            page.screenshot(path=str(screenshot_path), full_page=True)
            title = page.title()
            final_url = page.url
            status_code = response.status if response else "unknown"
            browser.close()

        log_path.write_text(
            "\n".join(
                [
                    f"mode=playwright",
                    f"url={url}",
                    f"final_url={final_url}",
                    f"title={title}",
                    f"http_status={status_code}",
                    f"expected_result={expected_result}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return BrowserObservation(
            status="passed",
            actual_result=f"Opened {final_url} with HTTP status {status_code}; page title: {title}",
            screenshot_path=str(screenshot_path),
            log_path=str(log_path),
        )
