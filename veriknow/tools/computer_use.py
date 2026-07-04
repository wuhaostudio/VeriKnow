from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from veriknow.tools.computer_runtime import ComputerRuntime, TraceOnlyRuntime


DEFAULT_APPROVAL_KEYWORDS = (
    "login",
    "sign in",
    "password",
    "billing",
    "payment",
    "purchase",
    "delete",
    "destructive",
    "account change",
    "account settings",
)


@dataclass(frozen=True)
class ComputerUseSafetyConfig:
    allowed_domains: tuple[str, ...] = ()
    approval_keywords: tuple[str, ...] = DEFAULT_APPROVAL_KEYWORDS
    max_steps: int = 12
    max_seconds: int = 180
    read_only: bool = True
    store_screenshots: bool = True
    require_approval_for_forms: bool = True

    def is_domain_allowed(self, url: str) -> bool:
        host = urlparse(url).hostname
        if not host:
            return False
        normalized_host = host.lower()
        for domain in self.allowed_domains:
            normalized_domain = domain.lower().strip()
            if not normalized_domain:
                continue
            if normalized_host == normalized_domain or normalized_host.endswith(f".{normalized_domain}"):
                return True
        return False

    def approval_reason(self, text: str) -> str | None:
        lowered = text.lower()
        for keyword in self.approval_keywords:
            normalized = keyword.lower().strip()
            if normalized and normalized in lowered:
                return f"approval keyword matched: {keyword}"
        return None


@dataclass(frozen=True)
class ComputerUseObservation:
    status: str
    actual_result: str
    screenshot_path: str | None = None
    log_path: str | None = None
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


class ComputerUseVerifier:
    def __init__(
        self,
        safety: ComputerUseSafetyConfig | None = None,
        runtime: ComputerRuntime | None = None,
    ):
        self.safety = safety or ComputerUseSafetyConfig()
        self.runtime = runtime or TraceOnlyRuntime()

    def verify_step(
        self,
        url: str,
        *,
        instruction: str,
        expected_result: str,
        screenshot_path: Path,
        log_path: Path,
        allow_approval_required: bool = False,
    ) -> ComputerUseObservation:
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        actions = [
            "open isolated computer-use browser",
            f"navigate to {url}",
            f"observe expected result: {expected_result}",
        ]
        observations = [
            f"instruction={instruction}",
            f"url={url}",
            f"runtime={self.runtime.name}",
            f"read_only={self.safety.read_only}",
            f"max_steps={self.safety.max_steps}",
            f"max_seconds={self.safety.max_seconds}",
        ]

        if not self.safety.is_domain_allowed(url):
            reason = "Domain is not in computer-use allowlist."
            self._write_log(log_path, url, expected_result, actions, observations, "blocked", reason)
            return ComputerUseObservation(
                status="blocked",
                actual_result=reason,
                log_path=str(log_path),
                actions=actions,
                observations=observations,
            )

        approval_reason = self.safety.approval_reason(f"{instruction} {expected_result}")
        if approval_reason and not allow_approval_required:
            reason = f"Computer-use step requires explicit approval because {approval_reason}."
            self._write_log(log_path, url, expected_result, actions, observations, "blocked", reason)
            return ComputerUseObservation(
                status="blocked",
                actual_result=reason,
                log_path=str(log_path),
                actions=actions,
                observations=observations,
            )

        try:
            runtime_observation = self.runtime.inspect_url(
                url,
                expected_result=expected_result,
                screenshot_path=screenshot_path,
                log_path=log_path,
            )
        except Exception as exc:
            runtime_observation = TraceOnlyRuntime().inspect_url(
                url,
                expected_result=expected_result,
                screenshot_path=screenshot_path,
                log_path=log_path,
            )
            observations.append(f"runtime error: {exc.__class__.__name__}: {exc}")

        if runtime_observation.observations:
            observations.extend(runtime_observation.observations)
        reason = self._result_message(runtime_observation)
        self._write_log(log_path, url, expected_result, actions, observations, runtime_observation.status, reason)
        return ComputerUseObservation(
            status=runtime_observation.status,
            actual_result=reason,
            screenshot_path=runtime_observation.screenshot_path,
            log_path=str(log_path),
            actions=actions,
            observations=observations,
        )

    def _result_message(self, observation) -> str:
        if observation.status == "passed":
            return (
                f"Computer-use runtime opened {observation.final_url} "
                f"with HTTP status {observation.http_status}; page title: {observation.title}"
            )
        if self.runtime.name == "trace-only":
            return (
                "Computer-use verification boundary recorded the planned action sequence; "
                "no live computer-use runtime is configured in this local build."
            )
        return f"Computer-use runtime recorded a {observation.status} result for {observation.final_url}."

    def _write_log(
        self,
        log_path: Path,
        url: str,
        expected_result: str,
        actions: list[str],
        observations: list[str],
        status: str,
        reason: str,
    ) -> None:
        lines = [
            "mode=computer-use",
            "isolation=required",
            f"url={url}",
            f"expected_result={expected_result}",
            f"status={status}",
            f"reason={reason}",
        ]
        lines.extend(f"action={action}" for action in actions)
        lines.extend(f"observation={observation}" for observation in observations)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

