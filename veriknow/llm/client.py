from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any, Protocol

from veriknow.config import Config


class LLMProviderError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class LLMCheckResult:
    provider: str
    model: str
    available: bool
    status: str
    message: str
    base_url: str = ""
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "available": self.available,
            "status": self.status,
            "message": self.message,
            "base_url": self.base_url,
            "error_code": self.error_code,
        }


@dataclass(frozen=True)
class LLMCallMetadata:
    provider: str
    model: str
    status: str
    error_code: str | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "error_code": self.error_code,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "attempts": self.attempts,
        }


class LLMClient(Protocol):
    provider: str
    model: str
    last_call_metadata: LLMCallMetadata | None

    def check(self) -> LLMCheckResult:
        ...

    def generate_text(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        ...

    def generate_json(self, prompt: str, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
        ...

    def classify(self, prompt: str, labels: list[str], *, context: dict[str, Any] | None = None) -> str:
        ...


class StubLLMClient:
    provider = "stub"

    def __init__(self, config: Config):
        self.config = config
        self.model = config.model_name or "stub-model"
        self.last_call_metadata: LLMCallMetadata | None = None

    def check(self) -> LLMCheckResult:
        self._record_call("completed", perf_counter())
        return LLMCheckResult(
            provider=self.provider,
            model=self.model,
            available=True,
            status="available",
            message="Deterministic stub provider is available.",
            base_url="local",
        )

    def generate_text(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        started = perf_counter()
        if context:
            result = json.dumps(
                {"prompt": prompt, "context": context},
                ensure_ascii=False,
                sort_keys=True,
            )
        else:
            result = prompt
        self._record_call("completed", started)
        return result

    def generate_json(self, prompt: str, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
        started = perf_counter()
        result = {"provider": self.provider, "prompt": prompt, "context": context or {}}
        self._record_call("completed", started)
        return result

    def classify(self, prompt: str, labels: list[str], *, context: dict[str, Any] | None = None) -> str:
        started = perf_counter()
        if not labels:
            self._record_call("failed", started, error_code="empty_labels")
            raise LLMProviderError("empty_labels", "at least one classification label is required")
        self._record_call("completed", started)
        return labels[0]

    def _record_call(
        self,
        status: str,
        started: float,
        *,
        error_code: str | None = None,
    ) -> None:
        self.last_call_metadata = LLMCallMetadata(
            provider=self.provider,
            model=self.model,
            status=status,
            error_code=error_code,
            latency_ms=_elapsed_ms(started),
        )


class BigModelLLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.provider = config.model_provider.strip().lower() or "bigmodel"
        self.model = config.model_name
        self.base_url = config.model_base_url.rstrip("/")
        self.api_key_env = config.model_api_key_env
        self.timeout_seconds = config.model_timeout_seconds
        self.max_output_tokens = config.model_max_output_tokens
        self.temperature = config.model_temperature
        self.max_retries = max(0, config.model_max_retries)
        self.retry_backoff_seconds = max(0.0, config.model_retry_backoff_seconds)
        self.last_call_metadata: LLMCallMetadata | None = None
        self._last_request_attempts = 1

    def check(self) -> LLMCheckResult:
        if not os.environ.get(self.api_key_env, ""):
            self.last_call_metadata = LLMCallMetadata(
                provider=self.provider,
                model=self.model,
                status="blocked",
                error_code="missing_api_key",
                latency_ms=0.0,
                attempts=0,
            )
            return LLMCheckResult(
                provider=self.provider,
                model=self.model,
                available=False,
                status="blocked",
                message=f"Missing environment variable: {self.api_key_env}.",
                base_url=self.base_url,
                error_code="missing_api_key",
            )

        try:
            self.generate_text("Reply with ok.", context={"purpose": "veriknow llm check"})
        except LLMProviderError as exc:
            return LLMCheckResult(
                provider=self.provider,
                model=self.model,
                available=False,
                status="failed",
                message=exc.message,
                base_url=self.base_url,
                error_code=exc.code,
            )

        return LLMCheckResult(
            provider=self.provider,
            model=self.model,
            available=True,
            status="available",
            message="BigModel provider responded successfully.",
            base_url=self.base_url,
        )

    def generate_text(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        started = perf_counter()
        self._last_request_attempts = 1
        usage: dict[str, int | None] = {}
        try:
            if not prompt.strip():
                raise LLMProviderError("empty_prompt", "prompt cannot be empty")
            content = prompt.strip()
            if context:
                content = (
                    f"{content}\n\nContext:\n"
                    f"{json.dumps(context, ensure_ascii=False, sort_keys=True)}"
                )

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "max_tokens": self.max_output_tokens,
            }
            data = self._request_json("/chat/completions", payload)
            usage = _usage_from_response(data)
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise LLMProviderError("missing_choices", "model response did not include choices")
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if not isinstance(message, dict):
                raise LLMProviderError("missing_message", "model response did not include a message")
            content_value = message.get("content")
            if content_value is None:
                raise LLMProviderError(
                    "missing_content",
                    "model response did not include message content",
                )
        except LLMProviderError as exc:
            self._record_call("failed", started, usage=usage, error_code=exc.code)
            raise
        except (TypeError, ValueError) as exc:
            self._record_call("failed", started, usage=usage, error_code="invalid_request")
            raise LLMProviderError("invalid_request", str(exc)) from exc

        self._record_call("completed", started, usage=usage)
        return str(content_value)

    def generate_json(self, prompt: str, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
        text = self.generate_text(
            f"{prompt}\n\nReturn only one valid JSON object.",
            context=context,
        )
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            self._mark_last_call_failed("invalid_json")
            raise LLMProviderError("invalid_json", "model response was not valid JSON") from exc
        if not isinstance(value, dict):
            self._mark_last_call_failed("invalid_json_object")
            raise LLMProviderError("invalid_json_object", "model JSON response was not an object")
        return value

    def classify(self, prompt: str, labels: list[str], *, context: dict[str, Any] | None = None) -> str:
        if not labels:
            self.last_call_metadata = LLMCallMetadata(
                provider=self.provider,
                model=self.model,
                status="failed",
                error_code="empty_labels",
                latency_ms=0.0,
                attempts=0,
            )
            raise LLMProviderError("empty_labels", "at least one classification label is required")
        label_text = ", ".join(labels)
        result = self.generate_text(
            f"{prompt}\n\nChoose exactly one label from: {label_text}. Return only the label.",
            context=context,
        ).strip()
        for label in labels:
            if result.lower() == label.lower():
                return label
        self._mark_last_call_failed("invalid_label")
        raise LLMProviderError("invalid_label", f"model returned unsupported label: {result}")

    def _request_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise LLMProviderError("missing_api_key", f"Missing environment variable: {self.api_key_env}.")

        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        raw = ""
        for attempt in range(self.max_retries + 1):
            self._last_request_attempts = attempt + 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                retryable = exc.code == 429 or exc.code >= 500
                if retryable and attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise LLMProviderError(
                    "http_error",
                    f"BigModel HTTP {exc.code}: {detail}",
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise LLMProviderError(
                    "network_error",
                    f"BigModel network error: {exc.reason}",
                ) from exc
            except TimeoutError as exc:
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise LLMProviderError("timeout", "BigModel request timed out") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMProviderError("invalid_json", "BigModel response was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise LLMProviderError("invalid_response", "BigModel response was not a JSON object")

        error = parsed.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or "api_error")
            message = str(error.get("message") or "BigModel API error")
            raise LLMProviderError(code, message)
        code = parsed.get("code")
        if code not in (None, 0, "0"):
            raise LLMProviderError(str(code), str(parsed.get("msg") or parsed.get("message") or "BigModel API error"))
        return parsed

    def _wait_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2**attempt)
        if delay > 0:
            time.sleep(delay)

    def _record_call(
        self,
        status: str,
        started: float,
        *,
        usage: dict[str, int | None] | None = None,
        error_code: str | None = None,
    ) -> None:
        token_usage = usage or {}
        self.last_call_metadata = LLMCallMetadata(
            provider=self.provider,
            model=self.model,
            status=status,
            error_code=error_code,
            latency_ms=_elapsed_ms(started),
            input_tokens=token_usage.get("input_tokens"),
            output_tokens=token_usage.get("output_tokens"),
            total_tokens=token_usage.get("total_tokens"),
            attempts=self._last_request_attempts,
        )

    def _mark_last_call_failed(self, error_code: str) -> None:
        current = self.last_call_metadata
        self.last_call_metadata = LLMCallMetadata(
            provider=self.provider,
            model=self.model,
            status="failed",
            error_code=error_code,
            latency_ms=current.latency_ms if current else None,
            input_tokens=current.input_tokens if current else None,
            output_tokens=current.output_tokens if current else None,
            total_tokens=current.total_tokens if current else None,
            estimated_cost_usd=current.estimated_cost_usd if current else None,
            attempts=current.attempts if current else self._last_request_attempts,
        )


ZhipuLLMClient = BigModelLLMClient


def create_llm_client(config: Config) -> LLMClient:
    provider = config.model_provider.strip().lower()
    if provider == "stub":
        return StubLLMClient(config)
    if provider in {"zhipu", "bigmodel"}:
        return BigModelLLMClient(config)
    raise ValueError(f"unsupported model provider: {config.model_provider}")


def llm_call_metadata(client: LLMClient) -> dict[str, Any]:
    metadata = getattr(client, "last_call_metadata", None)
    if isinstance(metadata, LLMCallMetadata):
        return metadata.to_dict()
    return LLMCallMetadata(
        provider=str(getattr(client, "provider", "unknown")),
        model=str(getattr(client, "model", "")),
        status="unknown",
        attempts=0,
    ).to_dict()


def prompt_persistence(prompt: str, *, store_prompt: bool) -> dict[str, Any]:
    return {
        "prompt": prompt if store_prompt else None,
        "prompt_hash": sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_stored": store_prompt,
    }


def _usage_from_response(data: dict[str, Any]) -> dict[str, int | None]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    input_tokens = _optional_int(usage.get("prompt_tokens", usage.get("input_tokens")))
    output_tokens = _optional_int(
        usage.get("completion_tokens", usage.get("output_tokens"))
    )
    total_tokens = _optional_int(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)
