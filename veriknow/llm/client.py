from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
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


class LLMClient(Protocol):
    provider: str
    model: str

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

    def check(self) -> LLMCheckResult:
        return LLMCheckResult(
            provider=self.provider,
            model=self.model,
            available=True,
            status="available",
            message="Deterministic stub provider is available.",
            base_url="local",
        )

    def generate_text(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        if context:
            return json.dumps({"prompt": prompt, "context": context}, ensure_ascii=False, sort_keys=True)
        return prompt

    def generate_json(self, prompt: str, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"provider": self.provider, "prompt": prompt, "context": context or {}}

    def classify(self, prompt: str, labels: list[str], *, context: dict[str, Any] | None = None) -> str:
        if not labels:
            raise LLMProviderError("empty_labels", "at least one classification label is required")
        return labels[0]


class ZhipuLLMClient:
    provider = "zhipu"

    def __init__(self, config: Config):
        self.config = config
        self.model = config.model_name
        self.base_url = config.model_base_url.rstrip("/")
        self.api_key_env = config.model_api_key_env
        self.timeout_seconds = config.model_timeout_seconds
        self.max_output_tokens = config.model_max_output_tokens
        self.temperature = config.model_temperature

    def check(self) -> LLMCheckResult:
        if not os.environ.get(self.api_key_env, ""):
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
            message="Zhipu provider responded successfully.",
            base_url=self.base_url,
        )

    def generate_text(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        if not prompt.strip():
            raise LLMProviderError("empty_prompt", "prompt cannot be empty")
        content = prompt.strip()
        if context:
            content = f"{content}\n\nContext:\n{json.dumps(context, ensure_ascii=False, sort_keys=True)}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        data = self._request_json("/chat/completions", payload)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMProviderError("missing_choices", "model response did not include choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise LLMProviderError("missing_message", "model response did not include a message")
        content_value = message.get("content")
        if content_value is None:
            raise LLMProviderError("missing_content", "model response did not include message content")
        return str(content_value)

    def generate_json(self, prompt: str, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
        text = self.generate_text(
            f"{prompt}\n\nReturn only one valid JSON object.",
            context=context,
        )
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMProviderError("invalid_json", "model response was not valid JSON") from exc
        if not isinstance(value, dict):
            raise LLMProviderError("invalid_json_object", "model JSON response was not an object")
        return value

    def classify(self, prompt: str, labels: list[str], *, context: dict[str, Any] | None = None) -> str:
        if not labels:
            raise LLMProviderError("empty_labels", "at least one classification label is required")
        label_text = ", ".join(labels)
        result = self.generate_text(
            f"{prompt}\n\nChoose exactly one label from: {label_text}. Return only the label.",
            context=context,
        ).strip()
        for label in labels:
            if result.lower() == label.lower():
                return label
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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMProviderError("http_error", f"Zhipu HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMProviderError("network_error", f"Zhipu network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMProviderError("timeout", "Zhipu request timed out") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMProviderError("invalid_json", "Zhipu response was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise LLMProviderError("invalid_response", "Zhipu response was not a JSON object")

        error = parsed.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or "api_error")
            message = str(error.get("message") or "Zhipu API error")
            raise LLMProviderError(code, message)
        code = parsed.get("code")
        if code not in (None, 0, "0"):
            raise LLMProviderError(str(code), str(parsed.get("msg") or parsed.get("message") or "Zhipu API error"))
        return parsed


def create_llm_client(config: Config) -> LLMClient:
    provider = config.model_provider.strip().lower()
    if provider == "stub":
        return StubLLMClient(config)
    if provider in {"zhipu", "bigmodel"}:
        return ZhipuLLMClient(config)
    raise ValueError(f"unsupported model provider: {config.model_provider}")
