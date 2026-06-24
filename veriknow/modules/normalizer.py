from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from veriknow.config import Config
from veriknow.llm import LLMProviderError, LLMClient
from veriknow.schemas import TaskSpec


class RequirementNormalizer:
    def __init__(self, config: Config):
        self.config = config

    def normalize(self, raw_request: str) -> TaskSpec:
        request = normalize_request(raw_request)
        verification_required = self._needs_verification(request)
        method = self._method_for(request, verification_required)
        locale = "zh-CN" if _contains_cjk(request) else "en-US"
        target = self._extract_target(request)
        constraints = self._constraints_for(request)

        return TaskSpec(
            raw_request=request,
            objective=self._objective_for(request),
            target=target,
            scope=self.config.default_scope,
            verification_required=verification_required,
            verification_method=method,
            output_format=self.config.default_output_format,
            publish_target=self.config.default_publish_target,
            locale=locale,
            constraints=constraints,
        )

    def _objective_for(self, request: str) -> str:
        lowered = request.lower()
        if any(word in lowered for word in ["guide", "operation", "manual", "用法", "指南", "操作"]):
            return "Create a traceable operation guide from verified information."
        if any(word in lowered for word in ["research", "study", "调研", "研究"]):
            return "Research the target and produce a source-grounded summary."
        return "Clarify and document the requested knowledge task."

    def _extract_target(self, request: str) -> str:
        cleaned = request
        patterns = [
            r"^(帮我|请|please)\s*",
            r"(研究|调研|了解|分析|create|write|research|study)\s*",
            r"(最新|latest|current)\s*",
            r"(用法|做法|指南|operation guide|manual)\s*",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,。.")
        cleaned = re.sub(r"(的|about|for)$", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned or request

    def _needs_verification(self, request: str) -> bool:
        lowered = request.lower()
        verification_terms = [
            "latest",
            "current",
            "verify",
            "validated",
            "最新",
            "验证",
            "可用",
            "当前",
        ]
        return any(term in lowered for term in verification_terms)

    def _method_for(self, request: str, verification_required: bool) -> str:
        lowered = request.lower()
        if not verification_required:
            return "manual"
        if any(term in lowered for term in ["api", "接口"]):
            return "api"
        if any(term in lowered for term in ["cli", "命令行"]):
            return "cli"
        return "browser"

    def _constraints_for(self, request: str) -> list[str]:
        lowered = request.lower()
        constraints: list[str] = []
        if "最新" in request or "latest" in lowered or "current" in lowered:
            constraints.append("Prioritize recent and official sources.")
        if "操作" in request or "用法" in request or "guide" in lowered or "manual" in lowered:
            constraints.append("Output should be actionable and step-by-step.")
        return constraints


@dataclass(frozen=True)
class NormalizationArtifact:
    strategy: str
    provider: str
    status: str
    prompt: str
    model_output: dict[str, Any] | None = None
    fallback_used: bool = False
    error_code: str | None = None
    message: str = ""
    task: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "provider": self.provider,
            "status": self.status,
            "prompt": self.prompt,
            "model_output": self.model_output,
            "fallback_used": self.fallback_used,
            "error_code": self.error_code,
            "message": self.message,
            "task": self.task,
        }


@dataclass(frozen=True)
class NormalizationResult:
    task: TaskSpec
    artifact: NormalizationArtifact | None = None


class AIRequirementNormalizer:
    def __init__(
        self,
        config: Config,
        llm: LLMClient,
        fallback: RequirementNormalizer | None = None,
    ):
        self.config = config
        self.llm = llm
        self.fallback = fallback or RequirementNormalizer(config)

    def normalize(self, raw_request: str) -> NormalizationResult:
        request = normalize_request(raw_request)
        prompt = self._prompt_for(request)
        try:
            output = self.llm.generate_json(prompt, context=self._context_for(request))
            task = self._task_from_output(request, output)
            artifact = NormalizationArtifact(
                strategy="ai",
                provider=self.llm.provider,
                status="completed",
                prompt=prompt,
                model_output=output,
                fallback_used=False,
                message="AI normalization completed.",
                task=task.to_dict(),
            )
            return NormalizationResult(task=task, artifact=artifact)
        except (LLMProviderError, ValueError, TypeError) as exc:
            task = self.fallback.normalize(request)
            error_code = exc.code if isinstance(exc, LLMProviderError) else exc.__class__.__name__
            artifact = NormalizationArtifact(
                strategy="ai",
                provider=self.llm.provider,
                status="fallback",
                prompt=prompt,
                model_output=None,
                fallback_used=True,
                error_code=error_code,
                message=str(exc),
                task=task.to_dict(),
            )
            return NormalizationResult(task=task, artifact=artifact)

    def _prompt_for(self, request: str) -> str:
        return (
            "Normalize the user request into one VeriKnow TaskSpec JSON object. "
            "Return fields: raw_request, objective, target, scope, verification_required, "
            "verification_method, output_format, publish_target, locale, constraints. "
            "Use concise English values for objective and constraints. "
            "Allowed verification_method values are browser, api, cli, manual."
        )

    def _context_for(self, request: str) -> dict[str, Any]:
        return {
            "raw_request": request,
            "defaults": {
                "scope": self.config.default_scope,
                "output_format": self.config.default_output_format,
                "publish_target": self.config.default_publish_target,
            },
        }

    def _task_from_output(self, request: str, output: dict[str, Any]) -> TaskSpec:
        fallback_task = self.fallback.normalize(request)
        data = dict(output)
        missing_required = [name for name in ("objective", "target") if not str(data.get(name, "")).strip()]
        if missing_required:
            raise ValueError(f"model output missing required field(s): {', '.join(missing_required)}")
        data["raw_request"] = request
        data.setdefault("scope", self.config.default_scope)
        data.setdefault("verification_required", fallback_task.verification_required)
        data.setdefault("verification_method", fallback_task.verification_method)
        data.setdefault("output_format", self.config.default_output_format)
        data.setdefault("publish_target", self.config.default_publish_target)
        data.setdefault("locale", fallback_task.locale)
        data.setdefault("constraints", fallback_task.constraints)
        data["verification_required"] = bool(data["verification_required"])
        if data["verification_method"] not in {"browser", "api", "cli", "manual"}:
            raise ValueError(f"unsupported verification_method: {data['verification_method']}")
        if not isinstance(data["constraints"], list):
            raise ValueError("constraints must be a list")
        data["constraints"] = [str(item) for item in data["constraints"]]
        return TaskSpec.from_dict(data)


SUPPORTED_NORMALIZER_STRATEGIES = {"deterministic", "ai"}


def normalize_request(raw_request: str) -> str:
    request = " ".join(raw_request.strip().split())
    if not request:
        raise ValueError("raw request cannot be empty")
    return request


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)
