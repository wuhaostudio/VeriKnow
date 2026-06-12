from __future__ import annotations

import re

from veriknow.config import Config
from veriknow.schemas import TaskSpec


class RequirementNormalizer:
    def __init__(self, config: Config):
        self.config = config

    def normalize(self, raw_request: str) -> TaskSpec:
        request = " ".join(raw_request.strip().split())
        if not request:
            raise ValueError("raw request cannot be empty")

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


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)
