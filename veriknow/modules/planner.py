from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from veriknow.llm import LLMClient, LLMProviderError
from veriknow.schemas import EvidenceBundle, EvidenceClaim, EvidenceItem, TaskSpec, VerificationPlan, VerificationStep


RISK_TERMS = {
    "account",
    "billing",
    "credential",
    "delete",
    "login",
    "payment",
    "production",
    "secret",
    "token",
    "账号",
    "付款",
    "删除",
    "密钥",
    "登录",
}

SUPPORTED_PLANNING_STRATEGIES = {"deterministic", "ai"}


class VerificationPlanner:
    def plan(
        self,
        task: TaskSpec,
        evidence: EvidenceBundle | None,
        *,
        run_id: str,
    ) -> VerificationPlan:
        steps: list[VerificationStep] = []

        if evidence is None or not evidence.items:
            steps.append(
                VerificationStep(
                    description="Collect or attach public evidence before verification.",
                    expected_result="Evidence bundle contains at least one source URL.",
                    method="manual",
                    tools=["human_review"],
                    screenshot_required=False,
                    requires_approval=False,
                )
            )
            return VerificationPlan(task_id=run_id, steps=steps)

        for item in evidence.items:
            steps.append(self._source_step(task, item))

        steps.append(
            VerificationStep(
                description=f"Compare collected sources for conflicts about {task.target}.",
                expected_result=(
                    "Conflicting, outdated, or unsupported claims are identified before writing "
                    "the final guide."
                ),
                method="manual",
                tools=["human_review"],
                screenshot_required=False,
                requires_approval=False,
            )
        )
        return VerificationPlan(task_id=run_id, steps=steps)

    def _source_step(self, task: TaskSpec, item: EvidenceItem) -> VerificationStep:
        claim = item.snippet or item.title
        return VerificationStep(
            description=f"Open source and verify claim for {task.target}: {claim}",
            expected_result=(
                f"The source is reachable and directly supports the claim. URL: {item.url}"
            ),
            method=self._method_for(task, item),
            tools=self._tools_for(task, item),
            screenshot_required=self._screenshot_required(item),
            requires_approval=self._requires_approval(task, item),
        )

    def _method_for(self, task: TaskSpec, item: EvidenceItem) -> str:
        if task.verification_method in {"api", "cli"} and item.source_type == "official_github":
            return task.verification_method
        if item.url.startswith("http://") or item.url.startswith("https://"):
            return "browser"
        return "manual"

    def _tools_for(self, task: TaskSpec, item: EvidenceItem) -> list[str]:
        method = self._method_for(task, item)
        if method == "api":
            return ["api_client", "browser"]
        if method == "cli":
            return ["cli", "browser"]
        if method == "browser":
            return ["browser"]
        return ["human_review"]

    def _screenshot_required(self, item: EvidenceItem) -> bool:
        return item.url.startswith("http://") or item.url.startswith("https://")

    def _requires_approval(self, task: TaskSpec, item: EvidenceItem) -> bool:
        haystack = f"{task.raw_request} {item.title} {item.url} {item.snippet}".lower()
        return any(term in haystack for term in RISK_TERMS)


@dataclass(frozen=True)
class PlanningArtifact:
    strategy: str
    provider: str
    status: str
    prompt: str
    seed_plan: dict[str, Any]
    model_output: dict[str, Any] | None = None
    fallback_used: bool = False
    error_code: str | None = None
    message: str = ""
    plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "provider": self.provider,
            "status": self.status,
            "prompt": self.prompt,
            "seed_plan": self.seed_plan,
            "model_output": self.model_output,
            "fallback_used": self.fallback_used,
            "error_code": self.error_code,
            "message": self.message,
            "plan": self.plan,
        }


@dataclass(frozen=True)
class PlanningResult:
    plan: VerificationPlan
    artifact: PlanningArtifact | None = None


class AIVerificationPlanner:
    def __init__(
        self,
        llm: LLMClient,
        base: VerificationPlanner | None = None,
    ):
        self.llm = llm
        self.base = base or VerificationPlanner()

    def plan(
        self,
        task: TaskSpec,
        evidence: EvidenceBundle | None,
        *,
        run_id: str,
        claims: list[EvidenceClaim] | None = None,
        claim_conflicts: list[dict[str, Any]] | None = None,
    ) -> PlanningResult:
        seed = self.base.plan(task, evidence, run_id=run_id)
        prompt = self._prompt_for(task)
        try:
            output = self.llm.generate_json(
                prompt,
                context=self._context_for(
                    task,
                    evidence,
                    seed,
                    claims=claims,
                    claim_conflicts=claim_conflicts,
                ),
            )
            plan = self._plan_from_output(output, run_id=run_id)
            artifact = PlanningArtifact(
                strategy="ai",
                provider=self.llm.provider,
                status="completed",
                prompt=prompt,
                seed_plan=seed.to_dict(),
                model_output=output,
                fallback_used=False,
                message="AI verification planning completed.",
                plan=plan.to_dict(),
            )
            return PlanningResult(plan=plan, artifact=artifact)
        except (LLMProviderError, ValueError, TypeError) as exc:
            error_code = exc.code if isinstance(exc, LLMProviderError) else exc.__class__.__name__
            artifact = PlanningArtifact(
                strategy="ai",
                provider=self.llm.provider,
                status="fallback",
                prompt=prompt,
                seed_plan=seed.to_dict(),
                model_output=None,
                fallback_used=True,
                error_code=error_code,
                message=str(exc),
                plan=seed.to_dict(),
            )
            return PlanningResult(plan=seed, artifact=artifact)

    def _prompt_for(self, task: TaskSpec) -> str:
        return (
            "Create a VeriKnow VerificationPlan JSON object from the supplied task and seed plan. "
            "Return fields: steps. Each step must include description, expected_result, method, "
            "tools, screenshot_required, requires_approval. Allowed methods are browser, api, cli, manual, computer-use. "
            "Browser and computer-use steps must include a concrete http or https source URL in the description or expected_result. Use extracted_claims and claim_conflicts when present, and add manual checkpoints for unresolved conflicts. Prefer steps that are directly testable and that preserve approval gates for risky actions."
        )

    def _context_for(
        self,
        task: TaskSpec,
        evidence: EvidenceBundle | None,
        seed: VerificationPlan,
        claims: list[EvidenceClaim] | None = None,
        claim_conflicts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "task": task.to_dict(),
            "evidence": evidence.to_dict() if evidence is not None else None,
            "seed_plan": seed.to_dict(),
            "extracted_claims": [claim.to_dict() for claim in claims or []],
            "claim_conflicts": claim_conflicts or [],
        }

    def _plan_from_output(self, output: dict[str, Any], *, run_id: str) -> VerificationPlan:
        raw_steps = output.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("model output must include a non-empty steps list")

        steps: list[VerificationStep] = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                raise ValueError("each model verification step must be an object")
            description = str(raw_step.get("description", "")).strip()
            expected_result = str(raw_step.get("expected_result", "")).strip()
            method = str(raw_step.get("method", "manual")).strip()
            if not description or not expected_result:
                raise ValueError("model verification steps require description and expected_result")
            if method not in {"browser", "api", "cli", "manual", "computer-use"}:
                raise ValueError(f"unsupported verification method: {method}")
            if method in {"browser", "computer-use"} and not _has_url(description, expected_result):
                raise ValueError(f"{method} verification steps require a source URL")
            tools = raw_step.get("tools", [])
            if not isinstance(tools, list):
                raise ValueError("model verification step tools must be a list")
            steps.append(
                VerificationStep(
                    description=description,
                    expected_result=expected_result,
                    method=method,
                    tools=[str(item) for item in tools],
                    screenshot_required=bool(raw_step.get("screenshot_required", False)),
                    requires_approval=bool(raw_step.get("requires_approval", False)),
                )
            )

        return VerificationPlan(task_id=run_id, steps=steps)


def _has_url(*values: str) -> bool:
    haystack = " ".join(values).lower()
    return "http://" in haystack or "https://" in haystack

def render_verification_checklist(plan: VerificationPlan) -> str:
    lines = [
        f"# Verification Checklist",
        "",
        f"- Task ID: `{plan.task_id}`",
        f"- Steps: {len(plan.steps)}",
        "",
    ]
    for index, step in enumerate(plan.steps, start=1):
        approval = "yes" if step.requires_approval else "no"
        screenshot = "yes" if step.screenshot_required else "no"
        tools = ", ".join(step.tools) if step.tools else "none"
        lines.extend(
            [
                f"## {index}. {step.description}",
                "",
                f"- Expected result: {step.expected_result}",
                f"- Method: {step.method}",
                f"- Tools: {tools}",
                f"- Screenshot required: {screenshot}",
                f"- Requires approval: {approval}",
                "",
            ]
        )
    return "\n".join(lines)
