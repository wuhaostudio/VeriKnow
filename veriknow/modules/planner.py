from __future__ import annotations

from veriknow.schemas import EvidenceBundle, EvidenceItem, TaskSpec, VerificationPlan, VerificationStep


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
