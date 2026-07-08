from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from veriknow.llm import LLMClient, LLMProviderError
from veriknow.tools.computer_runtime import ComputerAction


@dataclass(frozen=True)
class ComputerActionPlan:
    actions: list[ComputerAction]
    observations: list[str] = field(default_factory=list)


class ComputerActionAgent(Protocol):
    name: str

    def plan_actions(
        self,
        url: str,
        *,
        instruction: str,
        expected_result: str,
        max_steps: int,
        store_screenshots: bool,
    ) -> ComputerActionPlan:
        ...


class DeterministicComputerActionAgent:
    name = "deterministic"

    def plan_actions(
        self,
        url: str,
        *,
        instruction: str,
        expected_result: str,
        max_steps: int,
        store_screenshots: bool,
    ) -> ComputerActionPlan:
        actions = [
            ComputerAction("open", target=url, reason="navigate to verification source URL"),
        ]
        if store_screenshots:
            actions.append(ComputerAction("screenshot", reason="capture page after navigation"))
        actions.append(
            ComputerAction(
                "observe",
                target="body",
                reason=f"compare public page content with: {expected_result}",
            )
        )
        if store_screenshots:
            actions.append(ComputerAction("scroll", target="page", reason="inspect additional public content"))
        actions.append(ComputerAction("finish", reason="record final verification status"))
        return ComputerActionPlan(
            actions=actions[:max_steps],
            observations=[
                "action_agent=deterministic",
                f"action_agent_instruction={instruction}",
            ],
        )


class AIComputerActionAgent:
    name = "ai"

    def __init__(self, llm: LLMClient, fallback: ComputerActionAgent | None = None):
        self.llm = llm
        self.fallback = fallback or DeterministicComputerActionAgent()

    def plan_actions(
        self,
        url: str,
        *,
        instruction: str,
        expected_result: str,
        max_steps: int,
        store_screenshots: bool,
    ) -> ComputerActionPlan:
        prompt = (
            "Create a read-only computer-use action plan for public documentation verification. "
            "Return JSON with an actions array. Each action may include action, target, text, "
            "reason, and requires_approval. Use only open, screenshot, observe, scroll, wait, "
            "finish, or fail unless explicit approval is required."
        )
        context = {
            "url": url,
            "instruction": instruction,
            "expected_result": expected_result,
            "max_steps": max_steps,
            "store_screenshots": store_screenshots,
            "safety": {
                "read_only": True,
                "blocked": ["login", "payment", "delete", "credentials", "file upload", "account changes"],
            },
        }
        try:
            response = self.llm.generate_json(prompt, context=context)
            actions = _actions_from_response(response, max_steps=max_steps)
        except (LLMProviderError, ValueError, TypeError) as exc:
            fallback_plan = self.fallback.plan_actions(
                url,
                instruction=instruction,
                expected_result=expected_result,
                max_steps=max_steps,
                store_screenshots=store_screenshots,
            )
            return ComputerActionPlan(
                actions=fallback_plan.actions,
                observations=[
                    "action_agent=ai",
                    "action_agent_status=fallback",
                    f"action_agent_error={exc.__class__.__name__}: {exc}",
                    *fallback_plan.observations,
                ],
            )

        return ComputerActionPlan(
            actions=actions,
            observations=[
                "action_agent=ai",
                "action_agent_status=planned",
                f"action_agent_provider={self.llm.provider}",
                f"action_agent_model={self.llm.model}",
            ],
        )


def create_computer_action_agent(name: str, *, llm: LLMClient | None = None) -> ComputerActionAgent:
    normalized = name.strip().lower()
    if normalized in {"", "deterministic", "static"}:
        return DeterministicComputerActionAgent()
    if normalized == "ai":
        if llm is None:
            raise ValueError("AI computer-use action agent requires an LLM client")
        return AIComputerActionAgent(llm)
    raise ValueError(f"unsupported computer-use action agent: {name}")


def _actions_from_response(response: dict[str, Any], *, max_steps: int) -> list[ComputerAction]:
    raw_actions = response.get("actions")
    if not isinstance(raw_actions, list):
        raise ValueError("model response did not include an actions list")

    actions: list[ComputerAction] = []
    for raw_action in raw_actions[:max_steps]:
        if not isinstance(raw_action, dict):
            raise ValueError("model action was not an object")
        action = str(raw_action.get("action", "")).strip().lower()
        if not action:
            raise ValueError("model action is missing action")
        actions.append(
            ComputerAction(
                action=action,
                target=str(raw_action.get("target", "") or ""),
                text=str(raw_action.get("text", "") or ""),
                reason=str(raw_action.get("reason", "") or ""),
                requires_approval=bool(raw_action.get("requires_approval", False)),
            )
        )

    if not actions:
        raise ValueError("model response did not include any actions")
    if actions[-1].action not in {"finish", "fail"} and len(actions) < max_steps:
        actions.append(ComputerAction("finish", reason="record final verification status"))
    return actions[:max_steps]
