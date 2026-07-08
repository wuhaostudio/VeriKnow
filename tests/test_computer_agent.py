from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from veriknow.config import Config
from veriknow.llm import StubLLMClient
from veriknow.tools.computer_agent import (
    AIComputerActionAgent,
    DeterministicComputerActionAgent,
    create_computer_action_agent,
)
from veriknow.tools.computer_runtime import RuntimeObservation
from veriknow.tools.computer_use import ComputerUseSafetyConfig, ComputerUseVerifier


class FakeJsonLLM(StubLLMClient):
    def __init__(self, payload):
        super().__init__(
            Config(
                data_dir=Path("data"),
                database_path=Path("data/memory.sqlite"),
                model_provider="stub",
                model_name="fake-json",
            )
        )
        self.payload = payload

    def generate_json(self, prompt, *, context=None):
        return self.payload


class RecordingRuntime:
    name = "recording"

    def __init__(self):
        self.action_plan = []

    def inspect_url(
        self,
        url: str,
        *,
        expected_result: str,
        screenshot_path: Path,
        log_path: Path,
        action_plan=None,
    ) -> RuntimeObservation:
        self.action_plan = list(action_plan or [])
        screenshot_path.write_bytes(b"fake screenshot")
        return RuntimeObservation(
            status="passed",
            final_url=url,
            title="Docs",
            http_status="200",
            screenshot_path=str(screenshot_path),
            log_path=str(log_path),
            observations=[],
        )


class ComputerActionAgentTests(unittest.TestCase):
    def test_deterministic_agent_builds_read_only_plan(self) -> None:
        plan = DeterministicComputerActionAgent().plan_actions(
            "https://example.com/docs",
            instruction="Open docs",
            expected_result="URL: https://example.com/docs",
            max_steps=5,
            store_screenshots=True,
        )

        self.assertEqual([action.action for action in plan.actions], ["open", "screenshot", "observe", "scroll", "finish"])
        self.assertIn("action_agent=deterministic", plan.observations)

    def test_ai_agent_uses_model_actions(self) -> None:
        agent = AIComputerActionAgent(
            FakeJsonLLM(
                {
                    "actions": [
                        {"action": "open", "target": "https://example.com/docs", "reason": "navigate"},
                        {"action": "observe", "target": "body", "reason": "inspect"},
                    ]
                }
            )
        )

        plan = agent.plan_actions(
            "https://example.com/docs",
            instruction="Open docs",
            expected_result="URL: https://example.com/docs",
            max_steps=5,
            store_screenshots=False,
        )

        self.assertEqual([action.action for action in plan.actions], ["open", "observe", "finish"])
        self.assertIn("action_agent_status=planned", plan.observations)

    def test_ai_agent_falls_back_on_invalid_model_output(self) -> None:
        agent = AIComputerActionAgent(FakeJsonLLM({"not_actions": []}))

        plan = agent.plan_actions(
            "https://example.com/docs",
            instruction="Open docs",
            expected_result="URL: https://example.com/docs",
            max_steps=3,
            store_screenshots=True,
        )

        self.assertEqual([action.action for action in plan.actions], ["open", "screenshot", "observe"])
        self.assertIn("action_agent_status=fallback", plan.observations)

    def test_ai_model_proposed_unsafe_action_is_blocked_by_safety(self) -> None:
        runtime = RecordingRuntime()
        agent = AIComputerActionAgent(
            FakeJsonLLM(
                {
                    "actions": [
                        {"action": "open", "target": "https://example.com/docs", "reason": "navigate"},
                        {"action": "click", "target": "#delete", "reason": "delete item"},
                    ]
                }
            )
        )
        verifier = ComputerUseVerifier(
            ComputerUseSafetyConfig(
                allowed_domains=("example.com",),
                action_allowlist=("open", "click", "finish"),
            ),
            runtime,
            agent,
        )

        with TemporaryDirectory() as directory:
            observation = verifier.verify_step(
                "https://example.com/docs",
                instruction="Open docs",
                expected_result="URL: https://example.com/docs",
                screenshot_path=Path(directory) / "step.png",
                log_path=Path(directory) / "step.log",
            )

        self.assertEqual(observation.status, "blocked")
        self.assertIn("read-only runtime disallows click", observation.actual_result)
        self.assertEqual(runtime.action_plan, [])

    def test_factory_requires_llm_for_ai_agent(self) -> None:
        with self.assertRaises(ValueError):
            create_computer_action_agent("ai")
