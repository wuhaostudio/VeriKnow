import unittest

from veriknow.modules.planner import AIVerificationPlanner, VerificationPlanner, render_verification_checklist
from veriknow.schemas import EvidenceBundle, EvidenceClaim, EvidenceItem, TaskSpec


class PlannerTests(unittest.TestCase):
    def test_planner_generates_steps_from_evidence(self) -> None:
        task = TaskSpec(
            raw_request="Research latest LangChain workflow",
            objective="Research",
            target="LangChain workflow",
            verification_required=True,
            verification_method="browser",
        )
        evidence = EvidenceBundle(
            task_id="run-test",
            items=[
                EvidenceItem(
                    title="Official docs",
                    url="https://example.com/docs",
                    source_type="official_doc",
                    snippet="Official documentation for the workflow.",
                    confidence="high",
                )
            ],
        )

        plan = VerificationPlanner().plan(task, evidence, run_id="run-test")

        self.assertEqual(plan.task_id, "run-test")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].method, "browser")
        self.assertEqual(plan.steps[0].tools, ["browser"])
        self.assertTrue(plan.steps[0].screenshot_required)
        self.assertIn("directly supports the claim", plan.steps[0].expected_result)

    def test_planner_marks_risky_steps_for_approval(self) -> None:
        task = TaskSpec(
            raw_request="Verify account billing setup",
            objective="Research",
            target="billing setup",
        )
        evidence = EvidenceBundle(
            task_id="run-test",
            items=[
                EvidenceItem(
                    title="Billing docs",
                    url="https://example.com/account/billing",
                    source_type="official_doc",
                )
            ],
        )

        plan = VerificationPlanner().plan(task, evidence, run_id="run-test")

        self.assertTrue(plan.steps[0].requires_approval)

    def test_render_verification_checklist(self) -> None:
        task = TaskSpec(raw_request="Research example", objective="Research", target="example")
        plan = VerificationPlanner().plan(task, None, run_id="run-test")

        checklist = render_verification_checklist(plan)

        self.assertIn("# Verification Checklist", checklist)
        self.assertIn("Expected result:", checklist)
        self.assertIn("Method: manual", checklist)

class FakePlannerLLM:
    provider = "fake"
    model = "fake-model"

    def __init__(self, payload: dict):
        self.payload = payload
        self.context = None

    def check(self):
        raise NotImplementedError

    def generate_text(self, prompt: str, *, context: dict | None = None) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, *, context: dict | None = None) -> dict:
        self.context = context
        return self.payload

    def classify(self, prompt: str, labels: list[str], *, context: dict | None = None) -> str:
        return labels[0]


class AIPlannerTests(unittest.TestCase):
    def test_ai_planner_validates_model_output(self) -> None:
        task = TaskSpec(
            raw_request="Research latest LangChain workflow",
            objective="Research",
            target="LangChain workflow",
            verification_required=True,
            verification_method="browser",
        )
        evidence = EvidenceBundle(
            task_id="run-test",
            items=[
                EvidenceItem(
                    title="Official docs",
                    url="https://example.com/docs",
                    source_type="official_doc",
                    snippet="Official documentation for the workflow.",
                    confidence="high",
                )
            ],
        )
        llm = FakePlannerLLM(
            {
                "steps": [
                    {
                        "description": "Open the official docs page.",
                        "expected_result": "The page loads successfully at https://example.com/docs.",
                        "method": "browser",
                        "tools": ["browser"],
                        "screenshot_required": True,
                        "requires_approval": False,
                    }
                ]
            }
        )

        result = AIVerificationPlanner(llm).plan(task, evidence, run_id="run-ai")

        self.assertEqual(result.plan.task_id, "run-ai")
        self.assertEqual(result.plan.steps[0].description, "Open the official docs page.")
        self.assertIsNotNone(result.artifact)
        self.assertEqual(result.artifact.status, "completed")
        self.assertFalse(result.artifact.fallback_used)

    def test_ai_planner_falls_back_on_invalid_output(self) -> None:
        task = TaskSpec(
            raw_request="Research latest LangChain workflow",
            objective="Research",
            target="LangChain workflow",
            verification_required=True,
            verification_method="browser",
        )
        evidence = EvidenceBundle(
            task_id="run-test",
            items=[
                EvidenceItem(
                    title="Official docs",
                    url="https://example.com/docs",
                    source_type="official_doc",
                    snippet="Official documentation for the workflow.",
                    confidence="high",
                )
            ],
        )
        llm = FakePlannerLLM({"steps": []})

        result = AIVerificationPlanner(llm).plan(task, evidence, run_id="run-ai")

        self.assertEqual(result.plan.steps[0].description, "Open source and verify claim for LangChain workflow: Official documentation for the workflow.")
        self.assertIsNotNone(result.artifact)
        self.assertEqual(result.artifact.status, "fallback")
        self.assertTrue(result.artifact.fallback_used)

    def test_ai_planner_rejects_browser_steps_without_source_url(self) -> None:
        task = TaskSpec(
            raw_request="Research latest LangChain workflow",
            objective="Research",
            target="LangChain workflow",
            verification_required=True,
            verification_method="browser",
        )
        evidence = EvidenceBundle(
            task_id="run-test",
            items=[
                EvidenceItem(
                    title="Official docs",
                    url="https://example.com/docs",
                    source_type="official_doc",
                    snippet="Official documentation for the workflow.",
                    confidence="high",
                )
            ],
        )
        llm = FakePlannerLLM(
            {
                "steps": [
                    {
                        "description": "Open the official docs page.",
                        "expected_result": "The page loads successfully.",
                        "method": "browser",
                        "tools": ["browser"],
                        "screenshot_required": True,
                        "requires_approval": False,
                    }
                ]
            }
        )

        result = AIVerificationPlanner(llm).plan(task, evidence, run_id="run-ai")

        self.assertEqual(result.plan.steps[0].description, "Open source and verify claim for LangChain workflow: Official documentation for the workflow.")
        self.assertIsNotNone(result.artifact)
        self.assertEqual(result.artifact.status, "fallback")
        self.assertEqual(result.artifact.error_code, "ValueError")
        self.assertIn("source URL", result.artifact.message)

    def test_ai_planner_context_includes_extracted_claims_and_conflicts(self) -> None:
        task = TaskSpec(
            raw_request="Research latest LangChain workflow",
            objective="Research",
            target="LangChain workflow",
            verification_required=True,
            verification_method="browser",
        )
        evidence = EvidenceBundle(
            task_id="run-test",
            items=[
                EvidenceItem(
                    title="Official docs",
                    url="https://example.com/docs",
                    source_type="official_doc",
                    snippet="Official documentation for the workflow.",
                    confidence="high",
                )
            ],
        )
        claim = EvidenceClaim(
            text="The workflow supports supervisor routing.",
            source_url="https://example.com/docs",
            source_title="Official docs",
        )
        conflict = {"topic": "workflow", "reason": "opposing claims"}
        llm = FakePlannerLLM(
            {
                "steps": [
                    {
                        "description": "Open the official docs page.",
                        "expected_result": "The page loads successfully at https://example.com/docs.",
                        "method": "browser",
                        "tools": ["browser"],
                        "screenshot_required": True,
                        "requires_approval": False,
                    }
                ]
            }
        )

        AIVerificationPlanner(llm).plan(
            task,
            evidence,
            run_id="run-ai",
            claims=[claim],
            claim_conflicts=[conflict],
        )

        self.assertIsNotNone(llm.context)
        self.assertEqual(llm.context["extracted_claims"][0]["text"], claim.text)
        self.assertEqual(llm.context["claim_conflicts"], [conflict])
