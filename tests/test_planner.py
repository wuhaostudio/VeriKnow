import unittest

from veriknow.modules.planner import VerificationPlanner, render_verification_checklist
from veriknow.schemas import EvidenceBundle, EvidenceItem, TaskSpec


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
