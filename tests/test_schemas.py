import unittest

from veriknow.schemas import EvidenceItem, PublicationJob, RunRecord, TaskSpec, VerificationResult


class SchemaTests(unittest.TestCase):
    def test_task_spec_round_trip(self) -> None:
        task = TaskSpec(raw_request="test task", objective="Research", target="test")
        data = task.to_dict()
        loaded = TaskSpec.from_dict(data)

        self.assertEqual(loaded.raw_request, "test task")
        self.assertEqual(loaded.target, "test")

    def test_run_record_round_trip_nested_task(self) -> None:
        task = TaskSpec(raw_request="test task", objective="Research", target="test")
        record = RunRecord(run_id="run-test", raw_request="test task", task=task)

        loaded = RunRecord.from_dict(record.to_dict())

        self.assertEqual(loaded.task.target, "test")

    def test_required_field_validation(self) -> None:
        with self.assertRaises(ValueError):
            EvidenceItem(title="", url="https://example.com")

    def test_publication_job_round_trip(self) -> None:
        job = PublicationJob(
            document_path="data/knowledge/general/example.md",
            target="feishu",
            status="blocked",
            target_document_id="doc-1",
            target_url="https://example.feishu.cn/docx/doc-1",
            error_code="missing_credentials",
            message="missing credentials",
        )

        loaded = PublicationJob.from_dict(job.to_dict())

        self.assertEqual(loaded.document_path, job.document_path)
        self.assertEqual(loaded.target, "feishu")
        self.assertEqual(loaded.status, "blocked")
        self.assertEqual(loaded.target_document_id, "doc-1")
        self.assertEqual(loaded.target_url, "https://example.feishu.cn/docx/doc-1")
        self.assertEqual(loaded.error_code, "missing_credentials")

    def test_verification_result_round_trip_with_actions(self) -> None:
        result = VerificationResult(
            step_description="Open isolated browser",
            status="partial",
            actual_result="Computer-use boundary recorded.",
            actions=["open isolated browser", "navigate to https://example.com"],
            observations=["runtime not configured"],
        )

        loaded = VerificationResult.from_dict(result.to_dict())

        self.assertEqual(loaded.actions, result.actions)
        self.assertEqual(loaded.observations, result.observations)
