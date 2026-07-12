import unittest

from veriknow.schemas import EvidenceClaim, EvidenceItem, FetchedDocument, KnowledgeMergeProposal, PublicationJob, PublicationMapping, RunRecord, TaskSpec, VerificationResult


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

    def test_evidence_item_round_trip_preserves_policy_metadata(self) -> None:
        item = EvidenceItem(
            title="Official docs",
            url="https://example.com/docs",
            source_type="official_doc",
            confidence="high",
            confidence_reason="Official and recently updated.",
            freshness="fresh",
        )

        loaded = EvidenceItem.from_dict(item.to_dict())

        self.assertEqual(loaded.freshness, "fresh")
        self.assertEqual(
            loaded.confidence_reason,
            "Official and recently updated.",
        )


    def test_knowledge_merge_proposal_round_trip(self) -> None:
        proposal = KnowledgeMergeProposal(
            run_id="run-test",
            operation="update",
            target_path="data/knowledge/general/example.md",
            target_title="Example",
            rationale="Update existing knowledge document.",
            evidence_urls=["https://example.com/docs"],
            conflicts=["Older source is deprecated."],
            diff="--- old\n+++ new\n",
            risk_level="high",
        )

        loaded = KnowledgeMergeProposal.from_dict(proposal.to_dict())

        self.assertEqual(loaded.operation, "update")
        self.assertEqual(loaded.evidence_urls, ["https://example.com/docs"])
        self.assertEqual(loaded.conflicts, ["Older source is deprecated."])
        self.assertEqual(loaded.risk_level, "high")
    def test_publication_job_round_trip(self) -> None:
        job = PublicationJob(
            document_path="data/knowledge/general/example.md",
            target="feishu",
            status="blocked",
            local_path="C:/project/VeriKnow/data/knowledge/general/example.md",
            local_content_hash="hash-1",
            target_document_id="doc-1",
            target_url="https://example.feishu.cn/docx/doc-1",
            last_published_at="2026-07-03T00:00:00+00:00",
            last_published_hash="hash-0",
            remote_revision="rev-1",
            error_code="missing_credentials",
            message="missing credentials",
        )

        loaded = PublicationJob.from_dict(job.to_dict())

        self.assertEqual(loaded.document_path, job.document_path)
        self.assertEqual(loaded.target, "feishu")
        self.assertEqual(loaded.status, "blocked")
        self.assertEqual(loaded.target_document_id, "doc-1")
        self.assertEqual(loaded.target_url, "https://example.feishu.cn/docx/doc-1")
        self.assertEqual(loaded.local_content_hash, "hash-1")
        self.assertEqual(loaded.last_published_hash, "hash-0")
        self.assertEqual(loaded.remote_revision, "rev-1")
        self.assertEqual(loaded.error_code, "missing_credentials")


    def test_publication_mapping_round_trip(self) -> None:
        mapping = PublicationMapping(
            local_path="data/knowledge/general/example.md",
            target="feishu",
            local_content_hash="hash-1",
            target_document_id="doc-1",
            target_url="https://example.feishu.cn/docx/doc-1",
            last_published_hash="hash-1",
            status="published",
        )

        loaded = PublicationMapping.from_dict(mapping.to_dict())

        self.assertEqual(loaded.local_path, mapping.local_path)
        self.assertEqual(loaded.target, "feishu")
        self.assertEqual(loaded.target_document_id, "doc-1")
        self.assertEqual(loaded.last_published_hash, "hash-1")

    def test_evidence_claim_round_trip(self) -> None:
        claim = EvidenceClaim(
            text="The API supports tool calling in version 2.0.",
            source_url="https://example.com/docs",
            source_title="Example Docs",
            quote="The API supports tool calling in version 2.0.",
            source_type="official_doc",
            confidence="high",
            freshness="dated",
            caveats=["version-specific"],
            conflicts=["Older source says unsupported."],
        )

        loaded = EvidenceClaim.from_dict(claim.to_dict())

        self.assertEqual(loaded.text, claim.text)
        self.assertEqual(loaded.source_url, claim.source_url)
        self.assertEqual(loaded.caveats, ["version-specific"])
        self.assertEqual(loaded.conflicts, ["Older source says unsupported."])

    def test_fetched_document_round_trip(self) -> None:
        document = FetchedDocument(
            url="https://example.com/docs",
            title="Example Docs",
            text="Readable documentation text.",
            fetched_at="2026-06-26T00:00:00+00:00",
            status_code=200,
            content_hash="abc123",
            raw_path="data/runs/run-test/raw_pages/example.html",
        )

        loaded = FetchedDocument.from_dict(document.to_dict())

        self.assertEqual(loaded.url, document.url)
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(loaded.content_hash, "abc123")
        self.assertEqual(loaded.raw_path, "data/runs/run-test/raw_pages/example.html")

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
