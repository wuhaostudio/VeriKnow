import json
from pathlib import Path
import unittest

from veriknow.schemas import EvidenceClaim, FetchedDocument
from veriknow.tools.claims import AIClaimExtractor, detect_claim_conflicts, extract_claims


class FakeClaimLLM:
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

class ClaimExtractionTests(unittest.TestCase):
    def test_extract_claims_uses_first_sentence_and_keyword_sentences(self) -> None:
        document = FetchedDocument(
            url="https://example.com/docs",
            title="Example Docs",
            text=(
                "The API is documented for production use. "
                "Version 2.0 supports tool calling. "
                "This filler sentence has no signal. "
                "The old workflow is deprecated."
            ),
            fetched_at="2026-06-26T00:00:00+00:00",
            status_code=200,
            content_hash="hash-1",
        )

        claims = extract_claims([document], max_claims_per_document=3)

        self.assertEqual(len(claims), 3)
        self.assertEqual(claims[0].source_url, document.url)
        self.assertEqual(claims[0].source_title, "Example Docs")
        self.assertIn("production use", claims[0].text)
        self.assertEqual(claims[1].freshness, "dated")
        self.assertIn("mentions deprecation", claims[2].caveats)

    def test_extract_claims_adds_source_dates_and_version_constraints(self) -> None:
        document = FetchedDocument(
            url="https://example.com/docs",
            title="Example Docs",
            text=(
                "Last updated: 2026-06-26. "
                "Version 2.0 supports tool calling and Python >=3.11 is recommended."
            ),
            fetched_at="2026-06-26T00:00:00+00:00",
            status_code=200,
            content_hash="hash-1",
            metadata={
                "source_type": "official_doc",
                "published_at": "2026-01-01",
                "confidence": "high",
            },
        )

        claims = extract_claims([document], max_claims_per_document=2)

        self.assertEqual(claims[0].source_dates["published_at"], "2026-01-01")
        self.assertEqual(claims[0].source_dates["updated_at"], "2026-06-26")
        self.assertEqual(claims[0].source_type, "official_doc")
        self.assertEqual(claims[0].confidence, "high")
        self.assertIn("Version 2.0", claims[1].version_constraints)
        self.assertIn("Python >=3.11", claims[1].version_constraints)

    def test_phase13_metadata_eval_fixture(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "phase13_metadata_eval.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        documents = [FetchedDocument.from_dict(item) for item in fixture["documents"]]

        claims = extract_claims(documents, max_claims_per_document=4)
        conflicts = detect_claim_conflicts(claims)

        current_claim = claims[0]
        expected = fixture["expected"]
        self.assertEqual(current_claim.source_dates, expected["source_dates"])
        for constraint in expected["version_constraints"]:
            self.assertTrue(any(constraint in claim.version_constraints for claim in claims))
        self.assertTrue(conflicts)
        self.assertIn(expected["conflict_reason_contains"], conflicts[0].reason)
    def test_extract_claims_skips_failed_or_empty_documents(self) -> None:
        failed = FetchedDocument(
            url="https://example.com/error",
            fetched_at="2026-06-26T00:00:00+00:00",
            error_code="network_error",
        )
        empty = FetchedDocument(
            url="https://example.com/empty",
            fetched_at="2026-06-26T00:00:00+00:00",
            text="",
        )

        self.assertEqual(extract_claims([failed, empty]), [])

    def test_detect_claim_conflicts_marks_opposing_source_claims(self) -> None:
        claims = [
            EvidenceClaim(
                text="Tool calling is stable and recommended for the Example API.",
                source_url="https://example.com/current",
                source_title="Current Docs",
            ),
            EvidenceClaim(
                text="Tool calling is deprecated for the Example API.",
                source_url="https://example.com/old",
                source_title="Old Docs",
            ),
        ]

        conflicts = detect_claim_conflicts(claims)

        self.assertEqual(len(conflicts), 1)
        self.assertIn("example", conflicts[0].topic)
        self.assertIn("opposing", conflicts[0].reason)
        self.assertTrue(claims[0].conflicts)
        self.assertTrue(claims[1].conflicts)
    def test_ai_claim_extractor_validates_model_output(self) -> None:
        document = FetchedDocument(
            url="https://example.com/docs",
            title="Example Docs",
            text="Version 2.0 supports tool calling.",
            fetched_at="2026-06-26T00:00:00+00:00",
            status_code=200,
            content_hash="hash-1",
        )
        llm = FakeClaimLLM(
            {
                "claims": [
                    {
                        "text": "Version 2.0 supports tool calling.",
                        "source_url": "https://example.com/docs",
                        "source_title": "Example Docs",
                        "quote": "Version 2.0 supports tool calling.",
                        "source_type": "official_doc",
                        "published_at": None,
                        "updated_at": "2026-06-26",
                        "confidence": "high",
                        "freshness": "dated",
                        "caveats": [],
                        "conflicts": [],
                    }
                ]
            }
        )

        result = AIClaimExtractor(llm).extract([document])

        self.assertEqual(result.claims[0].source_url, document.url)
        self.assertEqual(result.claims[0].confidence, "high")
        self.assertIsNotNone(result.artifact)
        self.assertEqual(result.artifact.status, "completed")
        self.assertEqual(llm.context["fetched_documents"][0]["url"], document.url)

    def test_ai_claim_extractor_falls_back_on_unknown_source_url(self) -> None:
        document = FetchedDocument(
            url="https://example.com/docs",
            title="Example Docs",
            text="Version 2.0 supports tool calling.",
            fetched_at="2026-06-26T00:00:00+00:00",
            status_code=200,
            content_hash="hash-1",
        )
        llm = FakeClaimLLM(
            {
                "claims": [
                    {
                        "text": "Unsupported claim.",
                        "source_url": "https://example.com/unknown",
                    }
                ]
            }
        )

        result = AIClaimExtractor(llm).extract([document])

        self.assertIn("Version 2.0", result.claims[0].text)
        self.assertIsNotNone(result.artifact)
        self.assertEqual(result.artifact.status, "fallback")
        self.assertTrue(result.artifact.fallback_used)
