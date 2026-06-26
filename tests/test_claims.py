import unittest

from veriknow.schemas import FetchedDocument
from veriknow.tools.claims import extract_claims


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
