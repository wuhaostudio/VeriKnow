from __future__ import annotations

import unittest

from veriknow.tools.web_fetch import WebPageFetcher, normalize_html


class WebFetchTests(unittest.TestCase):
    def test_normalize_html_extracts_title_and_visible_text(self) -> None:
        title, text = normalize_html(
            "<html><head><title>Example Docs</title><style>.x{}</style></head>"
            "<body><h1>Guide</h1><script>hidden()</script><p>Use the API.</p></body></html>"
        )

        self.assertEqual(title, "Example Docs")
        self.assertIn("Guide", text)
        self.assertIn("Use the API.", text)
        self.assertNotIn("hidden", text)

    def test_fetcher_maps_html_response_to_document(self) -> None:
        calls = []

        class FakeHeaders:
            def get_content_charset(self):
                return "utf-8"

        class FakeResponse:
            status = 200
            headers = FakeHeaders()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def read(self) -> bytes:
                return b"<html><head><title>Fetched</title></head><body>Readable page.</body></html>"

            def getcode(self) -> int:
                return 200

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return FakeResponse()

        import unittest.mock

        with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
            document = WebPageFetcher(timeout_seconds=4).fetch("https://example.com/docs")

        self.assertEqual(document.url, "https://example.com/docs")
        self.assertEqual(document.title, "Fetched")
        self.assertEqual(document.status_code, 200)
        self.assertIn("Readable page.", document.text)
        self.assertTrue(document.content_hash)
        self.assertEqual(calls[0][1], 4)


    def test_fetcher_stores_raw_html_when_raw_dir_is_configured(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        class FakeHeaders:
            def get_content_charset(self):
                return "utf-8"

        class FakeResponse:
            status = 200
            headers = FakeHeaders()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def read(self) -> bytes:
                return b"<html><head><title>Raw</title></head><body>Raw page.</body></html>"

            def getcode(self) -> int:
                return 200

        def fake_urlopen(request, timeout):
            return FakeResponse()

        import unittest.mock

        with TemporaryDirectory() as directory:
            raw_dir = Path(directory) / "raw_pages"
            with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
                document = WebPageFetcher(raw_dir=raw_dir).fetch("https://docs.example.com/guide")

            self.assertIsNotNone(document.raw_path)
            raw_path = Path(document.raw_path or "")
            self.assertTrue(raw_path.exists())
            self.assertEqual(raw_path.parent, raw_dir)
            self.assertIn("Raw page.", raw_path.read_text(encoding="utf-8"))
    def test_fetcher_records_unsupported_urls(self) -> None:
        document = WebPageFetcher().fetch("file:///tmp/example.html", title="Local")

        self.assertEqual(document.error_code, "unsupported_url")
        self.assertEqual(document.title, "Local")
        self.assertEqual(document.status_code, 0)
