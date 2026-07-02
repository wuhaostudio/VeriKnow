from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from veriknow.schemas import EvidenceItem, FetchedDocument, now_iso


class WebPageFetcher:
    def __init__(
        self,
        *,
        timeout_seconds: int = 20,
        max_text_chars: int = 20000,
        raw_dir: Path | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_text_chars = max_text_chars
        self.raw_dir = raw_dir

    def fetch(self, url: str, *, title: str = "") -> FetchedDocument:
        if not url.startswith(("http://", "https://")):
            return FetchedDocument(
                url=url,
                title=title,
                text="",
                fetched_at=now_iso(),
                status_code=0,
                content_hash="",
                error_code="unsupported_url",
                message="Only http and https URLs can be fetched.",
            )

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "VeriKnow/0.1 (+local knowledge verification)"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                charset = _response_charset(response) or "utf-8"
                html = raw.decode(charset, errors="replace")
                status_code = getattr(response, "status", None) or response.getcode()
        except urllib.error.HTTPError as exc:
            return _failed_document(url, title, "http_error", f"HTTP {exc.code}", status_code=exc.code)
        except urllib.error.URLError as exc:
            return _failed_document(url, title, "network_error", str(exc.reason))
        except UnicodeError as exc:
            return _failed_document(url, title, "decode_error", str(exc))

        raw_path = self._write_raw_html(url, html)
        parsed_title, text = normalize_html(html)
        text = text[: self.max_text_chars]
        return FetchedDocument(
            url=url,
            title=parsed_title or title,
            text=text,
            fetched_at=now_iso(),
            status_code=int(status_code or 0),
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            raw_path=str(raw_path) if raw_path is not None else None,
        )

    def _write_raw_html(self, url: str, html: str) -> Path | None:
        if self.raw_dir is None:
            return None
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / _raw_filename(url)
        path.write_text(html, encoding="utf-8")
        return path


def fetch_documents(
    items: list[EvidenceItem],
    *,
    limit: int | None = None,
    fetcher: WebPageFetcher | None = None,
    raw_dir: Path | None = None,
) -> list[FetchedDocument]:
    selected = items[:limit] if limit is not None else items
    page_fetcher = fetcher or WebPageFetcher(raw_dir=raw_dir)
    documents: list[FetchedDocument] = []
    for item in selected:
        document = page_fetcher.fetch(item.url, title=item.title)
        document.metadata.update(
            {
                "source_title": item.title,
                "source_type": item.source_type,
                "source_snippet": item.snippet,
                "published_at": item.published_at,
                "updated_at": item.updated_at,
                "confidence": item.confidence,
            }
        )
        documents.append(document)
    return documents


def normalize_html(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = unescape(" ".join(parser.parts))
    text = re.sub(r"\s+", " ", text).strip()
    return parser.title.strip(), text


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs):
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif lowered == "title":
            self._in_title = True

    def handle_endtag(self, tag: str):
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        elif lowered == "title":
            self._in_title = False
            self.title = " ".join(self.title_parts).strip()

    def handle_data(self, data: str):
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        else:
            self.parts.append(text)


def _response_charset(response) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is not None and hasattr(headers, "get_content_charset"):
        return headers.get_content_charset()
    return None


def _raw_filename(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = re.sub(r"[^a-zA-Z0-9._-]+", "-", parsed.netloc) or "page"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{host}-{digest}.html"


def _failed_document(
    url: str,
    title: str,
    error_code: str,
    message: str,
    *,
    status_code: int = 0,
) -> FetchedDocument:
    return FetchedDocument(
        url=url,
        title=title,
        text="",
        fetched_at=now_iso(),
        status_code=status_code,
        content_hash="",
        error_code=error_code,
        message=message,
    )
