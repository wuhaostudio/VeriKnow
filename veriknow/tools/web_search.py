from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote_plus

from veriknow.config import Config


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    source_type: str = "unknown"
    published_at: str | None = None
    updated_at: str | None = None


class SearchProviderError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class WebSearchProvider(Protocol):
    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        ...


class StaticSeedSearchProvider:
    """Deterministic provider used until a live web-search backend is configured."""

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        normalized = " ".join(query.strip().lower().split())
        if not normalized:
            raise ValueError("search query cannot be empty")

        results = self._catalog_results(normalized)
        if not results:
            results = self._fallback_results(query)
        return results[:limit]

    def _catalog_results(self, normalized_query: str) -> list[SearchResult]:
        if "langchain" in normalized_query:
            return [
                SearchResult(
                    title="LangChain multi-agent documentation",
                    url="https://docs.langchain.com/oss/python/langchain/multi-agent",
                    snippet="Official LangChain documentation for multi-agent patterns.",
                    source_type="official_doc",
                ),
                SearchResult(
                    title="LangGraph supervisor tutorial",
                    url="https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/",
                    snippet="Official LangGraph tutorial for supervisor-style multi-agent workflows.",
                    source_type="official_doc",
                ),
                SearchResult(
                    title="LangGraph repository",
                    url="https://github.com/langchain-ai/langgraph",
                    snippet="Official LangGraph source repository and examples.",
                    source_type="official_github",
                ),
            ]
        if "playwright" in normalized_query:
            return [
                SearchResult(
                    title="Playwright Python documentation",
                    url="https://playwright.dev/python/docs/intro",
                    snippet="Official Playwright for Python setup and usage documentation.",
                    source_type="official_doc",
                ),
                SearchResult(
                    title="Playwright repository",
                    url="https://github.com/microsoft/playwright",
                    snippet="Official Playwright browser automation source repository.",
                    source_type="official_github",
                ),
            ]
        if "openai" in normalized_query:
            return [
                SearchResult(
                    title="OpenAI API documentation",
                    url="https://platform.openai.com/docs",
                    snippet="Official OpenAI API documentation.",
                    source_type="official_doc",
                ),
                SearchResult(
                    title="OpenAI Python SDK",
                    url="https://github.com/openai/openai-python",
                    snippet="Official OpenAI Python SDK repository.",
                    source_type="official_github",
                ),
            ]
        return []

    def _fallback_results(self, query: str) -> list[SearchResult]:
        encoded = quote_plus(query)
        return [
            SearchResult(
                title=f"Search results for {query}",
                url=f"https://www.google.com/search?q={encoded}",
                snippet="Fallback search URL. Replace with concrete sources during live research.",
                source_type="search_result",
            )
        ]


class BraveSearchProvider:
    provider = "brave"

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = "https://api.search.brave.com/res/v1/web/search",
        timeout_seconds: int = 20,
    ):
        if not api_key.strip():
            raise SearchProviderError("missing_api_key", "Brave search requires an API key.")
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        normalized = " ".join(query.strip().split())
        if not normalized:
            raise ValueError("search query cannot be empty")
        params = urllib.parse.urlencode({"q": normalized, "count": max(1, min(limit, 20))})
        request = urllib.request.Request(
            f"{self.endpoint}?{params}",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise SearchProviderError("http_error", f"Brave search failed with HTTP {exc.code}.") from exc
        except urllib.error.URLError as exc:
            raise SearchProviderError("network_error", f"Brave search request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise SearchProviderError("invalid_json", "Brave search response was not valid JSON.") from exc

        return _brave_results(payload)[:limit]


def create_search_provider(config: Config, *, provider: str | None = None) -> WebSearchProvider:
    selected = (provider or config.search_provider).strip().lower()
    if selected in {"", "static", "seed"}:
        return StaticSeedSearchProvider()
    if selected == "brave":
        env_name = config.search_api_key_env or "BRAVE_SEARCH_API_KEY"
        return BraveSearchProvider(os.environ.get(env_name, ""))
    raise ValueError(f"unsupported search provider: {provider or config.search_provider}")


def _brave_results(payload: dict[str, Any]) -> list[SearchResult]:
    web = payload.get("web")
    raw_results = web.get("results") if isinstance(web, dict) else None
    if not isinstance(raw_results, list):
        return []

    results: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        if not title or not url:
            continue
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=str(item.get("description") or ""),
                source_type=_source_type_for_url(url),
                published_at=_optional_string(item.get("age")),
            )
        )
    return results


def _source_type_for_url(url: str) -> str:
    lowered = url.lower()
    if "github.com" in lowered:
        return "official_github"
    if any(host in lowered for host in ["docs.", "documentation", "/docs"]):
        return "official_doc"
    return "search_result"


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
