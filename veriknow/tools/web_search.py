from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
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
    raw: dict[str, Any] = field(default_factory=dict)


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


class SerpApiSearchProvider:
    provider = "serpapi"

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = "https://serpapi.com/search",
        timeout_seconds: int = 20,
    ):
        if not api_key.strip():
            raise SearchProviderError("missing_api_key", "SerpApi search requires an API key.")
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        normalized = " ".join(query.strip().split())
        if not normalized:
            raise ValueError("search query cannot be empty")
        params = urllib.parse.urlencode(
            {
                "engine": "google",
                "q": normalized,
                "api_key": self.api_key,
                "output": "json",
            }
        )
        request = urllib.request.Request(
            f"{self.endpoint}?{params}",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise SearchProviderError("http_error", f"SerpApi search failed with HTTP {exc.code}.") from exc
        except urllib.error.URLError as exc:
            raise SearchProviderError("network_error", f"SerpApi search request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise SearchProviderError("invalid_json", "SerpApi search response was not valid JSON.") from exc

        if isinstance(payload.get("error"), str):
            raise SearchProviderError("api_error", payload["error"])
        return _serpapi_results(payload)[:limit]


class HybridSearchProvider:
    provider = "hybrid"

    def __init__(self, providers: list[WebSearchProvider]):
        if not providers:
            raise SearchProviderError("missing_provider", "Hybrid search requires at least one provider.")
        self.providers = providers
        self.failures: list[dict[str, str]] = []

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        normalized = " ".join(query.strip().split())
        if not normalized:
            raise ValueError("search query cannot be empty")

        self.failures = []
        provider_results: list[list[SearchResult]] = []
        for provider in self.providers:
            try:
                found = provider.search(normalized, limit=limit)
            except SearchProviderError as exc:
                self.failures.append(
                    {
                        "provider": _provider_name(provider),
                        "code": exc.code,
                        "message": exc.message,
                    }
                )
                continue
            provider_results.append(found)

        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        position = 0
        while len(results) < limit:
            found_at_position = False
            for found in provider_results:
                if position >= len(found):
                    continue
                found_at_position = True
                result = found[position]
                normalized_url = normalize_search_url(result.url)
                if not normalized_url or normalized_url in seen_urls:
                    continue
                seen_urls.add(normalized_url)
                results.append(result)
                if len(results) >= limit:
                    break
            if not found_at_position:
                break
            position += 1

        if not results and self.failures:
            failed = ", ".join(f"{failure['provider']}:{failure['code']}" for failure in self.failures)
            raise SearchProviderError("all_providers_failed", f"Hybrid search failed for all providers: {failed}.")
        return results[:limit]


class UnavailableSearchProvider:
    def __init__(self, provider: str, error: SearchProviderError):
        self.provider = provider
        self.error = error

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        raise self.error


def create_search_provider(config: Config, *, provider: str | None = None) -> WebSearchProvider:
    selected = (provider or config.search_provider).strip().lower()
    if selected in {"", "static", "seed"}:
        return StaticSeedSearchProvider()
    if selected == "brave":
        env_name = config.search_api_key_env or "BRAVE_SEARCH_API_KEY"
        return BraveSearchProvider(os.environ.get(env_name, ""))
    if selected in {"serpapi", "serp"}:
        env_name = config.search_api_key_env or "SERPAPI_API_KEY"
        return SerpApiSearchProvider(os.environ.get(env_name, ""))
    if selected == "hybrid":
        providers: list[WebSearchProvider] = []
        for name in config.search_hybrid_providers:
            normalized_name = name.strip().lower()
            if not normalized_name or normalized_name == "hybrid":
                continue
            try:
                providers.append(create_search_provider(config, provider=normalized_name))
            except SearchProviderError as exc:
                providers.append(UnavailableSearchProvider(normalized_name, exc))
        return HybridSearchProvider(providers)
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
                raw=_compact_raw_result(item),
            )
        )
    return results


def _serpapi_results(payload: dict[str, Any]) -> list[SearchResult]:
    raw_results = payload.get("organic_results")
    if not isinstance(raw_results, list):
        return []

    results: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("link") or item.get("url") or "").strip()
        if not title or not url:
            continue
        source_date = _optional_string(item.get("date"))
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=str(item.get("snippet") or ""),
                source_type=_source_type_for_url(url),
                published_at=source_date,
                updated_at=source_date,
                raw=_compact_serpapi_result(item),
            )
        )
    return results


def _compact_raw_result(item: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "title",
        "url",
        "description",
        "age",
        "profile",
        "language",
        "family_friendly",
        "page_age",
        "page_fetched",
        "thumbnail",
    }
    return {key: value for key, value in item.items() if key in allowed_keys}


def _compact_serpapi_result(item: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "position",
        "title",
        "link",
        "displayed_link",
        "snippet",
        "date",
        "source",
        "about_this_result",
        "snippet_highlighted_words",
    }
    return {key: value for key, value in item.items() if key in allowed_keys}


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


def normalize_search_url(url: str) -> str:
    text = url.strip()
    if not text:
        return ""
    parsed = urllib.parse.urlsplit(text)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    query_pairs = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    query = urllib.parse.urlencode(query_pairs)
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def _provider_name(provider: WebSearchProvider) -> str:
    return str(getattr(provider, "provider", provider.__class__.__name__))
