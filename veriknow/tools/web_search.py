from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote_plus


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    source_type: str = "unknown"
    published_at: str | None = None
    updated_at: str | None = None


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
