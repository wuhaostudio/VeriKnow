from __future__ import annotations

from veriknow.schemas import EvidenceBundle, EvidenceItem, TaskSpec
from veriknow.tools.web_search import SearchResult, StaticSeedSearchProvider, WebSearchProvider


SOURCE_PRIORITY = {
    "official_doc": 100,
    "official_github": 90,
    "standard": 80,
    "vendor_blog": 65,
    "community": 40,
    "search_result": 20,
    "unknown": 10,
}


class Researcher:
    def __init__(self, provider: WebSearchProvider | None = None):
        self.provider = provider or StaticSeedSearchProvider()

    def research(self, task: TaskSpec, *, run_id: str, limit: int = 5) -> EvidenceBundle:
        query = self._query_for(task)
        results = self.provider.search(query, limit=limit)
        items = [self._result_to_item(result) for result in results]
        ranked_items = sorted(
            items,
            key=lambda item: (
                SOURCE_PRIORITY.get(item.source_type, SOURCE_PRIORITY["unknown"]),
                item.confidence,
                item.title.lower(),
            ),
            reverse=True,
        )
        return EvidenceBundle(
            task_id=run_id,
            items=ranked_items,
            summary=self._summary_for(task, ranked_items),
        )

    def _query_for(self, task: TaskSpec) -> str:
        if task.target and task.target != task.raw_request:
            return task.target
        return task.raw_request

    def _result_to_item(self, result: SearchResult) -> EvidenceItem:
        source_type = result.source_type or "unknown"
        return EvidenceItem(
            title=result.title,
            url=result.url,
            source_type=source_type,
            snippet=result.snippet,
            published_at=result.published_at,
            updated_at=result.updated_at,
            confidence=self._confidence_for(source_type),
        )

    def _confidence_for(self, source_type: str) -> str:
        if source_type in {"official_doc", "official_github", "standard"}:
            return "high"
        if source_type in {"vendor_blog", "community"}:
            return "medium"
        return "low"

    def _summary_for(self, task: TaskSpec, items: list[EvidenceItem]) -> str:
        if not items:
            return f"No public evidence collected for {task.target}."
        high_confidence_count = sum(1 for item in items if item.confidence == "high")
        return (
            f"Collected {len(items)} public source(s) for {task.target}; "
            f"{high_confidence_count} high-confidence source(s)."
        )
