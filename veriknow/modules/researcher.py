from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from veriknow.llm import LLMClient, LLMProviderError
from veriknow.schemas import EvidenceBundle, EvidenceClaim, EvidenceItem, TaskSpec
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

SUPPORTED_RESEARCH_STRATEGIES = {"deterministic", "ai"}


class Researcher:
    def __init__(self, provider: WebSearchProvider | None = None):
        self.provider = provider or StaticSeedSearchProvider()

    def research(self, task: TaskSpec, *, run_id: str, limit: int = 5) -> EvidenceBundle:
        query = self._query_for(task)
        results = self.provider.search(query, limit=limit)
        items = [self._result_to_item(result) for result in results]
        ranked_items = rank_evidence_items(items)
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
            confidence=confidence_for_source_type(source_type),
        )

    def _summary_for(self, task: TaskSpec, items: list[EvidenceItem]) -> str:
        return summary_for_evidence(task, items)


@dataclass(frozen=True)
class ResearchArtifact:
    strategy: str
    provider: str
    status: str
    prompt: str
    seed_evidence: dict[str, Any]
    model_output: dict[str, Any] | None = None
    fallback_used: bool = False
    error_code: str | None = None
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "provider": self.provider,
            "status": self.status,
            "prompt": self.prompt,
            "seed_evidence": self.seed_evidence,
            "model_output": self.model_output,
            "fallback_used": self.fallback_used,
            "error_code": self.error_code,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ResearchResult:
    bundle: EvidenceBundle
    artifact: ResearchArtifact | None = None


class AIResearcher:
    def __init__(
        self,
        llm: LLMClient,
        base: Researcher | None = None,
    ):
        self.llm = llm
        self.base = base or Researcher()

    def research(self, task: TaskSpec, *, run_id: str, limit: int = 5) -> ResearchResult:
        seed = self.base.research(task, run_id=run_id, limit=limit)
        prompt = self._prompt_for(task)
        try:
            output = self.llm.generate_json(prompt, context=self._context_for(task, seed))
            bundle = self._bundle_from_output(output, task=task, run_id=run_id, limit=limit)
            artifact = ResearchArtifact(
                strategy="ai",
                provider=self.llm.provider,
                status="completed",
                prompt=prompt,
                seed_evidence=seed.to_dict(),
                model_output=output,
                fallback_used=False,
                message="AI research evidence extraction completed.",
                evidence=bundle.to_dict(),
            )
            return ResearchResult(bundle=bundle, artifact=artifact)
        except (LLMProviderError, ValueError, TypeError) as exc:
            error_code = exc.code if isinstance(exc, LLMProviderError) else exc.__class__.__name__
            artifact = ResearchArtifact(
                strategy="ai",
                provider=self.llm.provider,
                status="fallback",
                prompt=prompt,
                seed_evidence=seed.to_dict(),
                model_output=None,
                fallback_used=True,
                error_code=error_code,
                message=str(exc),
                evidence=seed.to_dict(),
            )
            return ResearchResult(bundle=seed, artifact=artifact)

    def _prompt_for(self, task: TaskSpec) -> str:
        return (
            "Create a VeriKnow EvidenceBundle JSON object from the supplied seed search results. "
            "Return fields: summary and items. Each item must include title, url, source_type, "
            "snippet, published_at, updated_at, confidence. Prefer official and recent sources. "
            "Do not invent URLs that are not present in the seed evidence."
        )

    def _context_for(self, task: TaskSpec, seed: EvidenceBundle) -> dict[str, Any]:
        return {
            "task": task.to_dict(),
            "seed_evidence": seed.to_dict(),
        }

    def _bundle_from_output(
        self,
        output: dict[str, Any],
        *,
        task: TaskSpec,
        run_id: str,
        limit: int,
    ) -> EvidenceBundle:
        raw_items = output.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("model output must include a non-empty items list")

        items: list[EvidenceItem] = []
        for raw_item in raw_items[:limit]:
            if not isinstance(raw_item, dict):
                raise ValueError("each model evidence item must be an object")
            title = str(raw_item.get("title", "")).strip()
            url = str(raw_item.get("url", "")).strip()
            if not title or not url:
                raise ValueError("model evidence items require title and url")
            source_type = str(raw_item.get("source_type") or "unknown")
            confidence = str(raw_item.get("confidence") or confidence_for_source_type(source_type))
            items.append(
                EvidenceItem(
                    title=title,
                    url=url,
                    source_type=source_type,
                    snippet=str(raw_item.get("snippet") or ""),
                    published_at=_optional_string(raw_item.get("published_at")),
                    updated_at=_optional_string(raw_item.get("updated_at")),
                    confidence=confidence,
                )
            )

        ranked_items = rank_evidence_items(items)
        summary = str(output.get("summary") or summary_for_evidence(task, ranked_items))
        return EvidenceBundle(task_id=run_id, items=ranked_items, summary=summary)


def rank_evidence_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return sorted(
        items,
        key=lambda item: (
            SOURCE_PRIORITY.get(item.source_type, SOURCE_PRIORITY["unknown"]),
            item.confidence,
            item.title.lower(),
        ),
        reverse=True,
    )


def confidence_for_source_type(source_type: str) -> str:
    if source_type in {"official_doc", "official_github", "standard"}:
        return "high"
    if source_type in {"vendor_blog", "community"}:
        return "medium"
    return "low"


def summary_for_evidence(task: TaskSpec, items: list[EvidenceItem]) -> str:
    if not items:
        return f"No public evidence collected for {task.target}."
    high_confidence_count = sum(1 for item in items if item.confidence == "high")
    return (
        f"Collected {len(items)} public source(s) for {task.target}; "
        f"{high_confidence_count} high-confidence source(s)."
    )


def add_claim_summary(
    summary: str,
    claims: list[EvidenceClaim],
    *,
    conflict_count: int = 0,
) -> str:
    if not claims:
        return f"{summary} No extracted claims were available from fetched pages."

    source_count = len({claim.source_url for claim in claims})
    dated_count = sum(1 for claim in claims if claim.freshness == "dated")
    caveat_count = sum(1 for claim in claims if claim.caveats)
    claim_summary = (
        f" Extracted {len(claims)} claim(s) from {source_count} fetched source(s); "
        f"{dated_count} dated claim(s); {caveat_count} claim(s) with caveats; "
        f"{conflict_count} detected conflict(s)."
    )
    return f"{summary}{claim_summary}"

def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
