from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from veriknow.llm import LLMClient, LLMProviderError
from veriknow.schemas import EvidenceBundle, EvidenceClaim, EvidenceItem, TaskSpec
from veriknow.tools.web_search import (
    SearchResult,
    StaticSeedSearchProvider,
    WebSearchProvider,
    normalize_search_url,
)


SOURCE_PRIORITY = {
    "official_doc": 100,
    "official_github": 90,
    "standard": 80,
    "vendor_blog": 65,
    "community": 40,
    "search_result": 20,
    "unknown": 10,
}

CONFIDENCE_PRIORITY = {
    "high": 3,
    "medium": 2,
    "low": 1,
}

DEFAULT_FRESHNESS_DAYS = {
    "official_doc": 365,
    "official_github": 180,
    "standard": 730,
    "vendor_blog": 180,
    "community": 90,
    "search_result": 30,
    "unknown": 90,
}

SUPPORTED_RESEARCH_STRATEGIES = {"deterministic", "ai"}


class Researcher:
    def __init__(
        self,
        provider: WebSearchProvider | None = None,
        *,
        freshness_days: dict[str, int] | None = None,
        source_priority: dict[str, int] | None = None,
        query_count: int = 1,
        as_of: date | None = None,
    ):
        self.provider = provider or StaticSeedSearchProvider()
        self.last_raw_search_payloads: list[dict[str, Any]] = []
        self.freshness_days = dict(freshness_days or DEFAULT_FRESHNESS_DAYS)
        self.source_priority = dict(source_priority or SOURCE_PRIORITY)
        self.query_count = min(5, max(1, query_count))
        self.as_of = as_of

    def research(self, task: TaskSpec, *, run_id: str, limit: int = 5) -> EvidenceBundle:
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        self.last_raw_search_payloads = []
        queries = self._queries_for(task)
        for query in queries:
            query_results = self.provider.search(query, limit=limit)
            for payload in _raw_search_payloads(self.provider, query_results):
                self.last_raw_search_payloads.append(
                    {**payload, "query": query} if len(queries) > 1 else payload
                )
            for result in query_results:
                normalized_url = normalize_search_url(result.url)
                if not normalized_url or normalized_url in seen_urls:
                    continue
                seen_urls.add(normalized_url)
                results.append(result)
        items = [self._result_to_item(result) for result in results]
        ranked_items = rank_evidence_items(
            items,
            source_priority=self.source_priority,
        )[:limit]
        return EvidenceBundle(
            task_id=run_id,
            items=ranked_items,
            summary=self._summary_for(task, ranked_items),
        )

    def _query_for(self, task: TaskSpec) -> str:
        if task.target and task.target != task.raw_request:
            return task.target
        return task.raw_request

    def _queries_for(self, task: TaskSpec) -> list[str]:
        primary = self._query_for(task)
        haystack = f"{task.raw_request} {' '.join(task.constraints)}".casefold()
        candidates = [
            primary,
            f"{primary} official documentation",
            (
                f"{primary} latest release notes"
                if any(term in haystack for term in ["latest", "current", "最新"])
                else f"{primary} implementation guide"
            ),
            f"{primary} official GitHub repository",
            f"{primary} changelog",
        ]
        queries: list[str] = []
        seen: set[str] = set()
        for query in candidates:
            normalized = " ".join(query.split())
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                queries.append(normalized)
            if len(queries) >= self.query_count:
                break
        return queries

    def _result_to_item(self, result: SearchResult) -> EvidenceItem:
        source_type = result.source_type or "unknown"
        return self.enrich_item(
            EvidenceItem(
                title=result.title,
                url=result.url,
                source_type=source_type,
                snippet=result.snippet,
                published_at=result.published_at,
                updated_at=result.updated_at,
            )
        )

    def enrich_item(self, item: EvidenceItem) -> EvidenceItem:
        source_type = item.source_type.strip().lower() or "unknown"
        freshness = freshness_for_evidence(
            item,
            as_of=self.as_of,
            freshness_days=self.freshness_days,
        )
        confidence = confidence_for_source_type(source_type, freshness=freshness)
        return EvidenceItem(
            title=item.title,
            url=item.url,
            source_type=source_type,
            snippet=item.snippet,
            published_at=item.published_at,
            updated_at=item.updated_at,
            confidence=confidence,
            confidence_reason=confidence_reason_for(source_type, freshness),
            freshness=freshness,
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
            bundle = self._bundle_from_output(
                output,
                task=task,
                run_id=run_id,
                limit=limit,
                seed=seed,
            )
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
        seed: EvidenceBundle,
    ) -> EvidenceBundle:
        raw_items = output.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("model output must include a non-empty items list")

        allowed_urls = {item.url for item in seed.items}
        items: list[EvidenceItem] = []
        for raw_item in raw_items[:limit]:
            if not isinstance(raw_item, dict):
                raise ValueError("each model evidence item must be an object")
            title = str(raw_item.get("title", "")).strip()
            url = str(raw_item.get("url", "")).strip()
            if not title or not url:
                raise ValueError("model evidence items require title and url")
            if url not in allowed_urls:
                raise ValueError(f"model evidence URL was not present in seed evidence: {url}")
            source_type = str(raw_item.get("source_type") or "unknown").strip().lower()
            items.append(
                self.base.enrich_item(
                    EvidenceItem(
                        title=title,
                        url=url,
                        source_type=source_type,
                        snippet=str(raw_item.get("snippet") or ""),
                        published_at=_optional_string(raw_item.get("published_at")),
                        updated_at=_optional_string(raw_item.get("updated_at")),
                        confidence=str(raw_item.get("confidence") or "medium"),
                    )
                )
            )

        ranked_items = rank_evidence_items(
            items,
            source_priority=self.base.source_priority,
        )
        summary = str(output.get("summary") or summary_for_evidence(task, ranked_items))
        return EvidenceBundle(task_id=run_id, items=ranked_items, summary=summary)


def rank_evidence_items(
    items: list[EvidenceItem],
    *,
    source_priority: dict[str, int] | None = None,
) -> list[EvidenceItem]:
    priorities = source_priority or SOURCE_PRIORITY
    return sorted(items, key=lambda item: _evidence_sort_key(item, priorities))


def _evidence_sort_key(
    item: EvidenceItem,
    source_priority: dict[str, int],
) -> tuple[int, int, int, int, str, str]:
    source_date = _source_datetime(item)
    source_type = item.source_type.strip().lower()
    return (
        -source_priority.get(
            source_type,
            source_priority.get("unknown", SOURCE_PRIORITY["unknown"]),
        ),
        -CONFIDENCE_PRIORITY.get(item.confidence.strip().lower(), 0),
        -(1 if source_date is not None else 0),
        -_datetime_rank(source_date),
        item.title.casefold(),
        item.url,
    )


def _source_datetime(item: EvidenceItem) -> datetime | None:
    for value in (item.updated_at, item.published_at):
        parsed = _parse_source_datetime(value)
        if parsed is not None:
            return parsed
    return None


def _parse_source_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None

    if parsed is None:
        for date_format in ("%b %d, %Y", "%B %d, %Y", "%Y/%m/%d"):
            try:
                parsed = datetime.strptime(text, date_format)
                break
            except ValueError:
                continue

    if parsed is not None and parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _datetime_rank(value: datetime | None) -> int:
    if value is None:
        return 0
    microseconds_per_day = 86_400_000_000
    microseconds = (
        ((value.hour * 60 + value.minute) * 60 + value.second) * 1_000_000
        + value.microsecond
    )
    return value.toordinal() * microseconds_per_day + microseconds


def confidence_for_source_type(source_type: str, *, freshness: str = "unknown") -> str:
    if source_type in {"official_doc", "official_github", "standard"}:
        confidence = "high"
    elif source_type in {"vendor_blog", "community"}:
        confidence = "medium"
    else:
        confidence = "low"
    if freshness == "stale":
        return {"high": "medium", "medium": "low", "low": "low"}[confidence]
    return confidence


def freshness_for_evidence(
    item: EvidenceItem,
    *,
    as_of: date | None = None,
    freshness_days: dict[str, int] | None = None,
) -> str:
    source_datetime = _source_datetime(item)
    if source_datetime is None:
        return "unknown"
    today = as_of or datetime.now(timezone.utc).date()
    age_days = (today - source_datetime.date()).days
    if age_days < 0:
        return "unknown"
    thresholds = freshness_days or DEFAULT_FRESHNESS_DAYS
    source_type = item.source_type.strip().lower()
    fresh_days = thresholds.get(
        source_type,
        thresholds.get("unknown", DEFAULT_FRESHNESS_DAYS["unknown"]),
    )
    if age_days <= fresh_days:
        return "fresh"
    if age_days <= fresh_days * 2:
        return "aging"
    return "stale"


def confidence_reason_for(source_type: str, freshness: str) -> str:
    base = confidence_for_source_type(source_type)
    effective = confidence_for_source_type(source_type, freshness=freshness)
    authority = source_type if source_type in SOURCE_PRIORITY else "unknown"
    reason = f"{authority} source has {base} base confidence"
    if freshness == "stale" and effective != base:
        return f"{reason}; stale source date lowers confidence to {effective}."
    if freshness == "unknown":
        return f"{reason}; source date is unavailable or invalid."
    return f"{reason}; source freshness is {freshness}."


def summary_for_evidence(task: TaskSpec, items: list[EvidenceItem]) -> str:
    if not items:
        return f"No public evidence collected for {task.target}."
    high_confidence_count = sum(1 for item in items if item.confidence == "high")
    fresh_count = sum(1 for item in items if item.freshness == "fresh")
    stale_count = sum(1 for item in items if item.freshness == "stale")
    unknown_count = sum(1 for item in items if item.freshness == "unknown")
    return (
        f"Collected {len(items)} public source(s) for {task.target}; "
        f"{high_confidence_count} high-confidence source(s); "
        f"{fresh_count} fresh, {stale_count} stale, and {unknown_count} unknown-date source(s)."
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


def _raw_search_payloads(provider: WebSearchProvider, results: list[SearchResult]) -> list[dict[str, Any]]:
    payloads = [result.raw for result in results if result.raw]
    failures = getattr(provider, "failures", None)
    if isinstance(failures, list) and failures:
        payloads.append({"provider": getattr(provider, "provider", "unknown"), "failures": failures})
    return payloads


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
