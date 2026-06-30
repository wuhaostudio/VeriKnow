from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from veriknow.llm import LLMClient, LLMProviderError

from veriknow.schemas import EvidenceClaim, FetchedDocument

CLAIM_KEYWORDS = {
    "available",
    "deprecated",
    "introduced",
    "latest",
    "new",
    "recommended",
    "requires",
    "stable",
    "supports",
    "updated",
    "version",
}

DATE_OR_VERSION_PATTERN = re.compile(
    r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|20\d{2}|v?\d+\.\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")
TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")

NEGATIVE_TERMS = {"deprecated", "removed", "unsupported", "legacy", "obsolete"}
POSITIVE_TERMS = {"available", "recommended", "stable", "supports", "supported"}


@dataclass(frozen=True)
class ClaimConflict:
    topic: str
    claim_a: dict
    claim_b: dict
    reason: str

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "claim_a": self.claim_a,
            "claim_b": self.claim_b,
            "reason": self.reason,
        }

@dataclass(frozen=True)
class ClaimExtractionArtifact:
    strategy: str
    provider: str
    status: str
    prompt: str
    fetched_documents: list[dict[str, Any]]
    model_output: dict[str, Any] | None = None
    fallback_used: bool = False
    error_code: str | None = None
    message: str = ""
    claims: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "provider": self.provider,
            "status": self.status,
            "prompt": self.prompt,
            "fetched_documents": self.fetched_documents,
            "model_output": self.model_output,
            "fallback_used": self.fallback_used,
            "error_code": self.error_code,
            "message": self.message,
            "claims": self.claims,
        }


@dataclass(frozen=True)
class ClaimExtractionResult:
    claims: list[EvidenceClaim]
    artifact: ClaimExtractionArtifact | None = None


class AIClaimExtractor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract(self, documents: list[FetchedDocument], *, max_claims_per_document: int = 5) -> ClaimExtractionResult:
        seed_claims = extract_claims(documents, max_claims_per_document=max_claims_per_document)
        prompt = self._prompt()
        context = self._context(documents, max_claims_per_document=max_claims_per_document)
        try:
            output = self.llm.generate_json(prompt, context=context)
            claims = self._claims_from_output(output, documents, limit=len(seed_claims) or None)
            artifact = ClaimExtractionArtifact(
                strategy="ai",
                provider=self.llm.provider,
                status="completed",
                prompt=prompt,
                fetched_documents=context["fetched_documents"],
                model_output=output,
                fallback_used=False,
                message="AI claim extraction completed.",
                claims=[claim.to_dict() for claim in claims],
            )
            return ClaimExtractionResult(claims=claims, artifact=artifact)
        except (LLMProviderError, ValueError, TypeError) as exc:
            error_code = exc.code if isinstance(exc, LLMProviderError) else exc.__class__.__name__
            artifact = ClaimExtractionArtifact(
                strategy="ai",
                provider=self.llm.provider,
                status="fallback",
                prompt=prompt,
                fetched_documents=context["fetched_documents"],
                model_output=None,
                fallback_used=True,
                error_code=error_code,
                message=str(exc),
                claims=[claim.to_dict() for claim in seed_claims],
            )
            return ClaimExtractionResult(claims=seed_claims, artifact=artifact)

    def _prompt(self) -> str:
        return (
            "Extract VeriKnow EvidenceClaim records from fetched public source pages. "
            "Return a JSON object with a claims array. Each claim must include text, source_url, "
            "source_title, quote, source_type, published_at, updated_at, confidence, freshness, "
            "caveats, and conflicts. Use only source_url values present in fetched_documents. "
            "Do not invent claims that are not supported by the page text."
        )

    def _context(self, documents: list[FetchedDocument], *, max_claims_per_document: int) -> dict[str, Any]:
        return {
            "max_claims_per_document": max_claims_per_document,
            "fetched_documents": [
                {
                    "url": document.url,
                    "title": document.title,
                    "text": document.text[:6000],
                    "fetched_at": document.fetched_at,
                    "status_code": document.status_code,
                    "error_code": document.error_code,
                    "message": document.message,
                }
                for document in documents
                if not document.error_code and document.text.strip()
            ],
        }

    def _claims_from_output(
        self,
        output: dict[str, Any],
        documents: list[FetchedDocument],
        *,
        limit: int | None,
    ) -> list[EvidenceClaim]:
        raw_claims = output.get("claims")
        if not isinstance(raw_claims, list) or not raw_claims:
            raise ValueError("model output must include a non-empty claims list")

        allowed_urls = {document.url: document for document in documents if not document.error_code}
        claims: list[EvidenceClaim] = []
        for raw_claim in raw_claims:
            if not isinstance(raw_claim, dict):
                raise ValueError("each model claim must be an object")
            text = str(raw_claim.get("text") or "").strip()
            source_url = str(raw_claim.get("source_url") or "").strip()
            if not text or not source_url:
                raise ValueError("model claims require text and source_url")
            if source_url not in allowed_urls:
                raise ValueError(f"model claim used unknown source_url: {source_url}")
            document = allowed_urls[source_url]
            claims.append(
                EvidenceClaim(
                    text=text,
                    source_url=source_url,
                    source_title=str(raw_claim.get("source_title") or document.title),
                    quote=str(raw_claim.get("quote") or text)[:500],
                    source_type=str(raw_claim.get("source_type") or "fetched_document"),
                    published_at=_optional_string(raw_claim.get("published_at")),
                    updated_at=_optional_string(raw_claim.get("updated_at")),
                    confidence=str(raw_claim.get("confidence") or "medium"),
                    freshness=str(raw_claim.get("freshness") or _freshness_for_text(text)),
                    caveats=_string_list(raw_claim.get("caveats")),
                    conflicts=_string_list(raw_claim.get("conflicts")),
                )
            )
            if limit is not None and len(claims) >= limit:
                break

        return claims

def extract_claims(documents: list[FetchedDocument], *, max_claims_per_document: int = 5) -> list[EvidenceClaim]:
    claims: list[EvidenceClaim] = []
    for document in documents:
        if document.error_code or not document.text.strip():
            continue
        for sentence in _candidate_sentences(document, limit=max_claims_per_document):
            claims.append(_claim_from_sentence(document, sentence))
    return claims


def detect_claim_conflicts(claims: list[EvidenceClaim]) -> list[ClaimConflict]:
    conflicts: list[ClaimConflict] = []
    for index, left in enumerate(claims):
        for right in claims[index + 1 :]:
            if left.source_url == right.source_url:
                continue
            topic = _shared_topic(left.text, right.text)
            if not topic:
                continue
            reason = _conflict_reason(left.text, right.text)
            if reason:
                _append_conflict(left, right, reason)
                conflicts.append(
                    ClaimConflict(
                        topic=topic,
                        claim_a=_claim_reference(left),
                        claim_b=_claim_reference(right),
                        reason=reason,
                    )
                )
    return conflicts


def _candidate_sentences(document: FetchedDocument, *, limit: int) -> list[str]:
    sentences = [item.strip() for item in SENTENCE_PATTERN.split(document.text) if item.strip()]
    if not sentences and document.text.strip():
        sentences = [document.text.strip()]

    selected: list[str] = []
    for sentence in sentences:
        normalized = _clean_sentence(sentence)
        if not normalized:
            continue
        if not selected:
            selected.append(normalized)
            continue
        lowered = normalized.lower()
        if any(keyword in lowered for keyword in CLAIM_KEYWORDS) or DATE_OR_VERSION_PATTERN.search(normalized):
            selected.append(normalized)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _claim_from_sentence(document: FetchedDocument, sentence: str) -> EvidenceClaim:
    freshness = "dated" if DATE_OR_VERSION_PATTERN.search(sentence) else "unknown"
    caveats = []
    lowered = sentence.lower()
    if "deprecated" in lowered:
        caveats.append("mentions deprecation")
    if "preview" in lowered or "beta" in lowered:
        caveats.append("mentions preview or beta status")

    return EvidenceClaim(
        text=sentence,
        source_url=document.url,
        source_title=document.title,
        quote=sentence[:500],
        source_type="fetched_document",
        confidence="medium",
        freshness=freshness,
        caveats=caveats,
    )


def _clean_sentence(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip()


def _shared_topic(left: str, right: str) -> str:
    left_tokens = _topic_tokens(left)
    right_tokens = _topic_tokens(right)
    shared = sorted(left_tokens & right_tokens)
    if not shared:
        return ""
    return " ".join(shared[:5])


def _topic_tokens(text: str) -> set[str]:
    ignored = {
        "available",
        "deprecated",
        "introduced",
        "latest",
        "new",
        "recommended",
        "requires",
        "stable",
        "supports",
        "supported",
        "updated",
        "version",
    }
    return {
        match.group(0).lower()
        for match in TOKEN_PATTERN.finditer(text)
        if match.group(0).lower() not in ignored
    }


def _conflict_reason(left: str, right: str) -> str:
    left_status = _claim_status(left)
    right_status = _claim_status(right)
    if {left_status, right_status} == {"positive", "negative"}:
        return "different sources make opposing availability or support claims"
    return ""


def _claim_status(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in NEGATIVE_TERMS):
        return "negative"
    if any(term in lowered for term in POSITIVE_TERMS):
        return "positive"
    return "neutral"


def _append_conflict(left: EvidenceClaim, right: EvidenceClaim, reason: str) -> None:
    left_label = f"{right.source_title or right.source_url}: {reason}"
    right_label = f"{left.source_title or left.source_url}: {reason}"
    if left_label not in left.conflicts:
        left.conflicts.append(left_label)
    if right_label not in right.conflicts:
        right.conflicts.append(right_label)


def _claim_reference(claim: EvidenceClaim) -> dict:
    return {
        "text": claim.text,
        "source_url": claim.source_url,
        "source_title": claim.source_title,
    }

def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _freshness_for_text(text: str) -> str:
    return "dated" if DATE_OR_VERSION_PATTERN.search(text) else "unknown"
