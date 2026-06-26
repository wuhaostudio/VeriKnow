from __future__ import annotations

import re

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


def extract_claims(documents: list[FetchedDocument], *, max_claims_per_document: int = 5) -> list[EvidenceClaim]:
    claims: list[EvidenceClaim] = []
    for document in documents:
        if document.error_code or not document.text.strip():
            continue
        for sentence in _candidate_sentences(document, limit=max_claims_per_document):
            claims.append(_claim_from_sentence(document, sentence))
    return claims


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
