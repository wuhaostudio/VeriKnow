from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from veriknow.schemas import RunRecord


TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)


@dataclass(frozen=True)
class KnowledgeDocument:
    path: Path
    title: str
    content: str
    front_matter: dict[str, str] | None = None


@dataclass(frozen=True)
class KnowledgeSearchResult:
    path: Path
    title: str
    score: int
    snippet: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "path": str(self.path),
            "title": self.title,
            "score": self.score,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class StaleKnowledgeDocument:
    path: Path
    title: str
    next_verify_at: str | None
    reason: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "path": str(self.path),
            "title": self.title,
            "next_verify_at": self.next_verify_at,
            "reason": self.reason,
        }


class MarkdownKnowledgeIndex:
    def index(self, knowledge_dir: Path) -> list[KnowledgeDocument]:
        if not knowledge_dir.exists():
            return []

        documents: list[KnowledgeDocument] = []
        for path in sorted(knowledge_dir.rglob("*.md")):
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8")
            front_matter = parse_front_matter(content)
            documents.append(
                KnowledgeDocument(
                    path=path,
                    title=title_from_markdown(content, path),
                    content=content,
                    front_matter=front_matter,
                )
            )
        return documents

    def search(
        self,
        query: str,
        knowledge_dir: Path,
        *,
        limit: int = 10,
    ) -> list[KnowledgeSearchResult]:
        query_tokens = set(tokens(query))
        if not query_tokens:
            return []

        results: list[KnowledgeSearchResult] = []
        for document in self.index(knowledge_dir):
            searchable = f"{document.title}\n{document.content}"
            document_tokens = tokens(searchable)
            score = sum(1 for token in document_tokens if token in query_tokens)
            if score <= 0:
                continue
            results.append(
                KnowledgeSearchResult(
                    path=document.path,
                    title=document.title,
                    score=score,
                    snippet=snippet_for(document.content, query_tokens),
                )
            )
        return sorted(results, key=lambda item: (-item.score, str(item.path)))[:limit]

    def related_for_run(
        self,
        record: RunRecord,
        knowledge_dir: Path,
        *,
        extra_text: str = "",
        limit: int = 5,
    ) -> list[KnowledgeSearchResult]:
        query = " ".join(
            [
                record.task.target,
                record.task.objective,
                record.raw_request,
                extra_text[:2000],
            ]
        )
        return self.search(query, knowledge_dir, limit=limit)

    def write_related(
        self,
        results: list[KnowledgeSearchResult],
        run_dir: Path,
    ) -> Path:
        path = run_dir / "related_knowledge.json"
        path.write_text(
            json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def stale_documents(
        self,
        knowledge_dir: Path,
        *,
        as_of: date | None = None,
        include_missing: bool = True,
    ) -> list[StaleKnowledgeDocument]:
        today = as_of or datetime.now(timezone.utc).date()
        stale: list[StaleKnowledgeDocument] = []
        for document in self.index(knowledge_dir):
            next_verify_at = None
            if document.front_matter:
                next_verify_at = document.front_matter.get("next_verify_at")

            if not next_verify_at:
                if include_missing:
                    stale.append(
                        StaleKnowledgeDocument(
                            path=document.path,
                            title=document.title,
                            next_verify_at=None,
                            reason="missing next_verify_at",
                        )
                    )
                continue

            due_date = parse_front_matter_date(next_verify_at)
            if due_date is None:
                stale.append(
                    StaleKnowledgeDocument(
                        path=document.path,
                        title=document.title,
                        next_verify_at=next_verify_at,
                        reason="invalid next_verify_at",
                    )
                )
                continue

            if due_date <= today:
                stale.append(
                    StaleKnowledgeDocument(
                        path=document.path,
                        title=document.title,
                        next_verify_at=next_verify_at,
                        reason="due",
                    )
                )
        return stale


def title_from_markdown(content: str, path: Path) -> str:
    front_matter = parse_front_matter(content)
    if front_matter and front_matter.get("title"):
        return front_matter["title"]
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return path.stem.replace("-", " ").replace("_", " ").strip()


def parse_front_matter(content: str) -> dict[str, str] | None:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    values: dict[str, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            return values
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        values[key.strip()] = _unquote_front_matter(value.strip())
    return None


def parse_front_matter_date(value: str) -> date | None:
    normalized = value.strip().strip('"').strip("'")
    if not normalized:
        return None
    try:
        return date.fromisoformat(normalized[:10])
    except ValueError:
        return None


def tokens(value: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(value)]


def snippet_for(content: str, query_tokens: set[str]) -> str:
    best_line = ""
    best_score = 0
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        line_tokens = set(tokens(stripped))
        score = len(query_tokens & line_tokens)
        if score > best_score:
            best_line = stripped
            best_score = score
    if best_line:
        return best_line[:240]
    return content.strip().replace("\n", " ")[:240]


def _unquote_front_matter(value: str) -> str:
    if value[0:1] == value[-1:] and value[0:1] in {"'", '"'}:
        return value[1:-1]
    return value
