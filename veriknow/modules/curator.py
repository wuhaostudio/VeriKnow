from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from veriknow.llm import LLMClient, LLMProviderError
from veriknow.modules.knowledge import KnowledgeDocument, MarkdownKnowledgeIndex
from veriknow.schemas import KnowledgeMergeProposal, KnowledgePatch, RunRecord


SUPPORTED_CURATION_STRATEGIES = {"deterministic", "ai"}


class KnowledgeCurator:
    def __init__(self, indexer: MarkdownKnowledgeIndex | None = None):
        self.indexer = indexer or MarkdownKnowledgeIndex()

    def index(self, knowledge_dir: Path) -> list[KnowledgeDocument]:
        return self.indexer.index(knowledge_dir)

    def find_related(
        self,
        record: RunRecord,
        report_content: str,
        knowledge_dir: Path,
        *,
        limit: int = 5,
    ) -> list[KnowledgeDocument]:
        results = self.indexer.related_for_run(
            record,
            knowledge_dir,
            extra_text=report_content,
            limit=limit,
        )
        return [
            KnowledgeDocument(
                path=result.path,
                title=result.title,
                content=result.path.read_text(encoding="utf-8"),
            )
            for result in results
        ]

    def create_patch(
        self,
        record: RunRecord,
        report_path: Path,
        knowledge_dir: Path,
    ) -> KnowledgePatch:
        if not report_path.exists():
            raise FileNotFoundError(f"report not found: {report_path}")

        report_content = report_path.read_text(encoding="utf-8")
        related = self.find_related(record, report_content, knowledge_dir, limit=1)
        target_path = related[0].path if related else _new_knowledge_path(record, knowledge_dir)
        original_content = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
        diff = _unified_diff(
            original_content,
            report_content,
            fromfile=str(target_path),
            tofile=str(report_path),
        )
        return KnowledgePatch(
            run_id=record.run_id,
            target_path=str(target_path),
            diff=diff,
            approved=False,
        )

    def create_patch_for_target(
        self,
        record: RunRecord,
        report_path: Path,
        target_path: Path,
        knowledge_dir: Path,
    ) -> KnowledgePatch:
        if not report_path.exists():
            raise FileNotFoundError(f"report not found: {report_path}")

        target_resolved = target_path.resolve()
        knowledge_resolved = knowledge_dir.resolve()
        if not target_resolved.is_relative_to(knowledge_resolved):
            raise ValueError(f"patch target is outside knowledge directory: {target_path}")

        report_content = report_path.read_text(encoding="utf-8")
        original_content = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
        diff = _unified_diff(
            original_content,
            report_content,
            fromfile=str(target_path),
            tofile=str(report_path),
        )
        return KnowledgePatch(
            run_id=record.run_id,
            target_path=str(target_path),
            diff=diff,
            approved=False,
        )

    def create_merge_proposal(
        self,
        record: RunRecord,
        patch: KnowledgePatch,
        report_path: Path,
    ) -> KnowledgeMergeProposal:
        if not report_path.exists():
            raise FileNotFoundError(f"report not found: {report_path}")

        target_path = Path(patch.target_path)
        report_content = report_path.read_text(encoding="utf-8")
        target_exists = target_path.exists()
        operation = "update" if target_exists else "create"
        target_title = _title_from_content(report_content, target_path)
        evidence_urls = _extract_urls(report_content)
        conflicts = _extract_conflict_lines(report_content)
        risk_level = _risk_level_for(operation, conflicts)
        rationale = _proposal_rationale(operation, target_title, evidence_urls, conflicts)
        return KnowledgeMergeProposal(
            run_id=record.run_id,
            operation=operation,
            target_path=str(target_path),
            target_title=target_title,
            rationale=rationale,
            evidence_urls=evidence_urls,
            conflicts=conflicts,
            diff=patch.diff,
            risk_level=risk_level,
        )

    def write_patch_files(
        self,
        patch: KnowledgePatch,
        run_dir: Path,
        proposal: KnowledgeMergeProposal | None = None,
    ) -> tuple[Path, Path]:
        diff_path = run_dir / "patch.diff"
        patch_path = run_dir / "knowledge_patch.json"
        diff_path.write_text(patch.diff, encoding="utf-8")
        patch_path.write_text(
            json.dumps(patch.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if proposal is not None:
            proposal_path = run_dir / "knowledge_merge_proposal.json"
            proposal_path.write_text(
                json.dumps(proposal.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return diff_path, patch_path

    def apply_patch(
        self,
        patch: KnowledgePatch,
        report_path: Path,
        knowledge_dir: Path,
        patch_path: Path | None = None,
    ) -> KnowledgePatch:
        if not report_path.exists():
            raise FileNotFoundError(f"report not found: {report_path}")
        target_path = Path(patch.target_path)
        target_resolved = target_path.resolve()
        knowledge_resolved = knowledge_dir.resolve()
        if not target_resolved.is_relative_to(knowledge_resolved):
            raise ValueError(f"patch target is outside knowledge directory: {target_path}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

        approved = KnowledgePatch(
            run_id=patch.run_id,
            target_path=patch.target_path,
            diff=patch.diff,
            approved=True,
            created_at=patch.created_at,
        )
        if patch_path is not None:
            patch_path.write_text(
                json.dumps(approved.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return approved

@dataclass(frozen=True)
class CurationArtifact:
    strategy: str
    provider: str
    status: str
    prompt: str
    seed_proposal: dict[str, Any]
    model_output: dict[str, Any] | None = None
    fallback_used: bool = False
    error_code: str | None = None
    message: str = ""
    proposal: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "provider": self.provider,
            "status": self.status,
            "prompt": self.prompt,
            "seed_proposal": self.seed_proposal,
            "model_output": self.model_output,
            "fallback_used": self.fallback_used,
            "error_code": self.error_code,
            "message": self.message,
            "proposal": self.proposal,
        }


@dataclass(frozen=True)
class CurationResult:
    proposal: KnowledgeMergeProposal
    artifact: CurationArtifact | None = None


class AIKnowledgeCurator:
    def __init__(
        self,
        llm: LLMClient,
        base: KnowledgeCurator | None = None,
    ):
        self.llm = llm
        self.base = base or KnowledgeCurator()

    def create_merge_proposal(
        self,
        record: RunRecord,
        patch: KnowledgePatch,
        report_path: Path,
        *,
        related_documents: list[KnowledgeDocument] | None = None,
    ) -> CurationResult:
        seed = self.base.create_merge_proposal(record, patch, report_path)
        prompt = self._prompt_for()
        try:
            output = self.llm.generate_json(
                prompt,
                context=self._context_for(
                    record,
                    patch,
                    seed,
                    report_path,
                    related_documents=related_documents,
                ),
            )
            proposal = self._proposal_from_output(output, seed=seed, patch=patch)
            artifact = CurationArtifact(
                strategy="ai",
                provider=self.llm.provider,
                status="completed",
                prompt=prompt,
                seed_proposal=seed.to_dict(),
                model_output=output,
                fallback_used=False,
                message="AI knowledge merge proposal completed.",
                proposal=proposal.to_dict(),
            )
            return CurationResult(proposal=proposal, artifact=artifact)
        except (LLMProviderError, ValueError, TypeError) as exc:
            error_code = exc.code if isinstance(exc, LLMProviderError) else exc.__class__.__name__
            artifact = CurationArtifact(
                strategy="ai",
                provider=self.llm.provider,
                status="fallback",
                prompt=prompt,
                seed_proposal=seed.to_dict(),
                model_output=None,
                fallback_used=True,
                error_code=error_code,
                message=str(exc),
                proposal=seed.to_dict(),
            )
            return CurationResult(proposal=seed, artifact=artifact)

    def _prompt_for(self) -> str:
        return (
            "Create a VeriKnow KnowledgeMergeProposal JSON object from the supplied report, "
            "patch, seed proposal, and related knowledge documents. Return fields: operation, "
            "target_path, target_title, rationale, evidence_urls, conflicts, diff, risk_level. "
            "Allowed operations are create, update, append, replace_section, mark_stale. "
            "Every substantial new claim must be supported by an evidence URL. Preserve unresolved "
            "conflicts instead of silently resolving them. Do not change target_path outside the "
            "seed proposal target."
        )

    def _context_for(
        self,
        record: RunRecord,
        patch: KnowledgePatch,
        seed: KnowledgeMergeProposal,
        report_path: Path,
        *,
        related_documents: list[KnowledgeDocument] | None = None,
    ) -> dict[str, Any]:
        return {
            "run": record.to_dict(),
            "patch": patch.to_dict(),
            "seed_proposal": seed.to_dict(),
            "report": {
                "path": str(report_path),
                "content": report_path.read_text(encoding="utf-8"),
            },
            "related_documents": [
                {
                    "path": str(document.path),
                    "title": document.title,
                    "content": document.content,
                }
                for document in related_documents or []
            ],
        }

    def _proposal_from_output(
        self,
        output: dict[str, Any],
        *,
        seed: KnowledgeMergeProposal,
        patch: KnowledgePatch,
    ) -> KnowledgeMergeProposal:
        operation = str(output.get("operation", "")).strip()
        if operation not in {"create", "update", "append", "replace_section", "mark_stale"}:
            raise ValueError(f"unsupported merge operation: {operation}")

        target_path = str(output.get("target_path", "")).strip()
        if target_path != seed.target_path:
            raise ValueError("model merge proposal cannot change the selected target_path")

        target_title = str(output.get("target_title", "")).strip()
        rationale = str(output.get("rationale", "")).strip()
        diff = str(output.get("diff", ""))
        risk_level = str(output.get("risk_level", "medium")).strip()
        if not target_title or not rationale:
            raise ValueError("model merge proposal requires target_title and rationale")
        if risk_level not in {"low", "medium", "high"}:
            raise ValueError(f"unsupported merge risk_level: {risk_level}")
        if not diff.strip():
            raise ValueError("model merge proposal requires a diff")
        if not _same_diff_effect(diff, patch.diff):
            raise ValueError("model merge proposal diff must preserve the generated patch content")

        evidence_urls = _string_list(output.get("evidence_urls"))
        if operation in {"create", "update", "append", "replace_section"} and not evidence_urls:
            raise ValueError("model merge proposal requires evidence_urls for content changes")

        return KnowledgeMergeProposal(
            run_id=seed.run_id,
            operation=operation,
            target_path=seed.target_path,
            target_title=target_title,
            rationale=rationale,
            evidence_urls=evidence_urls,
            conflicts=_string_list(output.get("conflicts")),
            diff=diff,
            risk_level=risk_level,
        )



def load_knowledge_patch(path: Path) -> KnowledgePatch:
    if not path.exists():
        raise FileNotFoundError(f"knowledge patch not found: {path}")
    return KnowledgePatch.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _new_knowledge_path(record: RunRecord, knowledge_dir: Path) -> Path:
    slug = _slugify(record.task.target or record.run_id)
    return knowledge_dir / "general" / f"{slug}.md"


def _slugify(value: str) -> str:
    ascii_slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if ascii_slug:
        return ascii_slug[:80]
    safe = re.sub(r"\s+", "-", value.strip()).strip("-")
    safe = re.sub(r'[<>:"/\\|?*]+', "", safe)
    return (safe or "knowledge")[:80]


def _unified_diff(original: str, updated: str, *, fromfile: str, tofile: str) -> str:
    lines = difflib.unified_diff(
        original.splitlines(),
        updated.splitlines(),
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
    )
    return "\n".join(lines) + "\n"


def load_knowledge_merge_proposal(path: Path) -> KnowledgeMergeProposal:
    if not path.exists():
        raise FileNotFoundError(f"knowledge merge proposal not found: {path}")
    return KnowledgeMergeProposal.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _title_from_content(content: str, fallback_path: Path) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback_path.stem.replace("-", " ").replace("_", " ").strip()


def _extract_urls(content: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"https?://[^\s)\]>\"']+", content):
        url = match.group(0).rstrip(".,;")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _extract_conflict_lines(content: str) -> list[str]:
    conflicts: list[str] = []
    for line in content.splitlines():
        stripped = line.strip(" -\t")
        lowered = stripped.lower()
        if stripped and any(term in lowered for term in ["conflict", "contradict", "outdated", "deprecated"]):
            conflicts.append(stripped[:240])
    return conflicts[:20]


def _risk_level_for(operation: str, conflicts: list[str]) -> str:
    if conflicts:
        return "high"
    if operation == "update":
        return "medium"
    return "low"


def _proposal_rationale(operation: str, title: str, evidence_urls: list[str], conflicts: list[str]) -> str:
    action = "Update existing" if operation == "update" else "Create new"
    support = f"{len(evidence_urls)} evidence URL(s)" if evidence_urls else "no explicit evidence URLs"
    conflict_note = f" and {len(conflicts)} conflict marker(s)" if conflicts else ""
    return f"{action} knowledge document for {title} using {support}{conflict_note}."


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("model merge proposal list fields must be lists")
    return [str(item).strip() for item in value if str(item).strip()]


def _same_diff_effect(candidate: str, expected: str) -> bool:
    return candidate.strip() == expected.strip()
