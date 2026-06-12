from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from veriknow.modules.knowledge import KnowledgeDocument, MarkdownKnowledgeIndex
from veriknow.schemas import KnowledgePatch, RunRecord


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

    def write_patch_files(
        self,
        patch: KnowledgePatch,
        run_dir: Path,
    ) -> tuple[Path, Path]:
        diff_path = run_dir / "patch.diff"
        patch_path = run_dir / "knowledge_patch.json"
        diff_path.write_text(patch.diff, encoding="utf-8")
        patch_path.write_text(
            json.dumps(patch.to_dict(), ensure_ascii=False, indent=2),
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
