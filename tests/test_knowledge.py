from pathlib import Path
from datetime import date
import json
import unittest

from veriknow.modules.knowledge import MarkdownKnowledgeIndex, parse_front_matter, title_from_markdown
from veriknow.schemas import RunRecord, TaskSpec


class KnowledgeTests(unittest.TestCase):
    def test_keyword_search_returns_ranked_markdown_results(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            knowledge_dir = Path(directory) / "knowledge"
            langchain_path = knowledge_dir / "agents" / "langchain-supervisor.md"
            other_path = knowledge_dir / "notes" / "browser.md"
            langchain_path.parent.mkdir(parents=True)
            other_path.parent.mkdir(parents=True)
            langchain_path.write_text(
                "# LangChain Supervisor\n\nMulti-agent supervisor workflow with LangGraph.\n",
                encoding="utf-8",
            )
            other_path.write_text("# Browser Notes\n\nPlaywright screenshots.\n", encoding="utf-8")

            results = MarkdownKnowledgeIndex().search("LangChain multi-agent", knowledge_dir)

            self.assertEqual(results[0].path, langchain_path)
            self.assertEqual(results[0].title, "LangChain Supervisor")
            self.assertGreater(results[0].score, 0)
            self.assertIn("Multi-agent", results[0].snippet)

    def test_related_results_can_be_written_for_run(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            knowledge_dir = tmp_path / "knowledge"
            run_dir = tmp_path / "run"
            run_dir.mkdir()
            doc_path = knowledge_dir / "agents" / "langchain-supervisor.md"
            doc_path.parent.mkdir(parents=True)
            doc_path.write_text("# LangChain Supervisor\n\nSupervisor agent guide.\n", encoding="utf-8")
            record = RunRecord(
                run_id="run-test",
                raw_request="Research LangChain Supervisor",
                task=TaskSpec(
                    raw_request="Research LangChain Supervisor",
                    objective="Research",
                    target="LangChain Supervisor",
                ),
            )
            indexer = MarkdownKnowledgeIndex()

            results = indexer.related_for_run(record, knowledge_dir)
            related_path = indexer.write_related(results, run_dir)

            related = json.loads(related_path.read_text(encoding="utf-8"))
            self.assertEqual(related[0]["path"], str(doc_path))
            self.assertEqual(related[0]["title"], "LangChain Supervisor")

    def test_front_matter_title_and_stale_documents(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            knowledge_dir = Path(directory) / "knowledge"
            due_path = knowledge_dir / "due.md"
            future_path = knowledge_dir / "future.md"
            missing_path = knowledge_dir / "missing.md"
            due_path.parent.mkdir(parents=True)
            due_path.write_text(
                "---\n"
                'title: "Due Doc"\n'
                'next_verify_at: "2026-06-01"\n'
                "---\n\n"
                "# Ignored Heading\n",
                encoding="utf-8",
            )
            future_path.write_text(
                "---\n"
                'title: "Future Doc"\n'
                'next_verify_at: "2026-07-01"\n'
                "---\n\n"
                "# Future\n",
                encoding="utf-8",
            )
            missing_path.write_text("# Missing Metadata\n", encoding="utf-8")

            indexer = MarkdownKnowledgeIndex()
            stale = indexer.stale_documents(knowledge_dir, as_of=date(2026, 6, 12))

            self.assertEqual(title_from_markdown(due_path.read_text(encoding="utf-8"), due_path), "Due Doc")
            self.assertEqual(parse_front_matter(due_path.read_text(encoding="utf-8"))["next_verify_at"], "2026-06-01")
            self.assertEqual([item.path for item in stale], [due_path, missing_path])
            self.assertEqual(stale[0].reason, "due")
            self.assertEqual(stale[1].reason, "missing next_verify_at")
