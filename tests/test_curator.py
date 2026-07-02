from pathlib import Path
import unittest

from veriknow.modules.curator import KnowledgeCurator
from veriknow.schemas import RunRecord, TaskSpec


class CuratorTests(unittest.TestCase):
    def test_curator_generates_diff_without_overwriting_existing_knowledge(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            knowledge_dir = tmp_path / "knowledge"
            existing_path = knowledge_dir / "agents" / "langchain-supervisor.md"
            existing_path.parent.mkdir(parents=True)
            existing_path.write_text("# LangChain Supervisor\n\nOld workflow.\n", encoding="utf-8")
            report_path = tmp_path / "run" / "report.md"
            report_path.parent.mkdir(parents=True)
            report_path.write_text("# LangChain Supervisor\n\nUpdated workflow.\n", encoding="utf-8")
            record = RunRecord(
                run_id="run-test",
                raw_request="Research LangChain supervisor",
                task=TaskSpec(
                    raw_request="Research LangChain supervisor",
                    objective="Research",
                    target="LangChain Supervisor",
                ),
            )

            curator = KnowledgeCurator()
            patch = curator.create_patch(record, report_path, knowledge_dir)
            diff_path, patch_path = curator.write_patch_files(patch, report_path.parent)

            self.assertEqual(Path(patch.target_path), existing_path)
            self.assertFalse(patch.approved)
            self.assertIn("-Old workflow.", patch.diff)
            self.assertIn("+Updated workflow.", patch.diff)
            self.assertEqual(existing_path.read_text(encoding="utf-8"), "# LangChain Supervisor\n\nOld workflow.\n")
            self.assertTrue(diff_path.exists())
            self.assertTrue(patch_path.exists())

            approved = curator.apply_patch(patch, report_path, knowledge_dir, patch_path)

            self.assertTrue(approved.approved)
            self.assertEqual(existing_path.read_text(encoding="utf-8"), "# LangChain Supervisor\n\nUpdated workflow.\n")


    def test_curator_writes_deterministic_merge_proposal(self) -> None:
        from tempfile import TemporaryDirectory
        import json

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            knowledge_dir = tmp_path / "knowledge"
            existing_path = knowledge_dir / "general" / "example.md"
            existing_path.parent.mkdir(parents=True)
            existing_path.write_text("# Example\n\nOld workflow.\n", encoding="utf-8")
            report_path = tmp_path / "run" / "report.md"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                "# Example\n\nUpdated workflow from https://example.com/docs.\n\nDeprecated older source.\n",
                encoding="utf-8",
            )
            record = RunRecord(
                run_id="run-test",
                raw_request="Research example",
                task=TaskSpec(
                    raw_request="Research example",
                    objective="Research",
                    target="Example",
                ),
            )

            curator = KnowledgeCurator()
            patch = curator.create_patch(record, report_path, knowledge_dir)
            proposal = curator.create_merge_proposal(record, patch, report_path)
            curator.write_patch_files(patch, report_path.parent, proposal=proposal)

            proposal_path = report_path.parent / "knowledge_merge_proposal.json"
            data = json.loads(proposal_path.read_text(encoding="utf-8"))
            self.assertTrue(proposal_path.exists())
            self.assertEqual(data["operation"], "update")
            self.assertEqual(data["target_path"], str(existing_path))
            self.assertEqual(data["evidence_urls"], ["https://example.com/docs"])
            self.assertEqual(data["risk_level"], "high")
            self.assertIn("Deprecated", data["conflicts"][0])
            self.assertEqual(existing_path.read_text(encoding="utf-8"), "# Example\n\nOld workflow.\n")
    def test_apply_rejects_targets_outside_knowledge_directory(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            report_path = tmp_path / "report.md"
            report_path.write_text("# Report\n", encoding="utf-8")
            record = RunRecord(
                run_id="run-test",
                raw_request="Research example",
                task=TaskSpec(
                    raw_request="Research example",
                    objective="Research",
                    target="example",
                ),
            )
            patch = KnowledgeCurator().create_patch(record, report_path, tmp_path / "knowledge")
            patch.target_path = str(tmp_path / "outside.md")

            with self.assertRaises(ValueError):
                KnowledgeCurator().apply_patch(patch, report_path, tmp_path / "knowledge")
