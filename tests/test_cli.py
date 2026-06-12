from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import unittest
from unittest.mock import patch

from veriknow.cli import main


class CliTests(unittest.TestCase):
    def test_research_command_creates_evidence_output(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = StringIO()

            with redirect_stdout(stdout):
                main(["research", "LangChain multi-agent supervisor workflow", "--config", str(config_path)])

            output = json.loads(stdout.getvalue())
            self.assertTrue(output["items"])
            self.assertIn("url", output["items"][0])
            self.assertIn("confidence", output["items"][0])
            self.assertTrue((tmp_path / "data" / "runs").exists())

    def test_research_command_writes_related_knowledge_artifact(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data_dir = tmp_path / "data"
            knowledge_path = data_dir / "knowledge" / "agents" / "langchain-supervisor.md"
            knowledge_path.parent.mkdir(parents=True)
            knowledge_path.write_text(
                "# LangChain Supervisor\n\nMulti-agent supervisor workflow with LangGraph.\n",
                encoding="utf-8",
            )
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {data_dir}",
                        f"database_path: {data_dir / 'memory.sqlite'}",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = StringIO()

            with redirect_stdout(stdout):
                main(["research", "LangChain multi-agent supervisor workflow", "--config", str(config_path)])

            output = json.loads(stdout.getvalue())
            related_path = data_dir / "runs" / output["task_id"] / "related_knowledge.json"
            related = json.loads(related_path.read_text(encoding="utf-8"))
            self.assertEqual(related[0]["path"], str(knowledge_path))
            self.assertEqual(related[0]["title"], "LangChain Supervisor")

    def test_kb_search_command_returns_local_markdown_results(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data_dir = tmp_path / "data"
            knowledge_path = data_dir / "knowledge" / "agents" / "langchain-supervisor.md"
            knowledge_path.parent.mkdir(parents=True)
            knowledge_path.write_text(
                "# LangChain Supervisor\n\nMulti-agent supervisor workflow with LangGraph.\n",
                encoding="utf-8",
            )
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {data_dir}",
                        f"database_path: {data_dir / 'memory.sqlite'}",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = StringIO()

            with redirect_stdout(stdout):
                main(["kb", "search", "LangChain multi-agent", "--config", str(config_path)])

            output = json.loads(stdout.getvalue())
            self.assertEqual(output[0]["path"], str(knowledge_path))
            self.assertEqual(output[0]["title"], "LangChain Supervisor")
            self.assertIn("Multi-agent", output[0]["snippet"])

    def test_publish_command_records_feishu_stub_job(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data_dir = tmp_path / "data"
            config_path = tmp_path / "config.yaml"
            knowledge_path = data_dir / "knowledge" / "agents" / "langchain-supervisor.md"
            knowledge_path.parent.mkdir(parents=True)
            knowledge_path.write_text("# LangChain Supervisor\n\nOld workflow.\n", encoding="utf-8")
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {data_dir}",
                        f"database_path: {data_dir / 'memory.sqlite'}",
                    ]
                ),
                encoding="utf-8",
            )
            run_stdout = StringIO()
            with redirect_stdout(run_stdout):
                main(["run", "Research LangChain Supervisor", "--dry-run", "--config", str(config_path)])
            run_id = json.loads(run_stdout.getvalue())["run_id"]
            with redirect_stdout(StringIO()):
                main(["write", run_id, "--config", str(config_path)])
            with redirect_stdout(StringIO()):
                main(["curate", run_id, "--config", str(config_path)])
            apply_stdout = StringIO()
            with redirect_stdout(apply_stdout):
                main(["apply", run_id, "--config", str(config_path)])
            document_path = json.loads(apply_stdout.getvalue())["target_path"]
            stdout = StringIO()

            with patch.dict("os.environ", {}, clear=True):
                with redirect_stdout(stdout):
                    main(["publish", str(document_path), "--target", "feishu", "--config", str(config_path)])

            output = json.loads(stdout.getvalue())
            self.assertEqual(output["target"], "feishu")
            self.assertEqual(output["status"], "blocked")
            self.assertEqual(output["error_code"], "missing_credentials")
            self.assertIn("FEISHU_APP_ID", output["message"])

            publications_stdout = StringIO()
            with redirect_stdout(publications_stdout):
                main(["memory", "publications", "--config", str(config_path)])

            self.assertIn(str(document_path), publications_stdout.getvalue())
            self.assertIn("missing_credentials", publications_stdout.getvalue())

    def test_publish_command_rejects_unapproved_knowledge_document(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data_dir = tmp_path / "data"
            document_path = data_dir / "knowledge" / "general" / "example.md"
            document_path.parent.mkdir(parents=True)
            document_path.write_text("# Example\n", encoding="utf-8")
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {data_dir}",
                        f"database_path: {data_dir / 'memory.sqlite'}",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                main(["publish", str(document_path), "--target", "feishu", "--config", str(config_path)])

    def test_plan_command_creates_verification_artifacts(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                    ]
                ),
                encoding="utf-8",
            )
            research_stdout = StringIO()
            with redirect_stdout(research_stdout):
                main(["research", "LangChain multi-agent supervisor workflow", "--config", str(config_path)])
            run_id = json.loads(research_stdout.getvalue())["task_id"]

            plan_stdout = StringIO()
            with redirect_stdout(plan_stdout):
                main(["plan", run_id, "--config", str(config_path)])

            output = json.loads(plan_stdout.getvalue())
            run_dir = tmp_path / "data" / "runs" / run_id
            self.assertEqual(output["task_id"], run_id)
            self.assertTrue(output["steps"])
            self.assertIn("expected_result", output["steps"][0])
            self.assertTrue((run_dir / "verification_plan.json").exists())
            self.assertTrue((run_dir / "verification_checklist.md").exists())

    def test_verify_command_creates_verification_artifact(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                    ]
                ),
                encoding="utf-8",
            )
            research_stdout = StringIO()
            with redirect_stdout(research_stdout):
                main(["research", "LangChain multi-agent supervisor workflow", "--config", str(config_path)])
            run_id = json.loads(research_stdout.getvalue())["task_id"]

            with redirect_stdout(StringIO()):
                main(["plan", run_id, "--config", str(config_path)])

            verify_stdout = StringIO()
            with redirect_stdout(verify_stdout):
                main(["verify", run_id, "--config", str(config_path)])

            output = json.loads(verify_stdout.getvalue())
            run_dir = tmp_path / "data" / "runs" / run_id
            self.assertEqual(output["task_id"], run_id)
            self.assertTrue(output["results"])
            self.assertTrue((run_dir / "verification.json").exists())
            self.assertTrue((run_dir / "screenshots" / "step-01.png").exists())
            self.assertTrue((run_dir / "logs" / "step-01.log").exists())

    def test_verify_command_supports_computer_use_mode(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                        "computer_use_domain_allowlist: docs.langchain.com, langchain-ai.github.io, github.com",
                    ]
                ),
                encoding="utf-8",
            )
            research_stdout = StringIO()
            with redirect_stdout(research_stdout):
                main(["research", "LangChain multi-agent supervisor workflow", "--config", str(config_path)])
            run_id = json.loads(research_stdout.getvalue())["task_id"]

            with redirect_stdout(StringIO()):
                main(["plan", run_id, "--config", str(config_path)])

            verify_stdout = StringIO()
            with redirect_stdout(verify_stdout):
                main(["verify", run_id, "--mode", "computer-use", "--config", str(config_path)])

            output = json.loads(verify_stdout.getvalue())
            run_dir = tmp_path / "data" / "runs" / run_id
            self.assertEqual(output["task_id"], run_id)
            self.assertEqual(output["status"], "partial")
            self.assertIn("open isolated computer-use browser", output["results"][0]["actions"])
            self.assertTrue((run_dir / "verification.json").exists())
            self.assertTrue((run_dir / "screenshots" / "step-01.png").exists())
            self.assertTrue((run_dir / "logs" / "step-01.log").exists())

    def test_write_command_creates_operation_report(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                    ]
                ),
                encoding="utf-8",
            )
            research_stdout = StringIO()
            with redirect_stdout(research_stdout):
                main(["research", "LangChain multi-agent supervisor workflow", "--config", str(config_path)])
            run_id = json.loads(research_stdout.getvalue())["task_id"]

            with redirect_stdout(StringIO()):
                main(["plan", run_id, "--config", str(config_path)])
            with redirect_stdout(StringIO()):
                main(["verify", run_id, "--config", str(config_path)])

            write_stdout = StringIO()
            with redirect_stdout(write_stdout):
                main(["write", run_id, "--config", str(config_path)])

            output = json.loads(write_stdout.getvalue())
            report_path = tmp_path / "data" / "runs" / run_id / "report.md"
            report = report_path.read_text(encoding="utf-8")
            self.assertEqual(output["artifacts"]["report"], str(report_path))
            self.assertIn("## Step-by-Step Guide", report)
            self.assertIn("## Verification Summary", report)
            self.assertIn("## Screenshots", report)
            self.assertIn("screenshots/step-01.png", report)

    def test_curate_and_apply_commands_update_knowledge_only_after_apply(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            data_dir = tmp_path / "data"
            knowledge_path = data_dir / "knowledge" / "agents" / "langchain-supervisor.md"
            knowledge_path.parent.mkdir(parents=True)
            knowledge_path.write_text("# LangChain Supervisor\n\nOld workflow.\n", encoding="utf-8")
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {data_dir}",
                        f"database_path: {data_dir / 'memory.sqlite'}",
                    ]
                ),
                encoding="utf-8",
            )
            run_stdout = StringIO()
            with redirect_stdout(run_stdout):
                main(["run", "Research LangChain Supervisor", "--dry-run", "--config", str(config_path)])
            run_id = json.loads(run_stdout.getvalue())["run_id"]

            with redirect_stdout(StringIO()):
                main(["write", run_id, "--config", str(config_path)])

            curate_stdout = StringIO()
            with redirect_stdout(curate_stdout):
                main(["curate", run_id, "--config", str(config_path)])

            patch = json.loads(curate_stdout.getvalue())
            run_dir = data_dir / "runs" / run_id
            self.assertEqual(patch["target_path"], str(knowledge_path))
            self.assertFalse(patch["approved"])
            self.assertTrue((run_dir / "patch.diff").exists())
            self.assertTrue((run_dir / "knowledge_patch.json").exists())
            self.assertEqual(knowledge_path.read_text(encoding="utf-8"), "# LangChain Supervisor\n\nOld workflow.\n")

            apply_stdout = StringIO()
            with redirect_stdout(apply_stdout):
                main(["apply", run_id, "--config", str(config_path)])

            approved = json.loads(apply_stdout.getvalue())
            self.assertTrue(approved["approved"])
            self.assertIn("# LangChain Supervisor", knowledge_path.read_text(encoding="utf-8"))
            self.assertNotIn("Old workflow.", knowledge_path.read_text(encoding="utf-8"))

    def test_stale_command_lists_due_knowledge_documents(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data_dir = tmp_path / "data"
            knowledge_path = data_dir / "knowledge" / "agents" / "langchain-supervisor.md"
            knowledge_path.parent.mkdir(parents=True)
            knowledge_path.write_text(
                "---\n"
                'title: "LangChain Supervisor"\n'
                'next_verify_at: "2020-01-01"\n'
                "---\n\n"
                "# LangChain Supervisor\n",
                encoding="utf-8",
            )
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {data_dir}",
                        f"database_path: {data_dir / 'memory.sqlite'}",
                    ]
                ),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                main(["stale", "--config", str(config_path)])

            output = json.loads(stdout.getvalue())
            self.assertEqual(output[0]["path"], str(knowledge_path))
            self.assertEqual(output[0]["title"], "LangChain Supervisor")
            self.assertEqual(output[0]["reason"], "due")

    def test_reverify_command_generates_patch_without_applying(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data_dir = tmp_path / "data"
            knowledge_path = data_dir / "knowledge" / "agents" / "langchain-supervisor.md"
            knowledge_path.parent.mkdir(parents=True)
            original = (
                "---\n"
                'title: "LangChain Supervisor"\n'
                'next_verify_at: "2020-01-01"\n'
                "---\n\n"
                "# LangChain Supervisor\n\nOld workflow.\n"
            )
            knowledge_path.write_text(original, encoding="utf-8")
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {data_dir}",
                        f"database_path: {data_dir / 'memory.sqlite'}",
                        "default_reverify_interval_days: 14",
                    ]
                ),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                main(["reverify", str(knowledge_path), "--config", str(config_path)])

            output = json.loads(stdout.getvalue())
            run_dir = data_dir / "runs" / output["run_id"]
            self.assertEqual(output["status"], "reverify_curated")
            self.assertEqual(output["artifacts"]["source_document"], str(knowledge_path))
            self.assertTrue((run_dir / "evidence.json").exists())
            self.assertTrue((run_dir / "verification_plan.json").exists())
            self.assertTrue((run_dir / "verification.json").exists())
            self.assertTrue((run_dir / "report.md").exists())
            self.assertTrue((run_dir / "knowledge_patch.json").exists())
            patch = json.loads((run_dir / "knowledge_patch.json").read_text(encoding="utf-8"))
            self.assertEqual(patch["target_path"], str(knowledge_path))
            self.assertFalse(patch["approved"])
            self.assertEqual(knowledge_path.read_text(encoding="utf-8"), original)
