from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import unittest
from unittest.mock import patch

from veriknow.cli import main


class CliTests(unittest.TestCase):
    def test_llm_check_command_reports_stub_available(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                        "model_provider: stub",
                        "model_name: stub-model",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = StringIO()

            with redirect_stdout(stdout):
                main(["llm", "check", "--config", str(config_path)])

            output = json.loads(stdout.getvalue())
            self.assertEqual(output["provider"], "stub")
            self.assertTrue(output["available"])
            self.assertEqual(output["status"], "available")

    def test_llm_check_command_reports_missing_zhipu_key(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                        "model_provider: zhipu",
                        "model_api_key_env: ZHIPUAI_API_KEY",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = StringIO()

            with patch.dict("os.environ", {}, clear=True):
                with redirect_stdout(stdout):
                    main(["llm", "check", "--config", str(config_path)])

            output = json.loads(stdout.getvalue())
            self.assertEqual(output["provider"], "zhipu")
            self.assertFalse(output["available"])
            self.assertEqual(output["status"], "blocked")
            self.assertEqual(output["error_code"], "missing_api_key")

    def test_run_command_ai_normalizer_writes_artifact_on_fallback(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                        "model_provider: stub",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = StringIO()

            with redirect_stdout(stdout):
                main([
                    "run",
                    "帮我研究智谱 platform API 的最新用法",
                    "--normalizer",
                    "ai",
                    "--dry-run",
                    "--config",
                    str(config_path),
                ])

            output = json.loads(stdout.getvalue())
            artifact_path = Path(output["artifacts"]["llm_normalizer"])
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(output["status"], "dry_run")
            self.assertTrue(artifact_path.exists())
            self.assertEqual(artifact["strategy"], "ai")
            self.assertEqual(artifact["provider"], "stub")
            self.assertEqual(artifact["status"], "fallback")
            self.assertTrue(artifact["fallback_used"])
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

    def test_research_command_writes_fetched_documents_when_enabled(self) -> None:
        from tempfile import TemporaryDirectory

        from veriknow.schemas import FetchedDocument

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                        "search_fetch_pages: true",
                        "search_store_raw_pages: true",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = StringIO()

            def fake_fetch_documents(items, *, limit=None, raw_dir=None):
                return [
                    FetchedDocument(
                        url=items[0].url,
                        title=items[0].title,
                        text="Fetched page text. Version 2.0 supports tool calling.",
                        fetched_at="2026-06-26T00:00:00+00:00",
                        status_code=200,
                        content_hash="hash-1",
                        raw_path=str(raw_dir / "example.html") if raw_dir is not None else None,
                    ),
                    FetchedDocument(
                        url="https://example.com/old",
                        title="Old Docs",
                        text="Version 2.0 tool calling is deprecated.",
                        fetched_at="2026-06-26T00:00:00+00:00",
                        status_code=200,
                        content_hash="hash-2",
                    ),
                ]

            with patch("veriknow.cli.fetch_documents", fake_fetch_documents):
                with redirect_stdout(stdout):
                    main(["research", "LangChain multi-agent supervisor workflow", "--config", str(config_path)])

            output = json.loads(stdout.getvalue())
            run_dir = tmp_path / "data" / "runs" / output["task_id"]
            evidence_path = run_dir / "evidence.json"
            fetched_path = run_dir / "fetched_documents.json"
            claims_path = run_dir / "extracted_claims.json"
            conflicts_path = run_dir / "claim_conflicts.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            fetched = json.loads(fetched_path.read_text(encoding="utf-8"))
            claims = json.loads(claims_path.read_text(encoding="utf-8"))
            conflicts = json.loads(conflicts_path.read_text(encoding="utf-8"))
            self.assertTrue(evidence_path.exists())
            self.assertTrue(fetched_path.exists())
            self.assertTrue(claims_path.exists())
            self.assertTrue(conflicts_path.exists())
            self.assertEqual(fetched[0]["status_code"], 200)
            self.assertEqual(fetched[0]["content_hash"], "hash-1")
            self.assertTrue(fetched[0]["raw_path"].endswith("raw_pages\\example.html"))
            self.assertEqual(claims[0]["source_url"], fetched[0]["url"])
            self.assertIn("Version 2.0", claims[1]["text"])
            self.assertIn("opposing", conflicts[0]["reason"])
            self.assertTrue(claims[1]["conflicts"])
            self.assertIn("Extracted 3 claim(s)", evidence["summary"])
            self.assertIn("1 detected conflict(s)", evidence["summary"])

    def test_research_command_writes_raw_search_payloads_when_provider_exposes_them(self) -> None:
        from tempfile import TemporaryDirectory

        from veriknow.tools.web_search import SearchResult

        class RawProvider:
            def search(self, query: str, *, limit: int = 5):
                return [
                    SearchResult(
                        title="Official docs",
                        url="https://docs.example.com/guide",
                        snippet="Official guide.",
                        source_type="official_doc",
                        raw={"title": "Official docs", "url": "https://docs.example.com/guide", "age": "2026-01-01"},
                    )
                ]

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                        "search_fetch_pages: false",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = StringIO()

            with patch("veriknow.cli.create_search_provider", return_value=RawProvider()):
                with redirect_stdout(stdout):
                    main(["research", "example docs", "--config", str(config_path)])

            output = json.loads(stdout.getvalue())
            run_dir = tmp_path / "data" / "runs" / output["task_id"]
            raw_payloads_path = run_dir / "raw_search_payloads.json"
            raw_payloads = json.loads(raw_payloads_path.read_text(encoding="utf-8"))
            self.assertTrue(raw_payloads_path.exists())
            self.assertEqual(raw_payloads[0]["age"], "2026-01-01")
    def test_research_command_brave_provider_requires_key(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                        "search_provider: brave",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                main(["research", "LangChain", "--config", str(config_path)])

    def test_research_command_ai_strategy_writes_artifact_on_fallback(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                        "model_provider: stub",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = StringIO()

            with redirect_stdout(stdout):
                main([
                    "research",
                    "LangChain multi-agent supervisor workflow",
                    "--strategy",
                    "ai",
                    "--config",
                    str(config_path),
                ])

            output = json.loads(stdout.getvalue())
            run_dir = tmp_path / "data" / "runs" / output["task_id"]
            artifact_path = run_dir / "llm" / "research.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertTrue(output["items"])
            self.assertTrue(artifact_path.exists())
            self.assertEqual(artifact["strategy"], "ai")
            self.assertEqual(artifact["provider"], "stub")
            self.assertEqual(artifact["status"], "fallback")
            self.assertTrue(artifact["fallback_used"])
    def test_research_command_ai_claim_extractor_writes_artifact_on_fallback(self) -> None:
        from tempfile import TemporaryDirectory

        from veriknow.schemas import FetchedDocument

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                        "model_provider: stub",
                        "search_fetch_pages: true",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = StringIO()

            def fake_fetch_documents(items, *, limit=None, raw_dir=None):
                return [
                    FetchedDocument(
                        url=items[0].url,
                        title=items[0].title,
                        text="Version 2.0 supports tool calling.",
                        fetched_at="2026-06-26T00:00:00+00:00",
                        status_code=200,
                        content_hash="hash-1",
                    )
                ]

            with patch("veriknow.cli.fetch_documents", fake_fetch_documents):
                with redirect_stdout(stdout):
                    main([
                        "research",
                        "LangChain multi-agent supervisor workflow",
                        "--strategy",
                        "ai",
                        "--config",
                        str(config_path),
                    ])

            output = json.loads(stdout.getvalue())
            run_dir = tmp_path / "data" / "runs" / output["task_id"]
            artifact_path = run_dir / "llm" / "claim_extractor.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertTrue(artifact_path.exists())
            self.assertEqual(artifact["strategy"], "ai")
            self.assertEqual(artifact["provider"], "stub")
            self.assertEqual(artifact["status"], "fallback")
            self.assertTrue(artifact["fallback_used"])
            self.assertTrue((run_dir / "extracted_claims.json").exists())
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

    def test_plan_command_ai_strategy_writes_artifact_on_fallback(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                        "model_provider: stub",
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
                main(["plan", run_id, "--strategy", "ai", "--config", str(config_path)])

            output = json.loads(plan_stdout.getvalue())
            run_dir = tmp_path / "data" / "runs" / run_id
            artifact_path = run_dir / "llm" / "planner.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(output["task_id"], run_id)
            self.assertTrue(output["steps"])
            self.assertTrue(artifact_path.exists())
            self.assertEqual(artifact["strategy"], "ai")
            self.assertEqual(artifact["status"], "fallback")
            self.assertTrue(artifact["fallback_used"])

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
            proposal_path = run_dir / "knowledge_merge_proposal.json"
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            self.assertTrue(proposal_path.exists())
            self.assertEqual(proposal["operation"], "update")
            self.assertEqual(proposal["target_path"], str(knowledge_path))
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
            proposal_path = run_dir / "knowledge_merge_proposal.json"
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            self.assertTrue(proposal_path.exists())
            self.assertEqual(proposal["operation"], "update")
            patch = json.loads((run_dir / "knowledge_patch.json").read_text(encoding="utf-8"))
            self.assertEqual(patch["target_path"], str(knowledge_path))
            self.assertFalse(patch["approved"])
            self.assertEqual(knowledge_path.read_text(encoding="utf-8"), original)
