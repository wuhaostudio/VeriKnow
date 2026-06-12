from pathlib import Path
import unittest

from veriknow.config import Config
from veriknow.memory.store import MemoryStore
from veriknow.modules.adaptive_profile import AdaptiveProfile
from veriknow.modules.normalizer import RequirementNormalizer
from veriknow.schemas import PublicationJob


class MemoryTests(unittest.TestCase):
    def test_memory_store_creates_and_reads_run(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            task = RequirementNormalizer(config).normalize("Research LangChain latest workflow")
            store = MemoryStore(config)

            record = store.create_run(task.raw_request, task)
            store.update_run(record.run_id, status="dry_run")
            loaded = store.get_run(record.run_id)

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.status, "dry_run")
            self.assertTrue(loaded.task.target)
            self.assertTrue((store.run_dir(record.run_id) / "task.json").exists())

    def test_adaptive_profile_blocks_sensitive_preferences(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            store = MemoryStore(config)
            profile = AdaptiveProfile(store)

            profile.append_signal("output_structure", "prefer concise checklists")
            self.assertEqual(store.list_preferences()[0].key, "output_structure")
            with self.assertRaises(ValueError):
                profile.append_signal("personality", "introvert")

    def test_memory_store_records_publication_jobs(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            store = MemoryStore(config)
            job = PublicationJob(
                document_path=str(config.knowledge_dir / "general" / "example.md"),
                target="feishu",
                status="blocked",
                message="missing credentials",
            )

            store.append_publication_job(job)
            jobs = store.list_publication_jobs()

            self.assertEqual(jobs[0].document_path, job.document_path)
            self.assertEqual(jobs[0].target, "feishu")
            self.assertEqual(jobs[0].status, "blocked")

    def test_memory_store_identifies_approved_knowledge_documents(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            store = MemoryStore(config)
            task = RequirementNormalizer(config).normalize("Research LangChain")
            record = store.create_run(task.raw_request, task)
            document_path = config.knowledge_dir / "general" / "langchain.md"
            document_path.parent.mkdir(parents=True)
            document_path.write_text("# LangChain\n", encoding="utf-8")

            self.assertFalse(store.is_approved_knowledge_document(document_path))

            store.complete_run(record.run_id, artifacts={"knowledge_document": str(document_path)})

            self.assertTrue(store.is_approved_knowledge_document(document_path))
