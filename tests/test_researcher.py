from pathlib import Path
import json
import unittest

from veriknow.config import Config
from veriknow.memory.store import MemoryStore
from veriknow.modules.normalizer import RequirementNormalizer
from veriknow.modules.researcher import AIResearcher, Researcher
from veriknow.tools.web_search import SearchResult, WebSearchProvider


class FakeProvider(WebSearchProvider):
    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                title="Community note",
                url="https://example.com/community",
                source_type="community",
            ),
            SearchResult(
                title="Official docs",
                url="https://example.com/docs",
                source_type="official_doc",
            ),
        ][:limit]


class ResearcherTests(unittest.TestCase):
    def test_researcher_creates_ranked_evidence_bundle(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            task = RequirementNormalizer(config).normalize("Research LangChain latest workflow")
            bundle = Researcher(FakeProvider()).research(task, run_id="run-test")

            self.assertEqual(bundle.task_id, "run-test")
            self.assertEqual(bundle.items[0].source_type, "official_doc")
            self.assertEqual(bundle.items[0].confidence, "high")
            self.assertIn("Collected 2 public source", bundle.summary)

    def test_evidence_can_be_persisted_as_run_artifact(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            store = MemoryStore(config)
            task = RequirementNormalizer(config).normalize("Research LangChain latest workflow")
            record = store.create_run(task.raw_request, task)
            bundle = Researcher(FakeProvider()).research(task, run_id=record.run_id)
            evidence_path = store.run_dir(record.run_id) / "evidence.json"
            evidence_path.write_text(
                json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            loaded = store.update_run(
                record.run_id,
                status="researched",
                artifacts={"evidence": str(evidence_path)},
            )

            self.assertEqual(loaded.status, "researched")
            self.assertTrue(evidence_path.exists())
            self.assertEqual(loaded.artifacts["evidence"], str(evidence_path))

class FakeResearchLLM:
    provider = "fake"
    model = "fake-model"

    def __init__(self, payload: dict):
        self.payload = payload

    def check(self):
        raise NotImplementedError

    def generate_text(self, prompt: str, *, context: dict | None = None) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, *, context: dict | None = None) -> dict:
        return self.payload

    def classify(self, prompt: str, labels: list[str], *, context: dict | None = None) -> str:
        return labels[0]


class AIResearcherTests(unittest.TestCase):
    def test_ai_researcher_validates_model_evidence_output(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            task = RequirementNormalizer(config).normalize("Research LangChain latest workflow")
            llm = FakeResearchLLM(
                {
                    "summary": "Official evidence collected from model-refined seed results.",
                    "items": [
                        {
                            "title": "Official docs",
                            "url": "https://example.com/docs",
                            "source_type": "official_doc",
                            "snippet": "Official source supports the workflow.",
                            "published_at": None,
                            "updated_at": None,
                            "confidence": "high",
                        }
                    ],
                }
            )

            result = AIResearcher(llm, base=Researcher(FakeProvider())).research(task, run_id="run-ai")

            self.assertEqual(result.bundle.task_id, "run-ai")
            self.assertEqual(result.bundle.summary, "Official evidence collected from model-refined seed results.")
            self.assertEqual(result.bundle.items[0].url, "https://example.com/docs")
            self.assertIsNotNone(result.artifact)
            self.assertEqual(result.artifact.status, "completed")
            self.assertFalse(result.artifact.fallback_used)

    def test_ai_researcher_falls_back_on_invalid_model_output(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            task = RequirementNormalizer(config).normalize("Research LangChain latest workflow")
            llm = FakeResearchLLM({"summary": "No items"})

            result = AIResearcher(llm, base=Researcher(FakeProvider())).research(task, run_id="run-ai")

            self.assertEqual(result.bundle.items[0].source_type, "official_doc")
            self.assertIsNotNone(result.artifact)
            self.assertEqual(result.artifact.status, "fallback")
            self.assertTrue(result.artifact.fallback_used)
