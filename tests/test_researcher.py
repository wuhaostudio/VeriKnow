from pathlib import Path
from datetime import date
import json
import unittest

from veriknow.config import Config
from veriknow.memory.store import MemoryStore
from veriknow.modules.normalizer import RequirementNormalizer
from veriknow.modules.researcher import AIResearcher, Researcher, rank_evidence_items
from veriknow.schemas import EvidenceItem
from veriknow.tools.web_search import (
    BraveSearchProvider,
    HybridSearchProvider,
    SearchProviderError,
    SearchResult,
    SerpApiSearchProvider,
    WebSearchProvider,
    create_search_provider,
)


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


class FailingProvider(WebSearchProvider):
    provider = "failing"

    def __init__(self, code: str = "failed"):
        self.code = code

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        raise SearchProviderError(self.code, "provider failed")


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

    def test_rank_evidence_items_uses_explicit_confidence_order(self) -> None:
        items = [
            EvidenceItem(
                title="Low",
                url="https://example.com/low",
                source_type="official_doc",
                confidence="low",
            ),
            EvidenceItem(
                title="High",
                url="https://example.com/high",
                source_type="official_doc",
                confidence="high",
            ),
            EvidenceItem(
                title="Medium",
                url="https://example.com/medium",
                source_type="official_doc",
                confidence="medium",
            ),
        ]

        ranked = rank_evidence_items(items)

        self.assertEqual([item.confidence for item in ranked], ["high", "medium", "low"])

    def test_rank_evidence_items_prefers_recent_updated_or_published_date(self) -> None:
        items = [
            EvidenceItem(
                title="Older update",
                url="https://example.com/older",
                source_type="official_doc",
                confidence="high",
                published_at="2026-12-31",
                updated_at="2025-01-01",
            ),
            EvidenceItem(
                title="Published fallback",
                url="https://example.com/published",
                source_type="official_doc",
                confidence="high",
                published_at="Jul 1, 2026",
                updated_at="recently",
            ),
            EvidenceItem(
                title="Newer update",
                url="https://example.com/newer",
                source_type="official_doc",
                confidence="high",
                published_at="2020-01-01",
                updated_at="2026-07-02T08:00:00Z",
            ),
        ]

        ranked = rank_evidence_items(items)

        self.assertEqual(
            [item.title for item in ranked],
            ["Newer update", "Published fallback", "Older update"],
        )

    def test_rank_evidence_items_puts_missing_or_invalid_dates_last_stably(self) -> None:
        items = [
            EvidenceItem(
                title="Zulu missing",
                url="https://example.com/zulu",
                source_type="official_doc",
                confidence="high",
            ),
            EvidenceItem(
                title="Dated",
                url="https://example.com/dated",
                source_type="official_doc",
                confidence="high",
                updated_at="2026/07/01",
            ),
            EvidenceItem(
                title="Alpha invalid",
                url="https://example.com/alpha",
                source_type="official_doc",
                confidence="high",
                updated_at="recently",
            ),
        ]

        ranked = rank_evidence_items(items)

        self.assertEqual(
            [item.title for item in ranked],
            ["Dated", "Alpha invalid", "Zulu missing"],
        )

    def test_researcher_labels_freshness_and_downgrades_stale_confidence(self) -> None:
        class DatedProvider:
            def search(self, query: str, *, limit: int = 5):
                return [
                    SearchResult(
                        title="Fresh docs",
                        url="https://example.com/fresh",
                        source_type="official_doc",
                        updated_at="2026-06-15",
                    ),
                    SearchResult(
                        title="Stale docs",
                        url="https://example.com/stale",
                        source_type="official_doc",
                        updated_at="2025-01-01",
                    ),
                ]

        config = Config(data_dir=Path("data"), database_path=Path("data/memory.sqlite"))
        task = RequirementNormalizer(config).normalize("Research example")
        bundle = Researcher(
            DatedProvider(),
            freshness_days={"official_doc": 90, "unknown": 90},
            as_of=date(2026, 7, 11),
        ).research(task, run_id="run-freshness")

        self.assertEqual(bundle.items[0].freshness, "fresh")
        self.assertEqual(bundle.items[0].confidence, "high")
        self.assertEqual(bundle.items[1].freshness, "stale")
        self.assertEqual(bundle.items[1].confidence, "medium")
        self.assertIn("lowers confidence", bundle.items[1].confidence_reason)
        self.assertIn("1 fresh, 1 stale", bundle.summary)

    def test_researcher_can_expand_and_deduplicate_search_queries(self) -> None:
        class RecordingProvider:
            def __init__(self):
                self.queries = []

            def search(self, query: str, *, limit: int = 5):
                self.queries.append(query)
                return [
                    SearchResult(
                        title=query,
                        url=f"https://example.com/{len(self.queries)}",
                        source_type="official_doc",
                    )
                ]

        provider = RecordingProvider()
        config = Config(data_dir=Path("data"), database_path=Path("data/memory.sqlite"))
        task = RequirementNormalizer(config).normalize("Research the latest Example API")

        bundle = Researcher(provider, query_count=3).research(
            task,
            run_id="run-queries",
            limit=5,
        )

        self.assertEqual(len(provider.queries), 3)
        self.assertIn("official documentation", provider.queries[1])
        self.assertIn("latest release notes", provider.queries[2])
        self.assertEqual(len(bundle.items), 3)

    def test_rank_evidence_items_accepts_configured_source_priority(self) -> None:
        items = [
            EvidenceItem(
                title="Docs",
                url="https://example.com/docs",
                source_type="official_doc",
                confidence="high",
            ),
            EvidenceItem(
                title="Community",
                url="https://example.com/community",
                source_type="community",
                confidence="medium",
            ),
        ]

        ranked = rank_evidence_items(
            items,
            source_priority={"official_doc": 10, "community": 100, "unknown": 1},
        )

        self.assertEqual(ranked[0].source_type, "community")

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

    def test_create_search_provider_uses_static_by_default(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            provider = create_search_provider(config)
            results = provider.search("LangChain", limit=1)

            self.assertEqual(len(results), 1)
            self.assertIn("LangChain", results[0].title)

    def test_brave_search_provider_maps_web_results(self) -> None:
        calls = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "web": {
                            "results": [
                                {
                                    "title": "Official docs",
                                    "url": "https://docs.example.com/guide",
                                    "description": "Official guide.",
                                    "age": "2026-01-01",
                                }
                            ]
                        }
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return FakeResponse()

        import unittest.mock

        with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
            results = BraveSearchProvider("search-key", timeout_seconds=3).search("example query", limit=2)

        self.assertEqual(results[0].title, "Official docs")
        self.assertEqual(results[0].source_type, "official_doc")
        self.assertIn("q=example+query", calls[0][0].full_url)
        self.assertEqual(calls[0][0].headers["X-subscription-token"], "search-key")
        self.assertEqual(calls[0][1], 3)

    def test_brave_search_provider_requires_key(self) -> None:
        with self.assertRaises(SearchProviderError) as context:
            BraveSearchProvider("")

        self.assertEqual(context.exception.code, "missing_api_key")

    def test_serpapi_search_provider_maps_organic_results(self) -> None:
        calls = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "organic_results": [
                            {
                                "position": 1,
                                "title": "Project documentation",
                                "link": "https://docs.example.com/guide",
                                "snippet": "Official guide.",
                                "date": "Jul 1, 2026",
                                "displayed_link": "docs.example.com",
                            }
                        ]
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return FakeResponse()

        import unittest.mock

        with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
            results = SerpApiSearchProvider("search-key", timeout_seconds=4).search(
                "example query",
                limit=2,
            )

        self.assertEqual(results[0].title, "Project documentation")
        self.assertEqual(results[0].url, "https://docs.example.com/guide")
        self.assertEqual(results[0].source_type, "official_doc")
        self.assertEqual(results[0].published_at, "Jul 1, 2026")
        self.assertEqual(results[0].updated_at, "Jul 1, 2026")
        self.assertEqual(results[0].raw["position"], 1)
        self.assertIn("engine=google", calls[0][0].full_url)
        self.assertIn("q=example+query", calls[0][0].full_url)
        self.assertIn("api_key=search-key", calls[0][0].full_url)
        self.assertEqual(calls[0][1], 4)

    def test_serpapi_search_provider_requires_key(self) -> None:
        with self.assertRaises(SearchProviderError) as context:
            SerpApiSearchProvider("")

        self.assertEqual(context.exception.code, "missing_api_key")

    def test_create_search_provider_supports_serpapi_env_key(self) -> None:
        from tempfile import TemporaryDirectory
        import unittest.mock

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
                search_provider="serpapi",
                search_api_key_env="TEST_SERPAPI_KEY",
            )

            with unittest.mock.patch.dict("os.environ", {"TEST_SERPAPI_KEY": "env-key"}):
                provider = create_search_provider(config)

            self.assertIsInstance(provider, SerpApiSearchProvider)
            self.assertEqual(provider.api_key, "env-key")

    def test_hybrid_search_provider_deduplicates_urls(self) -> None:
        class FirstProvider:
            provider = "first"

            def search(self, query: str, *, limit: int = 5):
                return [
                    SearchResult(
                        title="Official docs",
                        url="https://docs.example.com/guide?utm_source=test",
                        source_type="official_doc",
                    ),
                    SearchResult(
                        title="Repository",
                        url="https://github.com/example/project",
                        source_type="official_github",
                    ),
                ]

        class SecondProvider:
            provider = "second"

            def search(self, query: str, *, limit: int = 5):
                return [
                    SearchResult(
                        title="Duplicate docs",
                        url="https://docs.example.com/guide",
                        source_type="official_doc",
                    ),
                    SearchResult(
                        title="Community result",
                        url="https://example.com/community",
                        source_type="community",
                    ),
                ]

        results = HybridSearchProvider([FirstProvider(), SecondProvider()]).search("example", limit=3)

        self.assertEqual([result.title for result in results], ["Official docs", "Repository", "Community result"])

    def test_hybrid_search_provider_interleaves_provider_results(self) -> None:
        class Provider:
            def __init__(self, name: str):
                self.provider = name

            def search(self, query: str, *, limit: int = 5):
                return [
                    SearchResult(
                        title=f"{self.provider}-{index}",
                        url=f"https://{self.provider}.example.com/{index}",
                    )
                    for index in range(limit)
                ]

        results = HybridSearchProvider([Provider("first"), Provider("second")]).search(
            "example",
            limit=4,
        )

        self.assertEqual(
            [result.title for result in results],
            ["first-0", "second-0", "first-1", "second-1"],
        )

    def test_hybrid_search_provider_continues_after_provider_failure(self) -> None:
        provider = HybridSearchProvider([FailingProvider("network_error"), FakeProvider()])

        results = provider.search("example", limit=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(provider.failures[0]["provider"], "failing")
        self.assertEqual(provider.failures[0]["code"], "network_error")

    def test_hybrid_search_provider_raises_when_all_providers_fail(self) -> None:
        provider = HybridSearchProvider([FailingProvider("network_error"), FailingProvider("api_error")])

        with self.assertRaises(SearchProviderError) as context:
            provider.search("example")

        self.assertEqual(context.exception.code, "all_providers_failed")

    def test_create_search_provider_supports_hybrid_with_static_fallback(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
                search_provider="hybrid",
                search_hybrid_providers=("static",),
            )

            provider = create_search_provider(config)
            results = provider.search("LangChain", limit=1)

            self.assertIsInstance(provider, HybridSearchProvider)
            self.assertEqual(len(results), 1)
            self.assertIn("LangChain", results[0].title)

    def test_default_hybrid_provider_falls_back_to_static_when_live_keys_are_missing(self) -> None:
        from tempfile import TemporaryDirectory
        import unittest.mock

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
                search_provider="hybrid",
            )

            with unittest.mock.patch.dict("os.environ", {}, clear=True):
                provider = create_search_provider(config)
                results = provider.search("LangChain", limit=1)

            self.assertIsInstance(provider, HybridSearchProvider)
            self.assertEqual(provider.failures[0]["provider"], "brave")
            self.assertEqual(provider.failures[0]["code"], "missing_api_key")
            self.assertEqual(provider.failures[1]["provider"], "serpapi")
            self.assertEqual(provider.failures[1]["code"], "missing_api_key")
            self.assertEqual(len(results), 1)
            self.assertIn("LangChain", results[0].title)

    def test_researcher_records_hybrid_provider_failures_as_raw_payloads(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            task = RequirementNormalizer(config).normalize("Research LangChain latest workflow")
            provider = HybridSearchProvider([FailingProvider("network_error"), FakeProvider()])
            researcher = Researcher(provider)

            researcher.research(task, run_id="run-test")

            self.assertEqual(researcher.last_raw_search_payloads[-1]["provider"], "hybrid")
            self.assertEqual(researcher.last_raw_search_payloads[-1]["failures"][0]["code"], "network_error")


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

    def test_ai_researcher_rejects_urls_not_present_in_seed_evidence(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            task = RequirementNormalizer(config).normalize("Research example")
            llm = FakeResearchLLM(
                {
                    "summary": "Invented source",
                    "items": [
                        {
                            "title": "Invented",
                            "url": "https://invented.example.com/docs",
                            "source_type": "official_doc",
                        }
                    ],
                }
            )

            result = AIResearcher(llm, base=Researcher(FakeProvider())).research(
                task,
                run_id="run-ai",
            )

            self.assertEqual(result.artifact.status, "fallback")
            self.assertTrue(result.artifact.fallback_used)
            self.assertNotEqual(
                result.bundle.items[0].url,
                "https://invented.example.com/docs",
            )
