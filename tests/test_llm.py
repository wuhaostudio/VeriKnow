from pathlib import Path
from tempfile import TemporaryDirectory
import json
import urllib.error
import unittest
from unittest.mock import patch

from veriknow.config import Config
from veriknow.llm import (
    BigModelLLMClient,
    LLMProviderError,
    StubLLMClient,
    ZhipuLLMClient,
    create_llm_client,
    prompt_persistence,
)


class FakeUrlopenResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class LLMTests(unittest.TestCase):
    def test_create_stub_client_and_check(self) -> None:
        config = Config(
            data_dir=Path("data"),
            database_path=Path("data/memory.sqlite"),
            model_provider="stub",
            model_name="stub-model",
        )

        client = create_llm_client(config)
        result = client.check()

        self.assertIsInstance(client, StubLLMClient)
        self.assertTrue(result.available)
        self.assertEqual(result.provider, "stub")
        self.assertEqual(client.classify("Pick one", ["first", "second"]), "first")

    def test_bigmodel_check_blocks_without_api_key(self) -> None:
        config = Config(
            data_dir=Path("data"),
            database_path=Path("data/memory.sqlite"),
            model_provider="bigmodel",
            model_api_key_env="ZHIPUAI_API_KEY",
        )

        with patch.dict("os.environ", {}, clear=True):
            result = create_llm_client(config).check()

        self.assertFalse(result.available)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.error_code, "missing_api_key")
        self.assertEqual(result.provider, "bigmodel")
        self.assertIn("ZHIPUAI_API_KEY", result.message)

    def test_bigmodel_generate_text_posts_openai_compatible_request(self) -> None:
        config = Config(
            data_dir=Path("data"),
            database_path=Path("data/memory.sqlite"),
            model_provider="bigmodel",
            model_name="glm-test",
            model_api_key_env="ZHIPUAI_API_KEY",
            model_base_url="https://open.bigmodel.cn/api/paas/v4",
            model_temperature=0,
            model_max_output_tokens=128,
        )
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeUrlopenResponse(
                {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "total_tokens": 6,
                    },
                }
            )

        client = BigModelLLMClient(config)
        with patch.dict("os.environ", {"ZHIPUAI_API_KEY": "test-key"}, clear=True):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                text = client.generate_text("hello")

        self.assertEqual(text, "ok")
        self.assertEqual(captured["url"], "https://open.bigmodel.cn/api/paas/v4/chat/completions")
        self.assertEqual(captured["payload"]["model"], "glm-test")
        self.assertEqual(captured["payload"]["messages"][0]["content"], "hello")
        self.assertEqual(captured["payload"]["max_tokens"], 128)
        self.assertEqual(client.last_call_metadata.status, "completed")
        self.assertEqual(client.last_call_metadata.input_tokens, 4)
        self.assertEqual(client.last_call_metadata.output_tokens, 2)
        self.assertEqual(client.last_call_metadata.total_tokens, 6)

    def test_bigmodel_records_missing_usage_as_none(self) -> None:
        config = Config(
            data_dir=Path("data"),
            database_path=Path("data/memory.sqlite"),
            model_provider="bigmodel",
            model_api_key_env="TEST_MODEL_KEY",
        )
        client = BigModelLLMClient(config)

        with patch.dict("os.environ", {"TEST_MODEL_KEY": "test-key"}, clear=True):
            with patch(
                "urllib.request.urlopen",
                return_value=FakeUrlopenResponse(
                    {"choices": [{"message": {"content": "ok"}}]}
                ),
            ):
                client.generate_text("hello")

        self.assertIsNone(client.last_call_metadata.input_tokens)
        self.assertIsNone(client.last_call_metadata.output_tokens)
        self.assertIsNone(client.last_call_metadata.total_tokens)

    def test_bigmodel_retries_transient_network_failure_once(self) -> None:
        config = Config(
            data_dir=Path("data"),
            database_path=Path("data/memory.sqlite"),
            model_provider="bigmodel",
            model_api_key_env="TEST_MODEL_KEY",
            model_max_retries=1,
            model_retry_backoff_seconds=0,
        )
        client = BigModelLLMClient(config)

        with patch.dict("os.environ", {"TEST_MODEL_KEY": "test-key"}, clear=True):
            with patch(
                "urllib.request.urlopen",
                side_effect=[
                    urllib.error.URLError("temporary"),
                    FakeUrlopenResponse(
                        {"choices": [{"message": {"content": "ok"}}]}
                    ),
                ],
            ) as urlopen:
                result = client.generate_text("hello")

        self.assertEqual(result, "ok")
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(client.last_call_metadata.attempts, 2)

    def test_prompt_persistence_can_suppress_raw_prompt(self) -> None:
        stored = prompt_persistence("sensitive prompt", store_prompt=True)
        suppressed = prompt_persistence("sensitive prompt", store_prompt=False)

        self.assertEqual(stored["prompt"], "sensitive prompt")
        self.assertIsNone(suppressed["prompt"])
        self.assertFalse(suppressed["prompt_stored"])
        self.assertEqual(stored["prompt_hash"], suppressed["prompt_hash"])
        self.assertEqual(len(suppressed["prompt_hash"]), 64)

    def test_unsupported_provider_fails_fast(self) -> None:
        config = Config(
            data_dir=Path("data"),
            database_path=Path("data/memory.sqlite"),
            model_provider="unknown",
        )

        with self.assertRaises(ValueError):
            create_llm_client(config)

    def test_bigmodel_generate_json_requires_object(self) -> None:
        config = Config(
            data_dir=Path("data"),
            database_path=Path("data/memory.sqlite"),
            model_provider="bigmodel",
        )
        client = BigModelLLMClient(config)

        with patch.object(client, "generate_text", return_value="[]"):
            with self.assertRaises(LLMProviderError):
                client.generate_json("return json")

        self.assertEqual(client.last_call_metadata.status, "failed")
        self.assertEqual(client.last_call_metadata.error_code, "invalid_json_object")

    def test_zhipu_provider_alias_still_uses_bigmodel_client(self) -> None:
        config = Config(
            data_dir=Path("data"),
            database_path=Path("data/memory.sqlite"),
            model_provider="zhipu",
        )

        client = create_llm_client(config)

        self.assertIsInstance(client, BigModelLLMClient)
        self.assertIsInstance(client, ZhipuLLMClient)
        self.assertEqual(client.provider, "zhipu")
