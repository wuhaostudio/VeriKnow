from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

from veriknow.config import Config
from veriknow.llm import BigModelLLMClient, LLMProviderError, StubLLMClient, ZhipuLLMClient, create_llm_client


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
                {"choices": [{"message": {"content": "ok"}}]}
            )

        with patch.dict("os.environ", {"ZHIPUAI_API_KEY": "test-key"}, clear=True):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                text = BigModelLLMClient(config).generate_text("hello")

        self.assertEqual(text, "ok")
        self.assertEqual(captured["url"], "https://open.bigmodel.cn/api/paas/v4/chat/completions")
        self.assertEqual(captured["payload"]["model"], "glm-test")
        self.assertEqual(captured["payload"]["messages"][0]["content"], "hello")
        self.assertEqual(captured["payload"]["max_tokens"], 128)

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
