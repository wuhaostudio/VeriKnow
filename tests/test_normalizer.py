from pathlib import Path
import unittest

from veriknow.config import Config
from veriknow.modules.normalizer import AIRequirementNormalizer, RequirementNormalizer


class NormalizerTests(unittest.TestCase):
    def test_normalizer_creates_traceable_task(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            task = RequirementNormalizer(config).normalize("帮我研究某个工具的最新用法")

            self.assertEqual(task.raw_request, "帮我研究某个工具的最新用法")
            self.assertTrue(task.target)
            self.assertEqual(task.locale, "zh-CN")
            self.assertTrue(task.verification_required)
            self.assertEqual(task.verification_method, "browser")
            self.assertIn("Prioritize recent and official sources.", task.constraints)

class FakeNormalizerLLM:
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


class AINormalizerTests(unittest.TestCase):
    def test_ai_normalizer_validates_model_task_output(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            llm = FakeNormalizerLLM(
                {
                    "objective": "Research Zhipu platform usage and produce a verified guide.",
                    "target": "Zhipu platform API usage",
                    "scope": "public_web",
                    "verification_required": True,
                    "verification_method": "api",
                    "output_format": "markdown",
                    "publish_target": "local",
                    "locale": "zh-CN",
                    "constraints": ["Prioritize official Zhipu documentation."],
                }
            )

            result = AIRequirementNormalizer(config, llm).normalize("研究智谱 platform API 的最新用法")

            self.assertEqual(result.task.raw_request, "研究智谱 platform API 的最新用法")
            self.assertEqual(result.task.target, "Zhipu platform API usage")
            self.assertEqual(result.task.verification_method, "api")
            self.assertIsNotNone(result.artifact)
            self.assertEqual(result.artifact.status, "completed")
            self.assertFalse(result.artifact.fallback_used)

    def test_ai_normalizer_falls_back_on_invalid_output(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            llm = FakeNormalizerLLM({"verification_method": "unknown"})

            result = AIRequirementNormalizer(config, llm).normalize("帮我研究某个工具的最新用法")

            self.assertEqual(result.task.locale, "zh-CN")
            self.assertEqual(result.task.verification_method, "browser")
            self.assertIsNotNone(result.artifact)
            self.assertEqual(result.artifact.status, "fallback")
            self.assertTrue(result.artifact.fallback_used)
