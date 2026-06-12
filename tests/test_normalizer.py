from pathlib import Path
import unittest

from veriknow.config import Config
from veriknow.modules.normalizer import RequirementNormalizer


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
