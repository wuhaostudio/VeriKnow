from pathlib import Path
import unittest

from veriknow.config import ensure_data_dirs, load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_and_create_dirs(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"data_dir: {tmp_path / 'data'}",
                        f"database_path: {tmp_path / 'data' / 'memory.sqlite'}",
                        "publisher_allow_stub: false",
                        "feishu_base_url: https://open.feishu.cn",
                        "feishu_folder_token: folder-token",
                        "feishu_document_url_template: https://example.feishu.cn/docx/{document_id}",
                        "feishu_title_strategy: front_matter",
                        "computer_use_domain_allowlist: example.com, docs.example.com",
                        "computer_use_approval_keywords: login,payment,delete",
                        "default_reverify_interval_days: 14",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            ensure_data_dirs(config)

            self.assertTrue(config.data_dir.exists())
            self.assertTrue(config.runs_dir.exists())
            self.assertTrue(config.knowledge_dir.exists())
            self.assertFalse(config.publisher_allow_stub)
            self.assertEqual(config.feishu_base_url, "https://open.feishu.cn")
            self.assertEqual(config.feishu_folder_token, "folder-token")
            self.assertEqual(
                config.feishu_document_url_template,
                "https://example.feishu.cn/docx/{document_id}",
            )
            self.assertEqual(config.feishu_title_strategy, "front_matter")
            self.assertEqual(config.computer_use_domain_allowlist, ("example.com", "docs.example.com"))
            self.assertEqual(config.computer_use_approval_keywords, ("login", "payment", "delete"))
            self.assertEqual(config.default_reverify_interval_days, 14)
