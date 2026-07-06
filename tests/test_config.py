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
                        "computer_use_runtime: playwright",
                        "computer_use_max_steps: 6",
                        "computer_use_max_seconds: 90",
                        "computer_use_read_only: true",
                        "computer_use_store_screenshots: false",
                        "computer_use_require_approval_for_forms: false",
                        "computer_use_action_allowlist: open,observe,finish",
                        "default_reverify_interval_days: 14",
                        "model_provider: stub",
                        "model_name: glm-test",
                        "model_api_key_env: TEST_MODEL_KEY",
                        "model_base_url: https://example.bigmodel.test/api/paas/v4",
                        "model_temperature: 0.2",
                        "model_timeout_seconds: 12",
                        "model_max_output_tokens: 256",
                        "model_store_prompts: false",
                        "search_provider: brave",
                        "search_api_key_env: TEST_SEARCH_KEY",
                        "search_result_limit: 8",
                        "search_fetch_pages: true",
                        "search_store_raw_pages: true",
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
            self.assertEqual(config.computer_use_runtime, "playwright")
            self.assertEqual(config.computer_use_max_steps, 6)
            self.assertEqual(config.computer_use_max_seconds, 90)
            self.assertTrue(config.computer_use_read_only)
            self.assertFalse(config.computer_use_store_screenshots)
            self.assertFalse(config.computer_use_require_approval_for_forms)
            self.assertEqual(config.computer_use_action_allowlist, ("open", "observe", "finish"))
            self.assertEqual(config.default_reverify_interval_days, 14)
            self.assertEqual(config.model_provider, "stub")
            self.assertEqual(config.model_name, "glm-test")
            self.assertEqual(config.model_api_key_env, "TEST_MODEL_KEY")
            self.assertEqual(config.model_base_url, "https://example.bigmodel.test/api/paas/v4")
            self.assertEqual(config.model_temperature, 0.2)
            self.assertEqual(config.model_timeout_seconds, 12)
            self.assertEqual(config.model_max_output_tokens, 256)
            self.assertFalse(config.model_store_prompts)
            self.assertEqual(config.search_provider, "brave")
            self.assertEqual(config.search_api_key_env, "TEST_SEARCH_KEY")
            self.assertEqual(config.search_result_limit, 8)
            self.assertTrue(config.search_fetch_pages)
            self.assertTrue(config.search_store_raw_pages)

