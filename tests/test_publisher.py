from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from veriknow.config import Config
from veriknow.modules.publisher import (
    FeishuApiError,
    FeishuPublisher,
    MarkdownToFeishuConverter,
    PublisherRegistry,
    publish_document,
    title_for_document,
)


class FakeFeishuClient:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.created_title = ""
        self.blocks = []

    def tenant_access_token(self, app_id: str, app_secret: str) -> str:
        if self.fail:
            raise FeishuApiError("api_failed", "token failed")
        return "tenant-token"

    def create_document(self, token: str, *, title: str, folder_token: str) -> dict[str, str]:
        self.created_title = title
        return {"document_id": "doc-123", "url": "https://example.feishu.cn/docx/doc-123"}

    def append_blocks(self, token: str, *, document_id: str, blocks: list[dict]) -> None:
        self.blocks = blocks


class PublisherTests(unittest.TestCase):
    def test_feishu_stub_blocks_when_credentials_are_missing(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            document_path = config.knowledge_dir / "general" / "example.md"
            document_path.parent.mkdir(parents=True)
            document_path.write_text("# Example\n", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=True):
                job = publish_document(document_path, target="feishu", config=config, approved=True)

            self.assertEqual(job.target, "feishu")
            self.assertEqual(job.status, "blocked")
            self.assertEqual(job.error_code, "missing_credentials")
            self.assertIn("FEISHU_APP_ID", job.message)

    def test_publish_rejects_documents_outside_knowledge_dir(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            document_path = tmp_path / "draft.md"
            document_path.write_text("# Draft\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                publish_document(document_path, target="feishu", config=config)

    def test_publish_rejects_unapproved_knowledge_documents(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            document_path = config.knowledge_dir / "general" / "example.md"
            document_path.parent.mkdir(parents=True)
            document_path.write_text("# Example\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                publish_document(document_path, target="feishu", config=config)

    def test_feishu_publisher_uploads_with_injected_client(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
                feishu_folder_token="folder-token",
                feishu_title_strategy="front_matter",
            )
            document_path = config.knowledge_dir / "general" / "example.md"
            document_path.parent.mkdir(parents=True)
            document_path.write_text(
                '---\ntitle: "Verified Guide"\n---\n\n# Heading\n\n- First step\n',
                encoding="utf-8",
            )
            client = FakeFeishuClient()
            publisher = FeishuPublisher(config, client=client)
            registry = PublisherRegistry(config, publishers=[publisher])

            with patch.dict(
                "os.environ",
                {"FEISHU_APP_ID": "app-id", "FEISHU_APP_SECRET": "app-secret"},
                clear=True,
            ):
                job = publish_document(
                    document_path,
                    target="feishu",
                    config=config,
                    approved=True,
                    registry=registry,
                )

            self.assertEqual(job.status, "published")
            self.assertEqual(job.target_document_id, "doc-123")
            self.assertEqual(job.target_url, "https://example.feishu.cn/docx/doc-123")
            self.assertEqual(client.created_title, "Verified Guide")
            self.assertTrue(client.blocks)

    def test_feishu_publisher_records_api_errors(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
                feishu_folder_token="folder-token",
            )
            document_path = config.knowledge_dir / "general" / "example.md"
            document_path.parent.mkdir(parents=True)
            document_path.write_text("# Example\n", encoding="utf-8")
            publisher = FeishuPublisher(config, client=FakeFeishuClient(fail=True))
            registry = PublisherRegistry(config, publishers=[publisher])

            with patch.dict(
                "os.environ",
                {"FEISHU_APP_ID": "app-id", "FEISHU_APP_SECRET": "app-secret"},
                clear=True,
            ):
                job = publish_document(
                    document_path,
                    target="feishu",
                    config=config,
                    approved=True,
                    registry=registry,
                )

            self.assertEqual(job.status, "failed")
            self.assertEqual(job.error_code, "api_failed")
            self.assertEqual(job.message, "token failed")

    def test_markdown_converter_and_title_strategy(self) -> None:
        content = '---\ntitle: "Front Matter Title"\n---\n\n# Heading\n\n1. Open [Docs](https://example.com)\n'

        self.assertEqual(
            title_for_document(content, Path("fallback.md"), "front_matter"),
            "Front Matter Title",
        )
        blocks = MarkdownToFeishuConverter().convert(content)
        block_text = blocks[1]["text"]["elements"][0]["text_run"]["content"]
        self.assertEqual(block_text, "Open Docs")
