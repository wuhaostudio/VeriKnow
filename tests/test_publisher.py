from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from veriknow.config import Config
from veriknow.modules.publisher import (
    FeishuApiClient,
    FeishuApiError,
    FeishuPublisher,
    MarkdownToFeishuConverter,
    PublisherRegistry,
    content_hash_for,
    publish_document,
    title_for_document,
)
from veriknow.schemas import PublicationJob



class RecordingFeishuApiClient(FeishuApiClient):
    def __init__(self):
        super().__init__("https://example.feishu.test")
        self.requests = []

    def _request_json(self, method, path, *, token=None, body=None):
        self.requests.append({"method": method, "path": path, "token": token, "body": body or {}})
        if path.endswith("/children") and method == "GET":
            return {"items": [{"block_id": "block-1"}, {"block_id": "block-2"}]}
        if path.endswith("/children") and method == "POST":
            return {"document_revision_id": 2}
        return {}

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

    def append_blocks(self, token: str, *, document_id: str, blocks: list[dict]) -> str:
        self.blocks = blocks
        return "rev-2"



class FakeUpdatingFeishuClient(FakeFeishuClient):
    def __init__(self, *, fail: bool = False, remote_revision: str | None = None):
        super().__init__(fail=fail)
        self.remote_revision = remote_revision
        self.updated_document_id = ""
        self.updated_blocks = []

    def document_metadata(self, token: str, *, document_id: str) -> dict[str, str]:
        if self.remote_revision is None:
            return {"document_id": document_id}
        return {"document_id": document_id, "revision": self.remote_revision}

    def update_document(self, token: str, *, document_id: str, blocks: list[dict]) -> dict[str, str]:
        self.updated_document_id = document_id
        self.updated_blocks = blocks
        return {
            "url": f"https://example.feishu.cn/docx/{document_id}",
            "revision": "rev-2",
        }


class PaginatedFeishuApiClient(FeishuApiClient):
    def __init__(self):
        super().__init__("https://example.feishu.test")
        self.requests = []

    def _request_json(self, method, path, *, token=None, body=None):
        self.requests.append({"method": method, "path": path, "token": token, "body": body or {}})
        if path.endswith("/children") and method == "GET":
            return {"items": [{"block_id": "block-1"}], "has_more": True, "page_token": "next"}
        if "page_token=next" in path:
            return {"children": [{"id": "block-2"}], "has_more": False}
        if "/children/batch_delete?" in path and method == "DELETE":
            return {"document_revision_id": 2}
        if path.endswith("/children") and method == "POST":
            return {"document_revision_id": 3}
        if path.endswith("/documents/doc-123"):
            return {"document": {"document_id": "doc-123", "revision_id": "rev-3"}}
        return {}


class EmptyFeishuApiClient(FeishuApiClient):
    def __init__(self):
        super().__init__("https://example.feishu.test")
        self.requests = []

    def _request_json(self, method, path, *, token=None, body=None):
        self.requests.append({"method": method, "path": path, "token": token, "body": body or {}})
        if path.endswith("/children") and method == "GET":
            return {"items": []}
        if path.endswith("/children") and method == "POST":
            return {"document_revision_id": 1}
        if path.endswith("/documents/doc-123"):
            return {"document": {"document_id": "doc-123", "revision": "rev-1"}}
        return {}


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
            self.assertEqual(job.remote_revision, "rev-2")
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

    def test_publish_records_content_hash_metadata(self) -> None:
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
            publisher = FeishuPublisher(config, client=FakeFeishuClient())
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
            self.assertEqual(job.local_path, str(document_path.resolve()))
            self.assertEqual(job.local_content_hash, content_hash_for(document_path))

    def test_publish_update_skips_unchanged_document(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            document_path = config.knowledge_dir / "general" / "example.md"
            document_path.parent.mkdir(parents=True)
            document_path.write_text("# Example\n", encoding="utf-8")
            last = publish_document(document_path, target="feishu", config=config, approved=True)
            last.status = "published"
            last.target_document_id = "doc-123"
            last.target_url = "https://example.feishu.cn/docx/doc-123"

            job = publish_document(
                document_path,
                target="feishu",
                config=config,
                approved=True,
                update=True,
                last_publication=last,
            )

            self.assertEqual(job.status, "skipped")
            self.assertEqual(job.target_document_id, "doc-123")
            self.assertEqual(job.last_published_hash, last.local_content_hash)

    def test_publish_update_blocks_changed_existing_document(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
            )
            document_path = config.knowledge_dir / "general" / "example.md"
            document_path.parent.mkdir(parents=True)
            document_path.write_text("# Example\n", encoding="utf-8")
            last = publish_document(document_path, target="feishu", config=config, approved=True)
            last.status = "published"
            last.target_document_id = "doc-123"
            document_path.write_text("# Example\n\nChanged.\n", encoding="utf-8")
            publisher = FeishuPublisher(config, client=FakeFeishuClient())
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
                    update=True,
                    last_publication=last,
                    registry=registry,
                )

            self.assertEqual(job.status, "blocked")
            self.assertEqual(job.error_code, "update_not_supported")
            self.assertEqual(job.target_document_id, "doc-123")
            self.assertNotEqual(job.local_content_hash, last.local_content_hash)


    def test_publish_update_updates_changed_existing_document(self) -> None:
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
            client = FakeUpdatingFeishuClient()
            publisher = FeishuPublisher(config, client=client)
            registry = PublisherRegistry(config, publishers=[publisher])
            last = publish_document(document_path, target="feishu", config=config, approved=True)
            last.status = "published"
            last.target_document_id = "doc-123"
            last.target_url = "https://example.feishu.cn/docx/doc-123"
            document_path.write_text("# Example\n\nChanged.\n", encoding="utf-8")

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
                    update=True,
                    last_publication=last,
                    registry=registry,
                )

            self.assertEqual(job.status, "published")
            self.assertEqual(job.target_document_id, "doc-123")
            self.assertEqual(job.remote_revision, "rev-2")
            self.assertEqual(client.updated_document_id, "doc-123")
            self.assertTrue(client.updated_blocks)
            self.assertEqual(client.created_title, "")
            self.assertNotEqual(job.local_content_hash, last.local_content_hash)

    def test_publish_update_blocks_remote_revision_conflict(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = Config(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "memory.sqlite",
                feishu_folder_token="folder-token",
            )
            document_path = config.knowledge_dir / "general" / "example.md"
            document_path.parent.mkdir(parents=True)
            document_path.write_text("# Example\n\nChanged.\n", encoding="utf-8")
            client = FakeUpdatingFeishuClient(remote_revision="rev-remote")
            publisher = FeishuPublisher(config, client=client)
            registry = PublisherRegistry(config, publishers=[publisher])
            last = PublicationJob(
                document_path=str(document_path),
                target="feishu",
                status="published",
                local_content_hash="old-hash",
                target_document_id="doc-123",
                remote_revision="rev-local",
                completed_at="2026-07-03T00:00:00+00:00",
            )

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
                    update=True,
                    last_publication=last,
                    registry=registry,
                )

            self.assertEqual(job.status, "blocked")
            self.assertEqual(job.error_code, "remote_revision_conflict")
            self.assertEqual(job.remote_revision, "rev-remote")
            self.assertEqual(client.updated_document_id, "")

    def test_markdown_converter_and_title_strategy(self) -> None:
        content = (
            '---\ntitle: "Front Matter Title"\n---\n\n'
            '# Heading\n\n'
            '- First [Docs](https://example.com)\n'
            '1. Open `CLI`\n\n'
            '![Screenshot](screenshots/step-1.png)\n\n'
            '```\nveriknow run demo\n```\n'
        )

        self.assertEqual(
            title_for_document(content, Path("fallback.md"), "front_matter"),
            "Front Matter Title",
        )
        blocks = MarkdownToFeishuConverter().convert(content)
        self.assertEqual(blocks[0]["block_type"], 3)
        self.assertEqual(blocks[1]["block_type"], 12)
        self.assertEqual(blocks[2]["block_type"], 13)
        self.assertEqual(blocks[3]["block_type"], 2)
        self.assertEqual(blocks[4]["block_type"], 14)
        self.assertEqual(
            blocks[1]["bullet"]["elements"][0]["text_run"]["content"],
            "First Docs (https://example.com)",
        )
        self.assertEqual(blocks[2]["ordered"]["elements"][0]["text_run"]["content"], "Open CLI")
        self.assertEqual(
            blocks[3]["text"]["elements"][0]["text_run"]["content"],
            "Image: Screenshot (screenshots/step-1.png)",
        )
        self.assertIn("veriknow run demo", blocks[4]["code"]["elements"][0]["text_run"]["content"])

    def test_markdown_converter_preserves_inline_image_targets(self) -> None:
        blocks = MarkdownToFeishuConverter().convert(
            "Use ![diagram](assets/flow.png) before [Docs](https://example.com/docs).\n"
        )

        self.assertEqual(
            blocks[0]["text"]["elements"][0]["text_run"]["content"],
            "Use diagram (assets/flow.png) before Docs (https://example.com/docs).",
        )

    def test_markdown_converter_strips_common_inline_markers(self) -> None:
        blocks = MarkdownToFeishuConverter().convert(
            "- **Bold** and _italic_ with ~~old~~ text, `code`, and [Docs](https://example.com).\n"
        )

        self.assertEqual(blocks[0]["block_type"], 12)
        self.assertEqual(
            blocks[0]["bullet"]["elements"][0]["text_run"]["content"],
            "Bold and italic with old text, code, and Docs (https://example.com).",
        )

    def test_markdown_converter_handles_common_block_fallbacks(self) -> None:
        blocks = MarkdownToFeishuConverter().convert(
            "> Quoted [Docs](https://example.com)\n"
            "- [x] Done\n"
            "- [ ] Todo\n"
            "---\n"
            "| Name | URL |\n"
            "| --- | --- |\n"
            "| Docs | https://example.com |\n"
        )
        text_keys = {2: "text", 15: "quote", 17: "todo"}
        contents = [
            block[text_keys[block["block_type"]]]["elements"][0]["text_run"]["content"]
            for block in blocks
            if block["block_type"] != 22
        ]

        self.assertEqual(contents, [
            "Quoted Docs (https://example.com)",
            "Done",
            "Todo",
            "Name | URL",
            "Docs | https://example.com",
        ])
        self.assertEqual(blocks[0]["block_type"], 15)
        self.assertEqual(blocks[1]["block_type"], 17)
        self.assertEqual(blocks[1]["todo"]["style"]["done"], True)
        self.assertEqual(blocks[2]["block_type"], 17)
        self.assertEqual(blocks[2]["todo"]["style"]["done"], False)
        self.assertEqual(blocks[3], {"block_type": 22, "divider": {}})

    def test_feishu_api_client_update_replaces_root_children(self) -> None:
        client = RecordingFeishuApiClient()

        result = client.update_document("tenant-token", document_id="doc-123", blocks=[{"block_type": 2}])

        self.assertEqual(result["document_id"], "doc-123")
        self.assertEqual(client.requests[0]["method"], "GET")
        self.assertIn("/blocks/doc-123/children", client.requests[0]["path"])
        self.assertEqual(client.requests[1]["method"], "DELETE")
        self.assertEqual(
            client.requests[1]["path"],
            "/open-apis/docx/v1/documents/doc-123/blocks/doc-123/children/batch_delete?document_revision_id=-1",
        )
        self.assertEqual(client.requests[1]["body"], {"start_index": 0, "end_index": 2})
        self.assertEqual(client.requests[2]["path"], "/open-apis/docx/v1/documents/doc-123/blocks/doc-123/children")
        self.assertEqual(client.requests[2]["body"]["children"], [{"block_type": 2}])

    def test_feishu_api_client_update_handles_paginated_children_and_revision(self) -> None:
        client = PaginatedFeishuApiClient()

        result = client.update_document("tenant-token", document_id="doc-123", blocks=[{"block_type": 2}])

        self.assertEqual(result["revision_id"], "rev-3")
        self.assertIn("page_token=next", client.requests[1]["path"])
        self.assertEqual(client.requests[2]["method"], "DELETE")
        self.assertEqual(client.requests[2]["body"], {"start_index": 0, "end_index": 2})

    def test_feishu_api_client_update_handles_empty_children(self) -> None:
        client = EmptyFeishuApiClient()

        result = client.update_document("tenant-token", document_id="doc-123", blocks=[{"block_type": 2}])

        self.assertEqual(result["revision"], "rev-1")
        self.assertFalse(any("/children/batch_delete" in request["path"] for request in client.requests))
        self.assertTrue(any(request["path"].endswith("/children") and request["method"] == "POST" for request in client.requests))

    def test_feishu_api_client_chunks_child_creation_at_official_limit(self) -> None:
        client = RecordingFeishuApiClient()
        blocks = [{"block_type": 2, "text": {"elements": [], "style": {}}} for _ in range(51)]

        revision = client.append_blocks("tenant-token", document_id="doc-123", blocks=blocks)

        self.assertEqual(revision, "2")
        self.assertEqual(len(client.requests), 2)
        self.assertTrue(all(request["method"] == "POST" for request in client.requests))
        self.assertTrue(all(request["path"].endswith("/blocks/doc-123/children") for request in client.requests))
        self.assertEqual(len(client.requests[0]["body"]["children"]), 50)
        self.assertEqual(len(client.requests[1]["body"]["children"]), 1)
