from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from veriknow.config import Config
from veriknow.schemas import PublicationJob, now_iso


class PublicationMetadata:
    def __init__(
        self,
        *,
        local_path: str,
        local_content_hash: str,
        last_published_at: str | None = None,
        last_published_hash: str | None = None,
        target_document_id: str | None = None,
        target_url: str | None = None,
        remote_revision: str | None = None,
    ):
        self.local_path = local_path
        self.local_content_hash = local_content_hash
        self.last_published_at = last_published_at
        self.last_published_hash = last_published_hash
        self.target_document_id = target_document_id
        self.target_url = target_url
        self.remote_revision = remote_revision


class Publisher:
    target = ""

    def __init__(self, config: Config):
        self.config = config

    def publish(self, document_path: Path, *, metadata: "PublicationMetadata | None" = None) -> PublicationJob:
        raise NotImplementedError

    def update(
        self,
        document_path: Path,
        *,
        document_id: str,
        metadata: "PublicationMetadata | None" = None,
    ) -> PublicationJob:
        return _publication_job(
            document_path,
            self.target,
            metadata=metadata,
            status="blocked",
            target_document_id=document_id,
            error_code="update_not_supported",
            message="Existing document update is not supported by this publisher; no remote changes were made.",
            completed_at=now_iso(),
        )


class FeishuPublisher(Publisher):
    target = "feishu"
    required_env = ("FEISHU_APP_ID", "FEISHU_APP_SECRET")

    def __init__(
        self,
        config: Config,
        *,
        client: "FeishuApiClient | None" = None,
        converter: "MarkdownToFeishuConverter | None" = None,
    ):
        super().__init__(config)
        self.client = client or FeishuApiClient(config.feishu_base_url)
        self.converter = converter or MarkdownToFeishuConverter()

    def publish(self, document_path: Path, *, metadata: "PublicationMetadata | None" = None) -> PublicationJob:
        app_id = os.environ.get("FEISHU_APP_ID", "")
        app_secret = os.environ.get("FEISHU_APP_SECRET", "")
        missing = [name for name, value in {"FEISHU_APP_ID": app_id, "FEISHU_APP_SECRET": app_secret}.items() if not value]
        if missing:
            status = "blocked" if self.config.publisher_allow_stub else "failed"
            return _publication_job(
                document_path,
                self.target,
                metadata=metadata,
                status=status,
                error_code="missing_credentials",
                message=(
                    "Feishu publishing cannot upload yet. "
                    f"Missing environment variables: {', '.join(missing)}."
                ),
                completed_at=now_iso(),
            )
        if not self.config.feishu_folder_token:
            return _publication_job(
                document_path,
                self.target,
                metadata=metadata,
                status="failed",
                error_code="missing_folder_token",
                message="Feishu folder token is required in config key `feishu_folder_token`.",
                completed_at=now_iso(),
            )

        try:
            content = document_path.read_text(encoding="utf-8")
            title = title_for_document(content, document_path, self.config.feishu_title_strategy)
            blocks = self.converter.convert(content)
            token = self.client.tenant_access_token(app_id, app_secret)
            document = self.client.create_document(
                token,
                title=title,
                folder_token=self.config.feishu_folder_token,
            )
            document_id = document.get("document_id") or document.get("document", {}).get("document_id")
            target_url = document.get("url") or document.get("document", {}).get("url")
            if document_id and not target_url and self.config.feishu_document_url_template:
                target_url = self.config.feishu_document_url_template.format(document_id=document_id)
            if not document_id:
                raise FeishuApiError("missing_document_id", "Feishu create document response did not include document_id")
            remote_revision = _revision_from_document(document)
            if blocks:
                append_result = self.client.append_blocks(token, document_id=document_id, blocks=blocks)
                if isinstance(append_result, dict):
                    remote_revision = _revision_from_document(append_result) or remote_revision
                elif append_result is not None:
                    remote_revision = str(append_result)
        except FeishuApiError as exc:
            return _publication_job(
                document_path,
                self.target,
                metadata=metadata,
                status="failed",
                error_code=exc.code,
                message=exc.message,
                completed_at=now_iso(),
            )
        except OSError as exc:
            return _publication_job(
                document_path,
                self.target,
                metadata=metadata,
                status="failed",
                error_code="io_error",
                message=str(exc),
                completed_at=now_iso(),
            )

        return _publication_job(
            document_path,
            self.target,
            metadata=metadata,
            status="published",
            target_document_id=str(document_id),
            target_url=target_url,
            remote_revision=remote_revision,
            message=f"Published to Feishu document {document_id}.",
            completed_at=now_iso(),
        )

    def update(
        self,
        document_path: Path,
        *,
        document_id: str,
        metadata: "PublicationMetadata | None" = None,
    ) -> PublicationJob:
        app_id = os.environ.get("FEISHU_APP_ID", "")
        app_secret = os.environ.get("FEISHU_APP_SECRET", "")
        missing = [name for name, value in {"FEISHU_APP_ID": app_id, "FEISHU_APP_SECRET": app_secret}.items() if not value]
        if missing:
            status = "blocked" if self.config.publisher_allow_stub else "failed"
            return _publication_job(
                document_path,
                self.target,
                metadata=metadata,
                status=status,
                target_document_id=document_id,
                error_code="missing_credentials",
                message=(
                    "Feishu publishing cannot update yet. "
                    f"Missing environment variables: {', '.join(missing)}."
                ),
                completed_at=now_iso(),
            )
        if not hasattr(self.client, "update_document"):
            return _publication_job(
                document_path,
                self.target,
                metadata=metadata,
                status="blocked",
                target_document_id=document_id,
                error_code="update_not_supported",
                message="Existing Feishu document update is not supported by the current adapter; no remote changes were made.",
                completed_at=now_iso(),
            )

        try:
            content = document_path.read_text(encoding="utf-8")
            blocks = self.converter.convert(content)
            token = self.client.tenant_access_token(app_id, app_secret)
            remote_revision = self._current_remote_revision(token, document_id)
            if metadata and metadata.remote_revision and remote_revision and metadata.remote_revision != remote_revision:
                return _publication_job(
                    document_path,
                    self.target,
                    metadata=metadata,
                    status="blocked",
                    target_document_id=document_id,
                    remote_revision=remote_revision,
                    error_code="remote_revision_conflict",
                    message=(
                        "Remote Feishu document revision differs from the last published revision; "
                        "no remote changes were made."
                    ),
                    completed_at=now_iso(),
                )
            result = self.client.update_document(token, document_id=document_id, blocks=blocks)
            target_url = None
            if isinstance(result, dict):
                target_url = result.get("url")
                remote_revision = _revision_from_document(result) or remote_revision
        except FeishuApiError as exc:
            return _publication_job(
                document_path,
                self.target,
                metadata=metadata,
                status="failed",
                target_document_id=document_id,
                error_code=exc.code,
                message=exc.message,
                completed_at=now_iso(),
            )
        except OSError as exc:
            return _publication_job(
                document_path,
                self.target,
                metadata=metadata,
                status="failed",
                target_document_id=document_id,
                error_code="io_error",
                message=str(exc),
                completed_at=now_iso(),
            )

        return _publication_job(
            document_path,
            self.target,
            metadata=metadata,
            status="published",
            target_document_id=document_id,
            target_url=target_url,
            remote_revision=remote_revision,
            message=f"Updated Feishu document {document_id}.",
            completed_at=now_iso(),
        )

    def _current_remote_revision(self, token: str, document_id: str) -> str | None:
        if not hasattr(self.client, "document_metadata"):
            return None
        result = self.client.document_metadata(token, document_id=document_id)
        if not isinstance(result, dict):
            return None
        return _revision_from_document(result)


class PublisherRegistry:
    def __init__(self, config: Config, publishers: list[Publisher] | None = None):
        self.publishers = {
            publisher.target: publisher
            for publisher in (publishers or [FeishuPublisher(config)])
        }

    def get(self, target: str) -> Publisher:
        try:
            return self.publishers[target]
        except KeyError as exc:
            raise ValueError(f"unsupported publish target: {target}") from exc


def publish_document(
    document_path: str | Path,
    *,
    target: str,
    config: Config,
    approved: bool = False,
    update: bool = False,
    last_publication: PublicationJob | None = None,
    registry: PublisherRegistry | None = None,
) -> PublicationJob:
    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"document not found: {path}")
    if path.suffix.lower() != ".md":
        raise ValueError(f"publish document must be Markdown: {path}")

    resolved_path = path.resolve()
    knowledge_dir = config.knowledge_dir.resolve()
    if not resolved_path.is_relative_to(knowledge_dir):
        raise ValueError(f"document must be an approved local knowledge document: {path}")
    if not approved:
        raise ValueError(f"document has not been approved with `veriknow apply`: {path}")

    local_hash = content_hash_for(resolved_path)
    metadata = PublicationMetadata(
        local_path=str(resolved_path),
        local_content_hash=local_hash,
        last_published_at=last_publication.completed_at if last_publication else None,
        last_published_hash=last_publication.local_content_hash if last_publication else None,
        target_document_id=last_publication.target_document_id if last_publication else None,
        target_url=last_publication.target_url if last_publication else None,
        remote_revision=last_publication.remote_revision if last_publication else None,
    )
    if update and last_publication is not None and last_publication.local_content_hash == local_hash:
        return _publication_job(
            resolved_path,
            target,
            metadata=metadata,
            status="skipped",
            target_document_id=last_publication.target_document_id,
            target_url=last_publication.target_url,
            message="Local content hash is unchanged; publishing skipped.",
            completed_at=now_iso(),
        )
    publisher = (registry or PublisherRegistry(config)).get(target)
    if update and last_publication is not None and last_publication.target_document_id:
        return publisher.update(
            resolved_path,
            document_id=last_publication.target_document_id,
            metadata=metadata,
        )

    return publisher.publish(resolved_path, metadata=metadata)


def content_hash_for(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publication_job(
    document_path: Path,
    target: str,
    *,
    metadata: PublicationMetadata | None = None,
    status: str,
    target_document_id: str | None = None,
    target_url: str | None = None,
    remote_revision: str | None = None,
    error_code: str | None = None,
    message: str = "",
    completed_at: str | None = None,
) -> PublicationJob:
    return PublicationJob(
        document_path=str(document_path),
        target=target,
        status=status,
        local_path=metadata.local_path if metadata else str(document_path),
        local_content_hash=metadata.local_content_hash if metadata else content_hash_for(document_path),
        target_document_id=target_document_id or (metadata.target_document_id if metadata else None),
        target_url=target_url or (metadata.target_url if metadata else None),
        last_published_at=metadata.last_published_at if metadata else None,
        last_published_hash=metadata.last_published_hash if metadata else None,
        remote_revision=remote_revision or (metadata.remote_revision if metadata else None),
        error_code=error_code,
        message=message,
        completed_at=completed_at,
    )


class FeishuApiError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class FeishuApiClient:
    def __init__(self, base_url: str = "https://open.feishu.cn", *, timeout_seconds: int = 20):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def tenant_access_token(self, app_id: str, app_secret: str) -> str:
        data = self._request_json(
            "POST",
            "/open-apis/auth/v3/tenant_access_token/internal",
            body={"app_id": app_id, "app_secret": app_secret},
        )
        token = data.get("tenant_access_token")
        if not token:
            raise FeishuApiError("missing_tenant_access_token", "Feishu token response did not include tenant_access_token")
        return str(token)

    def create_document(self, token: str, *, title: str, folder_token: str) -> dict[str, Any]:
        body: dict[str, str] = {"title": title}
        if folder_token:
            body["folder_token"] = folder_token
        data = self._request_json(
            "POST",
            "/open-apis/docx/v1/documents",
            token=token,
            body=body,
        )
        document = data.get("document", data)
        if not isinstance(document, dict):
            raise FeishuApiError("invalid_document_response", "Feishu create document response was not an object")
        return document

    def append_blocks(self, token: str, *, document_id: str, blocks: list[dict[str, Any]]) -> str | None:
        remote_revision = None
        for chunk in _chunks(blocks, 50):
            result = self._request_json(
                "POST",
                f"/open-apis/docx/v1/documents/{document_id}/blocks/{document_id}/children",
                token=token,
                body={"children": chunk},
            )
            remote_revision = _revision_from_document(result) or remote_revision
        return remote_revision

    def update_document(self, token: str, *, document_id: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        existing_block_ids = self.list_child_block_ids(token, document_id=document_id, block_id=document_id)
        if existing_block_ids:
            self.delete_child_blocks(
                token,
                document_id=document_id,
                block_id=document_id,
                child_count=len(existing_block_ids),
            )
        if blocks:
            self.append_blocks(token, document_id=document_id, blocks=blocks)
        metadata = self.document_metadata(token, document_id=document_id)
        metadata.setdefault("document_id", document_id)
        return metadata

    def document_metadata(self, token: str, *, document_id: str) -> dict[str, Any]:
        data = self._request_json(
            "GET",
            f"/open-apis/docx/v1/documents/{document_id}",
            token=token,
        )
        document = data.get("document", data)
        if not isinstance(document, dict):
            raise FeishuApiError("invalid_document_response", "Feishu document metadata response was not an object")
        return document

    def list_child_block_ids(self, token: str, *, document_id: str, block_id: str) -> list[str]:
        block_ids: list[str] = []
        page_token = ""
        while True:
            path = f"/open-apis/docx/v1/documents/{document_id}/blocks/{block_id}/children"
            if page_token:
                path = f"{path}?{urllib.parse.urlencode({'page_token': page_token})}"
            data = self._request_json("GET", path, token=token)
            items = _block_items_from_response(data)
            for item in items:
                block_id_value = item.get("block_id") or item.get("id")
                if block_id_value:
                    block_ids.append(str(block_id_value))
            has_more = bool(data.get("has_more") or data.get("has_more_children"))
            page_token = str(data.get("page_token") or data.get("next_page_token") or "")
            if not has_more or not page_token:
                break
        return block_ids

    def delete_child_blocks(
        self,
        token: str,
        *,
        document_id: str,
        block_id: str,
        child_count: int,
    ) -> str | None:
        if child_count <= 0:
            return None
        query = urllib.parse.urlencode({"document_revision_id": -1})
        result = self._request_json(
            "DELETE",
            (
                f"/open-apis/docx/v1/documents/{document_id}/blocks/{block_id}"
                f"/children/batch_delete?{query}"
            ),
            token=token,
            body={"start_index": 0, "end_index": child_count},
        )
        return _revision_from_document(result)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        token: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        encoded_body = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=encoded_body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise FeishuApiError("http_error", f"Feishu HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise FeishuApiError("network_error", f"Feishu network error: {exc.reason}") from exc

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise FeishuApiError("invalid_json", "Feishu response was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise FeishuApiError("invalid_response", "Feishu response was not a JSON object")
        code = parsed.get("code", 0)
        if code not in (0, "0"):
            raise FeishuApiError(str(code), str(parsed.get("msg") or parsed.get("message") or "Feishu API error"))
        data = parsed.get("data", parsed)
        if not isinstance(data, dict):
            raise FeishuApiError("invalid_data", "Feishu response data was not a JSON object")
        return data


class MarkdownToFeishuConverter:
    def convert(self, content: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        in_code = False
        code_lines: list[str] = []
        for line in _body_lines(content):
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code:
                    code = "\n".join(code_lines).strip("\n")
                    if code:
                        blocks.append(code_block(code))
                    code_lines = []
                    in_code = False
                else:
                    in_code = True
                continue
            if in_code:
                code_lines.append(line)
                continue
            if not stripped:
                continue
            if _is_table_separator(stripped):
                continue
            blocks.append(markdown_line_block(stripped))
        if code_lines:
            blocks.append(code_block("\n".join(code_lines).strip("\n")))
        return blocks


def markdown_line_block(line: str) -> dict[str, Any]:
    if re.match(r"^[-*_]{3,}$", line):
        return {"block_type": 22, "divider": {}}
    heading = re.match(r"^(#{1,6})\s+(.+)$", line)
    if heading:
        level = len(heading.group(1))
        return text_block(_inline_text(heading.group(2)), block_type=heading_block_type(level))
    task = re.match(r"^[-*]\s+\[([ xX])\]\s+(.+)$", line)
    if task:
        return text_block(
            _inline_text(task.group(2)),
            block_type=17,
            style={"done": task.group(1).lower() == "x"},
        )
    unordered = re.match(r"^[-*]\s+(.+)$", line)
    if unordered:
        return text_block(_inline_text(unordered.group(1)), block_type=12)
    ordered = re.match(r"^\d+\.\s+(.+)$", line)
    if ordered:
        return text_block(_inline_text(ordered.group(1)), block_type=13)
    quote = re.match(r"^>\s?(.+)$", line)
    if quote:
        return text_block(_inline_text(quote.group(1)), block_type=15)
    image = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
    if image:
        return text_block(f"Image: {_image_text(image.group(1), image.group(2))}", block_type=2)
    table_cells = _table_cells(line)
    if table_cells:
        return text_block(" | ".join(_inline_text(cell) for cell in table_cells))
    return text_block(_inline_text(line))


def heading_block_type(level: int) -> int:
    return {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}.get(level, 5)


def text_block(
    content: str,
    *,
    block_type: int = 2,
    style: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block_key = {
        2: "text",
        3: "heading1",
        4: "heading2",
        5: "heading3",
        6: "heading4",
        7: "heading5",
        8: "heading6",
        12: "bullet",
        13: "ordered",
        14: "code",
        15: "quote",
        17: "todo",
    }.get(block_type)
    if block_key is None:
        raise ValueError(f"unsupported Feishu text block type: {block_type}")
    return {
        "block_type": block_type,
        block_key: {
            "elements": [
                {
                    "text_run": {
                        "content": content,
                        "text_element_style": {},
                    }
                }
            ],
            "style": style or {},
        },
    }


def code_block(content: str) -> dict[str, Any]:
    return text_block(content, block_type=14)

def title_for_document(content: str, path: Path, strategy: str) -> str:
    if strategy == "front_matter":
        title = _front_matter_title(content)
        if title:
            return title
    if strategy in {"front_matter", "heading"}:
        title = _heading_title(content)
        if title:
            return title
    return path.stem.replace("-", " ").replace("_", " ").strip() or "VeriKnow Document"


def _front_matter_title(content: str) -> str | None:
    if not content.startswith("---"):
        return None
    for line in content.splitlines()[1:]:
        if line.strip() == "---":
            return None
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'") or None
    return None


def _heading_title(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
    return None


def _body_lines(content: str) -> list[str]:
    lines = content.splitlines()
    if lines and lines[0].strip() == "---":
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return lines[index + 1 :]
    return lines


def _plain_text(line: str) -> str:
    stripped = line.strip()
    stripped = re.sub(r"^#{1,6}\s+", "", stripped)
    stripped = re.sub(r"^[-*]\s+", "", stripped)
    stripped = re.sub(r"^\d+\.\s+", "", stripped)
    return _inline_text(stripped)


def _block_items_from_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    items = data.get("items")
    if items is None:
        items = data.get("children")
    if items is None and isinstance(data.get("block"), dict):
        items = data["block"].get("children")
    if items is None:
        items = []
    if not isinstance(items, list):
        raise FeishuApiError("invalid_blocks_response", "Feishu block children response was not a list")
    return [item for item in items if isinstance(item, dict)]


def _revision_from_document(document: dict[str, Any]) -> str | None:
    revision = (
        document.get("revision")
        or document.get("revision_id")
        or document.get("document_revision_id")
        or document.get("revision_version")
    )
    if revision is not None:
        return str(revision)
    nested = document.get("document")
    if isinstance(nested, dict):
        return _revision_from_document(nested)
    return None


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    cells = _table_cells(stripped)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if "|" not in stripped:
        return []
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    return cells if len(cells) > 1 else []


def _inline_text(value: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda match: _image_text(match.group(1), match.group(2)), value)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda match: _link_text(match.group(1), match.group(2)), text)
    text = _strip_markdown_emphasis(text)
    text = text.replace("`", "")
    return text.strip()


def _strip_markdown_emphasis(value: str) -> str:
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", value)
    text = re.sub(r"__([^_\n]+)__", r"\1", text)
    text = re.sub(r"~~([^~\n]+)~~", r"\1", text)
    text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", text)
    return text


def _link_text(label: str, url: str) -> str:
    clean_label = label.strip()
    clean_url = url.strip()
    if not clean_label:
        return clean_url
    if clean_label == clean_url:
        return clean_url
    return f"{clean_label} ({clean_url})"


def _image_text(alt: str, url: str) -> str:
    clean_alt = alt.strip()
    clean_url = url.strip()
    if not clean_alt:
        return clean_url
    return f"{clean_alt} ({clean_url})"
