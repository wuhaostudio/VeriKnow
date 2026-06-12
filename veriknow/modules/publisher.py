from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from veriknow.config import Config
from veriknow.schemas import PublicationJob, now_iso


class Publisher:
    target = ""

    def __init__(self, config: Config):
        self.config = config

    def publish(self, document_path: Path) -> PublicationJob:
        raise NotImplementedError


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

    def publish(self, document_path: Path) -> PublicationJob:
        app_id = os.environ.get("FEISHU_APP_ID", "")
        app_secret = os.environ.get("FEISHU_APP_SECRET", "")
        missing = [name for name, value in {"FEISHU_APP_ID": app_id, "FEISHU_APP_SECRET": app_secret}.items() if not value]
        if missing:
            status = "blocked" if self.config.publisher_allow_stub else "failed"
            return PublicationJob(
                document_path=str(document_path),
                target=self.target,
                status=status,
                error_code="missing_credentials",
                message=(
                    "Feishu publishing cannot upload yet. "
                    f"Missing environment variables: {', '.join(missing)}."
                ),
                completed_at=now_iso(),
            )
        if not self.config.feishu_folder_token:
            return PublicationJob(
                document_path=str(document_path),
                target=self.target,
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
            if blocks:
                self.client.append_blocks(token, document_id=document_id, blocks=blocks)
        except FeishuApiError as exc:
            return PublicationJob(
                document_path=str(document_path),
                target=self.target,
                status="failed",
                error_code=exc.code,
                message=exc.message,
                completed_at=now_iso(),
            )
        except OSError as exc:
            return PublicationJob(
                document_path=str(document_path),
                target=self.target,
                status="failed",
                error_code="io_error",
                message=str(exc),
                completed_at=now_iso(),
            )

        return PublicationJob(
            document_path=str(document_path),
            target=self.target,
            status="published",
            target_document_id=str(document_id),
            target_url=target_url,
            message=f"Published to Feishu document {document_id}.",
            completed_at=now_iso(),
        )


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

    publisher = (registry or PublisherRegistry(config)).get(target)
    return publisher.publish(resolved_path)


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

    def append_blocks(self, token: str, *, document_id: str, blocks: list[dict[str, Any]]) -> None:
        self._request_json(
            "POST",
            f"/open-apis/docx/v1/documents/{document_id}/blocks/{document_id}/children/batch_create",
            token=token,
            body={"children": blocks},
        )

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
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body or {}, ensure_ascii=False).encode("utf-8"),
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
        for line in _body_lines(content):
            text = _plain_text(line)
            if not text:
                continue
            blocks.append(text_block(text))
        return blocks


def text_block(content: str) -> dict[str, Any]:
    return {
        "block_type": 2,
        "text": {
            "elements": [
                {
                    "text_run": {
                        "content": content,
                        "text_element_style": {},
                    }
                }
            ],
            "style": {},
        },
    }


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
    stripped = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", stripped)
    stripped = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped)
    stripped = stripped.replace("`", "")
    return stripped.strip()
