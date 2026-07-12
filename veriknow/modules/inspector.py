from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from veriknow.schemas import RunRecord


SENSITIVE_KEY_PARTS = {
    "api_key",
    "app_secret",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "password",
    "secret",
    "tenant_access_token",
    "token",
}

TEXT_PREVIEW_LIMIT = 2000
NON_SECRET_TOKEN_KEYS = {"input_tokens", "output_tokens", "total_tokens"}


def inspect_run(record: RunRecord, run_dir: Path) -> dict[str, Any]:
    return {
        "run": redact(record.to_dict()),
        "run_dir": str(run_dir),
        "artifact_count": len(record.artifacts),
        "artifacts": [
            inspect_artifact(name, Path(path), run_dir)
            for name, path in sorted(record.artifacts.items())
        ],
        "run_files": _run_files(run_dir),
        "llm_usage": summarize_llm_usage(run_dir),
    }


def summarize_llm_usage(run_dir: Path) -> dict[str, Any]:
    artifacts = _llm_artifact_payloads(run_dir)
    metadata = [
        payload.get("call_metadata")
        for _, payload in artifacts
        if isinstance(payload.get("call_metadata"), dict)
    ]
    statuses = [str(item.get("status") or "unknown") for item in metadata]
    latencies = [_number(item.get("latency_ms")) for item in metadata]
    input_tokens = [_integer(item.get("input_tokens")) for item in metadata]
    output_tokens = [_integer(item.get("output_tokens")) for item in metadata]
    total_tokens = [_integer(item.get("total_tokens")) for item in metadata]
    costs = [_number(item.get("estimated_cost_usd")) for item in metadata]
    return {
        "call_count": len(metadata),
        "completed_count": statuses.count("completed"),
        "failed_count": statuses.count("failed"),
        "blocked_count": statuses.count("blocked"),
        "unknown_count": statuses.count("unknown"),
        "total_latency_ms": round(sum(value for value in latencies if value is not None), 3),
        "input_tokens": sum(value for value in input_tokens if value is not None),
        "output_tokens": sum(value for value in output_tokens if value is not None),
        "total_tokens": sum(value for value in total_tokens if value is not None),
        "estimated_cost_usd": round(
            sum(value for value in costs if value is not None),
            8,
        ),
        "cost_reported_count": sum(value is not None for value in costs),
        "usage_reported_count": sum(value is not None for value in total_tokens),
        "prompt_stored_count": sum(
            payload.get("prompt_stored") is True for _, payload in artifacts
        ),
        "prompt_suppressed_count": sum(
            payload.get("prompt_stored") is False for _, payload in artifacts
        ),
        "artifact_paths": [path.relative_to(run_dir).as_posix() for path, _ in artifacts],
    }


def inspect_artifact(name: str, path: Path, run_dir: Path) -> dict[str, Any]:
    exists = path.exists()
    result: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "relative_path": _relative_path(path, run_dir),
        "exists": exists,
    }
    if not exists:
        return result

    result["size_bytes"] = path.stat().st_size
    result["kind"] = _artifact_kind(path)
    preview = _artifact_preview(path)
    if preview is not None:
        result["preview"] = redact(preview)
    return result


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if _is_sensitive_key(text_key):
                redacted[text_key] = "[REDACTED]"
            else:
                redacted[text_key] = redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".txt", ".diff", ".log"}:
        return "text"
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    return suffix.lstrip(".") or "file"


def _artifact_preview(path: Path) -> Any | None:
    kind = _artifact_kind(path)
    if kind == "json":
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {"error": exc.__class__.__name__, "message": str(exc)}
    if kind in {"text"}:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return {"error": exc.__class__.__name__, "message": str(exc)}
        if len(content) > TEXT_PREVIEW_LIMIT:
            return content[:TEXT_PREVIEW_LIMIT] + "\n[TRUNCATED]"
        return content
    return None


def _run_files(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        files.append(
            {
                "path": _relative_path(path, run_dir),
                "size_bytes": path.stat().st_size,
                "kind": _artifact_kind(path),
            }
        )
    return files


def _llm_artifact_payloads(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    llm_dir = run_dir / "llm"
    if not llm_dir.exists():
        return []
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(llm_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            payloads.append((path, value))
    return payloads


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _relative_path(path: Path, run_dir: Path) -> str:
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in NON_SECRET_TOKEN_KEYS:
        return False
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _redact_text(value: str) -> str:
    text = re.sub(r"(?i)bearer\s+[a-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
    text = re.sub(
        r"(?i)(api[_-]?key|app[_-]?secret|password|token|authorization|cookie)(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2[REDACTED]",
        text,
    )
    return text
