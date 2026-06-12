from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, TypeVar, get_args, get_origin, get_type_hints


T = TypeVar("T", bound="JsonModel")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonModel:
    required_fields: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        for name in self.required_fields:
            value = getattr(self, name)
            if value is None or value == "":
                raise ValueError(f"{self.__class__.__name__}.{name} is required")

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        kwargs: dict[str, Any] = {}
        field_map = {item.name: item for item in fields(cls)}
        type_hints = get_type_hints(cls)
        for name, item in field_map.items():
            if name not in data:
                continue
            kwargs[name] = _coerce_value(type_hints.get(name, item.type), data[name])
        return cls(**kwargs)


@dataclass
class TaskSpec(JsonModel):
    required_fields: ClassVar[tuple[str, ...]] = ("raw_request", "objective", "target")

    raw_request: str
    objective: str
    target: str
    scope: str = "public_web"
    verification_required: bool = True
    verification_method: str = "browser"
    output_format: str = "markdown"
    publish_target: str = "local"
    locale: str = "zh-CN"
    constraints: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)


@dataclass
class EvidenceItem(JsonModel):
    required_fields: ClassVar[tuple[str, ...]] = ("title", "url")

    title: str
    url: str
    source_type: str = "unknown"
    snippet: str = ""
    published_at: str | None = None
    updated_at: str | None = None
    confidence: str = "medium"


@dataclass
class EvidenceBundle(JsonModel):
    task_id: str
    items: list[EvidenceItem] = field(default_factory=list)
    summary: str = ""
    created_at: str = field(default_factory=now_iso)


@dataclass
class VerificationStep(JsonModel):
    required_fields: ClassVar[tuple[str, ...]] = ("description", "expected_result")

    description: str
    expected_result: str
    method: str = "browser"
    tools: list[str] = field(default_factory=list)
    screenshot_required: bool = False
    requires_approval: bool = False


@dataclass
class VerificationPlan(JsonModel):
    task_id: str
    steps: list[VerificationStep] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)


@dataclass
class VerificationResult(JsonModel):
    step_description: str
    status: str
    actual_result: str = ""
    screenshot_path: str | None = None
    log_path: str | None = None
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    observed_at: str = field(default_factory=now_iso)


@dataclass
class VerificationRun(JsonModel):
    task_id: str
    status: str = "pending"
    results: list[VerificationResult] = field(default_factory=list)
    started_at: str = field(default_factory=now_iso)
    completed_at: str | None = None


@dataclass
class KnowledgePatch(JsonModel):
    run_id: str
    target_path: str
    diff: str
    approved: bool = False
    created_at: str = field(default_factory=now_iso)


@dataclass
class MarkdownDocument(JsonModel):
    run_id: str
    title: str
    path: str
    status: str = "draft"
    content: str = ""
    sources: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)


@dataclass
class PublicationJob(JsonModel):
    required_fields: ClassVar[tuple[str, ...]] = ("document_path", "target")

    document_path: str
    target: str
    status: str = "pending"
    target_document_id: str | None = None
    target_url: str | None = None
    error_code: str | None = None
    message: str = ""
    created_at: str = field(default_factory=now_iso)
    completed_at: str | None = None


@dataclass
class UserPreference(JsonModel):
    key: str
    value: str
    source: str = "explicit"
    task_id: str | None = None
    created_at: str = field(default_factory=now_iso)


@dataclass
class RunRecord(JsonModel):
    run_id: str
    raw_request: str
    task: TaskSpec
    status: str = "created"
    artifacts: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    completed_at: str | None = None


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    return value


def _coerce_value(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list and args:
        inner = args[0]
        return [_coerce_value(inner, item) for item in value]

    if origin is dict:
        return dict(value)

    if origin is None and isinstance(annotation, type) and issubclass(annotation, JsonModel):
        return annotation.from_dict(value)

    if origin is not None and type(None) in args:
        if value is None:
            return None
        non_none = [item for item in args if item is not type(None)]
        if non_none:
            return _coerce_value(non_none[0], value)

    return value
