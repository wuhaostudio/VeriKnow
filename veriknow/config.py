from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "data_dir": "data",
    "database_path": "data/memory.sqlite",
    "default_scope": "public_web",
    "default_output_format": "markdown",
    "default_publish_target": "local",
    "publisher_allow_stub": True,
    "feishu_base_url": "https://open.feishu.cn",
    "feishu_folder_token": "",
    "feishu_document_url_template": "",
    "feishu_title_strategy": "filename",
    "computer_use_domain_allowlist": "",
    "computer_use_approval_keywords": (
        "login,sign in,password,billing,payment,purchase,delete,destructive,"
        "account change,account settings"
    ),
    "default_reverify_interval_days": 30,
    "model_provider": "zhipu",
    "model_name": "glm-5.2",
    "model_api_key_env": "ZHIPUAI_API_KEY",
    "model_base_url": "https://open.bigmodel.cn/api/paas/v4",
    "model_temperature": 0,
    "model_timeout_seconds": 60,
    "model_max_output_tokens": 4000,
    "model_store_prompts": True,
}


@dataclass(frozen=True)
class Config:
    data_dir: Path
    database_path: Path
    default_scope: str = "public_web"
    default_output_format: str = "markdown"
    default_publish_target: str = "local"
    publisher_allow_stub: bool = True
    feishu_base_url: str = "https://open.feishu.cn"
    feishu_folder_token: str = ""
    feishu_document_url_template: str = ""
    feishu_title_strategy: str = "filename"
    computer_use_domain_allowlist: tuple[str, ...] = ()
    computer_use_approval_keywords: tuple[str, ...] = ()
    default_reverify_interval_days: int = 30
    model_provider: str = "zhipu"
    model_name: str = "glm-5.2"
    model_api_key_env: str = "ZHIPUAI_API_KEY"
    model_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    model_temperature: float = 0
    model_timeout_seconds: int = 60
    model_max_output_tokens: int = 4000
    model_store_prompts: bool = True

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def knowledge_dir(self) -> Path:
        return self.data_dir / "knowledge"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def screenshots_dir(self) -> Path:
        return self.data_dir / "screenshots"


def load_config(path: str | Path = "config.yaml") -> Config:
    config_path = Path(path)
    values = dict(DEFAULT_CONFIG)
    if config_path.exists():
        values.update(_read_simple_yaml(config_path))

    data_dir = Path(str(values["data_dir"]))
    database_path = Path(str(values["database_path"]))
    return Config(
        data_dir=data_dir,
        database_path=database_path,
        default_scope=str(values.get("default_scope", DEFAULT_CONFIG["default_scope"])),
        default_output_format=str(
            values.get("default_output_format", DEFAULT_CONFIG["default_output_format"])
        ),
        default_publish_target=str(
            values.get("default_publish_target", DEFAULT_CONFIG["default_publish_target"])
        ),
        publisher_allow_stub=_parse_bool(
            values.get("publisher_allow_stub", DEFAULT_CONFIG["publisher_allow_stub"])
        ),
        feishu_base_url=str(values.get("feishu_base_url", DEFAULT_CONFIG["feishu_base_url"])),
        feishu_folder_token=str(values.get("feishu_folder_token", DEFAULT_CONFIG["feishu_folder_token"])),
        feishu_document_url_template=str(
            values.get("feishu_document_url_template", DEFAULT_CONFIG["feishu_document_url_template"])
        ),
        feishu_title_strategy=str(
            values.get("feishu_title_strategy", DEFAULT_CONFIG["feishu_title_strategy"])
        ),
        computer_use_domain_allowlist=_parse_csv_setting(
            values.get(
                "computer_use_domain_allowlist",
                DEFAULT_CONFIG["computer_use_domain_allowlist"],
            )
        ),
        computer_use_approval_keywords=_parse_csv_setting(
            values.get(
                "computer_use_approval_keywords",
                DEFAULT_CONFIG["computer_use_approval_keywords"],
            )
        ),
        default_reverify_interval_days=int(
            values.get(
                "default_reverify_interval_days",
                DEFAULT_CONFIG["default_reverify_interval_days"],
            )
        ),
        model_provider=str(values.get("model_provider", DEFAULT_CONFIG["model_provider"])),
        model_name=str(values.get("model_name", DEFAULT_CONFIG["model_name"])),
        model_api_key_env=str(values.get("model_api_key_env", DEFAULT_CONFIG["model_api_key_env"])),
        model_base_url=str(values.get("model_base_url", DEFAULT_CONFIG["model_base_url"])),
        model_temperature=float(values.get("model_temperature", DEFAULT_CONFIG["model_temperature"])),
        model_timeout_seconds=int(
            values.get("model_timeout_seconds", DEFAULT_CONFIG["model_timeout_seconds"])
        ),
        model_max_output_tokens=int(
            values.get("model_max_output_tokens", DEFAULT_CONFIG["model_max_output_tokens"])
        ),
        model_store_prompts=_parse_bool(
            values.get("model_store_prompts", DEFAULT_CONFIG["model_store_prompts"])
        ),
    )


def ensure_data_dirs(config: Config) -> None:
    for path in [
        config.data_dir,
        config.runs_dir,
        config.knowledge_dir,
        config.logs_dir,
        config.screenshots_dir,
        config.database_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def create_default_config(path: str | Path = "config.yaml") -> bool:
    config_path = Path(path)
    if config_path.exists():
        return False
    lines = [f"{key}: {value}" for key, value in DEFAULT_CONFIG.items()]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = _parse_scalar(value.strip())
    return values


def _parse_scalar(value: str) -> Any:
    if not value:
        return ""
    if value[0:1] == value[-1:] and value[0:1] in {"'", '"'}:
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


def _parse_csv_setting(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
