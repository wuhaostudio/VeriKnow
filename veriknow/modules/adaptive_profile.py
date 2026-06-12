from __future__ import annotations

from veriknow.memory.store import MemoryStore
from veriknow.schemas import UserPreference


SENSITIVE_KEYWORDS = {
    "age",
    "gender",
    "race",
    "religion",
    "politics",
    "health",
    "personality",
    "身份",
    "年龄",
    "性别",
    "宗教",
    "政治",
    "健康",
    "人格",
}


class AdaptiveProfile:
    """Passive store for task-relevant preferences only."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def append_signal(
        self,
        key: str,
        value: str,
        *,
        source: str = "explicit",
        task_id: str | None = None,
    ) -> UserPreference:
        normalized_key = key.strip().lower()
        if not normalized_key or not value.strip():
            raise ValueError("preference key and value are required")
        if self._looks_sensitive(normalized_key, value):
            raise ValueError("sensitive or personality-label preferences are not stored")

        preference = UserPreference(
            key=normalized_key,
            value=value.strip(),
            source=source,
            task_id=task_id,
        )
        self.store.append_preference(preference)
        return preference

    def _looks_sensitive(self, key: str, value: str) -> bool:
        haystack = f"{key} {value}".lower()
        return any(keyword in haystack for keyword in SENSITIVE_KEYWORDS)
