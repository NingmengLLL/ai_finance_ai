from __future__ import annotations

from typing import Any

from memory.storage_backend import DiskBackend, StorageBackend, create_backend, sanitize_user_id


class LongTermMemory:
    """用户画像持久化——通过StorageBackend抽象层读写，支持Redis/Disk一键切换。"""

    def __init__(self, backend: StorageBackend | None = None):
        self.backend = backend or DiskBackend()

    def get_profile(self, user_id: str) -> dict[str, Any]:
        return self.backend.get_profile(sanitize_user_id(user_id))

    def update_profile(self, user_id: str, **kwargs) -> dict[str, Any]:
        profile = self.get_profile(user_id)
        for key, value in kwargs.items():
            if isinstance(profile.get(key), list):
                values = value if isinstance(value, list) else [value]
                for item in values:
                    if item and item not in profile[key]:
                        profile[key].append(item)
            elif value:
                profile[key] = value
        self.backend.save_profile(user_id, profile)
        return profile