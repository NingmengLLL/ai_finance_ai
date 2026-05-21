from __future__ import annotations

from typing import Any

from memory.long_term import LongTermMemory
from memory.storage_backend import sanitize_user_id


class UserProfileService:
    """用户画像服务——衔接短期记忆和长期画像的桥梁。"""

    def __init__(self, memory: LongTermMemory | None = None):
        self.memory = memory or LongTermMemory()

    def get(self, user_id: str = "default") -> dict[str, Any]:
        return self.memory.get_profile(sanitize_user_id(user_id))

    def remember_focus(self, user_id: str, companies: list[str], metrics: list[str]) -> dict[str, Any]:
        """从对话中沉淀画像——关注公司和偏好指标。"""
        updates = {}
        if companies:
            updates["watchlist"] = companies
        if metrics:
            updates["preferred_metrics"] = metrics
        return self.memory.update_profile(sanitize_user_id(user_id), **updates)

    def remember_style(self, user_id: str, language_style: str = "", risk_preference: str = "") -> dict[str, Any]:
        """从对话中沉淀风格偏好。"""
        updates = {}
        if language_style:
            updates["language_style"] = language_style
        if risk_preference:
            updates["risk_preference"] = risk_preference
        return self.memory.update_profile(sanitize_user_id(user_id), **updates)