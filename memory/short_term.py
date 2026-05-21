from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from memory.storage_backend import StorageBackend, create_backend


@dataclass
class ShortTermMemory:
    window_size: int = 8
    messages: deque[dict] = field(default_factory=deque)
    summary: str = ""   # 历史对话的摘要文本
    slots: dict = field(default_factory=dict)   # 结构化信息槽，存储关键实体/变量

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        while len(self.messages) > self.window_size:
            self.messages.popleft()     # 队列实现，超过窗口大小时丢弃最早的消息(最左侧的消息)

    def set_slot(self, key: str, value: Any) -> None:
        if value:
            self.slots[key] = value

    def snapshot(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "messages": list(self.messages),
            "slots": self.slots.copy(),
        }

    # ── L1持久化：通过StorageBackend读写，关浏览器不丢失 ──

    def persist(self, user_id: str, backend: StorageBackend | None = None) -> None:
        """持久化summary+slots到存储后端。messages不持久化（已有ConversationStore）。"""
        _backend = backend or create_backend()
        _backend.save_session_state(user_id, {
            "summary": self.summary,
            "slots": self.slots,
        })

    def restore(self, user_id: str, backend: StorageBackend | None = None) -> None:
        """从存储后端恢复summary+slots。messages由ConversationStore负责恢复。"""
        _backend = backend or create_backend()
        state = _backend.get_session_state(user_id)
        self.summary = state.get("summary", "")
        self.slots = state.get("slots", {})