"""ConversationStore：对话历史持久化服务。
通过StorageBackend抽象层读写，支持Redis/Disk一键切换。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from memory.storage_backend import StorageBackend, create_backend, sanitize_user_id


class ConversationStore:
    """按user_id持久化对话历史。上层代码只依赖此接口。"""

    def __init__(self, backend: StorageBackend | None = None):
        self.backend = backend or create_backend()

    def get_threads(self, user_id: str) -> list[dict[str, Any]]:
        """读取某用户的全部对话线程。"""
        return self.backend.get_threads(sanitize_user_id(user_id))

    def save_threads(self, user_id: str, threads: list[dict[str, Any]]) -> None:
        """保存某用户的全部对话线程。"""
        self.backend.save_threads(sanitize_user_id(user_id), threads)

    def list_users(self) -> list[str]:
        """列出所有已注册用户（按最近活跃排序）。"""
        return self.backend.list_users()

    def append_message(self, user_id: str, thread_id: str, role: str, content: str, state: dict | None = None) -> None:
        """追加一条消息到指定线程，并持久化。"""
        threads = self.get_threads(user_id)
        thread = next((t for t in threads if t["id"] == thread_id), None)
        if thread is None:
            thread = {"id": thread_id, "title": "新对话", "messages": [], "last_state": {}, "updated_at": ""}
            threads.append(thread)
        thread["messages"].append({"role": role, "content": content})
        if state:
            thread["last_state"] = state
        if thread["title"] == "新对话" and content:
            title = " ".join(content.split())
            thread["title"] = title if len(title) <= 28 else title[:28].rstrip() + "..."
        thread["updated_at"] = datetime.now().strftime("%H:%M")
        self.save_threads(user_id, threads)

    def get_thread(self, user_id: str, thread_id: str) -> dict[str, Any] | None:
        """读取某用户的指定线程。"""
        threads = self.get_threads(user_id)
        return next((t for t in threads if t["id"] == thread_id), None)