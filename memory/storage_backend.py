"""存储后端抽象层：ABC + DiskBackend + RedisBackend。
上层代码只调用 StorageBackend 协议，无需关心底层是磁盘还是Redis。
Redis不可用时自动降级到DiskBackend。"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from utils.config_handler import memory_cof
from utils.path_tool import get_abs_path

# ── user_id 安全过滤：只允许字母数字下划线中划线 ──

_SAFE_USER_ID_RE = re.compile(r"""^[a-zA-Z0-9_-]{1,64}$""")


def sanitize_user_id(user_id: str) -> str:
    """过滤危险字符，防止路径注入和Redis key冲突。"""
    user_id = user_id.strip()
    if not _SAFE_USER_ID_RE.match(user_id):
        raise ValueError(f"非法user_id '{user_id}'：只允许字母数字下划线中划线，长度1-64")
    return user_id


# ── ABC抽象协议 ──

class StorageBackend(ABC):
    """存储后端抽象接口——定义对话和画像的读写协议。
    上层代码（app.py/graph.py）只依赖此接口，可一键切换后端。"""

    # ── 对话历史 ──

    @abstractmethod
    def get_threads(self, user_id: str) -> list[dict[str, Any]]:
        """读取某用户的全部对话线程列表。"""

    @abstractmethod
    def save_threads(self, user_id: str, threads: list[dict[str, Any]]) -> None:
        """保存某用户的全部对话线程。"""

    @abstractmethod
    def list_users(self) -> list[str]:
        """列出所有已注册用户（按最近活跃排序）。"""

    # ── 用户画像 ──

    @abstractmethod
    def get_profile(self, user_id: str) -> dict[str, Any]:
        """读取某用户的画像，不存在则返回空模板。"""

    @abstractmethod
    def save_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        """保存某用户画像。"""

    # ── 会话短期记忆状态（summary+slots）──

    @abstractmethod
    def get_session_state(self, user_id: str) -> dict[str, Any]:
        """读取某用户的短期记忆状态（summary, slots）。
        不存在则返回空模板，用于恢复ShortTermMemory。"""

    @abstractmethod
    def save_session_state(self, user_id: str, state: dict[str, Any]) -> None:
        """保存某用户的短期记忆状态。"""


# ── DiskBackend：磁盘JSON文件存储 ──

class DiskBackend(StorageBackend):
    """磁盘存储后端——零依赖，开发/调试阶段首选。"""

    def __init__(self, conversation_dir: str | None = None, profile_path: str | None = None):
        conv_dir = get_abs_path(conversation_dir or memory_cof.get("conversation_store_path", "data/processed/conversations"))
        self._conv_dir = Path(conv_dir)
        self._conv_dir.mkdir(parents=True, exist_ok=True)

        profile_file = get_abs_path(profile_path or memory_cof.get("profile_store_path", "data/processed/user_profiles.json"))
        self._profile_path = Path(profile_file)
        self._profile_path.parent.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, Any] = self._load_profiles()

    # ── 对话 ──

    def _user_conv_file(self, user_id: str) -> Path:
        return self._conv_dir / f"{sanitize_user_id(user_id)}.json"

    def get_threads(self, user_id: str) -> list[dict[str, Any]]:
        f = self._user_conv_file(user_id)
        if not f.exists():
            return []
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def save_threads(self, user_id: str, threads: list[dict[str, Any]]) -> None:
        f = self._user_conv_file(user_id)
        f.write_text(json.dumps(threads, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_users(self) -> list[str]:
        if not self._conv_dir.exists():
            return []
        files = sorted(self._conv_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        return [f.stem for f in files]

    # ── 画像 ──

    def _load_profiles(self) -> dict[str, Any]:
        if not self._profile_path.exists():
            return {}
        try:
            return json.loads(self._profile_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_profiles_disk(self) -> None:
        self._profile_path.write_text(json.dumps(self._profiles, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_profile(self, user_id: str) -> dict[str, Any]:
        return self._profiles.setdefault(
            sanitize_user_id(user_id),
            {"watchlist": [], "preferred_metrics": [], "risk_preference": "neutral",
             "language_style": "professional", "history_topics": []},
        )

    def save_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        self._profiles[sanitize_user_id(user_id)] = profile
        self._save_profiles_disk()

    # ── 会话短期记忆 ──

    def _session_state_file(self, user_id: str) -> Path:
        return self._conv_dir / f"{sanitize_user_id(user_id)}_session.json"

    def get_session_state(self, user_id: str) -> dict[str, Any]:
        f = self._session_state_file(user_id)
        if not f.exists():
            return {"summary": "", "slots": {}}
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"summary": "", "slots": {}}

    def save_session_state(self, user_id: str, state: dict[str, Any]) -> None:
        f = self._session_state_file(user_id)
        f.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ── RedisBackend：Redis Hash/JSON存储 ──

class RedisBackend(StorageBackend):
    """Redis存储后端——生产环境首选，原子操作、天然并发安全。
    依赖redis-py包和redis-server服务。"""

    CONV_KEY_PREFIX = "fa:conv:"      # 对话：fa:conv:{user_id} → JSON string
    PROFILE_KEY_PREFIX = "fa:profile:" # 画像：fa:profile:{user_id} → JSON string
    SESSION_KEY_PREFIX = "fa:session:" # 短期记忆：fa:session:{user_id} → JSON string
    USERS_SET_KEY = "fa:users"         # 用户集合
    USER_ACTIVITY_KEY = "fa:activity"  # sorted set：最近活跃时间戳

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0,
                 password: str | None = None, socket_timeout: float = 5.0):
        import redis as _redis
        self._redis = _redis.Redis(
            host=host, port=port, db=db, password=password,
            socket_connect_timeout=socket_timeout, socket_timeout=socket_timeout,
        )
        # 连接验证
        self._redis.ping()

    # ── 对话 ──

    def get_threads(self, user_id: str) -> list[dict[str, Any]]:
        uid = sanitize_user_id(user_id)
        raw = self._redis.get(f"{self.CONV_KEY_PREFIX}{uid}")
        if raw is None:
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []

    def save_threads(self, user_id: str, threads: list[dict[str, Any]]) -> None:
        uid = sanitize_user_id(user_id)
        self._redis.set(f"{self.CONV_KEY_PREFIX}{uid}", json.dumps(threads, ensure_ascii=False))
        self._redis.sadd(self.USERS_SET_KEY, uid)
        self._redis.zadd(self.USER_ACTIVITY_KEY, {uid: time.time()})

    def list_users(self) -> list[str]:
        # 按最近活跃排序
        active = self._redis.zrevrange(self.USER_ACTIVITY_KEY, 0, -1)
        return [uid.decode() if isinstance(uid, bytes) else uid for uid in active]

    # ── 画像 ──

    def get_profile(self, user_id: str) -> dict[str, Any]:
        uid = sanitize_user_id(user_id)
        raw = self._redis.get(f"{self.PROFILE_KEY_PREFIX}{uid}")
        if raw is None:
            return {"watchlist": [], "preferred_metrics": [], "risk_preference": "neutral",
                    "language_style": "professional", "history_topics": []}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"watchlist": [], "preferred_metrics": [], "risk_preference": "neutral",
                    "language_style": "professional", "history_topics": []}

    def save_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        uid = sanitize_user_id(user_id)
        self._redis.set(f"{self.PROFILE_KEY_PREFIX}{uid}", json.dumps(profile, ensure_ascii=False))

    # ── 会话短期记忆 ──

    def get_session_state(self, user_id: str) -> dict[str, Any]:
        uid = sanitize_user_id(user_id)
        raw = self._redis.get(f"{self.SESSION_KEY_PREFIX}{uid}")
        if raw is None:
            return {"summary": "", "slots": {}}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"summary": "", "slots": {}}

    def save_session_state(self, user_id: str, state: dict[str, Any]) -> None:
        uid = sanitize_user_id(user_id)
        self._redis.set(f"{self.SESSION_KEY_PREFIX}{uid}", json.dumps(state, ensure_ascii=False))


# ── 自动选择后端：Redis优先，不可用则降级Disk ──

def create_backend() -> StorageBackend:
    """工厂函数：尝试连接Redis，成功则用RedisBackend，失败则降级DiskBackend。"""
    redis_conf = memory_cof.get("redis", {})
    if redis_conf.get("enabled", False):
        try:
            return RedisBackend(
                host=redis_conf.get("host", "localhost"),
                port=int(redis_conf.get("port", 6379)),
                db=int(redis_conf.get("db", 0)),
                password=redis_conf.get("password"),
                socket_timeout=float(redis_conf.get("timeout", 5.0)),
            )
        except Exception:
            # Redis连接失败 → 降级到Disk
            pass
    return DiskBackend()