from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from utils.path_tool import get_abs_path


LOG_ROOT = Path(get_abs_path("logs"))
LOG_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_LOGGING_CONFIG = {
    "logging_enabled": True,
    "log_level": "INFO",
    "log_llm_payload": False,
    "log_retrieval_details": True,
}


def _load_logging_config() -> dict[str, Any]:
    config_path = Path(get_abs_path("config/logging.yaml"))
    if not config_path.exists():
        return DEFAULT_LOGGING_CONFIG.copy()
    try:
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.load(f, Loader=yaml.FullLoader) or {}
        return {**DEFAULT_LOGGING_CONFIG, **loaded}
    except Exception:
        return DEFAULT_LOGGING_CONFIG.copy()


logging_cof = _load_logging_config()


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_log_level(value: Any) -> int:
    if isinstance(value, int):
        return value
    name = str(value or "INFO").upper()
    return getattr(logging, name, logging.INFO)


class ChineseTimeFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created)
        return f"{dt.year}年{dt.month}月{dt.day}日{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"


DEFAULT_LOGGING_FORMAT = ChineseTimeFormatter("%(asctime)s - %(levelname)s - %(message)s")


class DailyFileHandler(logging.Handler):
    """Write logs to logs/YYYY-MM-DD.log, rotating lazily when the date changes."""

    def __init__(self, log_root: Path) -> None:
        super().__init__()
        self.log_root = log_root
        self.current_date = ""
        self.stream = None

    def _target_path(self, record: logging.LogRecord) -> Path:
        date_text = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d")
        return self.log_root / f"{date_text}.log"

    def _ensure_stream(self, record: logging.LogRecord) -> None:
        date_text = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d")
        if self.stream is not None and self.current_date == date_text:
            return
        self._close_stream()
        self.current_date = date_text
        target = self._target_path(record)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.stream = target.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_stream(record)
            if self.stream is None:
                return
            self.stream.write(self.format(record) + "\n")
            self.stream.flush()
        except Exception:
            self.handleError(record)

    def _close_stream(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def close(self) -> None:
        self._close_stream()
        super().close()


def get_logger(name: str = "agent", log_file: str | None = None, level: int | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level or _parse_log_level(logging_cof.get("log_level")))
    logger.propagate = False
    logger.disabled = not _as_bool(logging_cof.get("logging_enabled"), default=True)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(DEFAULT_LOGGING_FORMAT)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        file_handler = DailyFileHandler(LOG_ROOT)
    file_handler.setFormatter(DEFAULT_LOGGING_FORMAT)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()


def safe_preview(text: Any, max_chars: int = 80) -> str:
    preview = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 3] + "..."


def _format_fields(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, str):
            value = safe_preview(value, 120)
            if not value:
                continue
        parts.append(f"{key}={value}")
    return f"，{', '.join(parts)}" if parts else ""


def log_stage_start(stage_name: str, **fields: Any) -> float:
    started_at = time.perf_counter()
    logger.info(f"开始 {stage_name}{_format_fields(fields)}")
    return started_at


def log_stage_done(stage_name: str, started_at: float, **fields: Any) -> None:
    elapsed = time.perf_counter() - started_at
    logger.info(f"完成 {stage_name}，耗时={elapsed:.2f}s{_format_fields(fields)}")


def log_stage_error(stage_name: str, started_at: float, error: Exception | str | None = None, **fields: Any) -> None:
    elapsed = time.perf_counter() - started_at
    if error is not None:
        fields = {**fields, "error": safe_preview(error, 200)}
    logger.error(
        f"失败 {stage_name}，耗时={elapsed:.2f}s{_format_fields(fields)}",
        exc_info=sys.exc_info()[0] is not None,
    )


class StageLogContext:
    def __init__(self, stage_name: str, **fields: Any) -> None:
        self.stage_name = stage_name
        self.started_at = 0.0
        self.done_fields: dict[str, Any] = {}
        self.start_fields = fields

    def __enter__(self) -> "StageLogContext":
        self.started_at = log_stage_start(self.stage_name, **self.start_fields)
        return self

    def add_done_fields(self, **fields: Any) -> None:
        self.done_fields.update(fields)

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            log_stage_error(self.stage_name, self.started_at, error=exc, **self.done_fields)
            return False
        log_stage_done(self.stage_name, self.started_at, **self.done_fields)
        return False


def log_stage(stage_name: str, **fields: Any) -> StageLogContext:
    return StageLogContext(stage_name, **fields)
