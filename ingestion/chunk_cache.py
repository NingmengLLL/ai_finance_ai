from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ingestion.schema import Chunk
from utils.logger_handler import logger
from utils.path_tool import get_abs_path, get_project_root


CACHE_SCHEMA_VERSION = "chunk_cache_v1"


def file_md5(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as f:
        while block := f.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def normalized_path_key(path: str) -> str:
    raw_path = Path(path)
    if not raw_path.is_absolute():
        raw_path = Path(get_project_root()) / raw_path
    return os.path.normcase(str(raw_path.resolve()))


def _cache_file_name(path_key: str) -> str:
    return hashlib.md5(path_key.encode("utf-8")).hexdigest() + ".json"


class ChunkCache:
    def __init__(self, signature: str, cache_dir: str = "data/processed/chunk_cache"):
        self.signature = signature
        self.cache_dir = Path(get_abs_path(cache_dir))
        self.manifest_path = self.cache_dir / "manifest.json"
        self.enabled = True
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"Chunk cache disabled, cannot create cache directory: {exc}")
            self.enabled = False
        self.dirty = False
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"schema_version": CACHE_SCHEMA_VERSION, "files": {}}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {"schema_version": CACHE_SCHEMA_VERSION, "files": {}}
        if data.get("schema_version") != CACHE_SCHEMA_VERSION:
            return {"schema_version": CACHE_SCHEMA_VERSION, "files": {}}
        data.setdefault("files", {})
        return data

    def get(self, source_file: str, file_hash: str, metadata_hash: str = "") -> list[Chunk] | None:
        if not self.enabled:
            return None
        path_key = normalized_path_key(source_file)
        entry = self.manifest.get("files", {}).get(path_key)
        if not entry:
            return None
        if entry.get("file_hash") != file_hash:
            return None
        if entry.get("metadata_hash", "") != metadata_hash:
            return None
        if entry.get("signature") != self.signature:
            return None

        cache_file = self.cache_dir / entry.get("cache_file", "")
        if not cache_file.exists():
            return None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            return None
        if payload.get("file_hash") != file_hash or payload.get("signature") != self.signature:
            return None
        return [Chunk.from_dict(item) for item in payload.get("chunks", [])]

    def set(self, source_file: str, file_hash: str, metadata_hash: str, chunks: list[Chunk]) -> None:
        if not self.enabled:
            return
        path_key = normalized_path_key(source_file)
        cache_file_name = _cache_file_name(path_key)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source_file": path_key,
            "file_hash": file_hash,
            "metadata_hash": metadata_hash,
            "signature": self.signature,
            "chunks": [chunk.to_dict() for chunk in chunks],
        }
        cache_file = self.cache_dir / cache_file_name
        tmp_file = cache_file.with_suffix(".tmp")
        try:
            tmp_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_file.replace(cache_file)
        except OSError as exc:
            logger.warning(f"Chunk cache write skipped for {Path(source_file).name}: {exc}")
            return

        self.manifest.setdefault("files", {})[path_key] = {
            "source_file": path_key,
            "file_hash": file_hash,
            "metadata_hash": metadata_hash,
            "signature": self.signature,
            "cache_file": cache_file_name,
            "chunk_count": len(chunks),
        }
        self.dirty = True

    def prune(self, active_files: list[str]) -> None:
        if not self.enabled:
            return
        active_keys = {normalized_path_key(path) for path in active_files}
        files = self.manifest.setdefault("files", {})
        stale_keys = [path_key for path_key in files if path_key not in active_keys]
        for path_key in stale_keys:
            cache_file = self.cache_dir / files[path_key].get("cache_file", "")
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except OSError as exc:
                    logger.warning(f"Chunk cache prune skipped for {cache_file.name}: {exc}")
                    continue
            del files[path_key]
            self.dirty = True

    def save(self) -> None:
        if not self.enabled or not self.dirty:
            return
        tmp_file = self.manifest_path.with_suffix(".tmp")
        try:
            tmp_file.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_file.replace(self.manifest_path)
        except OSError as exc:
            logger.warning(f"Chunk cache manifest save skipped: {exc}")
            return
        self.dirty = False
