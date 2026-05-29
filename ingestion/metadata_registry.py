from __future__ import annotations

import csv
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from ingestion.schema import SourceDocument
from utils.path_tool import get_abs_path, get_project_root


DOCUMENT_REGISTRY_PATH = "data/raw/financial_reports/document_registry.csv"
REGISTRY_FILE_NAMES = {"company_registry.csv", "document_registry.csv"}


def _normalize_path(path: str) -> str:
    raw_path = Path(path)
    if not raw_path.is_absolute():
        raw_path = Path(get_project_root()) / raw_path
    return os.path.normcase(str(raw_path.resolve()))


@lru_cache(maxsize=4)
def load_document_registry(registry_path: str = DOCUMENT_REGISTRY_PATH) -> dict[str, dict[str, str]]:
    path = Path(get_abs_path(registry_path))
    if not path.exists():
        return {}

    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            local_path = row.get("local_path", "").strip()
            if local_path:
                rows[_normalize_path(local_path)] = {key: (value or "") for key, value in row.items()}
    return rows


def should_skip_registry_file(path: str) -> bool:
    return Path(path).name in REGISTRY_FILE_NAMES


def is_registered_document(path: str) -> bool:
    return _normalize_path(path) in load_document_registry()


def registered_document_paths(
    data_path: str | None = None,
    supported_suffixes: set[str] | None = None,
) -> list[str]:
    root = Path(get_abs_path(data_path)) if data_path else None
    root_resolved = root.resolve() if root else None
    paths: list[str] = []

    for row in load_document_registry().values():
        local_path = row.get("local_path", "").strip()
        if not local_path:
            continue
        path = Path(local_path)
        if not path.is_absolute():
            path = Path(get_project_root()) / path
        path = path.resolve()
        if supported_suffixes and path.suffix.lower() not in supported_suffixes:
            continue
        if root_resolved:
            try:
                path.relative_to(root_resolved)
            except ValueError:
                continue
        if path.is_file() and not should_skip_registry_file(str(path)):
            paths.append(str(path))
    return sorted(paths)


def registry_row_fingerprint(path: str) -> str:
    row = load_document_registry().get(_normalize_path(path))
    if not row:
        return ""
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def enrich_documents_from_registry(path: str, docs: list[SourceDocument]) -> list[SourceDocument]:
    row = load_document_registry().get(_normalize_path(path))
    if not row:
        return docs

    for doc in docs:
        doc.doc_type = row.get("doc_type") or doc.doc_type
        doc.metadata.update(
            {
                "company_id": row.get("company_id", ""),
                "ticker": row.get("ticker", ""),
                "company_name": row.get("company_name", ""),
                "doc_type": row.get("doc_type", doc.doc_type),
                "report_period": row.get("report_period", ""),
                "publish_date": row.get("publish_date", ""),
                "source_url": row.get("source_url", ""),
                "cninfo_detail_url": row.get("cninfo_detail_url", ""),
                "language": row.get("language", ""),
                "source_priority": row.get("source_priority", ""),
                "text_normalization": row.get("text_normalization", ""),
                "is_core_document": row.get("is_core_document", ""),
                "parse_status": row.get("parse_status", ""),
                "registry_hash": row.get("hash", ""),
                "industry": row.get("industry", ""),
            }
        )
    return docs
