from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

from ingestion.metadata_registry import (
    DOCUMENT_REGISTRY_PATH,
    is_registered_document,
    load_document_registry,
    should_skip_registry_file,
)
from utils.logger_handler import logger
from utils.path_tool import get_abs_path, get_project_root


COMPANY_REGISTRY_PATH = "data/raw/financial_reports/company_registry.csv"
REGISTRY_FIELDS = [
    "doc_id",
    "company_id",
    "ticker",
    "company_name",
    "doc_type",
    "report_period",
    "publish_date",
    "source_url",
    "local_path",
    "file_ext",
    "language",
    "is_core_document",
    "parse_status",
    "hash",
    "source_priority",
    "text_normalization",
    "cninfo_detail_url",
]

COMPANY_ALIASES = {
    "BIDU_baidu": ("baidu", "bidu", "\u767e\u5ea6"),
    "0700.HK_tencent": ("tencent", "\u817e\u8baf", "\u9a30\u8a0a"),
    "3690.HK_meituan": ("meituan", "\u7f8e\u56e2", "\u7f8e\u5718"),
    "1024.HK_kuaishou": ("kuaishou", "\u5feb\u624b"),
    "BABA_alibaba": ("alibaba", "\u963f\u91cc", "\u963f\u88e1"),
    "JD_jd": ("jd.com", "jingdong", "\u4eac\u4e1c", "\u4eac\u6771"),
    "NTES_netease": ("netease", "\u7f51\u6613", "\u7db2\u6613"),
    "PDD_pdd": ("pdd", "pinduoduo", "\u62fc\u591a\u591a"),
    "9626.HK_bilibili": ("bilibili", "\u54d4\u54e9\u54d4\u54e9"),
}
UNKNOWN_COMPANY_ID = "UNKNOWN_local"
UNKNOWN_COMPANY_ROW = {
    "company_id": UNKNOWN_COMPANY_ID,
    "ticker": "LOCAL",
    "company_name": "Local Unclassified Document",
    "market": "local",
    "sector": "unclassified",
    "subsector": "unclassified",
    "size_bucket": "unknown",
    "listing_status": "unknown",
    "ir_url": "",
}


def _read_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    target = Path(get_abs_path(path))
    if not target.exists():
        return [], []
    with target.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    target = Path(get_abs_path(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_unknown_company() -> dict[str, str]:
    fields, rows = _read_csv(COMPANY_REGISTRY_PATH)
    if any(row.get("company_id") == UNKNOWN_COMPANY_ID for row in rows):
        return next(row for row in rows if row.get("company_id") == UNKNOWN_COMPANY_ID)

    fields = fields or list(UNKNOWN_COMPANY_ROW)
    for field in UNKNOWN_COMPANY_ROW:
        if field not in fields:
            fields.append(field)
    row = {field: "" for field in fields}
    row.update(UNKNOWN_COMPANY_ROW)
    rows.append(row)
    _write_csv(COMPANY_REGISTRY_PATH, fields, rows)
    logger.info(f"Auto-added fallback company registry row: {UNKNOWN_COMPANY_ID}")
    return row


def _ensure_fields(fields: list[str], rows: list[dict[str, str]]) -> list[str]:
    updated = list(fields)
    for field in REGISTRY_FIELDS:
        if field not in updated:
            updated.append(field)
    for row in rows:
        for field in updated:
            row.setdefault(field, "")
    return updated


def _file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(get_project_root()))).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _scan_unregistered_files(root: Path, supported_suffixes: set[str]) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if should_skip_registry_file(str(path)):
            continue
        if path.suffix.lower() not in supported_suffixes:
            continue
        if is_registered_document(str(path)):
            continue
        files.append(path.resolve())
    return sorted(files)


def _load_companies() -> dict[str, dict[str, str]]:
    _, rows = _read_csv(COMPANY_REGISTRY_PATH)
    return {row["company_id"]: row for row in rows if row.get("company_id")}


def _pdf_sample(path: Path, max_pages: int = 3) -> str:
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages[:max_pages])
    except Exception:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages[:max_pages])
        except Exception:
            return ""


def _text_sample(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _pdf_sample(path)
    if path.suffix.lower() in {".txt", ".md", ".html", ".htm", ".csv"}:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")[:5000]
        except Exception:
            return ""
    return ""


def _infer_company_id(path: Path, sample: str, companies: dict[str, dict[str, str]]) -> str:
    haystack = f"{path.name}\n{sample}".lower()
    for company_id, aliases in COMPANY_ALIASES.items():
        if company_id in companies and any(alias.lower() in haystack for alias in aliases):
            return company_id
    return ""


def _infer_doc_type(path: Path, sample: str) -> str:
    haystack = f"{path.name}\n{sample}".lower()
    if any(term in haystack for term in ("q1", "q2", "q3", "q4", "quarter", "earnings")):
        return "quarterly_results"
    if any(term in haystack for term in ("interim", "half-year", "\u4e2d\u671f\u62a5\u544a", "\u4e2d\u671f\u5831\u544a")):
        return "interim_report"
    if any(
        term in haystack
        for term in (
            "annual",
            "20-f",
            "20f",
            "ndgb",
            "\u5e74\u62a5",
            "\u5e74\u5831",
            "\u5e74\u5ea6\u62a5\u544a",
            "\u5e74\u5ea6\u5831\u544a",
        )
    ):
        return "annual_report"
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "financial_table"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".txt", ".md", ".html", ".htm"}:
        return "text"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image_or_chart"
    return "local_document"


def _infer_report_period(path: Path, sample: str, doc_type: str) -> str:
    haystack = f"{path.name}\n{sample[:8000]}"
    years = [int(match) for match in re.findall(r"20\d{2}", haystack)]
    years = [year for year in years if 2020 <= year <= 2030]
    if not years:
        return "UNKNOWN"
    fiscal_year = years[0]
    if doc_type == "quarterly_results":
        quarter_match = re.search(r"\bq([1-4])\b", haystack, flags=re.I)
        return f"FY{fiscal_year}_Q{quarter_match.group(1)}" if quarter_match else f"FY{fiscal_year}"
    return f"FY{fiscal_year}"


def _safe_id(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "local_document"


def _infer_language(sample: str) -> str:
    if any(term in sample for term in ("\u5e74\u5ea6\u5831\u544a", "\u8ca1\u52d9", "\u80a1\u4efd", "\u7d9c\u5408")):
        return "zh-Hant"
    if any(term in sample for term in ("\u5e74\u5ea6\u62a5\u544a", "\u8d22\u52a1", "\u80a1\u4efd", "\u7efc\u5408")):
        return "zh-Hans"
    return "zh-Hans"


def _normalization_for_language(language: str) -> str:
    return "traditional_to_simplified" if language == "zh-Hant" else ""


def _unique_doc_id(base: str, rows: list[dict[str, str]]) -> str:
    existing = {row.get("doc_id", "") for row in rows}
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _build_registry_row(path: Path, companies: dict[str, dict[str, str]], rows: list[dict[str, str]]) -> dict[str, str] | None:
    sample = _text_sample(path)
    company_id = _infer_company_id(path, sample, companies)
    if not company_id:
        company_id = UNKNOWN_COMPANY_ID
        if company_id not in companies:
            companies[company_id] = _ensure_unknown_company()
    doc_type = _infer_doc_type(path, sample)
    report_period = _infer_report_period(path, sample, doc_type)

    company = companies[company_id]
    language = _infer_language(sample)
    year = report_period.replace("FY", "").split("_")[0] or "UNKNOWN"
    ticker_id = company.get("ticker", company_id).replace(".", "")
    base_doc_id = _safe_id(f"{ticker_id}_{year}_{doc_type}_{path.stem}")
    return {
        "doc_id": _unique_doc_id(base_doc_id, rows),
        "company_id": company_id,
        "ticker": company.get("ticker", ""),
        "company_name": company.get("company_name", ""),
        "doc_type": doc_type,
        "report_period": report_period,
        "publish_date": "",
        "source_url": "local_file",
        "local_path": _relative_to_project(path),
        "file_ext": path.suffix.lower().lstrip("."),
        "language": language,
        "is_core_document": "true",
        "parse_status": "downloaded",
        "hash": _file_md5(path),
        "source_priority": "local",
        "text_normalization": _normalization_for_language(language),
        "cninfo_detail_url": "",
    }


def sync_registry_from_data_path(data_path: str, supported_suffixes: set[str]) -> dict[str, int]:
    root = Path(get_abs_path(data_path))
    if not root.exists():
        return {"scanned": 0, "registered": 0, "skipped": 0}

    fields, rows = _read_csv(DOCUMENT_REGISTRY_PATH)
    fields = _ensure_fields(fields or REGISTRY_FIELDS, rows)
    companies = _load_companies()
    candidates = _scan_unregistered_files(root, supported_suffixes)

    registered = 0
    skipped = 0
    for path in candidates:
        row = _build_registry_row(path, companies, rows)
        if row is None:
            skipped += 1
            continue
        for field in fields:
            row.setdefault(field, "")
        rows.append(row)
        registered += 1
        logger.info(f"Auto-registered local document: {row['doc_id']} -> {row['local_path']}")

    if registered:
        _write_csv(DOCUMENT_REGISTRY_PATH, fields, rows)
        load_document_registry.cache_clear()

    if candidates:
        logger.info(
            f"Registry sync finished: scanned={len(candidates)}, registered={registered}, skipped={skipped}"
        )
    return {"scanned": len(candidates), "registered": registered, "skipped": skipped}
