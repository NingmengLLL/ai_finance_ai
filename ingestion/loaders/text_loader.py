from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from ingestion.schema import SourceDocument


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_doc_type(path: str, text: str) -> str:
    name = os.path.basename(path).lower()
    if any(keyword in name for keyword in ["annual", "20f", "20-f", "年报", "财报"]):
        return "annual_report"
    if any(keyword in name for keyword in ["interim", "half_year", "half-year", "中期"]):
        return "interim_report"
    if any(keyword in name for keyword in ["quarter", "q1", "q2", "q3", "q4", "earnings", "业绩"]):
        return "quarterly_results"
    if "research" in name or "研报" in name or "深度" in name:
        return "research_report"
    if "call" in name or "纪要" in name or "conference" in name:
        return "earnings_call"
    if "news" in name or "新闻" in name:
        return "financial_news"
    return "text"


def load_text_file(path: str) -> list[SourceDocument]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    if file_path.suffix.lower() in {".html", ".htm"}:
        text = _strip_html(text)
    doc_id = hashlib.md5(str(file_path.resolve()).encode("utf-8")).hexdigest()
    return [
        SourceDocument(
            doc_id=doc_id,
            source_file=str(file_path),
            text=text,
            doc_type=detect_doc_type(str(file_path), text),
            page_number=1,
            metadata={"file_name": file_path.name},
        )
    ]
