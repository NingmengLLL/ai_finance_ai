from __future__ import annotations

import re
from collections import Counter


PAGE_NUMBER_PATTERNS = (
    re.compile(r"^\d{1,4}$"),
    re.compile(r"^\d{1,4}\s+.+(?:年度报告|年报)$"),
    re.compile(r"^.+(?:年度报告|年报)\s+\d{1,4}$"),
    re.compile(r"^二\s*零\s*二[零一二三四五六七八九]\s*年\s*年\s*报\s*\d{0,4}$"),
)
KEEP_HEADING_PATTERNS = (
    re.compile(r"^(?:目录|目錄|公司资料|公司資料|财务概要|財務概要|主席报告|主席報告)$"),
    re.compile(r"^(?:管理层讨论[及与]分析|管理層討論及分析|董事会报告|董事會報告)$"),
    re.compile(r"^(?:企业管治报告|企業管治報告|独立核数师报告|獨立核數師報告)$"),
    re.compile(r"^(?:综合(?:全面)?收益表|綜合(?:全面)?收益表|综合财务状况表|綜合財務狀況表)$"),
    re.compile(r"^(?:综合权益变动表|綜合權益變動表|综合现金流量表|綜合現金流量表)$"),
    re.compile(r"^(?:综合财务报表附注|綜合財務報表附註|释义|釋義|分部财务资料|分部財務資料)$"),
)


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _noise_key(line: str) -> str:
    return re.sub(r"\s+", "", line.strip())


def _is_keep_heading(line: str) -> bool:
    normalized = _normalize_line(line)
    return any(pattern.match(normalized) for pattern in KEEP_HEADING_PATTERNS)


def _is_page_number_noise(line: str) -> bool:
    normalized = _normalize_line(line)
    compact = _noise_key(normalized)
    if any(pattern.match(normalized) for pattern in PAGE_NUMBER_PATTERNS):
        return True
    report_label = r"(?:年度报告|年度報告|年报|年報)"
    if len(compact) <= 80 and re.match(rf"^\d{{1,4}}.+{report_label}.*\d{{0,4}}$", compact):
        return True
    if len(compact) <= 80 and re.match(rf"^.+{report_label}\d{{1,4}}$", compact):
        return True
    return False


def _edge_lines(text: str, edge_size: int = 3) -> list[str]:
    lines = [_normalize_line(line) for line in text.splitlines() if _normalize_line(line)]
    if len(lines) <= edge_size * 2:
        return lines
    return lines[:edge_size] + lines[-edge_size:]


def repeated_edge_noise_keys(page_texts: list[str]) -> set[str]:
    """Find lines that recur at page edges and are likely headers/footers."""

    if len(page_texts) < 3:
        return set()

    counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for text in page_texts:
        seen_on_page = set()
        for line in _edge_lines(text):
            if _is_keep_heading(line):
                continue
            key = _noise_key(line)
            if not 2 <= len(key) <= 80:
                continue
            seen_on_page.add(key)
            examples.setdefault(key, line)
        counts.update(seen_on_page)

    threshold = max(3, min(8, len(page_texts) // 8))
    return {
        key
        for key, count in counts.items()
        if count >= threshold and not _is_keep_heading(examples.get(key, ""))
    }


def clean_page_texts(page_texts: list[str]) -> list[str]:
    noise_keys = repeated_edge_noise_keys(page_texts)
    cleaned_pages: list[str] = []

    for text in page_texts:
        cleaned_lines: list[str] = []
        for line in text.splitlines():
            normalized = _normalize_line(line)
            if not normalized:
                continue
            if _is_page_number_noise(normalized):
                continue
            if not _is_keep_heading(normalized) and _noise_key(normalized) in noise_keys:
                continue
            cleaned_lines.append(normalized)
        cleaned_pages.append("\n".join(cleaned_lines).strip())

    return cleaned_pages
