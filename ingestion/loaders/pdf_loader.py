from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ingestion.parsers.page_cleaner import clean_page_texts
from ingestion.schema import SourceDocument


TEXT_SIGNAL_RE = re.compile(r"[A-Za-z\u3400-\u9fff]")
SECTION_HEADING_RE = re.compile(
    r"^(目录|目錄|公司资料|公司資料|财务概要|財務概要|主席报告|主席報告|"
    r"管理层讨论[及与]分析|管理層討論及分析|董事会报告|董事會報告|企业管治报告|企業管治報告|"
    r"独立核数师报告|獨立核數師報告|综合收益表|綜合收益表|综合全面收益表|綜合全面收益表|"
    r"综合财务状况表|綜合財務狀況表|综合权益变动表|綜合權益變動表|"
    r"综合现金流量表|綜合現金流量表|综合财务报表附注|綜合財務報表附註|"
    r"释义|釋義|分部财务资料|分部財務資料|业务回顾|業務回顧|财务回顾|財務回顧)$"
)


def _doc_id_prefix(file_path: Path) -> str:
    return hashlib.md5(str(file_path.resolve()).encode("utf-8")).hexdigest()


def _has_text_signal(docs: list[SourceDocument]) -> bool:
    sample = "\n".join(doc.text for doc in docs[:12])
    return len(TEXT_SIGNAL_RE.findall(sample)) >= 50


def _markdownize_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2 and parts[0] == parts[1]:
            line = parts[0]
        if SECTION_HEADING_RE.match(line):
            lines.append(f"## {line}")
        else:
            lines.append(line)
    return "\n".join(lines)


def _clean_cell(value) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    return text.replace("|", "\\|")


def _is_useful_table(rows: list[list[str]]) -> bool:
    if len(rows) < 2:
        return False
    max_cols = max((len(row) for row in rows), default=0)
    if max_cols < 2:
        return False
    populated = sum(1 for row in rows for cell in row if cell.strip())
    return populated >= 4


def _table_to_markdown(raw_table, table_id: str) -> str:
    rows = [[_clean_cell(cell) for cell in row] for row in raw_table if any(_clean_cell(cell) for cell in row)]
    if not _is_useful_table(rows):
        return ""

    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    header = [
        cell if cell else f"col_{idx + 1}"
        for idx, cell in enumerate(normalized_rows[0])
    ]
    body = normalized_rows[1:]
    lines = [
        f"### Table {table_id}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines).strip()


def _load_with_pdfplumber(file_path: Path, doc_id_prefix: str) -> list[SourceDocument]:
    import pdfplumber

    raw_pages: list[dict] = []
    docs: list[SourceDocument] = []
    with pdfplumber.open(str(file_path)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            raw_tables = page.extract_tables() or []
            tables = [
                markdown
                for table_index, raw_table in enumerate(raw_tables, start=1)
                if (markdown := _table_to_markdown(raw_table, f"p{idx}_t{table_index}"))
            ]
            raw_pages.append({"page_number": idx, "text": text, "tables": tables})

    cleaned_texts = clean_page_texts([page["text"] for page in raw_pages])

    for page, cleaned_text in zip(raw_pages, cleaned_texts):
        idx = page["page_number"]
        tables: list[str] = page["tables"]
        page_markdown = _markdownize_text(cleaned_text)
        if page_markdown:
            docs.append(
                SourceDocument(
                    doc_id=f"{doc_id_prefix}-p{idx}-text",
                    source_file=str(file_path),
                    text=page_markdown,
                    doc_type="pdf",
                    page_number=idx,
                    metadata={
                        "file_name": file_path.name,
                        "pdf_parser": "pdfplumber",
                        "parser_profile": "pdfplumber_markdown",
                        "block_type": "mixed" if tables else "text",
                        "table_count": len(tables),
                        "cleaning_status": "rules_applied",
                    },
                )
            )
        for table_index, table_markdown in enumerate(tables, start=1):
            table_id = f"p{idx}_t{table_index}"
            docs.append(
                SourceDocument(
                    doc_id=f"{doc_id_prefix}-{table_id}",
                    source_file=str(file_path),
                    text=table_markdown,
                    doc_type="pdf",
                    page_number=idx,
                    metadata={
                        "file_name": file_path.name,
                        "pdf_parser": "pdfplumber",
                        "parser_profile": "pdfplumber_markdown",
                        "block_type": "table",
                        "table_id": table_id,
                        "cleaning_status": "rules_applied",
                    },
                )
            )
    return docs


def _load_with_pypdf(file_path: Path, doc_id_prefix: str) -> list[SourceDocument]:
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    raw_pages = [page.extract_text() or "" for page in reader.pages]
    cleaned_texts = clean_page_texts(raw_pages)
    docs: list[SourceDocument] = []
    for idx, text in enumerate(cleaned_texts, start=1):
        page_markdown = _markdownize_text(text)
        if page_markdown:
            docs.append(
                SourceDocument(
                    doc_id=f"{doc_id_prefix}-p{idx}-text",
                    source_file=str(file_path),
                    text=page_markdown,
                    doc_type="pdf",
                    page_number=idx,
                    metadata={
                        "file_name": file_path.name,
                        "pdf_parser": "pypdf",
                        "parser_profile": "pypdf_markdown",
                        "block_type": "text",
                        "cleaning_status": "rules_applied",
                    },
                )
            )
    return docs


def load_pdf_file(path: str) -> list[SourceDocument]:
    """Load page-level PDF text, preferring pdfplumber for Chinese financial PDFs."""

    file_path = Path(path)
    doc_id_prefix = _doc_id_prefix(file_path)
    last_error = ""
    low_signal_docs: list[SourceDocument] = []

    for loader in (_load_with_pdfplumber, _load_with_pypdf):
        try:
            docs = loader(file_path, doc_id_prefix)
        except Exception as exc:
            last_error = f"{loader.__name__}: {exc}"
            continue
        if _has_text_signal(docs):
            return docs
        low_signal_docs = docs

    if low_signal_docs:
        for doc in low_signal_docs:
            doc.metadata["parse_status"] = "low_text_signal"
        return low_signal_docs

    return [
        SourceDocument(
            doc_id=doc_id_prefix,
            source_file=str(file_path),
            text=f"[PDF placeholder] Unable to parse {file_path.name}. Install pdfplumber or pypdf and rebuild ingestion.",
            doc_type="pdf_unparsed",
            page_number=1,
            metadata={
                "file_name": file_path.name,
                "parse_status": "missing_pdf_dependency",
                "parse_error": last_error,
            },
        )
    ]
