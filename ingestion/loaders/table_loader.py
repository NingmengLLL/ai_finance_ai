from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from ingestion.schema import SourceDocument
from ingestion.parsers.table_parser import table_to_text


def load_csv_file(path: str) -> list[SourceDocument]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rows = list(csv.DictReader(f))
    text = table_to_text(rows, table_name=file_path.stem)
    doc_id = hashlib.md5(str(file_path.resolve()).encode("utf-8")).hexdigest()
    return [
        SourceDocument(
            doc_id=doc_id,
            source_file=str(file_path),
            text=text,
            doc_type="financial_table",
            page_number=1,
            metadata={"file_name": file_path.name, "row_count": len(rows), "raw_rows": rows[:50]},
        )
    ]
