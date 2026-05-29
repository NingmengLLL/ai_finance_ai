from __future__ import annotations

import re

from ingestion.schema import SourceDocument


DISCLAIMER_PATTERNS = [
    r"免责声明.*",
    r"重要声明.*",
    r"本报告仅供.*",
]


def clean_financial_text(doc: SourceDocument) -> SourceDocument:
    text = doc.text.replace("\x00", " ")
    for pattern in DISCLAIMER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.S)
    text = re.sub(r"\n{3,}", "\n\n", text)
    doc.text = text.strip()
    return doc
