from __future__ import annotations

import hashlib
from pathlib import Path

from ingestion.schema import SourceDocument


def load_image_file(path: str) -> list[SourceDocument]:
    file_path = Path(path)
    doc_id = hashlib.md5(str(file_path.resolve()).encode("utf-8")).hexdigest()
    return [
        SourceDocument(
            doc_id=doc_id,
            source_file=str(file_path),
            text=f"[图像占位] {file_path.name} 需要视觉模型生成图表描述后再入库。",
            doc_type="image_or_chart",
            page_number=None,
            metadata={"file_name": file_path.name, "requires_vision_caption": True},
        )
    ]
