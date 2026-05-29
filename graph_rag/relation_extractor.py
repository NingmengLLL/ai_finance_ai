from __future__ import annotations

from ingestion.schema import Chunk
from graph_rag.entity_extractor import extract_entities
from knowledge import get_relation_templates


def extract_relations(chunks: list[Chunk]) -> list[dict]:
    """关系提取 — 关系模板从 knowledge YAML 读取。
    动态模板(head/tail用pattern匹配) + 固定模板(head/tail精确匹配) 都支持。"""
    relations: list[dict] = []
    templates = get_relation_templates()

    for chunk in chunks:
        entities = extract_entities(chunk.text)
        companies = entities.get("companies", [])
        metrics = entities.get("metrics", [])

        # ── 动态模板：公司→披露指标→指标 ──
        for company in companies:
            for metric in metrics:
                relations.append(
                    {
                        "head": company,
                        "relation": "披露指标",
                        "tail": metric,
                        "source_file": chunk.metadata.get("file_name", chunk.source_file),
                        "page_number": chunk.page_start,
                        "chunk_id": chunk.chunk_id,
                    }
                )

        # ── 固定模板：从 knowledge YAML 读取 ──
        for template in templates:
            head = template.get("head", "")
            relation = template.get("relation", "")
            tail = template.get("tail", "")
            # 只有 head 和 tail 都有值（非pattern），才做精确匹配
            if head and tail and not template.get("head_pattern") and not template.get("tail_pattern"):
                if head in chunk.text and tail in chunk.text:
                    relations.append(
                        {
                            "head": head,
                            "relation": relation,
                            "tail": tail,
                            "source_file": chunk.metadata.get("file_name", chunk.source_file),
                            "page_number": chunk.page_start,
                            "chunk_id": chunk.chunk_id,
                        }
                    )

    return relations