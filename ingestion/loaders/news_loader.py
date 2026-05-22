from __future__ import annotations

from datetime import datetime

from ingestion.schema import SourceDocument


def news_item_to_document(item: dict) -> SourceDocument:
    title = item.get("title", "")
    body = item.get("body", item.get("content", ""))
    source = item.get("source", "external_news_api")
    published_at = item.get("published_at", datetime.utcnow().isoformat())
    doc_id = item.get("id", f"{source}:{published_at}:{title}")
    text = f"标题：{title}\n发布时间：{published_at}\n来源：{source}\n正文：{body}"
    return SourceDocument(
        doc_id=doc_id,
        source_file=source,
        text=text,
        doc_type="financial_news",
        page_number=None,
        metadata={"published_at": published_at, "source": source, "url": item.get("url")},
    )
