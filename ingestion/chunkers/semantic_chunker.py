from __future__ import annotations

import hashlib
import re

from ingestion.schema import Chunk, SourceDocument


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_.%+-]+|[\u4e00-\u9fff]", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SemanticChunker:
    def __init__(self, max_chars: int = 1200, min_similarity: float = 0.12):
        self.max_chars = max_chars
        self.min_similarity = min_similarity

    def split(self, docs: list[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
            if not paragraphs:
                continue
            bucket: list[str] = []
            bucket_tokens: set[str] = set()
            for paragraph in paragraphs:
                paragraph_tokens = _tokens(paragraph)
                next_text = "\n\n".join(bucket + [paragraph])
                similarity = _jaccard(bucket_tokens, paragraph_tokens) if bucket else 1.0
                should_flush = bucket and (len(next_text) > self.max_chars or similarity < self.min_similarity)
                if should_flush:
                    chunks.append(self._make_chunk(doc, bucket, len(chunks)))
                    bucket = [paragraph]
                    bucket_tokens = paragraph_tokens
                else:
                    bucket.append(paragraph)
                    bucket_tokens |= paragraph_tokens
            if bucket:
                chunks.append(self._make_chunk(doc, bucket, len(chunks)))
        return chunks

    def _make_chunk(self, doc: SourceDocument, paragraphs: list[str], index: int) -> Chunk:
        text = "\n\n".join(paragraphs)
        digest = hashlib.md5(f"{doc.doc_id}:semantic:{index}:{text[:80]}".encode("utf-8")).hexdigest()
        return Chunk(
            chunk_id=digest,
            doc_id=doc.doc_id,
            text=text,
            source_file=doc.source_file,
            page_start=doc.page_number,
            page_end=doc.page_number,
            section_path=doc.metadata.get("section_path", ""),
            doc_type=doc.doc_type,
            metadata=doc.metadata.copy(),
        )
