from __future__ import annotations

import hashlib
import re

from ingestion.schema import Chunk, SourceDocument


HEADING_RE = re.compile(r"^(第[一二三四五六七八九十\d]+[章节部分].*|[一二三四五六七八九十\d]+[、.．].{2,40}|#{1,4}\s+.+)$")


class StructureAwareChunker:
    def __init__(self, max_chars: int = 1200, overlap_chars: int = 120):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def split(self, docs: list[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            chunks.extend(self._split_one(doc))
        return chunks

    def _split_one(self, doc: SourceDocument) -> list[Chunk]:
        sections: list[tuple[str, str]] = []
        current_heading = ""
        current_lines: list[str] = []
        for raw_line in doc.text.splitlines():
            line = raw_line.strip()
            if not line:
                current_lines.append("")
                continue
            if HEADING_RE.match(line):
                if current_lines:
                    sections.append((current_heading, "\n".join(current_lines).strip()))
                    current_lines = []
                current_heading = line.lstrip("#").strip()
            else:
                current_lines.append(line)
        if current_lines:
            sections.append((current_heading, "\n".join(current_lines).strip()))
        if not sections:
            sections = [("", doc.text)]

        chunks: list[Chunk] = []
        for section_path, section_text in sections:
            chunks.extend(self._window(doc, section_text, section_path))
        return chunks

    def _window(self, doc: SourceDocument, text: str, section_path: str) -> list[Chunk]:
        if not text:
            return []
        windows: list[Chunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(len(text), start + self.max_chars)
            piece = text[start:end].strip()
            if piece:
                digest = hashlib.md5(f"{doc.doc_id}:{section_path}:{index}:{piece[:80]}".encode("utf-8")).hexdigest()
                windows.append(
                    Chunk(
                        chunk_id=digest,
                        doc_id=doc.doc_id,
                        text=piece,
                        source_file=doc.source_file,
                        page_start=doc.page_number,
                        page_end=doc.page_number,
                        section_path=section_path,
                        doc_type=doc.doc_type,
                        metadata=doc.metadata.copy(),
                    )
                )
            if end == len(text):
                break
            start = max(0, end - self.overlap_chars)
            index += 1
        return windows
