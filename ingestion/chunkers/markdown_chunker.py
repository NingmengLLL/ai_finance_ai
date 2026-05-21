from __future__ import annotations

import hashlib
import re

from ingestion.schema import Chunk, SourceDocument


HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
SEPARATORS = ["\n\n", "\n", "。", "；", ";", "，", ",", " ", ""]


class MarkdownHierarchyChunker:
    def __init__(self, max_chars: int = 1200, overlap_chars: int = 120):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def split(self, docs: list[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            if doc.metadata.get("block_type") == "table":
                chunks.extend(self._split_table_doc(doc))
            else:
                chunks.extend(self._split_markdown_doc(doc))
        return chunks

    def _split_table_doc(self, doc: SourceDocument) -> list[Chunk]:
        table_id = str(doc.metadata.get("table_id", ""))
        section_path = doc.metadata.get("section_path") or table_id or "table"
        pieces = self._split_large_table(doc.text)
        return [self._make_chunk(doc, piece, section_path, idx, "table") for idx, piece in enumerate(pieces)]

    def _split_markdown_doc(self, doc: SourceDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section_path, section_text in self._sections(doc.text):
            for idx, piece in enumerate(self._recursive_split(section_text)):
                block_type = "table" if self._looks_like_table(piece) else str(doc.metadata.get("block_type", "text"))
                chunks.append(self._make_chunk(doc, piece, section_path, idx, block_type))
        return chunks

    def _sections(self, text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        heading_stack: list[str] = []
        current_lines: list[str] = []
        current_path = ""

        def flush() -> None:
            nonlocal current_lines
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append((current_path, section_text))
            current_lines = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                current_lines.append("")
                continue
            match = HEADING_RE.match(line)
            if match:
                flush()
                level = len(match.group(1))
                heading = match.group(2).strip()
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading)
                current_path = " > ".join(heading_stack)
                current_lines = [line]
            else:
                current_lines.append(line)
        flush()

        if sections:
            return sections
        return [("", text.strip())] if text.strip() else []

    def _recursive_split(self, text: str, separator_index: int = 0) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.max_chars:
            return [text]
        if separator_index >= len(SEPARATORS) - 1:
            return self._window(text)

        separator = SEPARATORS[separator_index]
        if separator and separator in text:
            raw_parts = [part.strip() for part in text.split(separator) if part.strip()]
            parts: list[str] = []
            for part in raw_parts:
                if len(part) > self.max_chars:
                    parts.extend(self._recursive_split(part, separator_index + 1))
                else:
                    parts.append(part)
            return self._merge_parts(parts, separator)

        return self._recursive_split(text, separator_index + 1)

    def _merge_parts(self, parts: list[str], separator: str) -> list[str]:
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else f"{current}{separator}{part}"
            if len(candidate) <= self.max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current.strip())
            current = part
        if current:
            chunks.append(current.strip())

        if self.overlap_chars <= 0 or len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for chunk in chunks[1:]:
            prefix = overlapped[-1][-self.overlap_chars :].strip()
            overlapped.append(f"{prefix}\n{chunk}".strip() if prefix else chunk)
        return overlapped

    def _window(self, text: str) -> list[str]:
        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.max_chars)
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end == len(text):
                break
            start = max(0, end - self.overlap_chars)
        return pieces

    def _split_large_table(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text.strip()] if text.strip() else []

        lines = [line for line in text.splitlines() if line.strip()]
        table_lines = [line for line in lines if line.lstrip().startswith("|")]
        prefix_lines = [line for line in lines if not line.lstrip().startswith("|")]
        if len(table_lines) < 4:
            return self._window(text)

        header = table_lines[:2]
        rows = table_lines[2:]
        chunks: list[str] = []
        current = prefix_lines + header
        for row in rows:
            candidate = "\n".join(current + [row])
            if len(candidate) <= self.max_chars:
                current.append(row)
                continue
            chunks.append("\n".join(current).strip())
            current = prefix_lines + header + [row]
        if current:
            chunks.append("\n".join(current).strip())
        return chunks

    def _make_chunk(self, doc: SourceDocument, text: str, section_path: str, index: int, block_type: str) -> Chunk:
        digest = hashlib.md5(f"{doc.doc_id}:{section_path}:{index}:{text[:80]}".encode("utf-8")).hexdigest()
        metadata = doc.metadata.copy()
        metadata["block_type"] = block_type
        if section_path:
            metadata["section_path"] = section_path
        return Chunk(
            chunk_id=digest,
            doc_id=doc.doc_id,
            text=text,
            source_file=doc.source_file,
            page_start=doc.page_number,
            page_end=doc.page_number,
            section_path=section_path,
            doc_type=doc.doc_type,
            metadata=metadata,
        )

    def _looks_like_table(self, text: str) -> bool:
        table_lines = [line for line in text.splitlines() if line.lstrip().startswith("|")]
        return len(table_lines) >= 3
