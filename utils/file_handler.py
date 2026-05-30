from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

from utils.logger_handler import logger


@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def get_file_md5_hex(file_path: str) -> str | None:
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return None
    if not os.path.isfile(file_path):
        logger.error(f"Path is not a file: {file_path}")
        return None

    md5_hash = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                md5_hash.update(byte_block)
        return md5_hash.hexdigest()
    except Exception as exc:
        logger.error(f"Failed to calculate md5 for {file_path}: {exc}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str, ...]) -> tuple[str, ...]:
    if not os.path.isdir(path):
        logger.error(f"{path} is not a directory")
        return tuple()
    normalized = tuple(t if t.startswith(".") else f".{t}" for t in allowed_types)
    files = [os.path.join(path, name) for name in os.listdir(path) if name.endswith(normalized)]
    return tuple(files)


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    try:
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(filepath, passwd).load()
    except Exception:
        from ingestion.loaders.pdf_loader import load_pdf_file

        return [
            Document(page_content=doc.text, metadata={"source": doc.source_file, "page": doc.page_number})
            for doc in load_pdf_file(filepath)
        ]


def txt_loader(filepath: str) -> list[Document]:
    try:
        from langchain_community.document_loaders import TextLoader

        return TextLoader(filepath, encoding="utf-8").load()
    except Exception:
        text = open(filepath, "r", encoding="utf-8", errors="ignore").read()
        return [Document(page_content=text, metadata={"source": filepath, "page": 1})]
