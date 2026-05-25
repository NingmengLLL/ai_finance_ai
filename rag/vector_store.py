from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ingestion.pipeline import read_chunks, rebuild_index
from ingestion.schema import Chunk
from model.factory import SimpleEmbeddings, embed_model
from rag.query_filters import matches_metadata_filter, to_chroma_filter
from utils.config_handler import chroma_cof, rag_cof
from utils.logger_handler import logger, log_stage, log_stage_done, log_stage_start, safe_preview
from utils.path_tool import get_abs_path
from utils.progress import progress_bar


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return numerator / denom if denom else 0.0


def _metadata_to_scalar(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    scalar: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            scalar[key] = value
        else:
            scalar[key] = json.dumps(value, ensure_ascii=False)
    return scalar


def _chunk_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    metadata = _metadata_to_scalar(chunk.metadata)
    metadata.update(
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "source_file": chunk.source_file,
            "page_start": chunk.page_start or 0,
            "page_end": chunk.page_end or chunk.page_start or 0,
            "section_path": chunk.section_path or "",
            "doc_type": chunk.doc_type or "unknown",
        }
    )
    return metadata


def _batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def _tracked_index_metadata(chunk: Chunk) -> dict[str, str]:
    metadata = _chunk_metadata(chunk)
    keys = (
        "source_file",
        "source_file_hash",
        "registry_metadata_hash",
        "chunk_cache_signature",
        "doc_type",
        "page_start",
        "page_end",
        "section_path",
    )
    return {key: str(metadata.get(key, "")) for key in keys}


def _metadata_needs_update(existing: dict[str, Any] | None, chunk: Chunk) -> bool:
    if not existing:
        return True
    expected = _tracked_index_metadata(chunk)
    return any(str(existing.get(key, "")) != value for key, value in expected.items())


def chunk_to_document(chunk: Chunk):
    from langchain_core.documents import Document

    return Document(page_content=chunk.text, metadata=_chunk_metadata(chunk))


def document_to_chunk(document) -> Chunk:
    metadata = dict(document.metadata or {})
    return Chunk(
        chunk_id=str(metadata.get("chunk_id", "")),
        doc_id=str(metadata.get("doc_id", "")),
        text=document.page_content,
        source_file=str(metadata.get("source_file", metadata.get("source", ""))),
        page_start=int(metadata.get("page_start", 0)) or None,
        page_end=int(metadata.get("page_end", 0)) or None,
        section_path=str(metadata.get("section_path", "")),
        doc_type=str(metadata.get("doc_type", "unknown")),
        metadata=metadata,
    )


class InMemoryVectorStore:
    """Fallback vector store used only if Chroma is unavailable."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embedding_model = embed_model
        self.embedding_error: str | None = None
        self.embeddings = self._embed_documents([chunk.text for chunk in chunks]) if chunks else []

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.embedding_model.embed_documents(texts)
        except Exception as exc:
            self.embedding_error = str(exc)
            self.embedding_model = SimpleEmbeddings()
            return self.embedding_model.embed_documents(texts)

    def _embed_query(self, query: str) -> list[float]:
        try:
            return self.embedding_model.embed_query(query)
        except Exception as exc:
            self.embedding_error = str(exc)
            self.embedding_model = SimpleEmbeddings()
            self.embeddings = self.embedding_model.embed_documents([chunk.text for chunk in self.chunks])
            return self.embedding_model.embed_query(query)

    def search(
        self,
        query: str,
        top_k: int = 8,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        chunks = [chunk for chunk in self.chunks if matches_metadata_filter(chunk.metadata, metadata_filter)]
        if not chunks:
            return []
        query_embedding = self._embed_query(query)
        indexed = [
            (chunk, embedding)
            for chunk, embedding in zip(self.chunks, self.embeddings)
            if matches_metadata_filter(chunk.metadata, metadata_filter)
        ]
        scored = [
            (chunk, cosine_similarity(query_embedding, embedding))
            for chunk, embedding in indexed
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(chunk, score) for chunk, score in scored[:top_k] if score > 0]


class VectorStoreService:
    """Chroma-backed persistent vector store for financial chunks."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or read_chunks()
        self.collection_name = chroma_cof.get("collection_name", "financial_agent")
        self.persist_directory = get_abs_path(chroma_cof.get("persist_directory", "data/indexes/chroma_db"))
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.using_chroma = True
        self.fallback_error: str | None = None
        with log_stage(
            "vector_store.init",
            collection=self.collection_name,
            persist_directory=safe_preview(self.persist_directory),
            chunks=len(self.chunks),
        ) as stage:
            self.vector_store = self._init_chroma()
            stage.add_done_fields(using_chroma=self.using_chroma, fallback_error=self.fallback_error)

    def _init_chroma(self):
        try:
            from langchain_chroma import Chroma

            return Chroma(
                collection_name=self.collection_name,
                embedding_function=embed_model,
                persist_directory=self.persist_directory,
            )
        except Exception as exc:
            self.using_chroma = False
            self.fallback_error = str(exc)
            logger.warning(f"Chroma unavailable, falling back to in-memory vectors: {exc}")
            return InMemoryVectorStore(self.chunks)

    def collection_count(self) -> int:
        if not self.using_chroma:
            return len(getattr(self.vector_store, "chunks", []))
        try:
            return int(self.vector_store._collection.count())
        except Exception:
            return 0

    def clear_collection(self) -> None:
        if not self.using_chroma:
            self.vector_store = InMemoryVectorStore([])
            return
        try:
            self.vector_store.delete_collection()
        except Exception:
            pass
        self.vector_store = self._init_chroma()

    def build_from_chunks(
        self,
        chunks: list[Chunk] | None = None,
        force: bool = True,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> int:
        self.chunks = chunks or read_chunks()
        batch_size = min(max(int(batch_size), 1), 10)
        started_at = log_stage_start(
            "build_chroma",
            chunks=len(self.chunks),
            force=force,
            batch_size=batch_size,
            collection=self.collection_name,
        )
        if force:
            self.clear_collection()
        elif self.collection_count() > 0:
            count = self.collection_count()
            log_stage_done("build_chroma", started_at, skipped=True, existing_count=count)
            return count

        if not self.chunks:
            log_stage_done("build_chroma", started_at, count=0)
            return 0

        if not self.using_chroma:
            self.vector_store = InMemoryVectorStore(self.chunks)
            log_stage_done("build_chroma", started_at, using_chroma=False, count=len(self.chunks))
            return len(self.chunks)

        documents = [chunk_to_document(chunk) for chunk in self.chunks]
        ids = [chunk.chunk_id for chunk in self.chunks]
        batch_starts = range(0, len(documents), batch_size)
        batch_progress = progress_bar(
            batch_starts,
            desc="Build Chroma",
            unit="batch",
            total=math.ceil(len(documents) / batch_size),
            enabled=show_progress,
        )
        for start in batch_progress:
            end = start + batch_size
            self.vector_store.add_documents(documents[start:end], ids=ids[start:end])
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()
        logger.info(f"Chroma index built: collection={self.collection_name}, count={len(documents)}")
        log_stage_done("build_chroma", started_at, using_chroma=True, count=len(documents))
        return len(documents)

    def sync_from_chunks(
        self,
        chunks: list[Chunk] | None = None,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> int:
        self.chunks = chunks or read_chunks()
        batch_size = min(max(int(batch_size), 1), 10)
        started_at = log_stage_start(
            "sync_chroma",
            chunks=len(self.chunks),
            batch_size=batch_size,
            collection=self.collection_name,
        )
        if not self.chunks:
            self.clear_collection()
            log_stage_done("sync_chroma", started_at, count=0, cleared=True)
            return 0

        if not self.using_chroma:
            self.vector_store = InMemoryVectorStore(self.chunks)
            log_stage_done("sync_chroma", started_at, using_chroma=False, count=len(self.chunks))
            return len(self.chunks)

        target_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        existing = self.vector_store._collection.get(include=["metadatas"])
        existing_ids = set(existing.get("ids", []))
        existing_metadata = dict(zip(existing.get("ids", []), existing.get("metadatas", [])))
        target_ids = set(target_by_id)

        stale_ids = sorted(existing_ids - target_ids)
        missing_ids = sorted(target_ids - existing_ids)
        changed_ids = sorted(
            chunk_id
            for chunk_id in target_ids & existing_ids
            if _metadata_needs_update(existing_metadata.get(chunk_id), target_by_id[chunk_id])
        )

        delete_ids = stale_ids + changed_ids
        delete_batches = _batches(delete_ids, batch_size)
        delete_progress = progress_bar(
            delete_batches,
            desc="Delete stale Chroma chunks",
            unit="batch",
            total=len(delete_batches),
            enabled=show_progress and bool(delete_batches),
        )
        for batch in delete_progress:
            self.vector_store._collection.delete(ids=batch)

        add_ids = missing_ids + changed_ids
        add_chunks = [target_by_id[chunk_id] for chunk_id in add_ids]
        documents = [chunk_to_document(chunk) for chunk in add_chunks]
        batch_starts = range(0, len(documents), batch_size)
        add_progress = progress_bar(
            batch_starts,
            desc="Sync Chroma",
            unit="batch",
            total=math.ceil(len(documents) / batch_size) if documents else 0,
            enabled=show_progress and bool(documents),
        )
        for start in add_progress:
            end = start + batch_size
            self.vector_store.add_documents(documents[start:end], ids=add_ids[start:end])
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()

        logger.info(
            "Chroma index synced: "
            f"collection={self.collection_name}, total={len(target_by_id)}, "
            f"added={len(missing_ids)}, updated={len(changed_ids)}, "
            f"deleted={len(stale_ids)}, unchanged={len(target_ids & existing_ids) - len(changed_ids)}"
        )
        log_stage_done(
            "sync_chroma",
            started_at,
            using_chroma=True,
            total=len(target_by_id),
            added=len(missing_ids),
            updated=len(changed_ids),
            deleted=len(stale_ids),
            unchanged=len(target_ids & existing_ids) - len(changed_ids),
        )
        return len(target_by_id)

    def ensure_index(self) -> None:
        if self.collection_count() == 0 and self.chunks:
            self.build_from_chunks(self.chunks, force=False)

    def get_retriever(self):
        self.ensure_index()
        if self.using_chroma:
            return self.vector_store.as_retriever(search_kwargs={"k": int(rag_cof.get("retriever_k", 8))})
        return self

    def invoke(self, query: str) -> list[Any]:
        return [chunk for chunk, _ in self.search(query, top_k=int(rag_cof.get("retriever_k", 8)))]

    def search(
        self,
        query: str,
        top_k: int = 8,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        with log_stage("vector_store.search", query=safe_preview(query), top_k=top_k) as stage:
            self.ensure_index()
            if not self.using_chroma:
                results = self.vector_store.search(query, top_k, metadata_filter=metadata_filter)
                stage.add_done_fields(using_chroma=False, hits=len(results), metadata_filter=metadata_filter or None)
                return results

            docs_and_scores = self.vector_store.similarity_search_with_score(
                query,
                k=top_k,
                filter=to_chroma_filter(metadata_filter),
            )
            results: list[tuple[Chunk, float]] = []
            for document, distance in docs_and_scores:
                score = 1.0 / (1.0 + float(distance))
                results.append((document_to_chunk(document), score))
            stage.add_done_fields(using_chroma=True, hits=len(results), metadata_filter=metadata_filter or None)
            return results

    def load_doucments(self) -> None:
        rebuild_index(build_vector=False)
        self.chunks = read_chunks()
        self.build_from_chunks(self.chunks, force=True)


if __name__ == "__main__":
    service = VectorStoreService()
    count = service.build_from_chunks(force=True)
    print(f"Chroma collection built: {service.collection_name}, count={count}, path={service.persist_directory}")
    for chunk, score in service.search("示例科技现金流", top_k=3):
        print(f"{score:.4f}", chunk.metadata.get("file_name"), chunk.page_start, chunk.text[:80].replace("\n", " "))
