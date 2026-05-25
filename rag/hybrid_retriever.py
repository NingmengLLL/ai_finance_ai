from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ingestion.pipeline import read_chunks
from ingestion.schema import Chunk
from rag.bm25_store import BM25Store
from rag.context_compressor import ContextCompressor
from rag.query_filters import (
    augment_query_for_retrieval,
    infer_metadata_filter,
    normalize_query_for_metadata_filter,
)
from rag.reranker import LocalReranker
from rag.vector_store import VectorStoreService
from utils.config_handler import rag_cof
from utils.logger_handler import log_stage, safe_preview


class RetrievalStep(ABC):
    @abstractmethod
    def run(self, query: str, candidates: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
        pass


class DenseSearchStep(RetrievalStep):
    """Dense通道：使用语义增强query（augment_query_for_retrieval），
    让embedding模型捕捉到更丰富的语义信息。"""

    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store

    def run(self, query: str, candidates: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
        top_k = context.get("top_k", 8)
        metadata_filter = context.get("metadata_filter")
        # Dense通道用语义增强query
        dense_query = context.get("dense_query", query)
        hits = self.vector_store.search(dense_query, top_k=top_k, metadata_filter=metadata_filter)
        context["dense_hits"] = hits
        for chunk, score in hits:
            entry = context["merged"].setdefault(chunk.chunk_id, {"chunk": chunk, "dense_score": 0.0, "bm25_score": 0.0})
            entry["dense_score"] = max(entry["dense_score"], score)
        return list(context["merged"].values())


class BM25SearchStep(RetrievalStep):
    """BM25通道：使用精准关键词query（normalize_query_for_metadata_filter），
    保留实体名和关键词，不做语义扩展，保证词汇匹配精度。"""

    def __init__(self, bm25_store: BM25Store):
        self.bm25_store = bm25_store

    def run(self, query: str, candidates: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
        top_k = context.get("top_k", 8)
        metadata_filter = context.get("metadata_filter")
        # BM25通道用精准关键词query
        bm25_query = context.get("bm25_query", query)
        hits = self.bm25_store.search(bm25_query, top_k=top_k, metadata_filter=metadata_filter)
        context["bm25_hits"] = hits
        max_bm25 = max([score for _, score in hits], default=0.0) or 1.0
        for chunk, score in hits:
            entry = context["merged"].setdefault(chunk.chunk_id, {"chunk": chunk, "dense_score": 0.0, "bm25_score": 0.0})
            entry["bm25_score"] = max(entry["bm25_score"], score / max_bm25)
        return list(context["merged"].values())


class RerankStep(RetrievalStep):
    def __init__(self, reranker: LocalReranker):
        self.reranker = reranker

    def run(self, query: str, candidates: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
        top_n = context.get("top_n", 6)
        # Rerank使用原始query（用户意图的忠实表达）
        rerank_query = context.get("original_query", query)
        reranked = self.reranker.rerank(rerank_query, candidates, top_n=top_n)
        context["reranked"] = reranked
        return reranked


class CompressStep(RetrievalStep):
    def __init__(self, compressor: ContextCompressor):
        self.compressor = compressor

    def run(self, query: str, candidates: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
        # Compress使用原始query做句子评分
        original_query = context.get("original_query", query)
        cards = self.compressor.compress(original_query, candidates)
        context["evidence_cards"] = cards
        return candidates


class RetrievalPipeline:
    def __init__(self, steps: list[RetrievalStep] | None = None):
        self.steps = steps or []

    def add_step(self, step: RetrievalStep) -> "RetrievalPipeline":
        self.steps.append(step)
        return self

    def execute(self, query: str, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        context = context or {}
        context.setdefault("merged", {})
        candidates: list[dict[str, Any]] = []
        for step in self.steps:
            candidates = step.run(query, candidates, context)
        return candidates


class HybridRetriever:
    """Backward-compatible facade over RetrievalPipeline.
    
    核心改造：双通道检索策略
    - Dense通道使用 augment_query_for_retrieval（语义丰富，提升向量匹配召回）
    - BM25通道使用 normalize_query_for_metadata_filter（精准关键词，保证词汇匹配）
    - Rerank/Compress使用原始query（忠实用户意图）
    """

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or read_chunks()
        self.vector_store = VectorStoreService(self.chunks)
        self.bm25_store = BM25Store(self.chunks)
        self.reranker = LocalReranker()
        self.compressor = ContextCompressor()
        self.pipeline = self._build_default_pipeline()

    def _build_default_pipeline(self) -> RetrievalPipeline:
        return RetrievalPipeline([
            DenseSearchStep(self.vector_store),
            BM25SearchStep(self.bm25_store),
            RerankStep(self.reranker),
            CompressStep(self.compressor),
        ])

    def _prepare_context(self, query: str, top_k: int, top_n: int) -> dict[str, Any]:
        """统一构造检索上下文，双通道query分别构造。"""
        metadata_filter = infer_metadata_filter(query)
        # BM25通道：减法式改写，保留精准关键词
        bm25_query = normalize_query_for_metadata_filter(query, metadata_filter)
        # Dense通道：加法式增强，语义丰富
        dense_query = augment_query_for_retrieval(query, metadata_filter)

        return {
            "top_k": top_k,
            "top_n": top_n,
            "metadata_filter": metadata_filter,
            "original_query": query,          # 原始query，给Rerank/Compress用
            "bm25_query": bm25_query,          # BM25精准关键词query
            "dense_query": dense_query,         # Dense语义增强query
            "merged": {},
        }

    def retrieve(self, query: str, top_k: int | None = None, top_n: int | None = None) -> list[dict]:
        with log_stage("rag.retrieve", query=safe_preview(query)) as stage:
            top_k = top_k or int(rag_cof.get("retriever_k", 8))
            top_n = top_n or int(rag_cof.get("rerank_top_n", 6))
            context = self._prepare_context(query, top_k, top_n)
            self.pipeline.execute(query, context)

            reranked = context.get("reranked", list(context.get("merged", {}).values()))
            stage.add_done_fields(
                top_k=top_k,
                top_n=top_n,
                metadata_filter=context.get("metadata_filter") or None,
                dense_query=safe_preview(context.get("dense_query", "")),
                bm25_query=safe_preview(context.get("bm25_query", "")),
                dense_hits=len(context.get("dense_hits", [])),
                bm25_hits=len(context.get("bm25_hits", [])),
                merged=len(context.get("merged", {})),
                reranked=len(reranked),
            )
            return reranked

    def retrieve_evidence(self, query: str, top_k: int | None = None, top_n: int | None = None):
        top_k = top_k or int(rag_cof.get("retriever_k", 8))
        top_n = top_n or int(rag_cof.get("rerank_top_n", 6))
        context = self._prepare_context(query, top_k, top_n)
        self.pipeline.execute(query, context)
        return context.get("evidence_cards", [])