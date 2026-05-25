from __future__ import annotations

import math
import os
from typing import Any

# 强制设置 Hugging Face 国内镜像站，解决网络连接问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from rag.bm25_store import tokenize
from utils.config_handler import model_cof, rag_cof
from utils.helpers import as_bool
from utils.logger_handler import logger


def _score_weight(name: str, default: float) -> float:
    try:
        return float(rag_cof.get(name, default))
    except (TypeError, ValueError):
        return default


def _int_config(name: str, default: int) -> int:
    try:
        return int(rag_cof.get(name, default))
    except (TypeError, ValueError):
        return default


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _as_score(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        value = max(value)
    if hasattr(value, "tolist"):
        return _as_score(value.tolist())
    return float(value)


class LocalReranker:
    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        cross_encoder: Any | None = None,
    ):
        self.provider = str(provider or model_cof.get("rerank_provider", "local_cross_encoder")).lower()
        self.model_name = str(model_name or model_cof.get("rerank_model_name", "BAAI/bge-reranker-base"))
        self.cross_encoder = cross_encoder
        self._model_load_attempted = cross_encoder is not None
        self._model_error: str | None = None

    def score(self, query: str, text: str, metadata: dict | None = None) -> float:
        metadata = metadata or {}
        query_terms = set(tokenize(query))
        text_terms = set(tokenize(text))
        if not query_terms or not text_terms:
            lexical = 0.0
        else:
            lexical = len(query_terms & text_terms) / len(query_terms)
        authority = 0.1 if metadata.get("doc_type") in {"annual_report", "research_report", "financial_table"} else 0.0
        freshness = 0.05 if "2024" in text or "2025" in text else 0.0
        return min(1.0, lexical + authority + freshness)

    def rerank(self, query: str, candidates: list[dict], top_n: int = 6) -> list[dict]:
        # 只要 provider 包含 cross_encoder，就会尝试走深度学习重排
        if self.provider in {"local_cross_encoder", "cross_encoder"}:
            try:
                return self._cross_encoder_rerank(query, candidates, top_n)
            except Exception as exc:
                self._model_error = str(exc)
                logger.warning(f"Cross encoder rerank unavailable, falling back to local reranker: {exc}")
        return self._fallback_rerank(query, candidates, top_n)

    def _load_cross_encoder(self):
        if self.cross_encoder is not None:
            return self.cross_encoder
        if self._model_load_attempted:
            if self._model_error:
                raise RuntimeError(self._model_error)
            return self.cross_encoder

        self._model_load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            max_length = _int_config("rerank_max_length", 512)

            local_files_only = as_bool(rag_cof.get("rerank_local_files_only"), default=False)
            
            logger.info(f"正在加载/下载 Cross Encoder 模型: {self.model_name}...")
            self.cross_encoder = CrossEncoder(
                self.model_name,
                max_length=max_length,
                local_files_only=local_files_only,
            )
            logger.info("Cross Encoder 模型加载成功！")
            return self.cross_encoder
        except Exception as exc:
            self._model_error = str(exc)
            raise

    def _cross_encoder_rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        if not candidates:
            return []
        model = self._load_cross_encoder()
        pairs = [(query, candidate["chunk"].text) for candidate in candidates]
        batch_size = _int_config("rerank_batch_size", 8)
        try:
            raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        except TypeError:
            raw_scores = model.predict(pairs)
        try:
            raw_scores = list(raw_scores)
        except TypeError:
            raw_scores = [raw_scores]
        if len(raw_scores) != len(candidates):
            raise RuntimeError(
                f"Cross encoder returned {len(raw_scores)} scores for {len(candidates)} candidates."
            )

        for candidate, raw_score in zip(candidates, raw_scores):
            rerank_score = _sigmoid(_as_score(raw_score))
            candidate["retrieval_score"] = max(candidate.get("dense_score", 0.0), candidate.get("bm25_score", 0.0))
            candidate["rerank_score"] = rerank_score
            candidate["final_score"] = rerank_score
            candidate["rerank_provider"] = "cross_encoder"
            candidate["rerank_model"] = self.model_name
        candidates.sort(key=lambda item: item["final_score"], reverse=True)
        return candidates[:top_n]

    def _fallback_rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        dense_weight = _score_weight("dense_weight", 0.35)
        bm25_weight = _score_weight("bm25_weight", 0.35)
        rerank_weight = _score_weight("rerank_weight", 0.30)
        retrieval_weight = max(0.0, dense_weight + bm25_weight)
        total_weight = retrieval_weight + max(0.0, rerank_weight)
        total_weight = total_weight or 1.0

        for candidate in candidates:
            chunk = candidate["chunk"]
            retrieval_score = max(candidate.get("dense_score", 0.0), candidate.get("bm25_score", 0.0))
            candidate["retrieval_score"] = retrieval_score
            candidate["rerank_score"] = self.score(query, chunk.text, chunk.metadata | {"doc_type": chunk.doc_type})
            candidate["final_score"] = (
                retrieval_weight * retrieval_score + rerank_weight * candidate.get("rerank_score", 0.0)
            ) / total_weight
            candidate["rerank_provider"] = "local_fallback"
            if self._model_error:
                candidate["rerank_error"] = self._model_error
        candidates.sort(key=lambda item: item["final_score"], reverse=True)
        return candidates[:top_n]