from __future__ import annotations

from rag.hybrid_retriever import HybridRetriever


def retrieve_financial_docs(query: str) -> list[dict]:
    return [card.to_dict() for card in HybridRetriever().retrieve_evidence(query)]
