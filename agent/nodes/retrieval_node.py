from __future__ import annotations

from agent.state import FinancialAgentState
from rag.hybrid_retriever import HybridRetriever
from utils.logger_handler import log_stage, safe_preview


_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def retrieval_node(state: FinancialAgentState) -> dict:
    sub_queries = state.get("sub_queries")
    if not sub_queries:  # 覆盖 None 和 [] 两种情况，但语义更清晰
        sub_queries = [state.get("user_query", "")]
        
    with log_stage("retrieval", sub_queries=len(sub_queries)) as stage:
        retriever = _get_retriever()
        cards = []
        for index, query in enumerate(sub_queries, start=1):
            with log_stage("retrieval.query", index=index, query=safe_preview(query)) as query_stage:
                query_cards = retriever.retrieve_evidence(query)
                cards.extend(query_cards)
                query_stage.add_done_fields(cards=len(query_cards))

        seen = set()
        unique_cards = []
        for card in sorted(cards, key=lambda item: item.score, reverse=True):
            if card.chunk_id not in seen:
                unique_cards.append(card)
                seen.add(card.chunk_id)
        final_cards = unique_cards[:8]
        stage.add_done_fields(raw_cards=len(cards), unique_cards=len(unique_cards), final_cards=len(final_cards))
        return {
            "evidence_cards": [card.to_dict() for card in final_cards],
            "retrieved_docs": [card.metadata for card in final_cards],
        }
