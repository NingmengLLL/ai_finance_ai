from __future__ import annotations

from typing import Any, TypedDict


class FinancialAgentState(TypedDict, total=False):
    messages: list[dict[str, str]]
    user_id: str
    user_profile: dict[str, Any]    # 画像回灌：watchlist/preferred_metrics/risk_preference/language_style
    user_query: str
    intent: str
    entities: dict[str, list[str]]
    query_plan: dict[str, Any]
    sub_queries: list[str]
    retrieved_docs: list[dict[str, Any]]
    evidence_cards: list[dict[str, Any]]
    graph_relations: list[dict[str, Any]]
    calculations: list[dict[str, Any]]
    draft_answer: str
    critique_result: dict[str, Any]
    citation_errors: list[str]
    final_answer: str
    needs_web_search: bool
    needs_more_evidence: bool
    reflection_round: int
    reflection_history: list[dict[str, Any]]
    memory_snapshot: dict[str, Any]


def merge_state(state: FinancialAgentState, update: dict[str, Any]) -> FinancialAgentState:
    merged = dict(state)
    merged.update(update)
    return merged