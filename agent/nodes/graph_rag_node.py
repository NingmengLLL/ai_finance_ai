from __future__ import annotations

from agent.state import FinancialAgentState
from graph_rag.graph_retriever import GraphRetriever
from utils.config_handler import rag_cof
from utils.helpers import as_bool
from utils.logger_handler import log_stage, safe_preview


_graph_retriever: GraphRetriever | None = None


def _get_graph_retriever() -> GraphRetriever:
    global _graph_retriever
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever()
    return _graph_retriever


def graph_rag_node(state: FinancialAgentState) -> dict:
    query = state.get("user_query", "")
    enabled = as_bool(rag_cof.get("enable_graph_rag"), default=True)
    with log_stage("graph_rag", enabled=enabled, query=safe_preview(query)) as stage:
        if not enabled:
            stage.add_done_fields(skipped=True, relations=0)
            return {"graph_relations": []}
        retriever = _get_graph_retriever()
        relations = retriever.retrieve(query)
        stage.add_done_fields(skipped=False, relations=len(relations))
        return {"graph_relations": relations}
