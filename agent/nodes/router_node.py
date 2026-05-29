from __future__ import annotations

import re

from agent.llm_utils import extract_json_object, invoke_fast_llm, using_real_llm
from agent.state import FinancialAgentState
from knowledge import get_all_aliases_flat, get_intent_keywords
from utils.config_handler import rag_cof
from utils.helpers import as_bool
from utils.logger_handler import log_stage, safe_preview

# ── LLM意图分类+实体抽取（合并为一次调用）──

INTENT_ENTITY_PROMPT = """你是一个金融研报分析系统的意图分类与实体抽取模块。
请对以下用户查询进行分析，输出严格的JSON（不要Markdown，不要额外解释）。

分析要求：
1. intent：从以下4种意图中选择最匹配的一种
   - realtime_financial_search：查询实时数据（股价、公告、最新新闻等）
   - calculation：需要数值计算（增长率、市盈率、同比、CAGR等）
   - graph_reasoning：涉及关系推理（上下游、供应商客户、产业链影响等）
   - financial_analysis：默认的金融分析（基本面、行业分析等）

2. confidence：意图判断的置信度（0.0-1.0）

3. entities：从查询中抽取的实体
   - companies：涉及的公司名（中文+英文都要列出）
   - metrics：涉及的金融指标术语
   - years：涉及的年份
   - doc_type：文档类型推断（annual_report / quarterly_results / research_report / unknown）

用户查询：
{query}

输出JSON格式如下（注意：所有花括号为JSON语法，不是占位符）：
{{"intent": "...", "confidence": 0.0-1.0, "entities": {{\"companies\": [...], \"metrics\": [...], \"years\": [...], \"doc_type\": \"...\"}}}}"""


def _llm_route_and_extract(query: str) -> dict:
    """一次LLM调用同时完成意图分类+实体抽取。"""
    prompt = INTENT_ENTITY_PROMPT.format(query=query)
    raw = invoke_fast_llm(prompt)
    data = extract_json_object(raw)

    # 确保intent是有效值
    valid_intents = {"realtime_financial_search", "calculation", "graph_reasoning", "financial_analysis"}
    intent = data.get("intent", "financial_analysis")
    if intent not in valid_intents:
        intent = "financial_analysis"

    # 确保entities格式正确
    entities = data.get("entities", {})
    if not isinstance(entities, dict):
        entities = {}
    for key in ["companies", "metrics", "years"]:
        if not isinstance(entities.get(key), list):
            entities[key] = []
    if not isinstance(entities.get("doc_type"), str):
        entities["doc_type"] = "unknown"

    confidence = float(data.get("confidence", 0.5))

    return {
        "intent": intent,
        "confidence": confidence,
        "entities": entities,
        "needs_web_search": intent == "realtime_financial_search",
        "llm_used": True,
        "llm_raw": raw,
    }


def _keyword_route_and_extract(query: str) -> dict:
    """关键词fast-path：零延迟兜底方案。
    意图关键词从 knowledge YAML 读取，实体用 aliases + regex 提取。"""
    realtime_kw = get_intent_keywords("realtime_financial_search")
    calc_kw = get_intent_keywords("calculation")
    graph_kw = get_intent_keywords("graph_reasoning")

    if any(keyword in query for keyword in realtime_kw):
        intent = "realtime_financial_search"
    elif any(keyword in query for keyword in calc_kw):
        intent = "calculation"
    elif any(keyword in query for keyword in graph_kw):
        intent = "graph_reasoning"
    else:
        intent = "financial_analysis"

    # ── 公司实体：先从knowledge aliases匹配，再regex兜底 ──
    alias_map = get_all_aliases_flat()  # {alias_lower: company_id}
    matched_companies = sorted({
        company_id for alias, company_id in alias_map.items()
        if alias in query.lower()
    })
    # regex兜底：带后缀的公司名（如"腾讯控股"）
    suffix_companies = sorted(
        set(re.findall(r"[\u4e00-\u9fffA-Za-z]{2,20}(?:科技|股份|银行|证券|集团|公司)", query))
    )
    companies = sorted(set(matched_companies + suffix_companies))

    tickers_regex = sorted(set(re.findall(r"\b\d{6}\.(?:SZ|SH|BJ|HK)\b", query.upper())))

    from knowledge import get_all_metric_synonyms_flat
    metric_terms = get_all_metric_synonyms_flat()
    metrics_regex = sorted({term for term in metric_terms if term in query})

    years_regex = sorted(set(re.findall(r"20\d{2}年?", query)))

    return {
        "intent": intent,
        "confidence": 1.0 if intent != "financial_analysis" else 0.5,
        "entities": {
            "companies": companies,
            "tickers": tickers_regex,
            "metrics": metrics_regex,
            "years": years_regex,
            "doc_type": "unknown",
        },
        "needs_web_search": intent == "realtime_financial_search",
        "llm_used": False,
    }


def router_node(state: FinancialAgentState) -> dict:
    """意图分类+实体抽取节点。
    默认使用LLM（语义理解），关键词作为fast-path fallback。"""
    query = state.get("user_query", "")
    enable_llm_router = as_bool(rag_cof.get("enable_llm_router"), default=True)

    with log_stage("router", query=safe_preview(query), enable_llm=enable_llm_router) as stage:
        if enable_llm_router and using_real_llm():
            try:
                result = _llm_route_and_extract(query)
            except Exception as exc:
                # LLM调用失败 → 降级到关键词fast-path
                result = _keyword_route_and_extract(query)
                result["llm_error"] = str(exc)
        else:
            result = _keyword_route_and_extract(query)

        entities = result.get("entities", {})
        companies = entities.get("companies", []) if isinstance(entities, dict) else []
        metrics = entities.get("metrics", []) if isinstance(entities, dict) else []

        stage.add_done_fields(
            intent=result["intent"],
            confidence=result.get("confidence", 0.5),
            companies=len(companies),
            metrics=len(metrics),
            needs_web_search=result["needs_web_search"],
            llm_used=result.get("llm_used", False),
        )

        return {
            "intent": result["intent"],
            "entities": entities,
            "needs_web_search": result["needs_web_search"],
        }