from __future__ import annotations

from agent.llm_utils import compact_json, extract_json_object, invoke_fast_llm
from agent.state import FinancialAgentState
from knowledge import get_metrics, resolve_metrics_from_query
from utils.config_handler import rag_cof
from utils.helpers import as_bool


def _metric_expansion(query: str) -> list[str]:
    """指标扩展 — 从 knowledge YAML 读取，不再硬编码4条if规则。
    自动匹配query中涉及的指标，返回对应的扩展词列表。"""
    expansions: list[str] = []
    matched = resolve_metrics_from_query(query)
    all_metrics = get_metrics()

    for metric_name in matched:
        info = all_metrics.get(metric_name, {})
        # expansion_terms 优先（如现金流→[经营活动现金流量净额,自由现金流,...]）
        expansion = info.get("expansion_terms", [])
        if expansion:
            expansions.extend(expansion)
        else:
            # fallback：related字段作为扩展
            expansions.extend(info.get("related", []))

    return list(dict.fromkeys(expansions))


def _reflection_context(state: FinancialAgentState) -> str:
    """Build a reflection hint from the previous critique result."""
    critique = state.get("critique_result", {})
    issues = critique.get("issues", [])
    blocking = [i for i in issues if i.startswith("BLOCKING:")]
    if not blocking:
        return ""
    round_num = state.get("reflection_round", 0)
    return (
        f"上一轮审查（第{round_num}轮）发现问题：\n"
        + "\n".join(f"- {i}" for i in blocking)
        + "\n请针对以上问题调整改写策略，重点弥补缺失的证据维度。"
    )


def _fallback_query_plan(query: str, entities: dict, reflection_hint: str = "", profile: dict | None = None, session_summary: str = "") -> dict:
    expansions = _metric_expansion(query)
    # 画像偏好增强：把用户偏好指标的同义词加入expansion
    if profile:
        preferred = profile.get("preferred_metrics", [])
        watchlist = profile.get("watchlist", [])
        for pm in preferred:
            pm_expansion = resolve_metrics_from_query(pm)
            for m in pm_expansion:
                info = get_metrics().get(m, {})
                expansions.extend(info.get("expansion_terms", info.get("related", [])))
        expansions = list(dict.fromkeys(expansions))
        # watchlist中的公司加入entity_text增强
        if watchlist:
            for c in watchlist:
                if c not in (entities.get("companies") or []):
                    (entities.setdefault("companies", [])).append(c)

    # entities dict 可能包含 str/int/list 混合值——统一转str再拼接
    entity_text = " ".join(
        " ".join(str(item) for item in v) if isinstance(v, list) else str(v)
        for v in (entities.values() if entities else [])
    )
    rewritten_parts = [query, entity_text, " ".join(expansions)]
    if reflection_hint:
        rewritten_parts.append(reflection_hint)
    # L2：摘要关键词补充到改写query中
    if session_summary:
        rewritten_parts.append(session_summary[:200])
    rewritten = " ".join(part for part in rewritten_parts if part)
    sub_queries = [query]
    enable_sub_queries = as_bool(rag_cof.get("enable_sub_queries"), default=True)
    if enable_sub_queries:
        if any(keyword in query for keyword in ["和", "影响", "如何", "为什么", "是否"]):
            for metric in expansions[:4]:
                sub_queries.append(f"{entity_text} {metric} {query}".strip())
    enable_hyde = as_bool(rag_cof.get("enable_hyde"), default=True)
    hyde = ""
    if enable_hyde:
        hyde = (
            "假设性金融分析文档：围绕"
            + query
            + "，需要检查收入、利润、毛利率、现金流、估值、"
            "管理层表述、研报预测和风险提示，并交叉验证来源页码。"
        )
        if reflection_hint:
            hyde += f" 特别注意弥补：{reflection_hint}"
        sub_queries.append(hyde)
    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "hyde_document": hyde,
        "entities": entities,
        "required_metrics": expansions,
        "sub_queries": list(dict.fromkeys(sub_queries)),
        "llm_used": False,
    }


def _llm_query_plan(query: str, entities: dict, reflection_hint: str = "", profile: dict | None = None, session_summary: str = "") -> dict:
    reflection_section = ""
    if reflection_hint:
        reflection_section = f"""
        反思上下文：
        {reflection_hint}

        请针对审查反馈调整改写策略，重点弥补上一轮缺失的证据维度。
        """

    # ── 画像偏好注入 ──
    profile_section = ""
    if profile:
        preferred_metrics = profile.get("preferred_metrics", [])
        watchlist = profile.get("watchlist", [])
        if watchlist or preferred_metrics:
            profile_section = f"""
        用户偏好画像：
        - 关注公司：{watchlist}
        - 常查指标：{preferred_metrics}
        - 风险偏好：{profile.get('risk_preference', 'neutral')}
        请优先围绕以上偏好进行改写和指标扩展。
        """

    # ── L2：会话摘要注入 ──
    summary_section = ""
    if session_summary:
        summary_section = f"""
        会话历史摘要（改写时参考上下文）：
        {session_summary}
        """

    prompt = f"""
    你是金融研报 RAG 系统的查询规划节点。请把用户问题改写成适合混合检索和多跳推理的结构化查询。

    要求：
    1. 输出严格 JSON，不要 Markdown，不要额外解释。
    2. rewritten_query 要包含金融专业术语、时间范围、公司/股票代码、指标。
    3. sub_queries 用于多路召回，最多 5 条，必须具体、可检索。
    4. hyde_document 是一段假设性金融分析文档，仅用于向量检索，不得当作最终事实。
    5. required_metrics 只列与问题直接相关的指标。

    用户问题：
    {query}

    已抽取实体：
    {compact_json(entities)}

    {reflection_section}
    {profile_section}
    {summary_section}
    输出 JSON schema：
    {{
    "rewritten_query": "...",
    "sub_queries": ["..."],
    "hyde_document": "...",
    "required_metrics": ["..."],
    "intent_hint": "financial_analysis|calculation|graph_reasoning|realtime_financial_search"
    }}
    """
    raw = invoke_fast_llm(prompt)
    data = extract_json_object(raw)
    sub_queries = data.get("sub_queries") or [query]
    if not isinstance(sub_queries, list):
        sub_queries = [str(sub_queries)]
    required_metrics = data.get("required_metrics") or _metric_expansion(query)
    if not isinstance(required_metrics, list):
        required_metrics = [str(required_metrics)]
    hyde = str(data.get("hyde_document") or "")
    enable_hyde = as_bool(rag_cof.get("enable_hyde"), default=True)
    if not enable_hyde:
        hyde = ""
    enable_sub_queries = as_bool(rag_cof.get("enable_sub_queries"), default=True)
    if not enable_sub_queries:
        sub_queries = [query]
    elif hyde:
        sub_queries.append(hyde)
    return {
        "original_query": query,
        "rewritten_query": str(data.get("rewritten_query") or query),
        "hyde_document": hyde,
        "entities": entities,
        "required_metrics": [str(item) for item in required_metrics],
        "sub_queries": list(dict.fromkeys([str(item) for item in sub_queries if str(item).strip()])),
        "intent_hint": str(data.get("intent_hint") or ""),
        "llm_used": True,
        "llm_raw": raw,
    }


def query_transform_node(state: FinancialAgentState) -> dict:
    query = state.get("user_query", "")
    entities = state.get("entities", {})
    profile = state.get("user_profile", {})     # 画像回灌
    # ── L2：从memory_snapshot提取会话摘要 ──
    snapshot = state.get("memory_snapshot", {})
    session_summary = snapshot.get("summary", "")
    enable_query_rewrite = as_bool(rag_cof.get("enable_query_rewrite"), default=True)
    reflection_hint = _reflection_context(state)
    new_round = state.get("reflection_round", 0) + 1 if reflection_hint else state.get("reflection_round", 0)

    if not enable_query_rewrite:
        plan = _fallback_query_plan(query, entities, reflection_hint, profile=profile, session_summary=session_summary)
        sub_queries = plan.get("sub_queries", [query])
    else:
        try:
            plan = _llm_query_plan(query, entities, reflection_hint, profile=profile, session_summary=session_summary)
        except Exception as exc:
            plan = _fallback_query_plan(query, entities, reflection_hint, profile=profile, session_summary=session_summary)
            plan["llm_error"] = str(exc)
        sub_queries = [query, plan.get("rewritten_query", query), *plan.get("sub_queries", [])]

    return {
        "query_plan": plan,
        "sub_queries": list(dict.fromkeys([item for item in sub_queries if item])),
        "reflection_round": new_round,
    }