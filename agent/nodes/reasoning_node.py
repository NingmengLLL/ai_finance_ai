from __future__ import annotations

from agent.llm_utils import compact_json, invoke_llm
from agent.state import FinancialAgentState
from rag.citation import EvidenceCard, citation_text
from utils.config_handler import compliance_cof
from utils.logger_handler import log_stage, safe_preview


def _cards(state: FinancialAgentState) -> list[EvidenceCard]:
    return [EvidenceCard.from_dict(card) for card in state.get("evidence_cards", [])]


def _fallback_reasoning(state: FinancialAgentState, error: str | None = None) -> dict:
    query = state.get("user_query", "")
    cards = _cards(state)
    relations = state.get("graph_relations", [])
    calculations = state.get("calculations", [])
    web_note = state.get("web_search_note")

    # ── L2：fallback也注入摘要 ──
    snapshot = state.get("memory_snapshot", {})
    summary = snapshot.get("summary", "")

    if not cards:
        context_hint = ""
        if summary:
            context_hint = "\n历史上下文摘要：" + summary[:300]
        no_evidence_msg = "当前资料不足以回答" + query + "。" + context_hint + "\n建议补充公司年报、券商研报、电话会议纪要或实时新闻数据后再分析。"
        return {
            "draft_answer": no_evidence_msg,
            "reasoning_llm_used": False,
            "reasoning_llm_error": error,
        }

    lines = [f"问题：{query}", "", "结论摘要："]
    for idx, card in enumerate(cards[:4], start=1):
        lines.append(f"{idx}. {card.claim} {citation_text(card)}")

    if calculations:
        lines.append("")
        lines.append("可复核计算：")
        for item in calculations:
            lines.append(
                f"- {item['metric']} = {item.get('value')} {item.get('unit', '')}，公式：{item.get('formula')}"
            )

    if relations:
        lines.append("")
        lines.append("GraphRAG 实体关系辅助推理：")
        for relation in relations[:5]:
            page = f"第{relation.get('page_number')}页" if relation.get("page_number") else "页码未知"
            lines.append(
                f"- {relation['head']} --{relation['relation']}--> {relation['tail']} "
                f"【来源：{relation.get('source_file')}，{page}】"
            )

    lines.append("")
    lines.append("分析判断：")
    lines.append(
        "综合已检索证据，若多个证据均指向同一趋势，可作为研究判断；若证据来自预测或管理层展望，应视为假设而非既成事实。"
    )
    if web_note:
        lines.append(web_note)
    lines.append(compliance_cof.get("risk_disclaimer", "以上内容不构成投资建议。"))
    return {
        "draft_answer": "\n".join(lines),
        "reasoning_llm_used": False,
        "reasoning_llm_error": error,
    }


def _llm_reasoning(state: FinancialAgentState) -> str:
    prompt = build_reasoning_prompt(state)
    return invoke_llm(prompt)


def build_reasoning_prompt(state: FinancialAgentState) -> str:
    """构建reasoning prompt（供流式模式复用）。
    拆出来是因为流式输出需要先构建prompt，再由app.py单独stream LLM调用。"""
    query = state.get("user_query", "")
    cards = state.get("evidence_cards", [])
    relations = state.get("graph_relations", [])
    calculations = state.get("calculations", [])
    web_note = state.get("web_search_note", "")
    risk_disclaimer = compliance_cof.get("risk_disclaimer", "以上内容不构成投资建议。")

    # ── L2：会话摘要注入（跨轮对话的上下文桥梁）──
    snapshot = state.get("memory_snapshot", {})
    summary = snapshot.get("summary", "")
    slots = snapshot.get("slots", {})
    summary_section = ""
    if summary:
        summary_section = f"""
        会话历史摘要（跨轮上下文）：
        {summary}
        """
    if slots:
        slot_items = "; ".join(f"{k}={v}" for k, v in slots.items())
        summary_section += f"\n        已确认信息：{slot_items}\n"

    # ── 画像风格偏好注入 ──
    profile = state.get("user_profile", {})
    style = profile.get("language_style", "professional")
    risk_pref = profile.get("risk_preference", "neutral")
    style_map = {
        "professional": "专业严谨，使用金融术语，数据驱动",
        "casual": "通俗易懂，比喻解释，减少术语堆砌",
        "academic": "学术规范，引用文献风格，逻辑链条完整",
    }
    risk_map = {
        "conservative": "侧重稳健性和下行风险",
        "aggressive": "侧重增长潜力和上行空间",
        "neutral": "客观中立，兼顾多空观点",
    }
    style_instruction = f"回复风格：{style_map.get(style, style_map['professional'])}"
    risk_instruction = f"风险偏好视角：{risk_map.get(risk_pref, risk_map['neutral'])}"

    prompt = f"""
        你是金融研报分析师 Agent。请基于给定 EvidenceCard、GraphRAG 关系和可复核计算回答用户问题。

        硬性约束：
        1. 所有财务数字、公司结论、预测假设必须来自 EvidenceCard 或计算结果。
        2. 每条关键结论必须附引用，格式为：【来源：文件名，第X页】。
        3. 区分"事实""预测/假设""分析判断"，不要把预测当成事实。
        4. 不得输出确定性买卖建议，不得承诺收益。
        5. 如果证据不足，明确说"当前资料不足以确认"，不要编造。
        6. 用中文输出，结构清晰，最后附风险提示。

        {style_instruction}
        {risk_instruction}
        {summary_section}

        用户问题：
        {query}

        EvidenceCard 列表：
        {compact_json(cards)}

        GraphRAG 关系：
        {compact_json(relations)}

        可复核计算：
        {compact_json(calculations)}

        外部搜索说明：
        {web_note}

        风险提示固定句：
        {risk_disclaimer}
        """
    return prompt


def reasoning_node(state: FinancialAgentState) -> dict:
    evidence_count = len(state.get("evidence_cards", []))
    with log_stage(
        "reasoning",
        evidence_cards=evidence_count,
        graph_relations=len(state.get("graph_relations", [])),
        calculations=len(state.get("calculations", [])),
    ) as stage:
        # ── 流式模式：只构建prompt，不调用LLM ──
        if state.get("skip_llm_reasoning"):
            prompt = build_reasoning_prompt(state) if evidence_count > 0 else ""
            stage.add_done_fields(llm_used=False, stream_mode=True, prompt_chars=len(prompt))
            return {"reasoning_prompt": prompt, "draft_answer": "", "reasoning_llm_used": False, "reasoning_llm_skipped": True}

        if not state.get("evidence_cards"):
            result = _fallback_reasoning(state)
            stage.add_done_fields(llm_used=False, fallback=True, answer_chars=len(result.get("draft_answer", "")))
            return result
        try:
            draft = _llm_reasoning(state)
            stage.add_done_fields(llm_used=True, fallback=False, answer_chars=len(draft))
            return {"draft_answer": draft, "reasoning_llm_used": True}
        except Exception as exc:
            result = _fallback_reasoning(state, error=str(exc))
            stage.add_done_fields(
                llm_used=False,
                fallback=True,
                answer_chars=len(result.get("draft_answer", "")),
                llm_error=safe_preview(exc, 160),
            )
            return result
