from __future__ import annotations

import re

from agent.llm_utils import compact_json, extract_json_object, invoke_fast_llm
from agent.state import FinancialAgentState
from rag.citation import has_valid_citation
from utils.logger_handler import log_stage, safe_preview


INVESTMENT_ADVICE_PATTERNS = ["建议买入", "强烈买入", "必然上涨", "无风险", "保证收益"]


def _deterministic_issues(state: FinancialAgentState) -> list[str]:
    draft = state.get("draft_answer", "")
    cards = state.get("evidence_cards", [])
    issues: list[str] = []

    if not cards:
        issues.append("BLOCKING:证据不足：没有检索到可引用的证据卡片。")
    invalid_cards = [card for card in cards if not has_valid_citation(card)]
    if invalid_cards:
        issues.append(f"引用不完整：{len(invalid_cards)} 条证据缺少来源、页码或 chunk_id。")
    if any(pattern in draft for pattern in INVESTMENT_ADVICE_PATTERNS):
        issues.append("BLOCKING:合规风险：草稿包含确定性投资建议或收益承诺。")

    number_lines = [line for line in draft.splitlines() if re.search(r"\d", line)]
    missing_citation_lines = [
        line
        for line in number_lines
        if "【来源：" not in line and "公式：" not in line and "问题：" not in line and "EvidenceCard" not in line
    ]
    if missing_citation_lines:
        issues.append("溯源风险：部分含数字表述未直接附来源。")
    return issues


def _llm_critique(state: FinancialAgentState) -> dict:
    prompt = f"""
你是金融合规审查 Agent。请审查分析师草稿是否满足强溯源和金融合规要求。

审查点：
1. 是否存在无来源财务数字。
2. 是否存在确定性投资建议、收益承诺或过度结论。
3. 是否把预测/假设表述成事实。
4. 是否遗漏风险提示。
5. 证据是否足以回答用户问题。

请只输出 JSON，不要 Markdown：
{{
  "passed": true,
  "issues": ["..."],
  "needs_more_evidence": false
}}

用户问题：
{state.get("user_query", "")}

分析师草稿：
{state.get("draft_answer", "")}

EvidenceCard：
{compact_json(state.get("evidence_cards", []))}
"""
    raw = invoke_fast_llm(prompt)
    data = extract_json_object(raw)
    issues = data.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    return {
        "passed": bool(data.get("passed", False)),
        "issues": [str(issue) for issue in issues if str(issue).strip()],
        "needs_more_evidence": bool(data.get("needs_more_evidence", False)),
        "llm_used": True,
        "llm_raw": raw,
    }


def critic_node(state: FinancialAgentState) -> dict:
    with log_stage(
        "critic",
        evidence_cards=len(state.get("evidence_cards", [])),
        draft_chars=len(state.get("draft_answer", "")),
    ) as stage:
        deterministic = _deterministic_issues(state)
        try:
            result = _llm_critique(state)
            merged_issues = list(dict.fromkeys(result.get("issues", []) + deterministic))
            result["issues"] = merged_issues
            result["passed"] = not merged_issues and bool(result.get("passed", False))
            blocking_issues = [i for i in merged_issues if i.startswith("BLOCKING:")]
            result["needs_more_evidence"] = bool(
                result.get("needs_more_evidence", False)
                or (blocking_issues and state.get("reflection_round", 0) < 2)
            )
        except Exception as exc:
            blocking_issues = [i for i in deterministic if i.startswith("BLOCKING:")]
            result = {
                "passed": not deterministic,
                "issues": deterministic,
                "needs_more_evidence": bool(blocking_issues and state.get("reflection_round", 0) < 2),
                "llm_used": False,
                "llm_error": str(exc),
            }

        stage.add_done_fields(
            llm_used=result.get("llm_used", False),
            passed=result.get("passed", False),
            issues=len(result.get("issues", [])),
            needs_more_evidence=result.get("needs_more_evidence", False),
            llm_error=safe_preview(result.get("llm_error"), 160) if result.get("llm_error") else None,
        )
        return {"critique_result": result}
