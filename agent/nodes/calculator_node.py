from __future__ import annotations

import re

from agent.llm_utils import extract_json_object, invoke_fast_llm, using_real_llm
from agent.state import FinancialAgentState
from knowledge import get_calc_template_by_keyword, get_metrics
from utils.config_handler import rag_cof
from utils.helpers import as_bool

# ── LLM数值计算模板 ──

CALC_PROMPT = """你是金融数值提取与计算模块。请从以下证据文本中提取与用户问题相关的数值，并执行计算。

要求：
1. 只输出严格JSON，不要Markdown，不要额外解释
2. 从证据文本中找到最匹配的数值（注意单位：亿元、百万元、千元等，注意数字中可能包含逗号分隔符如5,600）
3. 如果涉及多年数据，计算同比增长率或CAGR
4. 如果涉及市盈率，用 市值/净利润 计算
5. 无法提取到的数值不要虚构
6. 每个计算结果包含 metric、formula、value、unit、period

用户问题：{query}

证据文本：
{evidence_text}

输出JSON格式如下（注意：花括号为JSON语法，不是占位符）：
{{"calculations": [{{"metric": "指标名称", "formula": "计算公式", "value": 数值结果, "unit": "单位", "period": "时间范围"}}]}}"""


def _strip_comma_numbers(text: str) -> str:
    """将文本中的逗号分隔数字转为纯数字（如 5,600 → 5600），便于regex匹配。"""
    return re.sub(r"(\d),(\d)", r"\1\2", text)


def _pct_change(new: float, old: float) -> float:
    return (new - old) / old * 100 if old else 0.0


def _llm_calculate(query: str, evidence_text: str) -> dict:
    """LLM数值提取+计算：从evidence文本中提取数值并计算。"""
    # 截断过长的evidence文本（避免LLM token超限）
    truncated = evidence_text[:3000] if len(evidence_text) > 3000 else evidence_text
    prompt = CALC_PROMPT.format(query=query, evidence_text=truncated)
    raw = invoke_fast_llm(prompt)
    data = extract_json_object(raw)

    calculations = data.get("calculations", [])
    if not isinstance(calculations, list):
        calculations = []

    # 格式校验：确保每个计算结果都有必需字段
    validated: list[dict] = []
    for calc in calculations:
        if not isinstance(calc, dict):
            continue
        metric = str(calc.get("metric", ""))
        formula = str(calc.get("formula", ""))
        value = calc.get("value")
        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        unit = str(calc.get("unit", ""))
        period = str(calc.get("period", ""))
        validated.append({
            "metric": metric,
            "formula": formula,
            "value": value,
            "unit": unit,
            "period": period,
        })

    return {
        "calculations": validated,
        "llm_used": True,
        "llm_raw": raw,
    }


def _regex_calculate(query: str, evidence_text: str) -> dict:
    """regex模板fallback：从knowledge YAML读取计算模板和regex模式。"""
    # 先去除逗号分隔符，便于regex匹配数值
    clean_text = _strip_comma_numbers(evidence_text)
    calculations: list[dict] = []
    matched_templates = get_calc_template_by_keyword(query)
    all_metrics_info = get_metrics()

    # ── 营业收入同比增长率 ──
    revenue_yoy_template = None
    for t in matched_templates:
        if t.get("metric_label") == "营业收入同比":
            revenue_yoy_template = t
            break

    if revenue_yoy_template:
        revenue_patterns = all_metrics_info.get("营收", {}).get("regex_patterns", [])
        if not revenue_patterns:
            revenue_patterns = [r"20(2[2-5])年[^。\n]*?营业收入(?:为)?([0-9.]+)亿元"]

        revenue_values = []
        for pattern in revenue_patterns:
            revenue_values.extend(re.findall(pattern, clean_text))

        if len(revenue_values) >= 2:
            revenue_values = sorted((int("20" + year), float(value)) for year, value in revenue_values)
            old_year, old_value = revenue_values[-2]
            new_year, new_value = revenue_values[-1]
            calculations.append(
                {
                    "metric": revenue_yoy_template.get("metric_label", "营业收入同比"),
                    "formula": f"({new_value}-{old_value})/{old_value}*100",
                    "value": round(_pct_change(new_value, old_value), 2),
                    "unit": revenue_yoy_template.get("unit", "%"),
                    "period": f"{new_year} vs {old_year}",
                }
            )

    # ── 市盈率 ──
    pe_template = None
    for t in matched_templates:
        if t.get("metric_label") == "市盈率":
            pe_template = t
            break

    if pe_template:
        pe_patterns = all_metrics_info.get("市盈率", {}).get("regex_patterns", [])
        market_cap_pattern = pe_patterns[0] if len(pe_patterns) > 0 else r"市值(?:为)?([0-9.]+)亿元"
        net_profit_pattern = pe_patterns[1] if len(pe_patterns) > 1 else r"净利润(?:为)?([0-9.]+)亿元"

        market_cap_match = re.search(market_cap_pattern, clean_text)
        net_profit_match = re.search(net_profit_pattern, clean_text)
        if market_cap_match and net_profit_match:
            market_cap = float(market_cap_match.group(1))
            net_profit = float(net_profit_match.group(1))
            calculations.append(
                {
                    "metric": pe_template.get("metric_label", "市盈率"),
                    "formula": f"{market_cap}/{net_profit}",
                    "value": round(market_cap / net_profit, 2) if net_profit else None,
                    "unit": pe_template.get("unit", "x"),
                }
            )

    return {"calculations": calculations, "llm_used": False}


def calculator_node(state: FinancialAgentState) -> dict:
    """数值计算节点。
    默认使用LLM（开放域数值提取+计算），regex模板作为fallback。"""
    query = state.get("user_query", "")
    evidence_cards = state.get("evidence_cards", [])
    text = "\n".join(card.get("evidence", "") for card in evidence_cards)
    enable_llm_calculator = as_bool(rag_cof.get("enable_llm_calculator"), default=True)

    if enable_llm_calculator and using_real_llm() and text:
        try:
            result = _llm_calculate(query, text)
        except Exception as exc:
            result = _regex_calculate(query, text)
            result["llm_error"] = str(exc)
    else:
        result = _regex_calculate(query, text)

    return {"calculations": result["calculations"]}