from __future__ import annotations

import math
from typing import Callable

from rag.rag_service import RagSummaryService


def _tool(description: str = "") -> Callable:
    try:
        from langchain_core.tools import tool

        return tool(description=description)
    except Exception:
        def decorator(func):
            func.description = description
            return func

        return decorator


rag = RagSummaryService()


@_tool(description="从本地金融研报、财报、纪要和表格中检索证据，并返回带来源页码的摘要。")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@_tool(description="计算同比增长率，输入新值和旧值，返回百分比。")
def calculate_growth_rate(new_value: float, old_value: float) -> str:
    if old_value == 0:
        return "旧值为0，无法计算增长率。"
    return f"{(new_value - old_value) / old_value * 100:.2f}%"


@_tool(description="计算复合年增长率CAGR。")
def calculate_cagr(begin_value: float, end_value: float, years: float) -> str:
    if begin_value <= 0 or years <= 0:
        return "起始值和年数必须大于0。"
    return f"{(math.pow(end_value / begin_value, 1 / years) - 1) * 100:.2f}%"


@_tool(description="根据市值和净利润计算市盈率。")
def calculate_pe_ratio(market_cap: float, net_profit: float) -> str:
    if net_profit == 0:
        return "净利润为0，无法计算市盈率。"
    return f"{market_cap / net_profit:.2f}x"


@_tool(description="实时金融新闻搜索占位工具。需要配置外部API后才会返回真实新闻。")
def search_financial_news(query: str) -> str:
    return f"当前未配置实时新闻API，无法联网检索：{query}"
