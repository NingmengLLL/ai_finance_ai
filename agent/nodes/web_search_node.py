"""web_search 节点：从 placeholder → MCP Client。

根据 config/mcp.yaml 的 web_search.mode 选择：
- mcp_client：通过 MCP 工具获取实时金融数据，注入 evidence_cards
- placeholder：旧逻辑（返回提示文本）

MCP 工具调用流程：
1. 从 mcp_client.py 获取所有 MCP 工具列表
2. 根据用户查询意图 + 实体信息，选择匹配的工具和参数
3. 调用工具，解析结果为 evidence_card 格式
4. 注入到 state["evidence_cards"]
"""
from __future__ import annotations

import logging
import re
from typing import Any

from agent.mcp_client import get_mcp_tools
from agent.state import FinancialAgentState
from utils.config_handler import mcp_cof
from utils.helpers import as_bool
from utils.logger_handler import log_stage, safe_preview

logger = logging.getLogger(__name__)

PLACEHOLDER_NOTE_TEMPLATE = (
    '当前未接入实时网页/行情搜索 API；问题\u201c{query}\u201d仅基于本地知识库回答。'
    '如需实时行情或最新公告，可在 config/mcp.yaml 开启 MCP Client 模式。'
)


# ── 工具选择策略：根据查询意图+实体匹配 MCP 工具 ──

# AKShare 工具名 → 参数模板映射
AKSHARE_TOOL_MAP: dict[str, dict[str, Any]] = {
    # 沪深A股实时行情（无参数）
    "stock_zh_a_spot_em": {},
    # 业绩报表（需要 date 参数）
    "stock_yjbb_em": {"date": "20241231"},
    # 涨停股池（需要 date 参数）
    "stock_zt_pool_em": {"date": "20241008"},
    # 指数行情（需要 symbol 参数）
    "stock_zh_index_spot_em": {"symbol": "上证系列指数"},
    # 全球指数（无参数）
    "index_global_spot_em": {},
}

# 查询关键词 → 工具名映射
QUERY_TOOL_HINTS: dict[str, list[str]] = {
    "实时行情": ["stock_zh_a_spot_em"],
    "股价": ["stock_zh_a_spot_em"],
    "行情": ["stock_zh_a_spot_em"],
    "涨停": ["stock_zt_pool_em"],
    "业绩": ["stock_yjbb_em"],
    "财报": ["stock_yjbb_em"],
    "年报": ["stock_yjbb_em"],
    "季报": ["stock_yjbb_em"],
    "营收": ["stock_yjbb_em"],
    "净利润": ["stock_yjbb_em"],
    "指数": ["stock_zh_index_spot_em", "index_global_spot_em"],
    "全球指数": ["index_global_spot_em"],
    "上证": ["stock_zh_index_spot_em"],
    "深证": ["stock_zh_index_spot_em"],
}


def _extract_date_from_query(query: str) -> str:
    """从用户查询中提取报告期日期，默认当年年末。

    支持格式：2024年、2024Q3、2024年第三季度、20240930 等。
    """
    # 精确日期格式：20240930、20241231
    exact_match = re.search(r"\b20\d{2}(0[3-9]|1[0-2])(?:0|3)\b", query)
    if exact_match:
        return exact_match.group()

    # 年份 + 季度
    year_match = re.search(r"(20\d{2})", query)
    if year_match:
        year = year_match.group(1)
        if "Q1" in query or "第一季度" in query or "一季报" in query or "季报" in query and "年报" not in query:
            return f"{year}0331"
        if "Q2" in query or "半年报" in query or "中期" in query or "第二季度" in query:
            return f"{year}0630"
        if "Q3" in query or "第三季度" in query or "三季报" in query:
            return f"{year}0930"
        # 默认年报
        return f"{year}1231"

    # 无年份 → 用当前年份
    from datetime import datetime
    current_year = datetime.now().year
    return f"{current_year}1231"


def _select_tools(query: str, entities: dict[str, Any], intent: str) -> list[tuple[str, dict[str, Any]]]:
    """根据查询意图+实体，选择要调用的 MCP 工具及其参数。

    返回 [(tool_name, kwargs), ...] 列表。
    """
    # 获取可用工具名
    available_tools = get_mcp_tools()
    available_names = {t.name for t in available_tools}

    # 优先按关键词匹配
    selected: list[tuple[str, dict[str, Any]]] = []
    matched_tool_names: set[str] = set()

    for keyword, tool_names in QUERY_TOOL_HINTS.items():
        if keyword in query:
            for tn in tool_names:
                if tn in available_names and tn not in matched_tool_names:
                    matched_tool_names.add(tn)
                    # 构建参数：从模板复制，动态替换日期
                    kwargs = dict(AKSHARE_TOOL_MAP.get(tn, {}))
                    if "date" in kwargs:
                        kwargs["date"] = _extract_date_from_query(query)
                    if "symbol" in kwargs:
                        # 尝试从 entities 匹配指数类型
                        for kw in ["上证", "深证", "沪深", "中证"]:
                            if kw in query:
                                kwargs["symbol"] = kw + "系列指数"
                                break
                    selected.append((tn, kwargs))

    # 如果 intent 是 realtime_financial_search 且没匹配到任何工具
    # → 尝试调 stock_zh_a_spot_em（最通用的实时行情工具）
    if not selected and intent == "realtime_financial_search":
        if "stock_zh_a_spot_em" in available_names:
            selected.append(("stock_zh_a_spot_em", {}))

    # 最多调 3 个工具（避免超时）
    return selected[:3]


def _parse_tool_result(tool_name: str, raw_result: Any, query: str) -> list[dict]:
    """解析 MCP 工具返回结果，转为 evidence_card 格式。

    AKShare 工具通常返回 markdown 表格字符串（因为 akshare-mcp 的 format=markdown）。
    """
    cards: list[dict] = []

    if isinstance(raw_result, str):
        # markdown 表格文本 → 作为 evidence 整体注入
        # 截断过长结果（避免注入过多 token 到 reasoning）
        max_chars = 3000
        evidence = raw_result[:max_chars] if len(raw_result) > max_chars else raw_result
        if len(raw_result) > max_chars:
            evidence += "\n...（已截断，原始数据较长）"

        cards.append({
            "evidence": evidence,
            "claim": f"[{tool_name}] 实时数据查询结果",
            "source_file": "web_search_mcp",
            "chunk_id": f"mcp_{tool_name}",
            "page_number": None,
        })

    elif isinstance(raw_result, list):
        for item in raw_result[:5]:
            content = str(item.get("body", item.get("content", str(item))))[:500]
            cards.append({
                "evidence": content,
                "claim": item.get("title", f"[{tool_name}] 搜索结果"),
                "source_file": "web_search_mcp",
                "chunk_id": f"mcp_{tool_name}",
                "page_number": None,
            })

    elif isinstance(raw_result, dict):
        content = str(raw_result.get("body", raw_result.get("content", str(raw_result))))[:500]
        cards.append({
            "evidence": content,
            "claim": raw_result.get("title", f"[{tool_name}] 搜索结果"),
            "source_file": "web_search_mcp",
            "chunk_id": f"mcp_{tool_name}",
            "page_number": None,
        })

    return cards


def _mcp_search(query: str, entities: dict[str, Any], intent: str, max_results: int) -> list[dict]:
    """通过 MCP 工具执行搜索，返回 evidence_card 格式列表。

    每个工具调用有 25s 超时保护，避免 AKShare 等接口卡住整个管线。
    """
    tools_to_call = _select_tools(query, entities, intent)
    if not tools_to_call:
        logger.info("web_search: 无匹配的 MCP 工具")
        return []

    # 从 mcp.yaml 读取超时配置，默认 25s
    mcp_timeout = int(mcp_cof.get("web_search", {}).get("tool_timeout", 25))

    results: list[dict] = []
    for tool_name, kwargs in tools_to_call:
        try:
            import asyncio
            # invoke_tool 内部用 asyncio.run，需要加超时保护
            async def _call_with_timeout():
                from agent.mcp_client import invoke_tool_async
                return await asyncio.wait_for(invoke_tool_async(tool_name, kwargs), timeout=mcp_timeout)

            try:
                loop = asyncio.get_running_loop()
                # 已在 async 上下文 → 无法 asyncio.run，直接 fallback
                logger.warning("web_search: 无法在 async 上下文中调用 MCP 工具 %s", tool_name)
                continue
            except RuntimeError:
                raw = asyncio.run(_call_with_timeout())

            cards = _parse_tool_result(tool_name, raw, query)
            results.extend(cards)
            logger.info(
                "web_search: MCP 工具 %s 调用成功，返回 %d 条 evidence",
                tool_name, len(cards),
            )
        except Exception as exc:
            logger.error("web_search: MCP 工具 %s 调用失败：%s", tool_name, exc)

    return results[:max_results]


def web_search_node(state: FinancialAgentState) -> dict:
    """web_search 节点：MCP Client 模式或 placeholder 模式。"""
    query = state.get("user_query", "")
    entities = state.get("entities", {})
    intent = state.get("intent", "")
    mode = mcp_cof.get("web_search", {}).get("mode", "placeholder")
    max_results = int(mcp_cof.get("web_search", {}).get("max_results", 5))
    inject_to_evidence = as_bool(mcp_cof.get("web_search", {}).get("inject_to_evidence"), default=True)

    with log_stage("web_search", query=safe_preview(query), mode=mode, intent=intent) as stage:
        # ── MCP Client 模式 ──
        if mode == "mcp_client":
            mcp_results = _mcp_search(query, entities, intent, max_results)
            if mcp_results:
                evidence_cards = state.get("evidence_cards", [])
                if inject_to_evidence:
                    evidence_cards = evidence_cards + mcp_results
                note = "已通过 MCP 获取 " + str(len(mcp_results)) + " 条实时数据。"
                stage.add_done_fields(mode="mcp_client", results=len(mcp_results))
                return {
                    "evidence_cards": evidence_cards,
                    "web_search_note": note,
                }
            # MCP 调用无结果 → fallback 到 placeholder
            logger.warning("web_search: MCP 搜索无结果，fallback 到 placeholder")

        # ── Placeholder 模式（或 MCP fallback）──
        note = PLACEHOLDER_NOTE_TEMPLATE.format(query=query)
        stage.add_done_fields(mode="placeholder")
        return {
            "evidence_cards": state.get("evidence_cards", []),
            "web_search_note": note,
        }