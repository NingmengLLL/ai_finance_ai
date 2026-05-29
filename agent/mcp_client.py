"""MCP Client 懒加载管理器。

按 config/mcp.yaml 连接 MCP Server，返回 LangChain BaseTool 列表。
- 懒加载：首次调用时才连接 MCP Server（避免 import 时卡住）
- 按配置过滤：只连接 enabled=True 的 server
- 错误容忍：单个 server 连接失败不影响其他 server
- 模块级缓存：进程生命周期内只连接一次
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils.config_handler import mcp_cof
from utils.helpers import as_bool

logger = logging.getLogger(__name__)

# ── 模块级缓存 ──
_tools_cache: list[BaseTool] | None = None
_client: MultiServerMCPClient | None = None


def _build_connections() -> dict[str, Any]:
    """从 mcp_cof 构造 MultiServerMCPClient 所需的 connections 参数。

    只包含 enabled=True 的 server。
    返回格式与 langchain_mcp_adapters.sessions 的连接类型对齐。
    MultiServerMCPClient 的 connections 参数期望 TypedDict 子类，
    但 Python 运行时普通 dict 也能正常工作（TypedDict 只是类型提示）。
    """
    connections: dict[str, Any] = {}
    for name, spec in mcp_cof.get("servers", {}).items():
        if not as_bool(spec.get("enabled"), default=False):
            logger.debug(f"MCP: server '{name}' 已禁用，跳过")
            continue

        transport = spec.get("transport", "stdio")
        entry: dict[str, Any] = {"transport": transport}

        if transport == "stdio":
            entry["command"] = spec["command"]
            entry["args"] = spec.get("args", [])
            if spec.get("env"):
                # 过滤掉空字符串的 env 值（避免传入无效 API Key）
                filtered_env = {k: v for k, v in spec["env"].items() if v}
                if filtered_env:
                    entry["env"] = filtered_env
        elif transport == "sse":
            entry["url"] = spec["url"]
        elif transport == "http":
            entry["url"] = spec["url"]
        else:
            logger.warning(f"MCP: server '{name}' 未知 transport='{transport}'，跳过")
            continue

        connections[name] = entry

    return connections


def get_mcp_tools() -> list[BaseTool]:
    """获取所有已启用 MCP Server 的工具列表（懒加载 + 缓存）。

    在同步上下文中调用，内部用 asyncio.run() 包装异步连接。
    首次调用会启动 MCP Server 进程并发现工具，后续调用直接返回缓存。
    """
    global _tools_cache, _client

    if _tools_cache is not None:
        return _tools_cache

    connections = _build_connections()
    if not connections:
        logger.info("MCP: 无已启用的 server，返回空工具列表")
        _tools_cache = []
        return _tools_cache

    try:
        _client = MultiServerMCPClient(connections)
        # get_tools() 是 async coroutine，在同步上下文中用 asyncio.run 包装
        # 注意：如果当前已在 async 上下文中（如 Streamlit async handler），会报错
        # 此处 try/except 覆盖两种情况
        try:
            loop = asyncio.get_running_loop()
            # 已在 async 上下文中——不能用 asyncio.run，需要 create_task
            logger.warning("MCP: 当前已在 async 上下文中，无法用 asyncio.run 初始化 MCP")
            _tools_cache = []
        except RuntimeError:
            # 无 running loop——可以用 asyncio.run
            _tools_cache = asyncio.run(_client.get_tools())

        tool_names = [t.name for t in _tools_cache]
        logger.info(
            "MCP: 已连接 %d 个 server，发现 %d 个工具：%s",
            len(connections), len(_tools_cache), tool_names,
        )
    except Exception as exc:
        logger.error("MCP 连接失败：%s", exc, exc_info=True)
        _tools_cache = []

    return _tools_cache


def get_mcp_tools_async() -> list[BaseTool]:
    """获取 MCP 工具列表（async 版本，供 async 上下文调用）。"""
    global _tools_cache, _client

    if _tools_cache is not None:
        return _tools_cache

    connections = _build_connections()
    if not connections:
        _tools_cache = []
        return _tools_cache

    try:
        _client = MultiServerMCPClient(connections)
        _tools_cache = asyncio.run(_client.get_tools())
        tool_names = [t.name for t in _tools_cache]
        logger.info(
            "MCP(async): 已连接 %d 个 server，发现 %d 个工具：%s",
            len(connections), len(_tools_cache), tool_names,
        )
    except Exception as exc:
        logger.error("MCP(async) 连接失败：%s", exc, exc_info=True)
        _tools_cache = []

    return _tools_cache


def close_mcp() -> None:
    """清理 MCP 连接（进程退出时调用）。"""
    global _client, _tools_cache
    if _client is not None:
        try:
            # MultiServerMCPClient 没有显式 close 方法，进程会自然退出
            pass
        except Exception:
            pass
    _client = None
    _tools_cache = None


def get_tool_by_name(name: str) -> BaseTool | None:
    """按工具名查找 MCP 工具。"""
    for tool in get_mcp_tools():
        if tool.name == name:
            return tool
    return None


def invoke_tool(tool_name: str, kwargs: dict[str, Any]) -> Any:
    """同步调用 MCP 工具。

    MCP 工具是 StructuredTool，不支持同步 invoke()，
    必须用 asyncio.run() 包装 ainvoke()。
    """
    tool = get_tool_by_name(tool_name)
    if tool is None:
        raise ValueError("MCP 工具 '" + tool_name + "' 不存在")

    try:
        loop = asyncio.get_running_loop()
        # 已在 async 上下文中——不能 asyncio.run
        raise RuntimeError("invoke_tool 不能在 async 上下文中调用，请使用 invoke_tool_async")
    except RuntimeError as e:
        if "invoke_tool" in str(e):
            raise
        # 无 running loop → 可以 asyncio.run
        return asyncio.run(tool.ainvoke(kwargs))


async def invoke_tool_async(tool_name: str, kwargs: dict[str, Any]) -> Any:
    """异步调用 MCP 工具（供 async 上下文使用）。"""
    tool = get_tool_by_name(tool_name)
    if tool is None:
        raise ValueError("MCP 工具 '" + tool_name + "' 不存在")
    return await tool.ainvoke(kwargs)