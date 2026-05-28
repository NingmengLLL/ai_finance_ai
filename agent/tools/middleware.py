from __future__ import annotations

from typing import Callable

from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


try:
    from langchain.agents.middleware import before_model, dynamic_prompt, wrap_tool_call
except Exception:
    def wrap_tool_call(func):
        return func

    def before_model(func):
        return func

    def dynamic_prompt(func):
        return func


@wrap_tool_call
def monitor_tool(request, handler: Callable):
    logger.info("[tool monitor] executing tool")
    return handler(request)


@before_model
def log_before_model(state, runtime=None):
    messages = state.get("messages", []) if isinstance(state, dict) else []
    logger.info(f"[log_before_model] messages={len(messages)}")
    return None


@dynamic_prompt
def report_prompt_switch(request):
    try:
        is_report = request.runtime.context.get("report", False)
    except Exception:
        is_report = False
    return load_report_prompts() if is_report else load_system_prompts()
