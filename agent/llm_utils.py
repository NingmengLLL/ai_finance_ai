from __future__ import annotations

import json
import re
import time
from typing import Any

from model.factory import SimpleChatModel, chat_model, fast_model
from utils.logger_handler import log_stage, logger


class LLMCallError(RuntimeError):
    pass


# ── 主模型检测 ──

def using_real_llm() -> bool:
    """主模型是否可用（非SimpleChatModel）。
    _LazyModel会在首次调用invoke时才初始化，这里检查底层实例。"""
    from model.factory import _LazyModel, SimpleChatModel
    model = chat_model
    if isinstance(model, _LazyModel):
        underlying = model._instance
        if underlying is None:
            # 未初始化 → 尝试初始化一次来判断
            underlying = model._get_instance()
        return not isinstance(underlying, SimpleChatModel)
    return not isinstance(model, SimpleChatModel)


def using_real_fast_llm() -> bool:
    """快模型是否可用（非SimpleChatModel）。
    如果快模型fallback到主模型配置，则与主模型相同。"""
    from model.factory import _LazyModel, SimpleChatModel
    model = fast_model
    if isinstance(model, _LazyModel):
        underlying = model._instance
        if underlying is None:
            underlying = model._get_instance()
        return not isinstance(underlying, SimpleChatModel)
    return not isinstance(model, SimpleChatModel)


# ── 主模型调用（深度推理）──

def invoke_llm(prompt: str, max_retries: int = 3, retry_delay: float = 1.0) -> str:
    """调用主模型（qwen3.6-plus等），用于 reasoning_node 等需要深度推理的场景。"""
    with log_stage("llm.invoke", prompt_chars=len(prompt), model=chat_model.__class__.__name__, tier="main") as stage:
        if not using_real_llm():
            raise LLMCallError("chat_model is SimpleChatModel fallback, not a real LLM provider.")
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = chat_model.invoke(prompt)
                content = getattr(response, "content", response)
                if isinstance(content, list):
                    content = "\n".join(str(item) for item in content)
                content = str(content).strip()
                if not content:
                    raise LLMCallError("LLM returned empty content.")
                stage.add_done_fields(response_chars=len(content), attempt=attempt)
                return content
            except LLMCallError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning(f"LLM invoke attempt {attempt}/{max_retries} failed: {exc}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
        raise LLMCallError(f"LLM invoke failed after {max_retries} attempts: {last_exc}") from last_exc


# ── 快模型调用（分类/提取/改写）──

def invoke_fast_llm(prompt: str, max_retries: int = 3, retry_delay: float = 0.5) -> str:
    """调用快模型（qwen-turbo/deepseek-v4-flash等），用于路由/提取/改写等快速任务。
    retry_delay更短（0.5s vs 1.0s），因为快模型响应快、成本低，重试代价小。"""
    with log_stage("llm.fast_invoke", prompt_chars=len(prompt), model=fast_model.__class__.__name__, tier="fast") as stage:
        if not using_real_fast_llm():
            # 快模型不可用时，自动升级到主模型
            logger.warning("fast_model is SimpleChatModel, falling back to main model for this call.")
            return invoke_llm(prompt, max_retries=max_retries, retry_delay=retry_delay)
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = fast_model.invoke(prompt)
                content = getattr(response, "content", response)
                if isinstance(content, list):
                    content = "\n".join(str(item) for item in content)
                content = str(content).strip()
                if not content:
                    raise LLMCallError("Fast LLM returned empty content.")
                stage.add_done_fields(response_chars=len(content), attempt=attempt)
                return content
            except LLMCallError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning(f"Fast LLM invoke attempt {attempt}/{max_retries} failed: {exc}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
        raise LLMCallError(f"Fast LLM invoke failed after {max_retries} attempts: {last_exc}") from last_exc


def _repair_json_string(text: str) -> str:
    """Attempt basic repairs on malformed JSON output from LLM."""
    cleaned = text.strip()
    # Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    # Remove inline comments like // or #
    cleaned = re.sub(r"//[^\n]*", "", cleaned)
    # Unescape double-escaped quotes
    cleaned = cleaned.replace("\\\"", "\"")
    # Try to close unclosed braces
    open_braces = cleaned.count("{") - cleaned.count("}")
    open_brackets = cleaned.count("[") - cleaned.count("]")
    if open_braces > 0:
        cleaned += "}" * open_braces
    if open_brackets > 0:
        cleaned += "]" * open_brackets
    return cleaned


def extract_json_object(text: str, max_repair_attempts: int = 2) -> dict[str, Any]:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.S)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]

    for attempt in range(max_repair_attempts + 1):
        try:
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                raise LLMCallError("LLM JSON output is not an object.")
            return data
        except json.JSONDecodeError as exc:
            if attempt < max_repair_attempts:
                logger.warning(f"JSON parse attempt {attempt + 1} failed: {exc}, attempting repair...")
                cleaned = _repair_json_string(cleaned)
            else:
                raise LLMCallError(f"Failed to parse JSON from LLM output after {max_repair_attempts + 1} attempts: {text[:500]}") from exc


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


# ── 主模型流式调用（reasoning_node 等需要逐token渲染的场景）──

def invoke_llm_stream(prompt: str) -> Any:
    """流式调用主模型，返回 generator 逐 token yield。
    用于 Streamlit st.write_stream() 逐字渲染最终答案。
    如果主模型不可用（SimpleChatModel），直接 yield invoke_llm 的完整结果。"""
    if not using_real_llm():
        # fallback：非流式，一次性yield完整文本
        yield invoke_llm(prompt)
        return

    with log_stage("llm.stream", prompt_chars=len(prompt), tier="main-stream") as stage:
        full_text = ""
        try:
            for chunk in chat_model.stream(prompt):
                token = getattr(chunk, "content", chunk)
                if isinstance(token, list):
                    token = "".join(str(item) for item in token)
                token = str(token)
                if token:
                    full_text += token
                    yield token
            stage.add_done_fields(response_chars=len(full_text))
        except Exception as exc:
            # 流式失败 → yield fallback完整结果
            if full_text:
                stage.add_done_fields(response_chars=len(full_text), stream_partial=True)
            else:
                try:
                    fallback = invoke_llm(prompt)
                    yield fallback
                    stage.add_done_fields(response_chars=len(fallback), stream_fallback=True)
                except LLMCallError:
                    yield f"[流式输出失败：{exc}]"