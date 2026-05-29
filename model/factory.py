from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from utils.config_handler import model_cof, rag_cof


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Any:
        pass


@dataclass
class SimpleMessage:
    content: str


class SimpleChatModel:
    """Deterministic fallback used when LangChain providers are unavailable."""

    def __init__(self, model: str = "local-rule-based"):
        self.model = model

    def invoke(self, prompt: Any) -> SimpleMessage:
        text = prompt if isinstance(prompt, str) else str(prompt)
        return SimpleMessage(
            content=(
                "当前环境未安装在线大模型依赖，已启用本地规则模型。"
                "安装 requirements.txt 后可接入真实 LLM。\n\n"
                f"输入摘要：{text[:500]}"
            )
        )


class SimpleEmbeddings:
    """Small hashing embedding fallback with the LangChain embedding interface."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[A-Za-z0-9_.%+-]+|[\u4e00-\u9fff]", text.lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dimensions
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]


def _build_chat_model(provider: str, model_name: str, base_url: str | None = None, api_key: str | None = None, enable_thinking: bool | None = None) -> Any:
    """通用构建函数，根据provider创建ChatModel实例。
    主模型和快模型共用此函数，避免重复逻辑。
    enable_thinking: Qwen3系列模型是否开启thinking模式。
        None=不设置（模型默认），True=开启深度推理，False=关闭加速。"""
    if provider in {"dashscope_compatible", "openai_compatible"}:
        try:
            from langchain_openai import ChatOpenAI

            resolved_base_url = base_url or (
                os.getenv("DASHSCOPE_BASE_URL")
                or os.getenv("DASH_SCOPE_BASE_URL")
                or os.getenv("ZZZ_BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
            # ── Qwen3 thinking模式控制 ──
            kwargs: dict[str, Any] = {
                "model": model_name,
                "base_url": resolved_base_url,
                "api_key": resolved_api_key,
            }
            if enable_thinking is not None:
                # DashScope OpenAI兼容接口通过extra_body传递enable_thinking
                kwargs["extra_body"] = {"enable_thinking": enable_thinking}
            return ChatOpenAI(**kwargs)
        except Exception:
            return SimpleChatModel(model=model_name)

    if provider == "dashscope":
        try:
            from langchain_community.chat_models.tongyi import ChatTongyi
            return ChatTongyi(model=model_name)
        except Exception:
            return SimpleChatModel(model=model_name)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name)
        except Exception:
            return SimpleChatModel(model=model_name)

    return SimpleChatModel(model=model_name)


# ── 主模型（深度推理、复杂生成）──

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Any:
        provider = model_cof.get("chat_provider", rag_cof.get("chat_provider", "dashscope"))
        model_name = model_cof.get("chat_model_name", rag_cof.get("chat_model_name", "qwen3-max"))
        base_url = model_cof.get("chat_base_url")
        api_key = model_cof.get("fast_model_api_key") or None
        # 主模型thinking开关：默认False（加速推理），设为True时开启深度思考
        enable_thinking = model_cof.get("enable_thinking", None)
        if isinstance(enable_thinking, str):
            enable_thinking = enable_thinking.lower() in ("true", "1", "yes")
        return _build_chat_model(provider, model_name, base_url, api_key, enable_thinking=enable_thinking)

# ── 快模型（分类、提取、结构化输出）──

class FastChatModelFactory(BaseModelFactory):
    """快模型工厂：独立配置的轻量级LLM，用于意图路由、实体抽取、query改写等快速任务。
    典型选用：deepseek-v4-flash、qwen-turbo 等低成本高速度模型。"""

    def generator(self) -> Any:
        # 快模型配置：优先读fast_model段，fallback到主模型配置
        provider = model_cof.get("fast_model_provider", model_cof.get("chat_provider", "dashscope_compatible"))
        model_name = model_cof.get("fast_model_name", model_cof.get("chat_model_name", "qwen3-max"))
        base_url = model_cof.get("fast_model_base_url", model_cof.get("chat_base_url"))
        # 快模型可能用不同的API key（如DeepSeek的key），如果没配则共用主模型的key
        api_key = model_cof.get("fast_model_api_key") or None
        # 快模型始终关闭thinking（分类/提取任务不需要深度推理）
        enable_thinking = False
        return _build_chat_model(provider, model_name, base_url, api_key, enable_thinking=enable_thinking)


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Any:
        provider = model_cof.get("embedding_provider", rag_cof.get("embedding_provider", "dashscope"))
        model_name = model_cof.get("embedding_model_name", rag_cof.get("embedding_model_name", "text-embedding-v4"))
        chunk_size = min(int(model_cof.get("embedding_chunk_size", 8)), 10)

        if provider in {"dashscope_compatible", "openai_compatible"}:
            try:
                from langchain_openai import OpenAIEmbeddings

                base_url = (
                    model_cof.get("embedding_base_url")
                    or model_cof.get("chat_base_url")
                    or os.getenv("DASHSCOPE_BASE_URL")
                    or os.getenv("DASH_SCOPE_BASE_URL")
                    or os.getenv("ZZZ_BASE_URL")
                    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
                return OpenAIEmbeddings(
                    model=model_name,
                    base_url=base_url,
                    api_key=api_key,
                    tiktoken_enabled=False,
                    check_embedding_ctx_length=False,
                    chunk_size=chunk_size,
                )
            except Exception:
                return SimpleEmbeddings()

        if provider == "dashscope":
            try:
                from langchain_community.embeddings import DashScopeEmbeddings
                return DashScopeEmbeddings(model=model_name)
            except Exception:
                return SimpleEmbeddings()

        if provider == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(model=model_name)
            except Exception:
                return SimpleEmbeddings()

        return SimpleEmbeddings()


# ── 全局单例（懒加载，避免import时初始化真实LLM连接）──

class _LazyModel:
    """懒加载包装器：首次访问时才创建真实模型实例，避免import时触发API连接。"""

    def __init__(self, factory_cls: type[BaseModelFactory]):
        self._factory_cls = factory_cls
        self._instance: Any = None

    def _get_instance(self) -> Any:
        if self._instance is None:
            self._instance = self._factory_cls().generator()
        return self._instance

    @property
    def model(self) -> Any:
        return self._get_instance()

    # 代理所有属性和方法到真实实例
    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)

    def __repr__(self) -> str:
        if self._instance is not None:
            return f"_LazyModel({self._instance.__class__.__name__})"
        return f"_LazyModel({self._factory_cls.__name__}, uninitialized)"


chat_model = _LazyModel(ChatModelFactory)      # 主模型：深度推理（reasoning_node）
fast_model = _LazyModel(FastChatModelFactory)   # 快模型：分类/提取/改写（router/query_transform/calculator/critic）
embed_model = _LazyModel(EmbeddingsFactory)