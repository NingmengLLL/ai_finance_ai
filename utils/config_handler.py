from __future__ import annotations

import yaml

from utils.path_tool import get_abs_path


def _load_yaml(config_path: str, default: dict | None = None, encoding: str = "utf-8") -> dict:
    default = default or {}
    try:
        with open(config_path, "r", encoding=encoding) as f:
            data = yaml.load(f, Loader=yaml.FullLoader) or {}
        return {**default, **data}
    except FileNotFoundError:
        return default.copy()


def load_rag_config(config_path: str = get_abs_path("config/rag.yaml"), encoding: str = "utf-8") -> dict:
    default = {
        "chat_model_name": "qwen3-max",
        "embedding_model_name": "text-embedding-v4",
        "data_path": "data/raw",
        "processed_path": "data/processed/chunks.jsonl",
        "index_path": "data/indexes",
        "require_document_registry": True,
        "auto_register_local_files": False,
        "retriever_k": 8,
        "rerank_top_n": 6,
        "rerank_batch_size": 8,
        "rerank_max_length": 512,
        "rerank_local_files_only": True,
        "chunk_max_chars": 1200,
        "chunk_overlap_chars": 120,
        "enable_hyde": True,
        "enable_query_rewrite": True,
        "enable_sub_queries": True,
        "enable_graph_rag": True,
    }
    return _load_yaml(config_path, default, encoding)


def load_chroma_config(config_path: str = get_abs_path("config/chroma.yaml"), encoding: str = "utf-8") -> dict:
    default = {
        "collection_name": "financial_agent",
        "persist_directory": "data/indexes/chroma_db",
        "retriever_k": 8,
        "data_path": "data/raw",
        "md5_hex_store": "data/indexes/md5.txt",
        "allow_knowledge_file_type": ["txt", "md", "pdf", "csv"],
        "chunk_size": 1200,
        "chunk_overlap": 120,
        "separators": ["\n\n", "\n", "。", "；", ".", ";", " ", ""],
    }
    return _load_yaml(config_path, default, encoding)


def load_prompts_config(config_path: str = get_abs_path("config/prompts.yaml"), encoding: str = "utf-8") -> dict:
    default = {
        "main_prompt_path": "prompts/system_prompt.txt",
        "rag_summary_prompt_path": "prompts/rag_summarize.txt",
        "report_prompt_path": "prompts/analyst_prompt.txt",
        "query_rewrite_prompt_path": "prompts/query_rewrite_prompt.txt",
        "hyde_prompt_path": "prompts/hyde_prompt.txt",
        "critic_prompt_path": "prompts/critic_prompt.txt",
        "citation_prompt_path": "prompts/citation_prompt.txt",
    }
    return _load_yaml(config_path, default, encoding)


def load_agent_config(config_path: str = get_abs_path("config/agent.yaml"), encoding: str = "utf-8") -> dict:
    default = {
        "external_data_path": "data/external/records.csv",
        "max_reflection_rounds": 2,
        "default_user_id": "default",
    }
    return _load_yaml(config_path, default, encoding)


def load_model_config(config_path: str = get_abs_path("config/model.yaml"), encoding: str = "utf-8") -> dict:
    default = {
        "chat_provider": "dashscope_compatible",
        "chat_model_name": "qwen3.6-plus",
        "chat_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embedding_provider": "dashscope_compatible",
        "embedding_model_name": "text-embedding-v4",
        "embedding_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "rerank_provider": "local",
        "rerank_model_name": "local-lexical-reranker",
    }
    return _load_yaml(config_path, default, encoding)


def load_memory_config(config_path: str = get_abs_path("config/memory.yaml"), encoding: str = "utf-8") -> dict:
    default = {
        "short_term_window": 8,
        "heartbeat_message_threshold": 12,
        "heartbeat_char_threshold": 16000,
        "profile_store_path": "data/processed/user_profiles.json",
    }
    return _load_yaml(config_path, default, encoding)


def load_graph_config(config_path: str = get_abs_path("config/graph.yaml"), encoding: str = "utf-8") -> dict:
    default = {
        "enable_web_fallback": False,
        "enable_critic": True,
        "enable_citation_guard": True,
        "max_reflection_rounds": 2,
    }
    return _load_yaml(config_path, default, encoding)


def load_compliance_config(config_path: str = get_abs_path("config/compliance.yaml"), encoding: str = "utf-8") -> dict:
    default = {
        "require_citation_for_numbers": True,
        "forbid_direct_investment_advice": True,
        "risk_disclaimer": "以上内容仅基于已检索资料进行研究分析，不构成投资建议。",
    }
    return _load_yaml(config_path, default, encoding)


def load_mcp_config(config_path: str = get_abs_path("config/mcp.yaml"), encoding: str = "utf-8") -> dict:
    default = {
        "servers": {},
        "web_search": {
            "mode": "placeholder",
            "max_results": 5,
            "inject_to_evidence": True,
        },
    }
    return _load_yaml(config_path, default, encoding)


rag_cof = load_rag_config()
chroma_cof = load_chroma_config()
prompts_cof = load_prompts_config()
agent_cof = load_agent_config()
model_cof = load_model_config()
memory_cof = load_memory_config()
graph_cof = load_graph_config()
compliance_cof = load_compliance_config()
mcp_cof = load_mcp_config()


if __name__ == "__main__":
    print("RAG config:", rag_cof)
