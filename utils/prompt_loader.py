from __future__ import annotations

from utils.config_handler import prompts_cof
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


def _load_prompt(config_key: str) -> str:
    try:
        prompt_path = get_abs_path(prompts_cof[config_key])
    except KeyError as exc:
        logger.error(f"Missing prompt config key: {config_key}")
        raise exc
    try:
        return open(prompt_path, "r", encoding="utf-8").read()
    except Exception as exc:
        logger.error(f"Failed to load prompt {prompt_path}: {exc}")
        raise exc


def load_system_prompts() -> str:
    return _load_prompt("main_prompt_path")


def load_rag_summary_prompts() -> str:
    return _load_prompt("rag_summary_prompt_path")


def load_report_prompts() -> str:
    return _load_prompt("report_prompt_path")


def load_query_rewrite_prompt() -> str:
    return _load_prompt("query_rewrite_prompt_path")


def load_hyde_prompt() -> str:
    return _load_prompt("hyde_prompt_path")


def load_critic_prompt() -> str:
    return _load_prompt("critic_prompt_path")


def load_citation_prompt() -> str:
    return _load_prompt("citation_prompt_path")
