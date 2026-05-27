"""knowledge_loader.py — 统一知识中枢的Python访问层

所有模块通过此文件读取领域知识，不再直接硬编码。
YAML文件变更后，调用 reload() 即可刷新（或重启进程自动重载）。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_YAML_PATH = Path(__file__).resolve().parent / "financial_knowledge.yaml"


@lru_cache(maxsize=1)
def _load_yaml() -> dict[str, Any]:
    with _YAML_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def reload() -> None:
    """强制重新加载YAML（用于热更新场景）。"""
    _load_yaml.cache_clear()


def _data() -> dict[str, Any]:
    return _load_yaml()


# ── 公司知识 ──────────────────────────────────

def get_companies() -> dict[str, dict]:
    """返回完整公司字典 {company_id: {aliases, industry, sector, tickers}}"""
    return _data().get("companies", {})


def get_company_aliases() -> dict[str, tuple]:
    """返回 {company_id: (aliases...)} 格式，兼容旧接口。"""
    return {cid: tuple(info["aliases"]) for cid, info in get_companies().items()}


def get_company_industry(company_id: str) -> str:
    """返回指定公司的行业分类。"""
    return get_companies().get(company_id, {}).get("industry", "")


def get_all_aliases_flat() -> dict[str, str]:
    """返回 {alias_lower: company_id} 的扁平映射，用于快速查找。"""
    flat: dict[str, str] = {}
    for cid, info in get_companies().items():
        for alias in info["aliases"]:
            flat[alias.lower()] = cid
    return flat


# ── 行业知识 ──────────────────────────────────

def get_industries() -> dict[str, dict]:
    """返回完整行业字典 {industry_name: {aliases, companies}}"""
    return _data().get("industries", {})


def get_industry_aliases() -> dict[str, list[str]]:
    """返回 {industry_name: [aliases...]} 格式。"""
    return {name: info.get("aliases", []) for name, info in get_industries().items()}


def get_industry_companies(industry_name: str) -> list[str]:
    """返回指定行业下的company_id列表。"""
    return get_industries().get(industry_name, {}).get("companies", [])


def resolve_industry(keyword: str) -> list[str] | None:
    """从关键词推断行业，返回对应的company_id列表。
    支持子串匹配：'电商行业' 中的 '电商' 能命中 aliases=['电商',...]"""
    industries = get_industries()
    for ind_name, info in industries.items():
        aliases = info.get("aliases", [])
        # 精确匹配（alias完全等于keyword，或keyword完全等于行业名）
        if keyword in aliases or keyword == ind_name:
            return info.get("companies")
        # 子串匹配（alias是keyword的子串，如"电商"∈"电商行业"）
        for alias in aliases:
            if alias in keyword:
                return info.get("companies")
    return None


# ── 文档类型 ──────────────────────────────────

def get_doc_types() -> dict[str, dict]:
    """返回 {doc_type: {aliases: [...]}}"""
    return _data().get("doc_types", {})


def get_doc_type_aliases() -> dict[str, list[str]]:
    """返回 {keyword: doc_type} 的反向映射，用于从query推断doc_type。"""
    reverse: dict[str, str] = {}
    for doc_type, info in get_doc_types().items():
        for alias in info.get("aliases", []):
            reverse[alias] = doc_type
    return reverse


def resolve_doc_type(query: str) -> str | None:
    """从query中推断文档类型。"""
    for keyword, doc_type in get_doc_type_aliases().items():
        if keyword in query:
            return doc_type
    return None


# ── 意图关键词 ──────────────────────────────────

def get_intents() -> dict[str, dict]:
    """返回 {intent_name: {keywords: [...]}}"""
    return _data().get("intents", {})


def get_intent_keywords(intent_name: str) -> list[str]:
    """返回指定意图的关键词列表。"""
    return get_intents().get(intent_name, {}).get("keywords", [])


# ── 金融指标 ──────────────────────────────────

def get_metrics() -> dict[str, dict]:
    """返回完整指标字典 {metric_name: {synonyms, related, unit, ...}}"""
    return _data().get("metrics", {})


def get_metric_names() -> list[str]:
    """返回所有指标名列表（用于entity_extractor）。"""
    return list(get_metrics().keys())


def get_all_metric_synonyms_flat() -> list[str]:
    """返回所有指标名+同义词的扁平列表（用于entity_extractor的匹配）。"""
    terms: list[str] = []
    for name, info in get_metrics().items():
        terms.append(name)
        terms.extend(info.get("synonyms", []))
    return list(dict.fromkeys(terms))


def get_metric_search_terms(metric_name: str) -> list[str]:
    """返回指定指标的检索词列表（用于context_compressor的_metric_terms）。"""
    info = get_metrics().get(metric_name, {})
    # search_terms 优先，fallback 到 synonyms
    return info.get("search_terms") or info.get("synonyms", [metric_name])


def get_metric_expansion_terms(metric_name: str) -> list[str]:
    """返回指定指标的扩展词列表（用于query_transform的_metric_expansion）。"""
    info = get_metrics().get(metric_name, {})
    return info.get("expansion_terms", [])


def resolve_metrics_from_query(query: str) -> list[str]:
    """从query中匹配所有涉及到的指标名。"""
    matched: list[str] = []
    for name, info in get_metrics().items():
        if name in query:
            matched.append(name)
            continue
        for synonym in info.get("synonyms", []):
            if synonym in query:
                matched.append(name)
                break
    return matched


def get_metric_regex_patterns(metric_name: str) -> list[str]:
    """返回指定指标的regex模式列表（用于calculator_node）。"""
    info = get_metrics().get(metric_name, {})
    return info.get("regex_patterns", [])


# ── 计算模板 ──────────────────────────────────

def get_calc_templates() -> list[dict]:
    """返回计算模板列表。YAML中calc_templates是dict，需取values。"""
    raw = _data().get("calc_templates", [])
    # YAML中calc_templates是keyed dict（revenue_yoy/pe_ratio），不是list
    if isinstance(raw, dict):
        return list(raw.values())
    return raw


def get_calc_template_by_keyword(query: str) -> list[dict]:
    """从query中匹配计算模板。"""
    matched: list[dict] = []
    for template in get_calc_templates():
        for keyword in template.get("trigger_keywords", []):
            if keyword in query:
                matched.append(template)
                break
    return matched


# ── 关系模板 ──────────────────────────────────

def get_relation_templates() -> list[dict]:
    """返回关系模板列表。"""
    return _data().get("relation_templates", [])


# ── 噪声模式 ──────────────────────────────────

def get_noise_patterns() -> list[str]:
    """返回噪声文本模式列表。"""
    return _data().get("noise_patterns", [])


# ── 中文数字映射 ──────────────────────────────────

def get_chinese_digit_map() -> dict[str, str]:
    """返回中文数字→阿拉伯数字映射。"""
    return _data().get("chinese_digit_map", {})