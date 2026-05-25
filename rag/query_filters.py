from __future__ import annotations

import re
from difflib import SequenceMatcher

from knowledge import (
    get_all_aliases_flat,
    get_companies,
    get_industry_aliases,
    resolve_doc_type,
    resolve_industry,
)


# ──────────────────────────────────────────────
# 1. 公司/行业/文档类型推断 — 全部从 knowledge YAML 读取
# ──────────────────────────────────────────────
# 旧的硬编码字典已移除，所有知识统一在 knowledge/financial_knowledge.yaml 维护


def _match_companies(query: str) -> list[str]:
    """从query中匹配公司，返回所有匹配到的company_id列表。
    支持精确→别名→regex兜底→fuzzy_match四级匹配。"""
    lowered = query.lower()
    companies = get_companies()
    matched: list[str] = []

    # 精确+别名匹配
    for company_id, info in companies.items():
        for alias in info.get("aliases", []):
            if alias.lower() in lowered:
                matched.append(company_id)
                break  # 一个公司只需匹配一次

    if not matched:
        # regex兜底：匹配"XX公司""XX集团"等后缀模式
        company_suffix_pattern = re.compile(r"([\u4e00-\u9fff]{2,4})(公司|集团|控股|科技|有限)")
        for m in company_suffix_pattern.finditer(query):
            core_name = m.group(1)
            for company_id, info in companies.items():
                for alias in info.get("aliases", []):
                    if core_name in alias:
                        matched.append(company_id)
                        break

    if not matched:
        # fuzzy_match兜底：用SequenceMatcher处理拼写偏差
        query_words = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]{2,}", query)
        for word in query_words:
            for company_id, info in companies.items():
                for alias in info.get("aliases", []):
                    ratio = SequenceMatcher(None, word.lower(), alias.lower()).ratio()
                    if ratio >= 0.85:
                        matched.append(company_id)
                        break

    return matched


def _match_industry(query: str) -> list[str] | None:
    """从query推断行业分类，返回对应的company_id列表。
    知识来源：knowledge YAML 的 industries 部分。支持子串匹配。"""
    # 先尝试resolve_industry（子串匹配）
    result = resolve_industry(query)
    if result:
        return result
    # 如果整个query没命中，尝试逐词拆分匹配
    query_words = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]+", query)
    for word in query_words:
        result = resolve_industry(word)
        if result:
            return result
    return None


def _chinese_num_to_int(text: str) -> int | None:
    """中文数字年份转换：'二零二四' → 2024"""
    from knowledge import get_chinese_digit_map
    cn_map = get_chinese_digit_map()
    table = str.maketrans(cn_map)
    digits = text.translate(table)
    if re.fullmatch(r"20\d{2}", digits):
        return int(digits)
    return None


def _parse_time_range(query: str) -> dict | None:
    """解析时间范围。支持：
    - 单年："2024年" → {"report_period": "FY2024"}
    - 近N年："近3年""最近5年" → {"report_period": {"$in": ["FY2022","FY2023","FY2024"]}}
    - 跨年："2022到2024""2022-2024年" → {"report_period": {"$in": ["FY2022","FY2023","FY2024"]}}
    """
    # 近N年 / 最近N年
    near_match = re.search(r"近(\d+)年|最近(\d+)年", query)
    if near_match:
        n = int(near_match.group(1) or near_match.group(2))
        current_year = 2025
        years = [f"FY{current_year - i}" for i in range(n)]
        return {"report_period": {"$in": years}}

    # 跨年：2022到2024 / 2022-2024 / 2022至2024
    range_match = re.search(r"(20\d{2})\s*(?:到|至|-|—|~)\s*(20\d{2})", query)
    if range_match:
        start_y, end_y = int(range_match.group(1)), int(range_match.group(2))
        if start_y <= end_y:
            years = [f"FY{y}" for y in range(start_y, end_y + 1)]
            return {"report_period": {"$in": years}}

    # 中文年份范围：二零二二年到二零二四年
    cn_range_match = re.search(r"(二[零〇][零〇一二三四五六七八九]{2})\s*(?:到|至|-|—)\s*(二[零〇][零〇一二三四五六七八九]{2})", query)
    if cn_range_match:
        start_y = _chinese_num_to_int(cn_range_match.group(1))
        end_y = _chinese_num_to_int(cn_range_match.group(2))
        if start_y and end_y and start_y <= end_y:
            years = [f"FY{y}" for y in range(start_y, end_y + 1)]
            return {"report_period": {"$in": years}}

    # 单年：2024年 / FY2024
    year_match = re.search(r"(20\d{2})\s*年?", query)
    if year_match:
        return {"report_period": f"FY{year_match.group(1)}"}

    # 中文单年：二零二四年
    cn_year_match = re.search(r"二[零〇][零〇一二三四五六七八九]{2}年", query)
    if cn_year_match:
        y = _chinese_num_to_int(cn_year_match.group(0)[:-1])
        if y:
            return {"report_period": f"FY{y}"}

    return None


def infer_metadata_filter(query: str) -> dict:
    """多层级智能元数据过滤推断。知识来源：knowledge YAML。
    
    降级策略：精确→行业→时间→全量，当推断失败时返回空dict（不设过滤）。
    """
    filter_result: dict = {}

    # ── 公司匹配 ──
    companies = _match_companies(query)
    if companies:
        if len(companies) == 1:
            filter_result["company_id"] = companies[0]
        else:
            filter_result["company_id"] = {"$in": companies}
    else:
        # 公司未命中 → 尝试行业匹配（降级策略第1级）
        industry_companies = _match_industry(query)
        if industry_companies:
            if len(industry_companies) == 1:
                filter_result["company_id"] = industry_companies[0]
            else:
                filter_result["company_id"] = {"$in": industry_companies}

    # ── 时间匹配 ──
    time_filter = _parse_time_range(query)
    if time_filter:
        filter_result.update(time_filter)

    # ── 文档类型匹配 ──
    doc_type = resolve_doc_type(query)
    if doc_type:
        filter_result["doc_type"] = doc_type

    return filter_result


# ──────────────────────────────────────────────
# 2. 查询改写：减法式(normalize) + 加法式(augment)
# ──────────────────────────────────────────────

def normalize_query_for_metadata_filter(query: str, metadata_filter: dict | None) -> str:
    """减法式改写 — 删减已被metadata filter覆盖的词项，BM25通道专用。"""
    if not metadata_filter:
        return query

    normalized = query
    company_id = metadata_filter.get("company_id")
    if company_id:
        company_ids = company_id if isinstance(company_id, dict) and "$in" in company_id else [company_id]
        if isinstance(company_ids, dict):
            company_ids = company_ids.get("$in", [])
        for cid in company_ids:
            info = get_companies().get(cid)
            if info:
                for alias in info.get("aliases", []):
                    normalized = re.sub(re.escape(alias), " ", normalized, flags=re.I)

    report_period = metadata_filter.get("report_period", "")
    if isinstance(report_period, str):
        year_match = re.match(r"FY(20\d{2})", report_period)
        if year_match:
            normalized = re.sub(rf"{year_match.group(1)}\s*年?", " ", normalized)
    elif isinstance(report_period, dict) and "$in" in report_period:
        for period in report_period["$in"]:
            year_match = re.match(r"FY(20\d{2})", period)
            if year_match:
                normalized = re.sub(rf"{year_match.group(1)}\s*年?", " ", normalized)

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or query


def augment_query_for_retrieval(query: str, metadata_filter: dict | None = None) -> str:
    """加法式查询增强 — Dense通道专用。补充公司全称+行业语义+指标同义词。"""
    augmented = query

    # ── 补充公司全称 ──
    company_id = metadata_filter.get("company_id") if metadata_filter else None
    if company_id:
        company_ids = [company_id] if isinstance(company_id, str) else company_id.get("$in", []) if isinstance(company_id, dict) else [company_id]
        for cid in company_ids:
            info = get_companies().get(cid)
            if info:
                aliases = info.get("aliases", [])
                # 如果query只有简称，补充全称（最长的alias通常是全称）
                for alias in aliases:
                    if alias.lower() in query.lower() and len(alias) < len(aliases[0]):
                        augmented = augmented + " " + aliases[0]
                        break

    # ── 补充行业语义 ──
    industry_companies = _match_industry(query)
    if industry_companies and not company_id:
        for cid in industry_companies:
            info = get_companies().get(cid)
            if info:
                augmented = augmented + " " + info.get("industry", "")

    # ── 补充财务指标同义扩展（从knowledge YAML读取）──
    from knowledge import resolve_metrics_from_query, get_metrics
    matched_metrics = resolve_metrics_from_query(query)
    for metric_name in matched_metrics:
        info = get_metrics().get(metric_name, {})
        synonyms = info.get("synonyms", [])
        for syn in synonyms:
            if syn not in augmented:
                augmented = augmented + " " + syn

    augmented = re.sub(r"\s+", " ", augmented).strip()
    return augmented


# ──────────────────────────────────────────────
# 3. 元数据匹配：支持多值过滤（$in / $or）
# ──────────────────────────────────────────────

def matches_metadata_filter(metadata: dict, metadata_filter: dict | None) -> bool:
    """检查一条chunk的metadata是否满足过滤条件。支持单值(str)和多值($in)。"""
    if not metadata_filter:
        return True

    for key, condition in metadata_filter.items():
        actual_value = str(metadata.get(key, ""))

        if isinstance(condition, str):
            if actual_value != condition:
                return False
        elif isinstance(condition, dict):
            if "$in" in condition:
                if actual_value not in [str(v) for v in condition["$in"]]:
                    return False
            elif "$or" in condition:
                if actual_value not in [str(v) for v in condition["$or"]]:
                    return False
        else:
            if actual_value != str(condition):
                return False

    return True


def to_chroma_filter(metadata_filter: dict | None) -> dict | None:
    """将metadata_filter转换为Chroma兼容的过滤表达式。支持$in多值和$and组合。"""
    if not metadata_filter:
        return None

    conditions: list[dict] = []
    for key, value in metadata_filter.items():
        if isinstance(value, dict):
            conditions.append({key: value})
        else:
            conditions.append({key: value})

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}