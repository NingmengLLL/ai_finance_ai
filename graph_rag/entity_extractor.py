from __future__ import annotations

import re

from knowledge import get_all_metric_synonyms_flat, get_companies


def extract_entities(text: str) -> dict[str, list[str]]:
    """实体提取 — 指标词表从 knowledge YAML 读取，不再硬编码9个词。
    公司名仍用regex提取（含后缀模式），后续可升级为NER模型。"""
    # 公司：regex提取含后缀的名称
    companies = sorted(
        set(re.findall(r"[\u4e00-\u9fffA-Za-z]{2,20}(?:科技|股份|银行|证券|集团|公司)", text))
    )
    tickers = sorted(set(re.findall(r"\b\d{6}\.(?:SZ|SH|BJ|HK)\b", text.upper())))

    # 指标：从knowledge YAML读取全部指标名+同义词
    metric_terms = get_all_metric_synonyms_flat()
    metrics = sorted({term for term in metric_terms if term in text})

    years = sorted(set(re.findall(r"20\d{2}年?", text)))
    return {"companies": companies, "tickers": tickers, "metrics": metrics, "years": years}