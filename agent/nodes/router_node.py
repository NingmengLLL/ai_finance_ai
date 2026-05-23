from __future__ import annotations


# ── LLM意图分类+实体抽取（合并为一次调用）──

INTENT_ENTITY_PROMPT = """你是一个金融研报分析系统的意图分类与实体抽取模块。
请对以下用户查询进行分析，输出严格的JSON（不要Markdown，不要额外解释）。

分析要求：
1. intent：从以下4种意图中选择最匹配的一种
   - realtime_financial_search：查询实时数据（股价、公告、最新新闻等）
   - calculation：需要数值计算（增长率、市盈率、同比、CAGR等）
   - graph_reasoning：涉及关系推理（上下游、供应商客户、产业链影响等）
   - financial_analysis：默认的金融分析（基本面、行业分析等）

2. confidence：意图判断的置信度（0.0-1.0）

3. entities：从查询中抽取的实体
   - companies：涉及的公司名（中文+英文都要列出）
   - metrics：涉及的金融指标术语
   - years：涉及的年份
   - doc_type：文档类型推断（annual_report / quarterly_results / research_report / unknown）

用户查询：
{query}

输出JSON格式如下（注意：所有花括号为JSON语法，不是占位符）：
{{"intent": "...", "confidence": 0.0-1.0, "entities": {{\"companies\": [...], \"metrics\": [...], \"years\": [...], \"doc_type\": \"...\"}}}}"""


