from __future__ import annotations


def search_financial_news(query: str) -> list[dict]:
    return [
        {
            "title": "未配置实时新闻API",
            "body": f"当前运行环境未启用外部金融新闻搜索，查询为：{query}",
            "source": "local_placeholder",
        }
    ]
