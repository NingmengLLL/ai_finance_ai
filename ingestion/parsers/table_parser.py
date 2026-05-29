from __future__ import annotations

from typing import Any


def table_to_text(rows: list[dict[str, Any]], table_name: str = "financial_table") -> str:
    if not rows:
        return f"表格 {table_name} 为空。"

    headers = list(rows[0].keys())
    lines = [f"表格名称：{table_name}", f"字段：{', '.join(headers)}"]
    for idx, row in enumerate(rows, start=1):
        cells = [f"{key}={row.get(key, '')}" for key in headers]
        lines.append(f"第{idx}行：" + "；".join(cells))
    return "\n".join(lines)


def normalize_financial_metric(value: str) -> float | None:
    text = str(value).replace(",", "").replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return None
