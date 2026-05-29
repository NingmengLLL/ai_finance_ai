from __future__ import annotations


def describe_chart_placeholder(file_name: str, caption: str = "") -> str:
    suffix = f" 图注：{caption}" if caption else ""
    return f"图表 {file_name} 需要视觉模型识别趋势、坐标轴、单位和异常点。{suffix}"
