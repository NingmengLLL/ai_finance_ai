from __future__ import annotations


def normalize_ticker(ticker: str) -> str:
    text = ticker.strip().upper()
    if len(text) == 6 and text.startswith(("0", "3")):
        return f"{text}.SZ"
    if len(text) == 6 and text.startswith("6"):
        return f"{text}.SH"
    return text
