from __future__ import annotations


def growth_rate(new_value: float, old_value: float) -> float:
    return (new_value - old_value) / old_value * 100 if old_value else 0.0


def gross_margin(gross_profit: float, revenue: float) -> float:
    return gross_profit / revenue * 100 if revenue else 0.0


def pe_ratio(market_cap: float, net_profit: float) -> float | None:
    return market_cap / net_profit if net_profit else None
