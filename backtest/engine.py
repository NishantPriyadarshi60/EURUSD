"""Backtest execution engine (spec §6) — simulates a single trade signal to its
resolution against real bar data, with realistic fills and costs.

Intrabar sequencing: on any bar where both stop and target fall inside that bar's
[low, high] range, a bar-only backtest cannot know which was hit first. Per spec §6's
documented fallback (used here because this account's M1 history is too thin for true
intrabar stepping — see memory: validation-date-split), we assume the WORSE outcome on
a same-bar collision: the stop, for either direction. This is deliberately pessimistic
by construction, not a coin flip, so it can never flatter results.

Fill realism (§6): entry_mode="limit" requires a genuine trade-through of the limit
price (a strict inequality, not a mere touch) before considering it filled, modeling
that a limit touch is not a guaranteed fill. entry_mode="market" fills unconditionally
at the signal's own specified price on the signal bar.

Cost accounting is entirely in account-currency deductions from gross P&L (spread,
commission, slippage, swap — see backtest/costs.py) rather than price-path adjustments,
so nothing here double-charges a cost that's also baked into an adjusted fill price.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backtest.costs import (
    CostConfig, TradeCosts, spread_cost, commission_cost, slippage_cost,
    is_spike_bar, count_swap_nights, swap_cost,
)
from data.symbol_info import SymbolMeta
from risk.position_sizing import compute_position_size


@dataclass(frozen=True)
class TradeSignal:
    signal_bar_index: int          # bar whose close generated this signal
    direction: str                   # "long" | "short"
    entry_mode: str                    # "market" | "limit"
    entry_price: float                   # market: filled here, on signal_bar_index.
                                           # limit: price to wait for a trade-through of.
    stop_price: float
    target_price: float
    limit_expiry_bars: Optional[int] = None  # None = wait until data ends


@dataclass(frozen=True)
class TradeResult:
    signal_bar_index: int
    entry_bar_index: Optional[int]        # None if a limit order never filled
    entry_time: Optional[pd.Timestamp]
    entry_price: Optional[float]
    exit_bar_index: Optional[int]
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    exit_reason: str                        # "stop" | "target" | "end_of_data" | "expired_unfilled"
    direction: str
    lots: float
    stop_distance_price: float
    gross_pnl: float
    costs: TradeCosts
    net_pnl: float
    r_multiple: Optional[float]               # net_pnl / risk_amount; None if unfilled
    spread_at_entry_points: Optional[float]
    spread_pct_of_stop: Optional[float]         # spec §6 cheap-viability signal


def _unfilled_result(signal: TradeSignal, reason: str) -> TradeResult:
    return TradeResult(
        signal_bar_index=signal.signal_bar_index, entry_bar_index=None, entry_time=None,
        entry_price=None, exit_bar_index=None, exit_time=None, exit_price=None,
        exit_reason=reason, direction=signal.direction, lots=0.0, stop_distance_price=0.0,
        gross_pnl=0.0, costs=TradeCosts(0.0, 0.0, 0.0, 0.0), net_pnl=0.0, r_multiple=None,
        spread_at_entry_points=None, spread_pct_of_stop=None,
    )


def simulate_trade(
    df: pd.DataFrame,
    signal: TradeSignal,
    equity: float,
    risk_fraction: float,
    symbol_meta: SymbolMeta,
    cost_cfg: CostConfig,
    atr: pd.Series,
) -> TradeResult:
    """df needs time/open/high/low/close/spread, positionally indexed 0..len-1. atr
    aligned to the same index (primitives.atr.compute_atr)."""
    n_bars = len(df)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    spreads = df["spread"].to_numpy()
    times = df["time"].to_numpy()

    is_long = signal.direction == "long"

    if signal.entry_mode == "market":
        entry_bar = signal.signal_bar_index
        if entry_bar >= n_bars:
            return _unfilled_result(signal, "end_of_data")
        is_market_entry = True
    elif signal.entry_mode == "limit":
        start = signal.signal_bar_index + 1
        end = n_bars if signal.limit_expiry_bars is None else min(start + signal.limit_expiry_bars, n_bars)
        entry_bar = None
        for j in range(start, end):
            traded_through = (lows[j] < signal.entry_price) if is_long else (highs[j] > signal.entry_price)
            if traded_through:
                entry_bar = j
                break
        if entry_bar is None:
            return _unfilled_result(signal, "expired_unfilled")
        is_market_entry = False
    else:
        raise ValueError(f"Unknown entry_mode: {signal.entry_mode!r}")

    entry_price = signal.entry_price
    entry_time = pd.Timestamp(times[entry_bar])
    stop_distance_price = abs(entry_price - signal.stop_price)

    size = compute_position_size(
        equity=equity, risk_fraction=risk_fraction, stop_distance_price=stop_distance_price,
        pip_value_per_lot=symbol_meta.pip_value_per_lot, pip_size=symbol_meta.pip_size,
        volume_min=symbol_meta.volume_min, volume_max=symbol_meta.volume_max,
        volume_step=symbol_meta.volume_step,
    )

    exit_bar, exit_price, exit_reason = None, None, "end_of_data"
    for j in range(entry_bar + 1, n_bars):
        hit_stop = (lows[j] <= signal.stop_price) if is_long else (highs[j] >= signal.stop_price)
        hit_target = (highs[j] >= signal.target_price) if is_long else (lows[j] <= signal.target_price)
        if hit_stop:  # conservative: stop wins any same-bar collision with target
            exit_bar, exit_price, exit_reason = j, signal.stop_price, "stop"
            break
        if hit_target:
            exit_bar, exit_price, exit_reason = j, signal.target_price, "target"
            break
    if exit_bar is None:
        exit_bar = n_bars - 1
        exit_price = float(closes[exit_bar])
        exit_reason = "end_of_data"
    exit_time = pd.Timestamp(times[exit_bar])

    spread_at_entry_points = float(spreads[entry_bar])
    s_cost = spread_cost(spread_at_entry_points, symbol_meta.point, symbol_meta.pip_size,
                          symbol_meta.pip_value_per_lot, size.lots)
    c_cost = commission_cost(size.lots, cost_cfg)

    exit_bar_range = float(highs[exit_bar] - lows[exit_bar])
    atr_at_exit = float(atr.iloc[exit_bar]) if exit_bar < len(atr) else float("nan")
    spike = is_spike_bar(exit_bar_range, atr_at_exit, cost_cfg)
    sl_cost = slippage_cost(
        size.lots, symbol_meta.pip_size, symbol_meta.pip_value_per_lot, cost_cfg,
        is_spike=spike, is_stop_exit=(exit_reason == "stop"), is_target_exit=(exit_reason == "target"),
        is_market_entry=is_market_entry,
    )

    nights = count_swap_nights(entry_time, exit_time, cost_cfg)
    sw_cost = swap_cost(signal.direction, nights, symbol_meta.point, symbol_meta.pip_size,
                         symbol_meta.pip_value_per_lot, size.lots, symbol_meta.swap_long, symbol_meta.swap_short)

    costs = TradeCosts(spread_cost=s_cost, commission_cost=c_cost, slippage_cost=sl_cost, swap_cost=sw_cost)

    price_diff = (exit_price - entry_price) if is_long else (entry_price - exit_price)
    gross_pnl = (price_diff / symbol_meta.pip_size) * symbol_meta.pip_value_per_lot * size.lots
    net_pnl = gross_pnl - costs.total
    r_multiple = net_pnl / size.risk_amount if size.risk_amount > 0 else None
    spread_pct_of_stop = (
        (spread_at_entry_points * symbol_meta.point) / stop_distance_price if stop_distance_price > 0 else None
    )

    return TradeResult(
        signal_bar_index=signal.signal_bar_index, entry_bar_index=entry_bar, entry_time=entry_time,
        entry_price=entry_price, exit_bar_index=exit_bar, exit_time=exit_time, exit_price=exit_price,
        exit_reason=exit_reason, direction=signal.direction, lots=size.lots,
        stop_distance_price=stop_distance_price, gross_pnl=gross_pnl, costs=costs, net_pnl=net_pnl,
        r_multiple=r_multiple, spread_at_entry_points=spread_at_entry_points,
        spread_pct_of_stop=spread_pct_of_stop,
    )
