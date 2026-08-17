"""Per-run backtest output (spec §6: "trade log, equity curve, and full metrics
(§7.4)"). Deflated Sharpe and out-of-sample framing belong to §7's validation harness
(they need to know the number of trials tested, and which window is in/out of sample) —
not computed here; this module is the honest per-run arithmetic those checks build on.

Sharpe/Sortino are reported BOTH per-trade and annualized, with the annualization factor
(trades/year) stated explicitly — spec §7.4 flags this as "where numbers get quietly
inflated 3-4x" when the factor is hidden or assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backtest.engine import TradeResult


@dataclass(frozen=True)
class EquityPoint:
    time: pd.Timestamp
    equity: float


@dataclass(frozen=True)
class DrawdownStats:
    max_drawdown_pct: float
    max_drawdown_amount: float
    max_drawdown_duration: pd.Timedelta


@dataclass(frozen=True)
class BacktestMetrics:
    trade_count: int
    unfilled_count: int
    win_rate: float
    expectancy_pnl: float
    expectancy_r: float
    profit_factor: float
    sharpe_per_trade: float
    trades_per_year: float          # the annualization factor, stated explicitly
    sharpe_annualized: float
    sortino_per_trade: float
    sortino_annualized: float
    drawdown: DrawdownStats
    exposure_fraction: float
    spread_pct_of_stop_median: Optional[float]  # spec §6 cheap TF-viability signal


def build_equity_curve(trades: list[TradeResult], starting_equity: float) -> list[EquityPoint]:
    filled = sorted((t for t in trades if t.entry_bar_index is not None), key=lambda t: t.exit_time)
    curve = [EquityPoint(time=filled[0].entry_time if filled else pd.NaT, equity=starting_equity)]
    equity = starting_equity
    for t in filled:
        equity += t.net_pnl
        curve.append(EquityPoint(time=t.exit_time, equity=equity))
    return curve


def _drawdown_stats(curve: list[EquityPoint]) -> DrawdownStats:
    if len(curve) < 2:
        return DrawdownStats(0.0, 0.0, pd.Timedelta(0))
    peak, peak_time = curve[0].equity, curve[0].time
    max_dd_amount, max_dd_pct, max_dd_duration = 0.0, 0.0, pd.Timedelta(0)
    for point in curve[1:]:
        if point.equity > peak:
            peak, peak_time = point.equity, point.time
            continue
        dd_amount = peak - point.equity
        dd_pct = dd_amount / peak if peak > 0 else 0.0
        duration = point.time - peak_time
        if dd_amount > max_dd_amount:
            max_dd_amount, max_dd_pct = dd_amount, dd_pct
        if duration > max_dd_duration:
            max_dd_duration = duration
    return DrawdownStats(max_dd_pct, max_dd_amount, max_dd_duration)


def compute_metrics(
    trades: list[TradeResult],
    starting_equity: float,
    data_start_time: pd.Timestamp,
    data_end_time: pd.Timestamp,
) -> BacktestMetrics:
    filled = [t for t in trades if t.entry_bar_index is not None]
    unfilled_count = len(trades) - len(filled)
    trade_count = len(filled)

    if trade_count == 0:
        return BacktestMetrics(
            trade_count=0, unfilled_count=unfilled_count, win_rate=float("nan"),
            expectancy_pnl=float("nan"), expectancy_r=float("nan"), profit_factor=float("nan"),
            sharpe_per_trade=float("nan"), trades_per_year=float("nan"), sharpe_annualized=float("nan"),
            sortino_per_trade=float("nan"), sortino_annualized=float("nan"),
            drawdown=DrawdownStats(0.0, 0.0, pd.Timedelta(0)), exposure_fraction=float("nan"),
            spread_pct_of_stop_median=None,
        )

    pnls = [t.net_pnl for t in filled]
    r_multiples = [t.r_multiple for t in filled if t.r_multiple is not None]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / trade_count
    expectancy_pnl = sum(pnls) / trade_count
    expectancy_r = sum(r_multiples) / len(r_multiples) if r_multiples else float("nan")
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else float("nan"))

    if len(r_multiples) >= 2:
        mean_r = sum(r_multiples) / len(r_multiples)
        variance = sum((r - mean_r) ** 2 for r in r_multiples) / (len(r_multiples) - 1)
        std_r = math.sqrt(variance)
        sharpe_per_trade = mean_r / std_r if std_r > 0 else float("nan")

        downside = [min(r, 0.0) for r in r_multiples]
        downside_dev = math.sqrt(sum(d ** 2 for d in downside) / len(downside))
        sortino_per_trade = mean_r / downside_dev if downside_dev > 0 else float("nan")
    else:
        sharpe_per_trade = float("nan")
        sortino_per_trade = float("nan")

    span_days = max((data_end_time - data_start_time).total_seconds() / 86400.0, 1e-9)
    trades_per_year = trade_count / (span_days / 365.25)
    annualization = math.sqrt(trades_per_year) if trades_per_year > 0 else float("nan")
    sharpe_annualized = sharpe_per_trade * annualization if not math.isnan(sharpe_per_trade) else float("nan")
    sortino_annualized = sortino_per_trade * annualization if not math.isnan(sortino_per_trade) else float("nan")

    curve = build_equity_curve(filled, starting_equity)
    drawdown = _drawdown_stats(curve)

    total_open_time = sum((t.exit_time - t.entry_time for t in filled), pd.Timedelta(0))
    total_span = data_end_time - data_start_time
    exposure_fraction = total_open_time / total_span if total_span > pd.Timedelta(0) else float("nan")

    spread_pcts = [t.spread_pct_of_stop for t in filled if t.spread_pct_of_stop is not None]
    spread_pct_median = pd.Series(spread_pcts).median() if spread_pcts else None

    return BacktestMetrics(
        trade_count=trade_count, unfilled_count=unfilled_count, win_rate=win_rate,
        expectancy_pnl=expectancy_pnl, expectancy_r=expectancy_r, profit_factor=profit_factor,
        sharpe_per_trade=sharpe_per_trade, trades_per_year=trades_per_year,
        sharpe_annualized=sharpe_annualized, sortino_per_trade=sortino_per_trade,
        sortino_annualized=sortino_annualized, drawdown=drawdown,
        exposure_fraction=exposure_fraction, spread_pct_of_stop_median=spread_pct_median,
    )
