"""Sequential backtest orchestrator (spec §6) — ties position sizing, the fill/cost
engine, and hard risk limits together over a list of trade signals.

Scope limitation, stated rather than faked: this only supports
max_concurrent_positions == 1 (one trade open at a time, fully resolved before the next
candidate signal is even considered). True concurrent-position equity accounting — where
sizing for a new trade depends on other still-open trades' eventual (not yet known)
outcomes — is a materially harder simulation problem and is not implemented. Asking for
more than 1 raises rather than silently producing an incorrect equity curve.

Signals are resolved in signal_bar_index order, one fully to completion (fill/expiry and
exit) before the next is evaluated — including waiting out a pending limit order. This is
a simplification (a live system would keep evaluating new signals while one is pending),
documented so it's visible if/when Phase 2's strategy layer needs something richer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtest.costs import CostConfig
from backtest.engine import TradeSignal, TradeResult, simulate_trade
from data.symbol_info import SymbolMeta
from risk.limits import RiskLimitsConfig, RiskManager


@dataclass(frozen=True)
class SkippedSignal:
    signal_bar_index: int
    reason: str   # "overlap" | "risk_halted"


@dataclass(frozen=True)
class BacktestRun:
    """Every signal decision is accounted for here (spec §0.6: log every signal decision
    with the inputs behind it) — trades includes expired_unfilled entries, and skipped
    covers signals that never even reached simulation (overlap or a risk halt)."""
    trades: list[TradeResult] = field(default_factory=list)
    skipped: list[SkippedSignal] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    signals: list[TradeSignal],
    starting_equity: float,
    risk_fraction: float,
    symbol_meta: SymbolMeta,
    cost_cfg: CostConfig,
    atr: pd.Series,
    limits_cfg: RiskLimitsConfig,
) -> BacktestRun:
    if limits_cfg.max_concurrent_positions != 1:
        raise NotImplementedError(
            "run_backtest only supports max_concurrent_positions=1 -- true concurrent-"
            "position equity accounting is not implemented (see module docstring)."
        )

    manager = RiskManager(config=limits_cfg, starting_equity=starting_equity)
    times = df["time"].to_numpy()
    run = BacktestRun()

    ordered_signals = sorted(signals, key=lambda s: s.signal_bar_index)
    next_available_bar = 0

    for signal in ordered_signals:
        if signal.signal_bar_index < next_available_bar:
            run.skipped.append(SkippedSignal(signal.signal_bar_index, "overlap"))
            continue

        signal_time = pd.Timestamp(times[signal.signal_bar_index])
        if not manager.can_open(signal_time):
            run.skipped.append(SkippedSignal(signal.signal_bar_index, "risk_halted"))
            continue

        result = simulate_trade(df, signal, manager.equity, risk_fraction, symbol_meta, cost_cfg, atr)
        run.trades.append(result)

        if result.entry_bar_index is None:
            continue  # expired unfilled -- no position was ever opened, nothing to register

        manager.register_open()
        manager.register_close(result.exit_time, result.net_pnl)
        next_available_bar = result.exit_bar_index + 1

    return run
