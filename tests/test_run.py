import pandas as pd
import pytest

from backtest.engine import TradeSignal
from backtest.run import run_backtest
from backtest.costs import CostConfig
from risk.limits import RiskLimitsConfig
from data.symbol_info import SymbolMeta


def _symbol_meta():
    return SymbolMeta(
        symbol="EURUSD", point=0.00001, digits=5, trade_tick_size=0.00001,
        trade_tick_value=1.0, trade_contract_size=100000.0, pip_size=0.0001,
        pip_value_per_lot=10.0, volume_min=0.01, volume_max=500.0, volume_step=0.01,
        swap_long=-0.7, swap_short=-1.0,
    )


def _cost_cfg():
    return CostConfig(
        commission_per_lot_round_trip=7.0, slippage_base_pips=0.2,
        slippage_spike_atr_multiple=2.0, slippage_spike_extra_pips=1.0,
        slippage_apply_to_target_exits=False, slippage_apply_to_market_entries=True,
        swap_rollover_hour_utc=21, swap_triple_swap_weekday=2,
    )


def _limits(max_concurrent=1):
    return RiskLimitsConfig(max_daily_loss_fraction=0.5, max_weekly_loss_fraction=0.5,
                             max_concurrent_positions=max_concurrent)


def _df(rows, spreads=None):
    n = len(rows)
    opens, highs, lows, closes = zip(*rows)
    times = pd.date_range("2024-01-02 10:00", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "time": times, "open": opens, "high": highs, "low": lows, "close": closes,
        "spread": spreads if spreads is not None else [2] * n,
    })


def _flat_atr(n, value=0.001):
    return pd.Series([value] * n)


def test_two_non_overlapping_trades_both_execute_and_equity_compounds():
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),   # 0: signal 1 entry
        (1.1000, 1.1030, 1.0998, 1.1020),    # 1: signal 1 target hit
        (1.1020, 1.1025, 1.1015, 1.1020),     # 2: signal 2 entry
        (1.1020, 1.1050, 1.1018, 1.1040),      # 3: signal 2 target hit
    ]
    df = _df(rows)
    signals = [
        TradeSignal(0, "long", "market", 1.1000, 1.0980, 1.1020),
        TradeSignal(2, "long", "market", 1.1020, 1.1000, 1.1040),
    ]
    run = run_backtest(df, signals, starting_equity=10000, risk_fraction=0.01,
                        symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(len(rows)),
                        limits_cfg=_limits())
    assert len(run.trades) == 2
    assert run.skipped == []
    assert all(t.exit_reason == "target" for t in run.trades)
    assert all(t.net_pnl > 0 for t in run.trades)


def test_overlapping_signal_is_skipped_with_reason():
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),   # 0: signal 1 entry
        (1.1000, 1.1008, 1.0997, 1.1003),    # 1: no touch yet
        (1.1003, 1.1030, 1.1000, 1.1020),     # 2: signal 1 target hit; signal 2 tries to enter here too
        (1.1020, 1.1040, 1.1010, 1.1030),      # 3
    ]
    df = _df(rows)
    signals = [
        TradeSignal(0, "long", "market", 1.1000, 1.0980, 1.1020),
        TradeSignal(2, "long", "market", 1.1020, 1.1000, 1.1050),  # overlaps trade 1 (still open through bar 2)
    ]
    run = run_backtest(df, signals, starting_equity=10000, risk_fraction=0.01,
                        symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(len(rows)),
                        limits_cfg=_limits())
    assert len(run.trades) == 1
    assert len(run.skipped) == 1
    assert run.skipped[0].signal_bar_index == 2
    assert run.skipped[0].reason == "overlap"


def test_risk_halt_skips_subsequent_signal_same_day():
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),   # 0: signal 1 entry
        (1.0940, 1.0945, 1.0935, 1.0940),    # 1: big stop-out (deliberately breaches daily cap)
        (1.0940, 1.0945, 1.0935, 1.0940),     # 2: signal 2 would enter here, same day
        (1.0940, 1.0970, 1.0930, 1.0960),      # 3
    ]
    df = _df(rows)
    signals = [
        TradeSignal(0, "long", "market", 1.1000, 1.0940, 1.1100),
        TradeSignal(2, "long", "market", 1.0940, 1.0900, 1.1000),
    ]
    tight_limits = RiskLimitsConfig(max_daily_loss_fraction=0.01, max_weekly_loss_fraction=0.5, max_concurrent_positions=1)
    run = run_backtest(df, signals, starting_equity=10000, risk_fraction=0.05,  # large risk_fraction -> big loss
                        symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(len(rows)),
                        limits_cfg=tight_limits)
    assert len(run.trades) == 1  # only the first (losing) trade executed
    assert run.trades[0].net_pnl < 0
    assert len(run.skipped) == 1
    assert run.skipped[0].reason == "risk_halted"


def test_max_concurrent_positions_other_than_one_raises():
    df = _df([(1.1, 1.1, 1.1, 1.1)])
    with pytest.raises(NotImplementedError):
        run_backtest(df, [], starting_equity=10000, risk_fraction=0.01,
                     symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(1),
                     limits_cfg=_limits(max_concurrent=2))
