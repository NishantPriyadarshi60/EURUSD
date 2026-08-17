import math

import pandas as pd
import pytest

from backtest.engine import TradeResult
from backtest.costs import TradeCosts
from backtest.metrics import compute_metrics, build_equity_curve


def _trade(entry_time, exit_time, net_pnl, r_multiple, spread_pct=None, gross_pnl=None):
    return TradeResult(
        signal_bar_index=0, entry_bar_index=1, entry_time=entry_time, entry_price=1.1,
        exit_bar_index=2, exit_time=exit_time, exit_price=1.1, exit_reason="target",
        direction="long", lots=1.0, stop_distance_price=0.002,
        gross_pnl=gross_pnl if gross_pnl is not None else net_pnl,
        costs=TradeCosts(0, 0, 0, 0), net_pnl=net_pnl, r_multiple=r_multiple,
        spread_at_entry_points=2.0, spread_pct_of_stop=spread_pct,
    )


def _unfilled():
    return TradeResult(
        signal_bar_index=0, entry_bar_index=None, entry_time=None, entry_price=None,
        exit_bar_index=None, exit_time=None, exit_price=None, exit_reason="expired_unfilled",
        direction="long", lots=0.0, stop_distance_price=0.0, gross_pnl=0.0,
        costs=TradeCosts(0, 0, 0, 0), net_pnl=0.0, r_multiple=None,
        spread_at_entry_points=None, spread_pct_of_stop=None,
    )


def _four_trade_scenario():
    start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    pnls = [200, 200, 200, -100]
    rs = [2.0, 2.0, 2.0, -1.0]
    spreads = [0.05, 0.10, 0.15, 0.20]
    trades = []
    for i in range(4):
        entry = start + pd.Timedelta(days=i)
        exit_ = entry + pd.Timedelta(hours=1)
        trades.append(_trade(entry, exit_, pnls[i], rs[i], spread_pct=spreads[i]))
    data_start = start
    data_end = start + pd.Timedelta(days=365.25)
    return trades, data_start, data_end


def test_win_rate_expectancy_and_profit_factor():
    trades, data_start, data_end = _four_trade_scenario()
    m = compute_metrics(trades, starting_equity=10000, data_start_time=data_start, data_end_time=data_end)
    assert m.trade_count == 4
    assert m.unfilled_count == 0
    assert m.win_rate == 0.75
    assert m.expectancy_pnl == 125.0
    assert m.expectancy_r == 1.25
    assert m.profit_factor == 6.0


def test_sharpe_and_sortino_hand_verified():
    # r = [2, 2, 2, -1]: mean=1.25, sample std=1.5 -> sharpe=0.8333...
    # downside = [0,0,0,-1] -> downside_dev = sqrt(1/4) = 0.5 -> sortino = 2.5
    trades, data_start, data_end = _four_trade_scenario()
    m = compute_metrics(trades, starting_equity=10000, data_start_time=data_start, data_end_time=data_end)
    assert round(m.sharpe_per_trade, 4) == round(1.25 / 1.5, 4)
    assert round(m.sortino_per_trade, 4) == 2.5


def test_annualization_factor_stated_and_applied_explicitly():
    # span = 365.25 days, 4 trades -> trades_per_year = 4 -> sqrt(4) = 2x multiplier
    trades, data_start, data_end = _four_trade_scenario()
    m = compute_metrics(trades, starting_equity=10000, data_start_time=data_start, data_end_time=data_end)
    assert round(m.trades_per_year, 4) == 4.0
    assert round(m.sharpe_annualized, 4) == round(m.sharpe_per_trade * 2, 4)
    assert round(m.sortino_annualized, 4) == round(m.sortino_per_trade * 2, 4)


def test_max_drawdown_amount_pct_and_duration():
    # equity: 10000 -> 10200 -> 10400 -> 10600 (peak) -> 10500 (drawdown of 100)
    trades, data_start, data_end = _four_trade_scenario()
    m = compute_metrics(trades, starting_equity=10000, data_start_time=data_start, data_end_time=data_end)
    assert round(m.drawdown.max_drawdown_amount, 6) == 100.0
    assert round(m.drawdown.max_drawdown_pct, 6) == round(100 / 10600, 6)
    assert m.drawdown.max_drawdown_duration == pd.Timedelta(days=1)


def test_exposure_fraction():
    trades, data_start, data_end = _four_trade_scenario()
    m = compute_metrics(trades, starting_equity=10000, data_start_time=data_start, data_end_time=data_end)
    expected = pd.Timedelta(hours=4) / (data_end - data_start)
    assert round(m.exposure_fraction, 8) == round(expected, 8)


def test_spread_pct_of_stop_median():
    trades, data_start, data_end = _four_trade_scenario()
    m = compute_metrics(trades, starting_equity=10000, data_start_time=data_start, data_end_time=data_end)
    assert round(m.spread_pct_of_stop_median, 4) == 0.125  # median of [0.05,0.10,0.15,0.20]


def test_unfilled_trades_excluded_from_trade_count_but_tracked_separately():
    trades, data_start, data_end = _four_trade_scenario()
    trades_with_unfilled = trades + [_unfilled(), _unfilled()]
    m = compute_metrics(trades_with_unfilled, starting_equity=10000, data_start_time=data_start, data_end_time=data_end)
    assert m.trade_count == 4
    assert m.unfilled_count == 2


def test_zero_filled_trades_is_nan_safe_not_a_crash():
    data_start = pd.Timestamp("2024-01-01", tz="UTC")
    data_end = pd.Timestamp("2024-06-01", tz="UTC")
    m = compute_metrics([_unfilled(), _unfilled()], starting_equity=10000,
                         data_start_time=data_start, data_end_time=data_end)
    assert m.trade_count == 0
    assert m.unfilled_count == 2
    assert math.isnan(m.win_rate)
    assert math.isnan(m.sharpe_per_trade)
    assert m.spread_pct_of_stop_median is None


def test_profit_factor_is_infinite_with_no_losses():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    trades = [_trade(start, start + pd.Timedelta(hours=1), 100.0, 1.0)]
    m = compute_metrics(trades, starting_equity=10000, data_start_time=start, data_end_time=start + pd.Timedelta(days=30))
    assert m.profit_factor == float("inf")


def test_equity_curve_starts_at_starting_equity_and_compounds():
    trades, data_start, data_end = _four_trade_scenario()
    curve = build_equity_curve(trades, starting_equity=10000)
    assert curve[0].equity == 10000
    assert curve[-1].equity == 10500  # 10000 + 200 + 200 + 200 - 100
