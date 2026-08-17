import pandas as pd
import pytest

from backtest.engine import TradeSignal, simulate_trade
from backtest.costs import CostConfig
from data.symbol_info import SymbolMeta


def _symbol_meta(**overrides):
    defaults = dict(
        symbol="EURUSD", point=0.00001, digits=5, trade_tick_size=0.00001,
        trade_tick_value=1.0, trade_contract_size=100000.0, pip_size=0.0001,
        pip_value_per_lot=10.0, volume_min=0.01, volume_max=500.0, volume_step=0.01,
        swap_long=-0.7, swap_short=-1.0,
    )
    defaults.update(overrides)
    return SymbolMeta(**defaults)


def _cost_cfg(**overrides):
    defaults = dict(
        commission_per_lot_round_trip=7.0, slippage_base_pips=0.2,
        slippage_spike_atr_multiple=2.0, slippage_spike_extra_pips=1.0,
        slippage_apply_to_target_exits=False, slippage_apply_to_market_entries=True,
        swap_rollover_hour_utc=21, swap_triple_swap_weekday=2,
    )
    defaults.update(overrides)
    return CostConfig(**defaults)


def _df(rows, spreads=None):
    """rows: list of (open, high, low, close). Bars 15 minutes apart, no overnight span
    (so swap defaults to 0 nights) unless a test needs otherwise."""
    n = len(rows)
    opens, highs, lows, closes = zip(*rows)
    times = pd.date_range("2024-01-02 10:00", periods=n, freq="15min", tz="UTC")  # a Tuesday
    return pd.DataFrame({
        "time": times, "open": opens, "high": highs, "low": lows, "close": closes,
        "spread": spreads if spreads is not None else [2] * n,
    })


def _flat_atr(n, value=0.001):
    return pd.Series([value] * n)


def test_market_long_hits_target_cleanly():
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),   # 0: signal/entry bar
        (1.1000, 1.1010, 1.0998, 1.1005),    # 1: no touch
        (1.1005, 1.1030, 1.1003, 1.1020),     # 2: target (1.1020) touched
    ]
    df = _df(rows)
    signal = TradeSignal(signal_bar_index=0, direction="long", entry_mode="market",
                          entry_price=1.1000, stop_price=1.0980, target_price=1.1020)
    result = simulate_trade(df, signal, equity=10000, risk_fraction=0.01,
                             symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(len(rows)))
    assert result.entry_bar_index == 0
    assert result.exit_bar_index == 2
    assert result.exit_reason == "target"
    assert result.exit_price == 1.1020
    assert result.net_pnl > 0


def test_market_long_hits_stop_cleanly():
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),
        (1.1000, 1.1010, 1.0998, 1.1005),
        (1.0985, 1.0990, 1.0975, 1.0980),   # 2: stop (1.0980) touched, target untouched
    ]
    df = _df(rows)
    signal = TradeSignal(signal_bar_index=0, direction="long", entry_mode="market",
                          entry_price=1.1000, stop_price=1.0980, target_price=1.1020)
    result = simulate_trade(df, signal, equity=10000, risk_fraction=0.01,
                             symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(len(rows)))
    assert result.exit_reason == "stop"
    assert result.exit_price == 1.0980
    assert result.net_pnl < 0


def test_same_bar_collision_conservatively_resolves_to_stop():
    # Bar 1's range contains BOTH the stop (1.0980) and the target (1.1020) -- a
    # bar-only backtest can't know which hit first, so the engine must assume the worse.
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),
        (1.1000, 1.1030, 1.0970, 1.1010),   # both stop and target inside [0.0970, 1.1030]
    ]
    df = _df(rows)
    signal = TradeSignal(signal_bar_index=0, direction="long", entry_mode="market",
                          entry_price=1.1000, stop_price=1.0980, target_price=1.1020)
    result = simulate_trade(df, signal, equity=10000, risk_fraction=0.01,
                             symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(len(rows)))
    assert result.exit_reason == "stop"
    assert result.exit_price == 1.0980


def test_short_trade_directions_are_correctly_mirrored():
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),
        (1.1000, 1.1005, 1.0998, 1.1002),
        (1.0975, 1.0980, 1.0970, 1.0975),   # target (1.0980) for a short is touched via low<=... wait high check
    ]
    df = _df(rows)
    signal = TradeSignal(signal_bar_index=0, direction="short", entry_mode="market",
                          entry_price=1.1000, stop_price=1.1020, target_price=1.0980)
    result = simulate_trade(df, signal, equity=10000, risk_fraction=0.01,
                             symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(len(rows)))
    assert result.exit_reason == "target"
    assert result.exit_price == 1.0980
    assert result.net_pnl > 0


def test_limit_entry_requires_genuine_trade_through_not_a_mere_touch():
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),   # 0: signal bar
        (1.1000, 1.1005, 1.0990, 1.0995),    # 1: low == limit (1.0990) exactly -> touch, NOT trade-through
        (1.0995, 1.1000, 1.0988, 1.0992),     # 2: low < limit (1.0990) -> genuine trade-through, fills here
        (1.0992, 1.1030, 1.0985, 1.1020),      # 3: target hit after fill
    ]
    df = _df(rows)
    signal = TradeSignal(signal_bar_index=0, direction="long", entry_mode="limit",
                          entry_price=1.0990, stop_price=1.0970, target_price=1.1020)
    result = simulate_trade(df, signal, equity=10000, risk_fraction=0.01,
                             symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(len(rows)))
    assert result.entry_bar_index == 2   # not bar 1, where it only touched
    assert result.exit_reason == "target"


def test_limit_entry_expires_unfilled_when_price_never_trades_through():
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),
        (1.1000, 1.1010, 1.0995, 1.1005),
        (1.1005, 1.1015, 1.0998, 1.1010),
    ]
    df = _df(rows)
    signal = TradeSignal(signal_bar_index=0, direction="long", entry_mode="limit",
                          entry_price=1.0900, stop_price=1.0880, target_price=1.0950,
                          limit_expiry_bars=2)
    result = simulate_trade(df, signal, equity=10000, risk_fraction=0.01,
                             symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(len(rows)))
    assert result.exit_reason == "expired_unfilled"
    assert result.entry_bar_index is None
    assert result.net_pnl == 0.0
    assert result.r_multiple is None


def test_neither_stop_nor_target_hit_closes_at_end_of_data():
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),
        (1.1000, 1.1008, 1.0997, 1.1003),
        (1.1003, 1.1006, 1.0999, 1.1004),
    ]
    df = _df(rows)
    signal = TradeSignal(signal_bar_index=0, direction="long", entry_mode="market",
                          entry_price=1.1000, stop_price=1.0950, target_price=1.1100)
    result = simulate_trade(df, signal, equity=10000, risk_fraction=0.01,
                             symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(len(rows)))
    assert result.exit_reason == "end_of_data"
    assert result.exit_bar_index == 2
    assert result.exit_price == 1.1004


def test_net_pnl_equals_gross_minus_total_costs():
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),
        (1.1000, 1.1030, 1.0998, 1.1020),
    ]
    df = _df(rows, spreads=[15, 15])
    signal = TradeSignal(signal_bar_index=0, direction="long", entry_mode="market",
                          entry_price=1.1000, stop_price=1.0980, target_price=1.1020)
    result = simulate_trade(df, signal, equity=10000, risk_fraction=0.01,
                             symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(len(rows)))
    assert round(result.net_pnl, 6) == round(result.gross_pnl - result.costs.total, 6)
    assert result.costs.spread_cost > 0
    assert result.costs.commission_cost > 0


def test_stop_hit_r_multiple_is_worse_than_exactly_minus_one_due_to_costs():
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),
        (1.0985, 1.0990, 1.0975, 1.0980),
    ]
    df = _df(rows)
    signal = TradeSignal(signal_bar_index=0, direction="long", entry_mode="market",
                          entry_price=1.1000, stop_price=1.0980, target_price=1.1050)
    result = simulate_trade(df, signal, equity=10000, risk_fraction=0.01,
                             symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(len(rows)))
    assert result.exit_reason == "stop"
    assert round(result.gross_pnl, 2) == round(-result.lots * (0.0020 / 0.0001) * 10.0, 2)  # -risk_amount
    assert result.r_multiple < -1.0  # costs make it worse than a "clean" -1R


def test_spread_pct_of_stop_computed_correctly():
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),
        (1.1000, 1.1030, 1.0998, 1.1020),
    ]
    df = _df(rows, spreads=[20, 20])  # 20 points = 0.0002 price
    signal = TradeSignal(signal_bar_index=0, direction="long", entry_mode="market",
                          entry_price=1.1000, stop_price=1.0980, target_price=1.1020)  # stop dist = 0.0020
    result = simulate_trade(df, signal, equity=10000, risk_fraction=0.01,
                             symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(len(rows)))
    assert round(result.spread_pct_of_stop, 4) == round(0.0002 / 0.0020, 4)  # 10%


def test_unknown_entry_mode_raises():
    df = _df([(1.1, 1.1, 1.1, 1.1)])
    signal = TradeSignal(signal_bar_index=0, direction="long", entry_mode="stop_order",
                          entry_price=1.1, stop_price=1.09, target_price=1.11)
    with pytest.raises(ValueError):
        simulate_trade(df, signal, equity=10000, risk_fraction=0.01,
                        symbol_meta=_symbol_meta(), cost_cfg=_cost_cfg(), atr=_flat_atr(1))
