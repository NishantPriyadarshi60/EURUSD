import pandas as pd

from backtest.costs import (
    CostConfig, spread_cost, commission_cost, slippage_cost, is_spike_bar,
    count_swap_nights, swap_cost,
)

# Standard EURUSD 5-digit feed: point_size=0.00001, pip_size=0.0001, pip_value_per_lot=10.0

def _cfg(**overrides):
    defaults = dict(
        commission_per_lot_round_trip=7.0,
        slippage_base_pips=0.2,
        slippage_spike_atr_multiple=2.0,
        slippage_spike_extra_pips=1.0,
        slippage_apply_to_target_exits=False,
        slippage_apply_to_market_entries=True,
        swap_rollover_hour_utc=21,
        swap_triple_swap_weekday=2,  # Wednesday
    )
    defaults.update(overrides)
    return CostConfig(**defaults)


def test_spread_cost_matches_hand_calculation():
    # 10 points = 1 pip; 1 pip on 1.0 lot EURUSD = $10.
    cost = spread_cost(spread_points=10, point_size=0.00001, pip_size=0.0001, pip_value_per_lot=10.0, lots=1.0)
    assert round(cost, 6) == 10.0


def test_commission_cost_scales_with_lots():
    assert commission_cost(lots=0.5, cfg=_cfg()) == 3.5


def test_slippage_applies_to_stop_exits_always():
    cfg = _cfg()
    cost = slippage_cost(lots=1.0, pip_size=0.0001, pip_value_per_lot=10.0, cfg=cfg,
                          is_spike=False, is_stop_exit=True, is_target_exit=False, is_market_entry=False)
    assert round(cost, 6) == 2.0  # 0.2 pips * $10/pip


def test_slippage_skips_target_exits_by_default():
    cfg = _cfg()
    cost = slippage_cost(lots=1.0, pip_size=0.0001, pip_value_per_lot=10.0, cfg=cfg,
                          is_spike=False, is_stop_exit=False, is_target_exit=True, is_market_entry=False)
    assert cost == 0.0


def test_slippage_applies_extra_on_spike():
    cfg = _cfg()
    cost = slippage_cost(lots=1.0, pip_size=0.0001, pip_value_per_lot=10.0, cfg=cfg,
                          is_spike=True, is_stop_exit=True, is_target_exit=False, is_market_entry=False)
    assert round(cost, 6) == 12.0  # (0.2 + 1.0) pips * $10/pip


def test_slippage_applies_to_market_entries_when_configured():
    cfg = _cfg(slippage_apply_to_market_entries=True)
    cost = slippage_cost(lots=1.0, pip_size=0.0001, pip_value_per_lot=10.0, cfg=cfg,
                          is_spike=False, is_stop_exit=False, is_target_exit=False, is_market_entry=True)
    assert round(cost, 6) == 2.0


def test_is_spike_bar_threshold_and_nan_safety():
    cfg = _cfg()
    assert is_spike_bar(bar_range_price=0.0025, atr_value=0.001, cfg=cfg) is True   # 2.5x >= 2.0x
    assert is_spike_bar(bar_range_price=0.0015, atr_value=0.001, cfg=cfg) is False   # 1.5x < 2.0x
    assert is_spike_bar(bar_range_price=0.01, atr_value=float("nan"), cfg=cfg) is False
    assert is_spike_bar(bar_range_price=0.01, atr_value=0.0, cfg=cfg) is False


def test_swap_nights_zero_for_same_day_trade_before_rollover():
    cfg = _cfg()
    entry = pd.Timestamp("2024-01-01 10:00", tz="UTC")  # Monday
    exit_ = pd.Timestamp("2024-01-01 15:00", tz="UTC")
    assert count_swap_nights(entry, exit_, cfg) == 0


def test_swap_nights_one_crossing_single_rollover():
    cfg = _cfg()
    entry = pd.Timestamp("2024-01-01 10:00", tz="UTC")  # Monday
    exit_ = pd.Timestamp("2024-01-02 10:00", tz="UTC")   # Tuesday, crosses Mon 21:00 rollover
    assert count_swap_nights(entry, exit_, cfg) == 1


def test_swap_nights_triple_on_configured_weekday():
    cfg = _cfg()
    entry = pd.Timestamp("2024-01-02 10:00", tz="UTC")  # Tuesday
    exit_ = pd.Timestamp("2024-01-04 10:00", tz="UTC")   # Thursday
    # crosses Tue 21:00 (weekday=1, +1) and Wed 21:00 (weekday=2, triple, +3) -> 4
    assert count_swap_nights(entry, exit_, cfg) == 4


def test_swap_cost_sign_matches_broker_convention():
    # Real observed demo-account values: swap_long=-0.7, swap_short=-1.0 points/lot/night.
    cost = swap_cost(direction="long", nights=1, point_size=0.00001, pip_size=0.0001,
                      pip_value_per_lot=10.0, lots=1.0, swap_long_points=-0.7, swap_short_points=-1.0)
    assert round(cost, 6) == 0.7  # a genuine cost (positive, deducted from P&L)


def test_swap_credit_flips_sign_when_broker_pays():
    cost = swap_cost(direction="short", nights=2, point_size=0.00001, pip_size=0.0001,
                      pip_value_per_lot=10.0, lots=1.0, swap_long_points=-0.7, swap_short_points=1.5)
    assert round(cost, 6) == -3.0  # a credit: negative "cost" over 2 nights at +1.5 pts/night
