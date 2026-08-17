import pytest

from risk.position_sizing import compute_position_size


# Standard EURUSD 5-digit feed: pip_size=0.0001, pip_value_per_lot=10.0 (1.0 lot,
# 100k contract, USD account) -- matches the real symbol_meta observed on the MT5 account.

def test_basic_sizing_matches_hand_calculation():
    # equity=10000, risk 1% -> $100 risk target. Stop = 20 pips. loss/lot = 20*10=$200.
    # raw_lots = 100/200 = 0.5, already on a clean 0.01 step -> no clamping.
    result = compute_position_size(
        equity=10000, risk_fraction=0.01, stop_distance_price=0.0020,
        pip_value_per_lot=10.0, pip_size=0.0001,
        volume_min=0.01, volume_max=500.0, volume_step=0.01,
    )
    assert result.lots == 0.5
    assert result.risk_amount == 100.0
    assert result.clamped is False


def test_clamped_up_to_volume_min():
    # Tiny risk target relative to a wide stop -> raw lots below the broker's minimum.
    result = compute_position_size(
        equity=100, risk_fraction=0.001, stop_distance_price=0.0500,  # $0.10 risk target, 500 pips
        pip_value_per_lot=10.0, pip_size=0.0001,
        volume_min=0.01, volume_max=500.0, volume_step=0.01,
    )
    assert result.lots == 0.01
    assert result.clamped is True


def test_clamped_down_to_volume_max():
    # Huge risk target relative to a tight stop -> raw lots above the broker's maximum.
    result = compute_position_size(
        equity=10_000_000, risk_fraction=0.5, stop_distance_price=0.0005,
        pip_value_per_lot=10.0, pip_size=0.0001,
        volume_min=0.01, volume_max=500.0, volume_step=0.01,
    )
    assert result.lots == 500.0
    assert result.clamped is True


def test_rounds_to_nearest_volume_step():
    # raw_lots lands off-step; must round to the nearest 0.01 without residual float noise.
    result = compute_position_size(
        equity=10037, risk_fraction=0.01, stop_distance_price=0.0020,
        pip_value_per_lot=10.0, pip_size=0.0001,
        volume_min=0.01, volume_max=500.0, volume_step=0.01,
    )
    # raw_lots = 100.37/200 = 0.50185 -> rounds to 0.50
    assert result.lots == 0.50
    assert result.clamped is False


def test_non_positive_equity_raises():
    with pytest.raises(ValueError):
        compute_position_size(
            equity=0, risk_fraction=0.01, stop_distance_price=0.002,
            pip_value_per_lot=10.0, pip_size=0.0001,
            volume_min=0.01, volume_max=500.0, volume_step=0.01,
        )


def test_non_positive_stop_distance_raises():
    with pytest.raises(ValueError):
        compute_position_size(
            equity=10000, risk_fraction=0.01, stop_distance_price=0,
            pip_value_per_lot=10.0, pip_size=0.0001,
            volume_min=0.01, volume_max=500.0, volume_step=0.01,
        )
