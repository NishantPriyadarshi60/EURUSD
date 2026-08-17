import pandas as pd

from primitives.swings import Swing
from primitives.premium_discount import compute_dealing_ranges, range_as_of, classify_position


def _swing(price, kind, bar_index, confirmed_at_index):
    return Swing(bar_index=bar_index, confirmed_at_index=confirmed_at_index,
                 time=pd.Timestamp("2023-01-01"), confirmed_at_time=pd.Timestamp("2023-01-01"),
                 price=price, kind=kind, role="external", n=5)


def _flat_atr(n, value=1.0):
    return pd.Series([value] * n)


# min_leg_atr = 3.0, flat ATR = 1.0 -> a leg must span >= 3.0 to qualify.

def test_valid_leg_forms_dealing_range():
    swings = [
        _swing(100.0, "low", bar_index=0, confirmed_at_index=5),
        _swing(104.0, "high", bar_index=2, confirmed_at_index=7),  # span 4.0 >= 3.0
    ]
    states = compute_dealing_ranges(swings, _flat_atr(10), min_leg_atr=3.0)
    assert len(states) == 1
    s = states[0]
    assert s.as_of_index == 7
    assert s.range_low == 100.0 and s.range_high == 104.0
    assert s.equilibrium == 102.0
    assert s.stale is False


def test_first_candidate_below_guard_produces_no_range_at_all():
    # Nothing to freeze yet -- a sub-threshold FIRST leg must not create a range.
    swings = [
        _swing(100.0, "low", bar_index=0, confirmed_at_index=5),
        _swing(101.0, "high", bar_index=2, confirmed_at_index=7),  # span 1.0 < 3.0
    ]
    states = compute_dealing_ranges(swings, _flat_atr(10), min_leg_atr=3.0)
    assert states == []
    assert range_as_of(states, 100) is None


def test_sub_threshold_candidate_freezes_prior_valid_range_as_stale():
    swings = [
        _swing(100.0, "low", bar_index=0, confirmed_at_index=5),
        _swing(104.0, "high", bar_index=2, confirmed_at_index=7),   # valid range: [100,104]
        _swing(102.5, "high", bar_index=8, confirmed_at_index=10),   # new candidate: [100,102.5] span 2.5 < 3.0
    ]
    states = compute_dealing_ranges(swings, _flat_atr(15), min_leg_atr=3.0)
    assert len(states) == 2
    frozen = states[1]
    assert frozen.as_of_index == 10
    assert frozen.stale is True
    # Frozen values must be the OLD valid range, not derived from the new candidate.
    assert frozen.range_low == 100.0 and frozen.range_high == 104.0
    assert frozen.high_swing_bar_index == 2  # still references the original high, not bar 8


def test_later_leg_clearing_guard_replaces_the_frozen_range():
    swings = [
        _swing(100.0, "low", bar_index=0, confirmed_at_index=5),
        _swing(104.0, "high", bar_index=2, confirmed_at_index=7),     # valid: [100, 104]
        _swing(102.5, "high", bar_index=8, confirmed_at_index=10),     # frozen (stale)
        _swing(95.0, "low", bar_index=12, confirmed_at_index=14),       # new candidate: [95, 102.5] span 7.5 >= 3.0
    ]
    states = compute_dealing_ranges(swings, _flat_atr(20), min_leg_atr=3.0)
    assert len(states) == 3
    fresh = states[2]
    assert fresh.as_of_index == 14
    assert fresh.stale is False
    assert fresh.range_low == 95.0 and fresh.range_high == 102.5
    assert fresh.equilibrium == 98.75
    assert fresh.low_swing_bar_index == 12 and fresh.high_swing_bar_index == 8


def test_range_as_of_looks_up_correct_historical_state():
    swings = [
        _swing(100.0, "low", bar_index=0, confirmed_at_index=5),
        _swing(104.0, "high", bar_index=2, confirmed_at_index=7),
        _swing(102.5, "high", bar_index=8, confirmed_at_index=10),
        _swing(95.0, "low", bar_index=12, confirmed_at_index=14),
    ]
    states = compute_dealing_ranges(swings, _flat_atr(20), min_leg_atr=3.0)

    assert range_as_of(states, 6) is None            # before any valid range
    assert range_as_of(states, 7).as_of_index == 7     # right at formation
    assert range_as_of(states, 9).as_of_index == 7      # still the same range
    assert range_as_of(states, 10).as_of_index == 10     # frozen/stale entry now applies
    assert range_as_of(states, 13).as_of_index == 10      # still frozen
    assert range_as_of(states, 14).as_of_index == 14       # fresh valid range
    assert range_as_of(states, 1000).as_of_index == 14      # stays the latest thereafter


def test_classify_position_discount_premium_equilibrium_and_outside():
    assert classify_position(101.0, 100.0, 104.0) == "discount"
    assert classify_position(103.0, 100.0, 104.0) == "premium"
    assert classify_position(102.0, 100.0, 104.0) == "equilibrium"
    assert classify_position(99.0, 100.0, 104.0) is None   # below range entirely
    assert classify_position(105.0, 100.0, 104.0) is None    # above range entirely
    assert classify_position(100.0, 100.0, 100.0) is None     # degenerate zero-width range
