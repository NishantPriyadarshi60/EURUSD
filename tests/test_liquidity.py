import pandas as pd

from primitives.swings import Swing
from primitives.liquidity import compute_sr_levels, compute_equal_highs_lows


def _swing(price, kind, bar_index, confirmed_at_index):
    return Swing(bar_index=bar_index, confirmed_at_index=confirmed_at_index,
                 time=pd.Timestamp("2023-01-01"), confirmed_at_time=pd.Timestamp("2023-01-01"),
                 price=price, kind=kind, role="external", n=5)


def _flat_atr(n, value=1.0):
    return pd.Series([value] * n)


# ---------------------------------------------------------------------------
# S/R (§2.7) -- mixes highs and lows together, strength = touch count
# ---------------------------------------------------------------------------

def test_sr_clusters_nearby_swings_regardless_of_kind():
    # tol = 0.15 * 1.0 = 0.15. swing2 (low) is 0.1 from swing1 (high) -> same cluster.
    swings = [
        _swing(100.0, "high", bar_index=0, confirmed_at_index=1),
        _swing(100.1, "low", bar_index=2, confirmed_at_index=3),
        _swing(105.0, "high", bar_index=4, confirmed_at_index=5),  # far away -> own cluster
    ]
    levels = compute_sr_levels(swings, _flat_atr(10), tol_atr=0.15)
    assert len(levels) == 2
    merged = next(l for l in levels if l.strength == 2)
    assert merged.price == 100.05  # running average of 100.0 and 100.1
    assert merged.touches == [0, 2]
    lone = next(l for l in levels if l.strength == 1)
    assert lone.price == 105.0


def test_sr_strength_as_of_reflects_point_in_time_touch_count():
    swings = [
        _swing(100.0, "high", bar_index=0, confirmed_at_index=5),
        _swing(100.05, "low", bar_index=10, confirmed_at_index=15),
    ]
    level = compute_sr_levels(swings, _flat_atr(20), tol_atr=0.15)[0]
    assert level.strength == 2
    assert level.strength_as_of(4) == 0
    assert level.strength_as_of(5) == 1
    assert level.strength_as_of(14) == 1
    assert level.strength_as_of(15) == 2


def test_sr_skips_swings_during_atr_warmup():
    atr = pd.Series([float("nan"), float("nan"), 1.0, 1.0])
    swings = [_swing(100.0, "high", bar_index=0, confirmed_at_index=0)]  # NaN ATR here
    levels = compute_sr_levels(swings, atr, tol_atr=0.15)
    assert levels == []


# ---------------------------------------------------------------------------
# EQH/EQL (§2.8) -- same-kind only, requires 2+ members, confirmed at the 2nd member
# ---------------------------------------------------------------------------

def test_single_swing_high_is_not_an_eqh():
    swings = [
        _swing(100.0, "high", bar_index=0, confirmed_at_index=2),
        _swing(110.0, "high", bar_index=5, confirmed_at_index=7),  # far away, own cluster
    ]
    pools = compute_equal_highs_lows(swings, _flat_atr(10), tol_atr=0.1)
    assert pools == []  # neither cluster has 2+ members


def test_two_close_swing_highs_form_an_eqh_confirmed_at_second_member():
    swings = [
        _swing(100.0, "high", bar_index=0, confirmed_at_index=2),
        _swing(100.05, "high", bar_index=6, confirmed_at_index=8),  # within 0.1 tol
    ]
    pools = compute_equal_highs_lows(swings, _flat_atr(10), tol_atr=0.1)
    assert len(pools) == 1
    pool = pools[0]
    assert pool.kind == "high"
    assert pool.price == 100.025
    assert pool.member_bar_indices == [0, 6]
    assert pool.confirmed_at_index == 8  # the 2nd member's confirm index, not the 1st


def test_eqh_and_eql_are_independent_even_at_the_same_price():
    # A high and a low at nearly the same price must NOT merge into one pool -- highs
    # and lows are clustered in separate passes. Neither kind here has 2+ members.
    swings = [
        _swing(100.0, "high", bar_index=0, confirmed_at_index=2),
        _swing(100.02, "low", bar_index=1, confirmed_at_index=3),
    ]
    pools = compute_equal_highs_lows(swings, _flat_atr(10), tol_atr=0.1)
    assert pools == []


def test_eql_symmetric_to_eqh():
    swings = [
        _swing(50.0, "low", bar_index=0, confirmed_at_index=2),
        _swing(50.03, "low", bar_index=4, confirmed_at_index=6),
    ]
    pools = compute_equal_highs_lows(swings, _flat_atr(10), tol_atr=0.1)
    assert len(pools) == 1
    assert pools[0].kind == "low"
    assert pools[0].confirmed_at_index == 6


def test_eqh_pool_is_duck_type_compatible_with_sweep_targets():
    # spec's stated purpose (§2.7/§2.8): EQH/EQL feed sweep.detect_sweeps as targets.
    from primitives.sweep import detect_sweeps

    swings = [
        _swing(10.0, "high", bar_index=0, confirmed_at_index=1),
        _swing(10.02, "high", bar_index=2, confirmed_at_index=3),
    ]
    pools = compute_equal_highs_lows(swings, _flat_atr(10), tol_atr=0.1)
    assert len(pools) == 1
    assert pools[0].confirmed_at_index == 3  # 2nd member's confirm index

    rows = [
        (10.0, 9.5, 9.8),      # 0: filler, before confirm
        (10.05, 9.7, 9.9),      # 1: filler, before confirm
        (10.05, 9.8, 9.9),       # 2: filler, before confirm
        (10.15, 9.9, 10.05),      # 3: confirm bar; pierces the EQH (10.15 > 10.01+0.1 tol)
        (10.2, 9.9, 9.98),          # 4: reclaims (close < pool price 10.01)
    ]
    highs, lows, closes = zip(*rows)
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    events = detect_sweeps(df, pools, _flat_atr(len(rows)), tol_atr=0.1, window_bars=3)
    assert len(events) == 1
    assert events[0].kind == "buy_side"
