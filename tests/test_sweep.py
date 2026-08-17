import pandas as pd

from primitives.swings import Swing
from primitives.sweep import detect_sweeps


def _df(rows):
    """rows: list of (high, low, close)."""
    n = len(rows)
    times = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    highs, lows, closes = zip(*rows)
    return pd.DataFrame({"time": times, "high": highs, "low": lows, "close": closes})


def _flat_atr(n, value=1.0):
    return pd.Series([value] * n)


def _swing_low(price, bar_index=0, confirmed_at_index=1):
    return Swing(bar_index=bar_index, confirmed_at_index=confirmed_at_index,
                 time=pd.Timestamp("2023-01-01"), confirmed_at_time=pd.Timestamp("2023-01-01"),
                 price=price, kind="low", role="internal", n=1)


def _swing_high(price, bar_index=0, confirmed_at_index=1):
    return Swing(bar_index=bar_index, confirmed_at_index=confirmed_at_index,
                 time=pd.Timestamp("2023-01-01"), confirmed_at_time=pd.Timestamp("2023-01-01"),
                 price=price, kind="high", role="internal", n=1)


# level=10.0, tol_atr=0.1, flat ATR=1.0 -> tolerance=0.1 everywhere in these tests.

def test_sell_side_sweep_same_candle_reclaim():
    rows = [
        (10.5, 10.4, 10.3),   # 0: filler (before confirm)
        (10.3, 9.95, 10.1),    # 1: confirm bar; low 9.95 >= 9.9 -> not pierced
        (10.2, 9.85, 10.05),    # 2: pierced (9.85 < 9.9); reclaimed same candle (10.05>10.0)
        (10.4, 10.1, 10.3),      # 3: irrelevant, level already resolved
    ]
    df = _df(rows)
    swing = _swing_low(10.0, bar_index=0, confirmed_at_index=1)
    events = detect_sweeps(df, [swing], _flat_atr(len(rows)), tol_atr=0.1, window_bars=3)
    assert len(events) == 1
    e = events[0]
    assert e.kind == "sell_side"
    assert e.pierce_bar_index == 2
    assert e.reclaim_bar_index == 2
    assert e.level_price == 10.0
    assert e.pierce_extreme == 9.85


def test_sell_side_sweep_multi_candle_reclaim_within_window():
    rows = [
        (10.5, 10.4, 10.3),   # 0: filler
        (10.3, 9.95, 10.1),    # 1: confirm bar; not pierced
        (10.2, 9.8, 9.95),      # 2: pierced, not reclaimed yet
        (10.0, 9.85, 9.98),      # 3: still not reclaimed
        (10.3, 9.9, 10.05),       # 4: reclaimed here (close > 10.0)
        (10.4, 10.0, 10.2),        # 5: filler
    ]
    df = _df(rows)
    swing = _swing_low(10.0, bar_index=0, confirmed_at_index=1)
    events = detect_sweeps(df, [swing], _flat_atr(len(rows)), tol_atr=0.1, window_bars=3)
    assert len(events) == 1
    assert events[0].pierce_bar_index == 2
    assert events[0].reclaim_bar_index == 4


def test_no_reclaim_within_window_produces_no_sweep_event():
    rows = [
        (10.5, 10.4, 10.3),   # 0: filler
        (10.3, 9.95, 10.1),    # 1: confirm bar; not pierced
        (10.2, 9.8, 9.95),      # 2: pierced
        (10.0, 9.85, 9.90),      # 3: not reclaimed
        (10.0, 9.85, 9.85),       # 4: still not reclaimed; window_bars=2 -> search ends here
    ]
    df = _df(rows)
    swing = _swing_low(10.0, bar_index=0, confirmed_at_index=1)
    events = detect_sweeps(df, [swing], _flat_atr(len(rows)), tol_atr=0.1, window_bars=2)
    assert events == []  # a close-beyond-and-stay is BOS territory, not a sweep


def test_shallow_wick_within_tolerance_is_not_a_sweep():
    rows = [(10.3, 9.95, 10.1)] * 5  # low always 9.95 >= 9.9 threshold -> never pierced
    df = _df(rows)
    swing = _swing_low(10.0, bar_index=0, confirmed_at_index=0)
    events = detect_sweeps(df, [swing], _flat_atr(len(rows)), tol_atr=0.1, window_bars=3)
    assert events == []


def test_buy_side_sweep_symmetric():
    rows = [
        (10.0, 9.5, 9.8),       # 0: filler
        (10.05, 9.7, 9.9),       # 1: confirm bar; high 10.05 <= 10.1 -> not pierced
        (10.15, 9.9, 10.05),      # 2: pierced (10.15 > 10.1); not reclaimed same candle
        (10.2, 9.9, 9.95),         # 3: reclaimed (close < 10.0)
    ]
    df = _df(rows)
    swing = _swing_high(10.0, bar_index=0, confirmed_at_index=1)
    events = detect_sweeps(df, [swing], _flat_atr(len(rows)), tol_atr=0.1, window_bars=3)
    assert len(events) == 1
    e = events[0]
    assert e.kind == "buy_side"
    assert e.pierce_bar_index == 2
    assert e.reclaim_bar_index == 3
    assert e.pierce_extreme == 10.15


def test_no_lookahead_pierce_before_confirmation_is_ignored():
    # Bar 0 has an obvious pierce+same-candle-reclaim pattern, but the swing doesn't
    # confirm until bar 3 -- that early pattern must be invisible. A genuine pierce+
    # reclaim after confirmation (bars 4-5) must still be detected correctly.
    rows = [
        (10.3, 9.7, 10.05),   # 0: would pierce+reclaim if tested -- must be ignored
        (10.3, 10.0, 10.2),    # 1: filler
        (10.3, 10.0, 10.2),     # 2: filler
        (10.3, 9.95, 10.1),      # 3: confirm bar; not pierced
        (10.2, 9.8, 9.9),         # 4: pierced, not reclaimed yet
        (10.3, 9.9, 10.05),        # 5: reclaimed here
    ]
    df = _df(rows)
    swing = _swing_low(10.0, bar_index=-1, confirmed_at_index=3)
    events = detect_sweeps(df, [swing], _flat_atr(len(rows)), tol_atr=0.1, window_bars=3)
    assert len(events) == 1
    assert events[0].pierce_bar_index == 4
    assert events[0].reclaim_bar_index == 5


def test_each_swing_tested_for_at_most_one_pierce():
    rows = [
        (10.3, 9.8, 10.05),   # 0: pierced + reclaimed same candle -> resolves the level
        (10.3, 9.7, 9.9),       # 1: would ALSO pierce if re-tested -- must not double-count
        (10.3, 10.0, 10.2),      # 2: filler
    ]
    df = _df(rows)
    swing = _swing_low(10.0, bar_index=0, confirmed_at_index=0)
    events = detect_sweeps(df, [swing], _flat_atr(len(rows)), tol_atr=0.1, window_bars=3)
    assert len(events) == 1
    assert events[0].pierce_bar_index == 0
