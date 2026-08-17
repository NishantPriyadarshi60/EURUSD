import pandas as pd

from primitives.structure import StructureEvent
from primitives.order_block import detect_order_blocks


def _df(rows):
    """rows: list of (open, high, low, close)."""
    opens, highs, lows, closes = zip(*rows)
    n = len(rows)
    times = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"time": times, "open": opens, "high": highs, "low": lows, "close": closes})


def _event(bar_index, direction, event_type="BOS"):
    return StructureEvent(
        bar_index=bar_index, time=pd.Timestamp("2023-01-01"), resolution="internal",
        event_type=event_type, direction=direction, broken_level=0.0, broken_swing_bar_index=0,
    )


def test_bullish_ob_is_last_down_close_candle_before_the_break():
    rows = [
        (10.0, 10.1, 9.8, 9.85),   # 0: down-close (OB candidate)
        (9.85, 10.5, 9.8, 10.4),    # 1: bullish break candle
    ]
    df = _df(rows)
    events = [_event(1, "bullish")]
    displacement = pd.Series([False, True])
    obs = detect_order_blocks(df, events, displacement)
    assert len(obs) == 1
    ob = obs[0]
    assert ob.candle_bar_index == 0
    assert ob.direction == "bullish"
    assert ob.zone_low == 9.8 and ob.zone_high == 10.1
    assert ob.body_low == 9.85 and ob.body_high == 10.0  # sorted(open=10.0, close=9.85)
    assert ob.break_bar_index == 1


def test_bearish_ob_is_last_up_close_candle_before_the_break():
    rows = [
        (9.85, 10.1, 9.8, 10.0),   # 0: up-close (OB candidate)
        (10.0, 10.05, 9.4, 9.5),    # 1: bearish break candle
    ]
    df = _df(rows)
    events = [_event(1, "bearish")]
    displacement = pd.Series([False, True])
    obs = detect_order_blocks(df, events, displacement)
    assert len(obs) == 1
    ob = obs[0]
    assert ob.candle_bar_index == 0
    assert ob.direction == "bearish"
    assert ob.zone_low == 9.8 and ob.zone_high == 10.1
    assert ob.body_low == 9.85 and ob.body_high == 10.0


def test_non_displacement_break_registers_no_order_block():
    rows = [
        (10.0, 10.1, 9.8, 9.85),
        (9.85, 10.5, 9.8, 10.4),
    ]
    df = _df(rows)
    events = [_event(1, "bullish")]
    displacement = pd.Series([False, False])  # break candle fails the quality gate
    obs = detect_order_blocks(df, events, displacement)
    assert obs == []


def test_no_opposite_colored_candle_before_break_yields_no_ob():
    rows = [(9.85, 10.5, 9.8, 10.4)]  # break is the very first bar; nothing precedes it
    df = _df(rows)
    events = [_event(0, "bullish")]
    displacement = pd.Series([True])
    obs = detect_order_blocks(df, events, displacement)
    assert obs == []


def test_mitigation_detected_on_first_touch_bar():
    rows = [
        (10.0, 10.1, 9.8, 9.85),   # 0: OB candle, zone [9.8, 10.1]
        (9.85, 10.5, 9.8, 10.4),    # 1: break candle
        (10.4, 10.45, 10.2, 10.3),   # 2: no touch (low 10.2 > zone_high 10.1)
        (10.3, 10.35, 9.9, 10.0),      # 3: touches zone (low 9.9 is inside [9.8,10.1])
        (10.0, 10.6, 9.95, 10.5),        # 4: also overlaps, but first touch already at bar 3
    ]
    df = _df(rows)
    events = [_event(1, "bullish")]
    displacement = pd.Series([False, True, False, False, False])
    obs = detect_order_blocks(df, events, displacement)
    assert len(obs) == 1
    assert obs[0].mitigated_at_index == 3


def test_never_mitigated_stays_none():
    rows = [
        (10.0, 10.1, 9.8, 9.85),
        (9.85, 10.5, 9.8, 10.4),
        (10.4, 10.45, 10.2, 10.3),  # never dips back into [9.8, 10.1]
    ]
    df = _df(rows)
    events = [_event(1, "bullish")]
    displacement = pd.Series([False, True, False])
    obs = detect_order_blocks(df, events, displacement)
    assert obs[0].mitigated_at_index is None


def test_is_unmitigated_at_respects_break_and_mitigation_bounds():
    rows = [
        (10.0, 10.1, 9.8, 9.85),
        (9.85, 10.5, 9.8, 10.4),
        (10.4, 10.45, 10.2, 10.3),
        (10.3, 10.35, 9.9, 10.0),  # mitigation touch at bar 3
    ]
    df = _df(rows)
    events = [_event(1, "bullish")]
    displacement = pd.Series([False, True, False, False])
    ob = detect_order_blocks(df, events, displacement)[0]

    assert ob.is_unmitigated_at(0) is False   # before the break, not yet knowable
    assert ob.is_unmitigated_at(1) is True     # right at the break bar
    assert ob.is_unmitigated_at(2) is True      # still unmitigated
    assert ob.is_unmitigated_at(3) is False       # mitigated exactly at this bar
    assert ob.is_unmitigated_at(4) is False        # and stays mitigated after
