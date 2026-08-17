import pandas as pd

from primitives.displacement import is_displacement


def _df(rows):
    opens, highs, lows, closes = zip(*rows)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})


def test_strong_body_and_wide_range_is_displacement():
    # body_ratio = |11-9|/(11.2-8.9) = 2/2.3 = 0.87 >= 0.6; range 2.3 >= 1.5*1.0=1.5
    df = _df([(9, 11.2, 8.9, 11)])
    atr = pd.Series([1.0])
    result = is_displacement(df, atr, body_ratio_min=0.6, atr_multiple_min=1.5)
    assert bool(result.iloc[0]) is True


def test_small_body_wide_range_is_not_displacement():
    # A wide-range doji: body_ratio = |10.05-10.0|/2.3 ~= 0.02, well under 0.6
    df = _df([(10.0, 11.2, 8.9, 10.05)])
    atr = pd.Series([1.0])
    result = is_displacement(df, atr, body_ratio_min=0.6, atr_multiple_min=1.5)
    assert bool(result.iloc[0]) is False


def test_strong_body_narrow_range_is_not_displacement():
    # Full-body candle but range too small relative to ATR: range 0.5 < 1.5*1.0
    df = _df([(9.0, 9.5, 9.0, 9.5)])
    atr = pd.Series([1.0])
    result = is_displacement(df, atr, body_ratio_min=0.6, atr_multiple_min=1.5)
    assert bool(result.iloc[0]) is False


def test_zero_range_candle_is_not_displacement_and_does_not_crash():
    df = _df([(10.0, 10.0, 10.0, 10.0)])
    atr = pd.Series([1.0])
    result = is_displacement(df, atr, body_ratio_min=0.6, atr_multiple_min=1.5)
    assert bool(result.iloc[0]) is False


def test_nan_atr_warmup_is_not_displacement_and_does_not_crash():
    df = _df([(9, 11.2, 8.9, 11)])
    atr = pd.Series([float("nan")])
    result = is_displacement(df, atr, body_ratio_min=0.6, atr_multiple_min=1.5)
    assert bool(result.iloc[0]) is False
