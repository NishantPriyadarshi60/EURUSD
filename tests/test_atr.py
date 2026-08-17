import pandas as pd

from primitives.atr import compute_atr


def test_atr_warmup_is_nan_and_seed_is_simple_mean():
    # Hand-verified: TR[1..5] = [1.5, 0.7, 1.5, 1.0, 0.8] -> seed mean = 1.1
    df = pd.DataFrame({
        "high":  [10, 11, 10.5, 11.5, 12, 11.8],
        "low":   [9, 9.5, 9.8, 10.5, 11, 11.0],
        "close": [9.5, 10.2, 10.0, 11.0, 11.5, 11.3],
    })
    atr = compute_atr(df, period=5)
    assert atr.iloc[:5].isna().all()
    assert atr.iloc[5] == 1.1


def test_atr_wilder_smoothing_step():
    # Continuing the same series one more bar: TR6 = 0.9 (hand-verified),
    # atr[6] = (1.1*4 + 0.9) / 5 = 1.06
    df = pd.DataFrame({
        "high":  [10, 11, 10.5, 11.5, 12, 11.8, 12.2],
        "low":   [9, 9.5, 9.8, 10.5, 11, 11.0, 11.5],
        "close": [9.5, 10.2, 10.0, 11.0, 11.5, 11.3, 12.0],
    })
    atr = compute_atr(df, period=5)
    assert round(atr.iloc[6], 6) == 1.06


def test_atr_short_series_all_nan():
    df = pd.DataFrame({"high": [10, 11], "low": [9, 9.5], "close": [9.5, 10.2]})
    atr = compute_atr(df, period=5)
    assert atr.isna().all()
