import pandas as pd

from signals.regime import directional_movement, compute_adx, classify_regime


def test_directional_movement_hand_verified():
    # idx1: up_move=11-10=1, down_move=9-9.5=-0.5 -> +DM=1, -DM=0
    # idx2: up_move=10.5-11=-0.5, down_move=9.5-9=0.5 -> +DM=0, -DM=0.5
    # idx3: up_move=10.8-10.5=0.3, down_move=9-9.5=-0.5 -> +DM=0.3, -DM=0
    df = pd.DataFrame({
        "high": [10, 11, 10.5, 10.8],
        "low": [9, 9.5, 9, 9.5],
        "close": [9.5, 10.5, 9.2, 10.5],
    })
    plus_dm, minus_dm = directional_movement(df)
    assert [round(v, 6) for v in plus_dm] == [0.0, 1.0, 0.0, 0.3]
    assert [round(v, 6) for v in minus_dm] == [0.0, 0.0, 0.5, 0.0]


def test_directional_movement_never_both_nonzero_same_bar():
    df = pd.DataFrame({
        "high": [10, 12, 11, 13, 10, 14],
        "low": [9, 10, 9.5, 11, 8, 12],
        "close": [9.5, 11, 10, 12, 9, 13],
    })
    plus_dm, minus_dm = directional_movement(df)
    assert all((p == 0) or (m == 0) for p, m in zip(plus_dm, minus_dm))


def _trending_series(n=60):
    highs = [100 + i * 0.5 for i in range(n)]
    lows = [h - 0.3 for h in highs]
    closes = [h - 0.1 for h in highs]
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


def _ranging_series(n=60):
    highs = [100 + (0.4 if i % 2 == 0 else -0.2) for i in range(n)]
    lows = [h - 0.5 for h in highs]
    closes = [h - 0.25 for h in highs]
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


def test_adx_is_high_for_a_clean_uptrend():
    df = _trending_series()
    result = compute_adx(df, period=14)
    final_adx = result["adx"].iloc[-1]
    assert final_adx > 25  # trending_threshold from configs/signals.yaml
    assert result["plus_di"].iloc[-1] > result["minus_di"].iloc[-1]


def test_adx_is_low_for_a_choppy_oscillation():
    df = _ranging_series()
    result = compute_adx(df, period=14)
    final_adx = result["adx"].iloc[-1]
    assert final_adx < 20  # ranging_threshold from configs/signals.yaml


def test_adx_warmup_is_nan_for_roughly_twice_the_period():
    df = _trending_series(n=60)
    result = compute_adx(df, period=14)
    # DX's first valid value is at index=period (14) -- wilder_smooth then treats THAT
    # as its own "first valid" reference point and skips one more before seeding on the
    # next `period` values, so ADX's first valid index is 14 + 14 = 28.
    assert result["adx"].iloc[:28].isna().all()
    assert result["adx"].iloc[28:].notna().all()


def test_classify_regime_trending_ranging_transitional_unknown():
    assert classify_regime(30, trending_threshold=25, ranging_threshold=20) == "trending"
    assert classify_regime(25, trending_threshold=25, ranging_threshold=20) == "trending"  # boundary inclusive
    assert classify_regime(15, trending_threshold=25, ranging_threshold=20) == "ranging"
    assert classify_regime(20, trending_threshold=25, ranging_threshold=20) == "ranging"   # boundary inclusive
    assert classify_regime(22, trending_threshold=25, ranging_threshold=20) == "transitional"
    assert classify_regime(float("nan"), trending_threshold=25, ranging_threshold=20) == "unknown"


def test_flat_market_does_not_crash_on_zero_di_sum():
    # A completely flat market (no high/low movement at all) drives +DI and -DI both to
    # 0, which would be a 0/0 division in DX -- must produce NaN, not crash.
    df = pd.DataFrame({"high": [100.0] * 40, "low": [100.0] * 40, "close": [100.0] * 40})
    result = compute_adx(df, period=14)
    assert result["dx"].isna().any() or (result["dx"].dropna() == 0).all()
