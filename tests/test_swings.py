import pandas as pd

from primitives.swings import detect_swings, compute_dual_resolution_swings


def _df(highs, lows):
    n = len(highs)
    times = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"time": times, "high": highs, "low": lows})


def test_swing_high_detected_with_correct_confirmation_lag():
    # pivot high at index 3 (value 5), n=2: window idx 1..5 = [2,3,5,3,2]
    highs = [1, 2, 3, 5, 3, 2, 1]
    lows = [0.5, 1, 1.5, 2, 1.5, 1, 0.5]  # flat/no swing lows here
    df = _df(highs, lows)
    swings = detect_swings(df, n=2, role="internal")
    highs_found = [s for s in swings if s.kind == "high"]
    assert len(highs_found) == 1
    s = highs_found[0]
    assert s.bar_index == 3
    assert s.price == 5
    assert s.confirmed_at_index == 3 + 2  # spec: confirmed at i+N, not before
    assert s.role == "internal"
    assert s.n == 2


def test_swing_low_detected_with_correct_confirmation_lag():
    lows = [5, 4, 3, 1, 3, 4, 5]  # pivot low at index 3
    highs = [10, 10, 10, 10, 10, 10, 10]  # flat, no swing highs
    df = _df(highs, lows)
    swings = detect_swings(df, n=2, role="external")
    lows_found = [s for s in swings if s.kind == "low"]
    assert len(lows_found) == 1
    s = lows_found[0]
    assert s.bar_index == 3
    assert s.price == 1
    assert s.confirmed_at_index == 5
    assert s.role == "external"


def test_no_lookahead_swing_not_detectable_before_confirmation_bar_exists():
    # Same swing-high setup as above (pivot at index 3, n=2, confirmed at index 5).
    # If the series is truncated to end exactly one bar short of the confirmation bar
    # (i.e. only through index 4), the swing must NOT appear at all — a signal running
    # live at that point in time genuinely does not yet know bar 3 is a swing.
    highs = [1, 2, 3, 5, 3, 2, 1]
    lows = [0.5, 1, 1.5, 2, 1.5, 1, 0.5]
    df_full = _df(highs, lows)

    truncated = df_full.iloc[:5].reset_index(drop=True)  # indices 0..4, missing index 5
    swings_truncated = detect_swings(truncated, n=2, role="internal")
    assert all(s.bar_index != 3 for s in swings_truncated)

    exactly_confirmed = df_full.iloc[:6].reset_index(drop=True)  # indices 0..5, index 5 present
    swings_confirmed = detect_swings(exactly_confirmed, n=2, role="internal")
    assert any(s.bar_index == 3 and s.kind == "high" for s in swings_confirmed)


def test_plateau_tie_is_not_a_swing():
    # Two equal highs adjacent to the "pivot" must NOT count — spec requires strict
    # inequality (high[i] > high[i±k]), not >=.
    highs = [1, 2, 5, 5, 2, 1]
    lows = [0, 0, 0, 0, 0, 0]
    df = _df(highs, lows)
    swings = detect_swings(df, n=2, role="internal")
    assert all(s.kind != "high" for s in swings)


def test_dual_resolution_role_tagging_and_relative_density():
    # A tighter fractal width (internal, small N) should find at least as many swings as
    # a wider one (external, large N) on the same series, and each must carry the right role.
    import random
    random.seed(7)
    highs = [1.0 + 0.01 * i + (0.05 if i % 4 == 0 else 0) for i in range(60)]
    lows = [h - 0.02 for h in highs]
    df = _df(highs, lows)

    result = compute_dual_resolution_swings(df, n_internal=2, n_external=5)
    assert all(s.role == "internal" and s.n == 2 for s in result["internal"])
    assert all(s.role == "external" and s.n == 5 for s in result["external"])
    assert len(result["internal"]) >= len(result["external"])
