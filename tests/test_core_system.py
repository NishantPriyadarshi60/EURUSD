import pandas as pd

from primitives.swings import detect_swings
from primitives.structure import run_single_resolution_structure
from signals.core_system import compute_system
from signals.session import load_session_windows
from signals.scoring import score_candidate
from configs.loader import load as load_config


def _zigzag_ohlc(n=300, seed_amplitude=0.001):
    """A long enough oscillating-with-drift series to clear ATR/ADX/structure warmup
    and actually generate some confirmed swings and structure events."""
    import math
    base = 1.1000
    closes = []
    for i in range(n):
        drift = i * 0.00005
        wave = seed_amplitude * math.sin(i / 6.0)
        closes.append(base + drift + wave)
    highs = [c + 0.0004 for c in closes]
    lows = [c - 0.0004 for c in closes]
    opens = [closes[i - 1] if i > 0 else closes[0] for i in range(n)]
    times = pd.date_range("2023-06-01 00:00", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"time": times, "open": opens, "high": highs, "low": lows, "close": closes,
                          "spread": [2] * n})


def _h4_from_m15(m15_df):
    # crude but sufficient H4 proxy for a smoke test: resample every 16th M15 bar as an H4 bar
    h4 = m15_df.iloc[::16].reset_index(drop=True).copy()
    h4["time"] = pd.date_range("2023-05-01 00:00", periods=len(h4), freq="4h", tz="UTC")
    return h4


def test_compute_system_runs_end_to_end_and_produces_sane_output():
    m15 = _zigzag_ohlc(n=300)
    h4 = _h4_from_m15(m15)

    from signals.htf_bias import compute_htf_direction_timeline
    sig_cfg = load_config("signals")
    prim_cfg = load_config("primitives")

    htf_swings = detect_swings(h4, n=sig_cfg["htf_bias"]["swing_n"], role="htf")
    htf_events = run_single_resolution_structure(h4, htf_swings, resolution="htf")
    htf_timeline = compute_htf_direction_timeline(h4, swing_n=sig_cfg["htf_bias"]["swing_n"],
                                                   htf_timeframe_hours=sig_cfg["htf_bias"]["htf_timeframe_hours"])
    session_windows = load_session_windows(sig_cfg)

    result = compute_system(m15, htf_timeline, session_windows, prim_cfg, sig_cfg)

    assert isinstance(result.candidates, list)
    assert len(result.conditions) == 6
    assert {c.name for c in result.conditions} == {
        "htf_pass", "session_pass", "regime_pass", "premium_discount_pass",
        "has_ob_retest", "has_recent_sweep",
    }
    assert all(c.weight == 1.0 for c in result.conditions)

    # every candidate must be scoreable without error, and score in [0, 1]
    for candidate in result.candidates[:50]:  # bound the check for speed
        r = score_candidate(candidate, result.conditions)
        assert 0.0 <= r.fraction <= 1.0
