"""Trend-vs-range regime filter (spec §3.3) — "central here, not optional" for EUR/USD,
which spends much of its life mean-reverting/range-bound, exactly where breakout/
continuation systems get chopped. Uses Wilder's ADX, the classic trend-strength measure.

ADX needs roughly 2x its period to warm up (it's Wilder-smoothed DX, and DX itself isn't
defined until +DI/-DI have already warmed up over one full period — see
primitives/atr.py's wilder_smooth docstring for how this warmup falls out naturally
rather than being hardcoded).
"""
from __future__ import annotations

import pandas as pd

from primitives.atr import compute_true_range, wilder_smooth


def directional_movement(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """+DM/-DM per Wilder: only the DOMINANT directional move on a bar counts, and only
    if positive — a bar can register +DM or -DM, never both. Index 0 is always 0 (no
    prior bar to compare)."""
    high = df["high"]
    low = df["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_mask = ((up_move > down_move) & (up_move > 0)).fillna(False)
    minus_mask = ((down_move > up_move) & (down_move > 0)).fillna(False)
    plus_dm[plus_mask] = up_move[plus_mask]
    minus_dm[minus_mask] = down_move[minus_mask]
    return plus_dm, minus_dm


def compute_adx(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """df needs 'high','low','close'. Returns a DataFrame (same index as df) with
    plus_di, minus_di, dx, adx columns. NaN until each measure's own warmup completes."""
    plus_dm, minus_dm = directional_movement(df)
    tr = compute_true_range(df)

    smoothed_tr = wilder_smooth(tr, period)
    smoothed_plus_dm = wilder_smooth(plus_dm, period)
    smoothed_minus_dm = wilder_smooth(minus_dm, period)

    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr

    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.where(di_sum != 0)  # avoid 0/0 on a flat market

    adx = wilder_smooth(dx, period)

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "dx": dx, "adx": adx})


def classify_regime(adx_value: float, trending_threshold: float, ranging_threshold: float) -> str:
    """'trending' | 'ranging' | 'transitional' (between the two thresholds) | 'unknown'
    (NaN, still warming up). transitional is deliberately its own label, not silently
    folded into either side — spec doesn't say what to do with the gray zone, so a
    strategy consuming this must make that choice explicitly, not by default."""
    if pd.isna(adx_value):
        return "unknown"
    if adx_value >= trending_threshold:
        return "trending"
    if adx_value <= ranging_threshold:
        return "ranging"
    return "transitional"
