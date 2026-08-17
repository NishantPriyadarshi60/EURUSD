"""Average True Range — Wilder's smoothing (the conventional default; MT5's built-in ATR
uses the same method). Used to scale sweep/S-R/EQH tolerances and the displacement gate.
"""
from __future__ import annotations

import pandas as pd


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """df needs 'high','low','close'. TR[i] uses bar i's own high/low and bar i-1's
    close (all known at bar i's own close — not lookahead). Shared by ATR and ADX
    (signals/regime.py) so the definition can't drift between the two."""
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's recursive smoothing, generalized to work on any series — including one
    that itself starts with leading NaNs (e.g. ADX smoothing DX, in signals/regime.py,
    where DX isn't defined until the underlying +DI/-DI have already warmed up).

    Seeding: starting from the series' first valid value, the NEXT `period` values
    (excluding that first one) are averaged to seed the smoothed series; every value
    after that follows smoothed[i] = (smoothed[i-1]*(period-1) + series[i]) / period.
    This "skip the boundary value" convention matches the original ATR seeding (which
    excludes bar 0's true range, since it has no real previous close to compare
    against) — expressed here as a general rule so ADX's warmup naturally comes out to
    ~2x the period (seed on DX, which itself already needed `period` bars to exist),
    matching standard ADX behavior, without hardcoding index 0 anywhere.
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    smoothed = pd.Series(float("nan"), index=series.index, dtype=float)

    first_valid = next((i for i in range(n) if not pd.isna(values[i])), None)
    if first_valid is None:
        return smoothed

    seed_start = first_valid + 1
    seed_end = first_valid + period  # inclusive
    if seed_end >= n:
        return smoothed

    smoothed.iloc[seed_end] = values[seed_start:seed_end + 1].mean()
    for i in range(seed_end + 1, n):
        smoothed.iloc[i] = (smoothed.iloc[i - 1] * (period - 1) + values[i]) / period
    return smoothed


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """First ~`period` values are NaN (insufficient history to warm up); callers must
    handle that, not silently zero-fill."""
    return wilder_smooth(compute_true_range(df), period)
