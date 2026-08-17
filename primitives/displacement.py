"""Displacement (spec §2.9) — a strong, structure-relevant candle. Gates order-block
validity (§2.6) and FVG (§2.11, not built in v1), and separates a genuine structure
break from price limping across a level.

Caveat carried forward from the spec: displacement clusters around news/volatility, so
it's correlated with session/event context — don't treat it as fully independent
confluence when this feeds into §4's scoring later.
"""
from __future__ import annotations

import pandas as pd


def is_displacement(df: pd.DataFrame, atr: pd.Series, body_ratio_min: float, atr_multiple_min: float) -> pd.Series:
    """Per-bar boolean Series. df needs 'open','high','low','close'; atr aligned by
    index. A zero-range bar (high == low) is never displacement (avoids 0/0); a bar
    during ATR warm-up (atr is NaN) is never displacement either — comparisons against
    NaN correctly evaluate False rather than raising.
    """
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    has_range = rng > 0
    body_ratio = body.where(has_range, 0) / rng.where(has_range, 1)
    return has_range & (body_ratio >= body_ratio_min) & (rng >= atr_multiple_min * atr)
