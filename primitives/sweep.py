"""Sweep / liquidity grab detection (spec §2.5) — distinct from a break (§2.3/§2.4).

Hard contrast per spec: BOS = close beyond a level and stay; sweep = wick beyond, close
back inside, within a bounded reclaim window. This module implements sweep detection
only, independently of structure.py's close-only BOS/CHoCH logic — the two are computed
from different rules and are not forced mutually exclusive here. If a reclaim spans
multiple bars, it's possible for a bar's close to have already tripped a structure.py
break before a later bar in the same window reclaims the level; that overlap is a real,
expected instance of the correlated-signal problem spec §4 asks the confluence layer to
detect and account for — not something to paper over inside the primitive itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from primitives.swings import Swing


@dataclass(frozen=True)
class SweepEvent:
    pierce_bar_index: int
    reclaim_bar_index: int    # bar whose close reclaimed the level; event is usable from here
    kind: str                   # "sell_side" | "buy_side"
    level_price: float
    level_swing_bar_index: int
    pierce_extreme: float        # the low (sell-side) / high (buy-side) that pierced


def detect_sweeps(
    df: pd.DataFrame,
    target_swings: list[Swing],
    atr: pd.Series,
    tol_atr: float,
    window_bars: int,
) -> list[SweepEvent]:
    """target_swings: confirmed swing highs/lows to test as liquidity levels — raw
    Swing objects (§2.1) or primitives.liquidity.LiquidityPool objects (§2.8 EQH/EQL,
    the prime sweep targets per spec §2.7). Both expose the same
    price/kind/confirmed_at_index/bar_index shape this function relies on, so either can
    be passed directly with no adapter. Each target is only tested from its own
    confirmed_at_index onward (§2.1 lag) and for at most one pierce — its first
    interaction resolves it (swept or not); we don't re-test the same level later.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n_bars = len(df)

    events: list[SweepEvent] = []

    for swing in target_swings:
        level = swing.price
        for i in range(swing.confirmed_at_index, n_bars):
            tol = tol_atr * atr.iloc[i]
            if pd.isna(tol):
                continue

            if swing.kind == "low":
                if lows[i] >= level - tol:
                    continue
                reclaim_idx = _find_reclaim(closes, i, window_bars, n_bars, level, above=True)
                if reclaim_idx is not None:
                    events.append(SweepEvent(i, reclaim_idx, "sell_side", level, swing.bar_index, lows[i]))
                break  # level resolved (pierced), whether reclaimed or not
            else:
                if highs[i] <= level + tol:
                    continue
                reclaim_idx = _find_reclaim(closes, i, window_bars, n_bars, level, above=False)
                if reclaim_idx is not None:
                    events.append(SweepEvent(i, reclaim_idx, "buy_side", level, swing.bar_index, highs[i]))
                break

    events.sort(key=lambda e: e.pierce_bar_index)
    return events


def _find_reclaim(closes, pierce_idx: int, window_bars: int, n_bars: int, level: float, above: bool) -> Optional[int]:
    end = min(pierce_idx + window_bars, n_bars - 1)
    for j in range(pierce_idx, end + 1):
        if (closes[j] > level) if above else (closes[j] < level):
            return j
    return None
