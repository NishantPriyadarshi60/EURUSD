"""Swing points — fractal method (spec §2.1). The atom everything else in §2 is built on.

A swing at bar i is only *confirmed* at bar i+N (spec's explicit no-lookahead rule for
swings) — a signal may reference a swing only from its confirmed_at_index onward, never
from bar_index itself. Every downstream consumer (structure, BOS/CHoCH, order blocks)
must key off confirmed_at_index, not bar_index.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

Role = str    # "internal" | "external"
Kind = str    # "high" | "low"


@dataclass(frozen=True)
class Swing:
    bar_index: int              # the pivot bar itself
    confirmed_at_index: int      # bar_index + n — earliest index a signal may reference this
    time: pd.Timestamp
    confirmed_at_time: pd.Timestamp
    price: float
    kind: Kind
    role: Role
    n: int


def detect_swings(df: pd.DataFrame, n: int, role: Role) -> list[Swing]:
    """df must have 'time', 'high', 'low' columns, positionally indexed 0..len-1.
    Strict inequality per spec: high[i] > high[i±k] for all k in 1..n (equal values do
    NOT count as a swing — avoids ambiguous ties).
    """
    if n < 1:
        raise ValueError("fractal width n must be >= 1")

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df["time"].to_numpy()
    n_bars = len(df)

    swings: list[Swing] = []
    for i in range(n, n_bars - n):
        window_highs = highs[i - n : i + n + 1]
        window_lows = lows[i - n : i + n + 1]
        pivot_high = highs[i]
        pivot_low = lows[i]

        is_swing_high = pivot_high == window_highs.max() and (window_highs == pivot_high).sum() == 1
        is_swing_low = pivot_low == window_lows.min() and (window_lows == pivot_low).sum() == 1

        confirmed_idx = i + n
        if is_swing_high:
            swings.append(Swing(
                bar_index=i, confirmed_at_index=confirmed_idx,
                time=pd.Timestamp(times[i]), confirmed_at_time=pd.Timestamp(times[confirmed_idx]),
                price=float(pivot_high), kind="high", role=role, n=n,
            ))
        if is_swing_low:
            swings.append(Swing(
                bar_index=i, confirmed_at_index=confirmed_idx,
                time=pd.Timestamp(times[i]), confirmed_at_time=pd.Timestamp(times[confirmed_idx]),
                price=float(pivot_low), kind="low", role=role, n=n,
            ))

    swings.sort(key=lambda s: (s.bar_index, s.kind))
    return swings


def compute_dual_resolution_swings(df: pd.DataFrame, n_internal: int, n_external: int) -> dict[str, list[Swing]]:
    """Role-assigned per spec §2.1/§2.2 — n_internal and n_external are fixed roles,
    not a single N searched/tuned to make structure 'look right' (§0.4, §11)."""
    return {
        "internal": detect_swings(df, n_internal, role="internal"),
        "external": detect_swings(df, n_external, role="external"),
    }
