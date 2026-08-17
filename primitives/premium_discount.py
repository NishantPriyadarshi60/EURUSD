"""Premium / discount (equilibrium) — spec §2.10.

The dealing range is the current EXTERNAL impulse leg (§2.2): from its most recently
confirmed external swing low to swing high (or vice versa) — using the external leg
makes the range stable by construction and removes the discretionary "which leg" choice.

This is DELIBERATELY separate state from structure.py's last_SH/last_SL: those get
cleared the instant they're broken (so BOS doesn't re-fire every bar — see
structure.py's test_break), which makes them wrong as a persistent dealing range. An
earlier attempt to reuse them for the §2.2 interaction rule's "inside the external leg"
check produced false negatives for exactly this reason (see memory:
primitives-design-decisions). Here we track the raw most-recently-confirmed external
swing of each kind, uncleared, purely to define the range.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from primitives.swings import Swing


@dataclass(frozen=True)
class DealingRangeState:
    as_of_index: int          # bar at which this determination became active
    range_low: float
    range_high: float
    equilibrium: float
    stale: bool                 # True if frozen from an earlier valid leg because the
                                  # newest candidate leg failed the min-leg-ATR guard
    low_swing_bar_index: int
    high_swing_bar_index: int


def compute_dealing_ranges(
    external_swings: list[Swing],
    atr: pd.Series,
    min_leg_atr: float,
) -> list[DealingRangeState]:
    """One entry per bar where the range genuinely changed: either a new valid leg
    cleared the min-leg-ATR guard, or a sub-threshold candidate arrived and got rejected
    (recorded as stale, freezing the prior valid range — spec's "mark stale in the log
    so its influence is visible," not a silent redraw). Look up the applicable range for
    a given bar via `range_as_of`.
    """
    events_by_confirm: dict[int, list[Swing]] = {}
    for s in external_swings:
        events_by_confirm.setdefault(s.confirmed_at_index, []).append(s)

    raw_last_high: Optional[Swing] = None
    raw_last_low: Optional[Swing] = None
    valid_range: Optional[DealingRangeState] = None
    states: list[DealingRangeState] = []

    for confirm_idx in sorted(events_by_confirm):
        for s in events_by_confirm[confirm_idx]:
            if s.kind == "high":
                raw_last_high = s
            else:
                raw_last_low = s

        if raw_last_high is None or raw_last_low is None:
            continue  # need both endpoints before any candidate leg exists

        candidate_low = min(raw_last_high.price, raw_last_low.price)
        candidate_high = max(raw_last_high.price, raw_last_low.price)
        span = candidate_high - candidate_low

        atr_value = atr.iloc[confirm_idx] if confirm_idx < len(atr) else float("nan")
        guard_ok = pd.notna(atr_value) and span >= min_leg_atr * atr_value

        if guard_ok:
            valid_range = DealingRangeState(
                as_of_index=confirm_idx, range_low=candidate_low, range_high=candidate_high,
                equilibrium=(candidate_low + candidate_high) / 2, stale=False,
                low_swing_bar_index=raw_last_low.bar_index, high_swing_bar_index=raw_last_high.bar_index,
            )
            states.append(valid_range)
        elif valid_range is not None:
            states.append(DealingRangeState(
                as_of_index=confirm_idx, range_low=valid_range.range_low, range_high=valid_range.range_high,
                equilibrium=valid_range.equilibrium, stale=True,
                low_swing_bar_index=valid_range.low_swing_bar_index,
                high_swing_bar_index=valid_range.high_swing_bar_index,
            ))
        # else: no valid range has ever formed and this candidate doesn't qualify either
        # -- nothing to record; range_as_of correctly returns None for bars this early.

    return states


def range_as_of(states: list[DealingRangeState], bar_index: int) -> Optional[DealingRangeState]:
    """Most recent range state usable at bar_index (its as_of_index <= bar_index).
    None if no qualifying leg had formed yet by this bar."""
    applicable = None
    for state in states:
        if state.as_of_index <= bar_index:
            applicable = state
        else:
            break
    return applicable


def classify_position(price: float, range_low: float, range_high: float) -> Optional[str]:
    """'discount' below equilibrium, 'premium' above, 'equilibrium' exactly at the
    midpoint. None for a degenerate zero-width range or a price outside [low, high] --
    callers must treat "outside the range" as its own case, not silently coerce it.
    """
    if range_high <= range_low:
        return None
    if price < range_low or price > range_high:
        return None
    equilibrium = (range_low + range_high) / 2
    if price < equilibrium:
        return "discount"
    if price > equilibrium:
        return "premium"
    return "equilibrium"
