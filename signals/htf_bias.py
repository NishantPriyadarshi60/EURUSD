"""HTF bias gate (spec §3.1) — directional gate computed on H4 (the locked primary
anchor; D1 is an optional off-by-default secondary — see memory: timeframe-architecture)
using only CLOSED H4 bars.

The no-lookahead requirement is explicit in the spec ("no lookahead into a forming HTF
candle") and is the entire reason this module exists rather than just reusing
structure.py directly on H4 data: an M15 bar must only see the H4 direction as of the
LAST H4 bar that had FULLY CLOSED by that M15 bar's own close — never the H4 bar still
forming. Alignment uses each bar's own CLOSE time (open + timeframe duration) via
pandas merge_asof(direction='backward'), which finds exactly "the latest H4 close-time
at or before this M15 bar's own close-time" without assuming bars are perfectly
contiguous (weekend gaps are a real concern here).

Assumes both htf_df and entry_df are already time-sorted ascending, which every
snapshot in this project already is (spec §1.5's monotonic-time quality gate).
"""
from __future__ import annotations

import pandas as pd

from primitives.swings import detect_swings
from primitives.structure import run_single_resolution_structure


def compute_htf_direction_timeline(htf_df: pd.DataFrame, swing_n: int, htf_timeframe_hours: int) -> pd.DataFrame:
    """One row per htf_df bar: 'close_time' (open + timeframe duration) and
    'direction' (the H4 structure direction AS OF that bar's own close — i.e.
    immediately after processing it, forward-filled between structure events since
    direction only changes ON an event)."""
    swings = detect_swings(htf_df, n=swing_n, role="htf")
    events = run_single_resolution_structure(htf_df, swings, resolution="htf")

    direction = pd.Series(index=htf_df.index, dtype=object)
    for event in events:
        direction.iloc[event.bar_index] = event.direction
    direction = direction.ffill().fillna("undefined")

    close_time = htf_df["time"] + pd.Timedelta(hours=htf_timeframe_hours)
    return pd.DataFrame({"close_time": close_time, "direction": direction})


def align_htf_direction_to_entry_tf(
    entry_df: pd.DataFrame, htf_timeline: pd.DataFrame, entry_timeframe_minutes: int,
) -> pd.Series:
    """For each entry-TF bar, the HTF direction as of the latest HTF bar that had
    FULLY CLOSED by this bar's own close."""
    entry_close_time = entry_df["time"] + pd.Timedelta(minutes=entry_timeframe_minutes)
    left = pd.DataFrame({"close_time": entry_close_time})
    right = htf_timeline[["close_time", "direction"]].sort_values("close_time")
    merged = pd.merge_asof(left, right, on="close_time", direction="backward")
    result = merged["direction"].fillna("undefined")
    result.index = entry_df.index
    return result


def compute_htf_bias(
    htf_df: pd.DataFrame, entry_df: pd.DataFrame,
    swing_n: int, htf_timeframe_hours: int, entry_timeframe_minutes: int,
) -> pd.Series:
    timeline = compute_htf_direction_timeline(htf_df, swing_n, htf_timeframe_hours)
    return align_htf_direction_to_entry_tf(entry_df, timeline, entry_timeframe_minutes)


def htf_gate_allows(htf_direction: str, trade_direction: str, allow_neutral: bool) -> bool:
    """§3.1: "entry-TF long setups allowed only when HTF bias is bullish (or neutral,
    per config); shorts only when bearish." 'neutral' maps to this system's "undefined"
    state (no HTF break has established a direction yet)."""
    if trade_direction not in ("long", "short"):
        raise ValueError(f"Unknown trade_direction: {trade_direction!r}")
    if htf_direction == "undefined":
        return allow_neutral
    required = "bullish" if trade_direction == "long" else "bearish"
    return htf_direction == required
