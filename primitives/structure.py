"""Market structure — dual-resolution BOS/CHoCH state machines (spec §2.2-2.4).

Two independent state machines (internal, external) run in lockstep over the same bar
sequence, each maintaining its own direction and last_SH/last_SL reference swings. Both
are fed confirmed swings only at the bar where they become confirmed (§2.1's lag), so
nothing here can reference a swing before it would genuinely be known live.

Design choices made where the spec's prose leaves a gap (documented here, not hidden):
- "last_SH"/"last_SL" are read literally as *most recently confirmed* swing of that
  kind, updated unconditionally whenever a new one confirms — not "the most extreme
  since the last break". This is simplest and matches the literal wording; fractal
  swings alternate high/low in practice for continuous price action, so this rarely
  diverges from "most extreme" anyway.
- The very first break (while direction == "undefined") establishes initial bias. It is
  logged as its own "initial" event type — not BOS (no trend to continue) or CHoCH (no
  trend to reverse), since both presuppose an existing direction.
- On a bar where both internal and external could fire, external is evaluated first, so
  an internal event's entry-eligibility check (below) sees the external state *after*
  any same-bar external flip — external bias is the coarser context internal timing
  operates inside of.

Entry eligibility (§2.2's interaction rule) is the conjunction of two independently
exposed conditions — `direction_aligned` and `inside_external_leg` — combined into
`entry_eligible`. They're kept separate (not just a single bool) so a caller can see
*why* a break wasn't an entry: direction misalignment and being outside the current
dealing range are different failure modes with different implications for the §4
confluence layer. `inside_external_leg` is checked against primitives.premium_discount's
dealing-range state, which is genuinely persistent (unlike this module's last_SH/last_SL,
which get cleared on every break — see test_break below) — an earlier version tried
reusing the cleared references for this and produced false negatives on aligned breaks;
see memory: primitives-design-decisions.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from primitives.swings import Swing
from primitives.premium_discount import DealingRangeState, range_as_of

Direction = str  # "bullish" | "bearish" | "undefined"


@dataclass(frozen=True)
class StructureEvent:
    bar_index: int
    time: pd.Timestamp
    resolution: str                 # "internal" | "external"
    event_type: str                  # "initial" | "BOS" | "CHoCH"
    direction: Direction              # direction AFTER this event
    broken_level: float
    broken_swing_bar_index: int
    direction_aligned: Optional[bool] = None    # meaningful only for internal-resolution events
    inside_external_leg: Optional[bool] = None    # meaningful only for internal-resolution events

    @property
    def entry_eligible(self) -> Optional[bool]:
        """§2.2: "in the direction of external bias, inside the external leg" — a
        conjunction. None (not just False) when either component wasn't evaluated,
        so a missing check is never silently read as ineligible."""
        if self.direction_aligned is None or self.inside_external_leg is None:
            return None
        return self.direction_aligned and self.inside_external_leg


@dataclass
class StructureResult:
    internal_events: list[StructureEvent]
    external_events: list[StructureEvent]


class _ResolutionState:
    """One resolution's (internal or external) running structure state."""

    def __init__(self, resolution: str):
        self.resolution = resolution
        self.direction: Direction = "undefined"
        self.last_SH: Optional[Swing] = None
        self.last_SL: Optional[Swing] = None

    def ingest_confirmed_swing(self, swing: Swing) -> None:
        if swing.kind == "high":
            self.last_SH = swing
        else:
            self.last_SL = swing

    def test_break(self, bar_index: int, time: pd.Timestamp, close: float) -> Optional[StructureEvent]:
        broke_high = self.last_SH is not None and close > self.last_SH.price
        broke_low = self.last_SL is not None and close < self.last_SL.price

        if self.direction == "undefined":
            if broke_high:
                self.direction = "bullish"
                event = StructureEvent(bar_index, time, self.resolution, "initial", "bullish",
                                        self.last_SH.price, self.last_SH.bar_index)
                self.last_SH = None  # broken level is "used up" — advances (§2.3) via the
                return event          # next confirmed swing high, not re-tested every bar
            if broke_low:
                self.direction = "bearish"
                event = StructureEvent(bar_index, time, self.resolution, "initial", "bearish",
                                        self.last_SL.price, self.last_SL.bar_index)
                self.last_SL = None
                return event
            return None

        if self.direction == "bullish":
            if broke_high:
                event = StructureEvent(bar_index, time, self.resolution, "BOS", "bullish",
                                        self.last_SH.price, self.last_SH.bar_index)
                self.last_SH = None
                return event
            if broke_low:
                self.direction = "bearish"
                event = StructureEvent(bar_index, time, self.resolution, "CHoCH", "bearish",
                                        self.last_SL.price, self.last_SL.bar_index)
                self.last_SL = None
                return event
            return None

        # self.direction == "bearish"
        if broke_low:
            event = StructureEvent(bar_index, time, self.resolution, "BOS", "bearish",
                                    self.last_SL.price, self.last_SL.bar_index)
            self.last_SL = None
            return event
        if broke_high:
            self.direction = "bullish"
            event = StructureEvent(bar_index, time, self.resolution, "CHoCH", "bullish",
                                    self.last_SH.price, self.last_SH.bar_index)
            self.last_SH = None
            return event
        return None


def _group_by_confirm_index(swings: list[Swing]) -> dict[int, list[Swing]]:
    groups: dict[int, list[Swing]] = {}
    for s in swings:
        groups.setdefault(s.confirmed_at_index, []).append(s)
    return groups


def _check_direction_aligned(int_event: StructureEvent, external_state: _ResolutionState) -> bool:
    """§2.2: an internal BOS/CHoCH's resulting direction must match the CURRENT external
    direction. An internal break against external bias is a candidate reversal only —
    not an entry — until external structure itself flips (the case spec §10 requires a
    negative test for)."""
    if external_state.direction == "undefined":
        return False
    return int_event.direction == external_state.direction


def _check_inside_external_leg(close: float, dealing_ranges: list[DealingRangeState], bar_index: int) -> bool:
    """§2.2/§2.10: price must be inside the current external dealing range. No range has
    ever formed yet, or the newest leg failed the min-leg-ATR guard and nothing valid was
    ever recorded (§2.10) -> not inside anything, so not eligible."""
    state = range_as_of(dealing_ranges, bar_index)
    if state is None:
        return False
    return state.range_low <= close <= state.range_high


def run_single_resolution_structure(df: pd.DataFrame, swings: list[Swing], resolution: str = "single") -> list[StructureEvent]:
    """One state machine over one swing set — for standalone single-timeframe use (e.g.
    signals/htf_bias.py's H4 structure direction), where there's no separate external
    context to check entry-eligibility against. direction_aligned/inside_external_leg
    stay None on every returned event (their default) — reusing the EXACT SAME
    BOS/CHoCH mechanics as the dual-resolution engine, not a re-implementation, so the
    two can never silently drift apart.
    """
    state = _ResolutionState(resolution)
    by_confirm = _group_by_confirm_index(swings)
    closes = df["close"].to_numpy()
    times = df["time"].to_numpy()

    events: list[StructureEvent] = []
    for i in range(len(df)):
        for s in by_confirm.get(i, []):
            state.ingest_confirmed_swing(s)
        event = state.test_break(i, pd.Timestamp(times[i]), float(closes[i]))
        if event is not None:
            events.append(event)
    return events


def run_dual_resolution_structure(
    df: pd.DataFrame,
    internal_swings: list[Swing],
    external_swings: list[Swing],
    dealing_ranges: list[DealingRangeState],
) -> StructureResult:
    """df must have 'time' and 'close' columns, positionally indexed 0..len-1, aligned
    with the same indexing used to produce internal_swings/external_swings.
    dealing_ranges: from primitives.premium_discount.compute_dealing_ranges, built from
    the SAME external_swings passed here — required (not optional) so the full §2.2
    interaction rule (direction + leg containment) is always evaluated, never silently
    reduced to the direction-only partial check.
    """
    internal_state = _ResolutionState("internal")
    external_state = _ResolutionState("external")

    internal_by_confirm = _group_by_confirm_index(internal_swings)
    external_by_confirm = _group_by_confirm_index(external_swings)

    internal_events: list[StructureEvent] = []
    external_events: list[StructureEvent] = []

    closes = df["close"].to_numpy()
    times = df["time"].to_numpy()

    for i in range(len(df)):
        for s in internal_by_confirm.get(i, []):
            internal_state.ingest_confirmed_swing(s)
        for s in external_by_confirm.get(i, []):
            external_state.ingest_confirmed_swing(s)

        ts = pd.Timestamp(times[i])
        close = float(closes[i])

        ext_event = external_state.test_break(i, ts, close)
        if ext_event is not None:
            external_events.append(ext_event)

        int_event = internal_state.test_break(i, ts, close)
        if int_event is not None:
            direction_aligned = _check_direction_aligned(int_event, external_state)
            inside_leg = _check_inside_external_leg(close, dealing_ranges, i)
            internal_events.append(dataclasses.replace(
                int_event, direction_aligned=direction_aligned, inside_external_leg=inside_leg,
            ))

    return StructureResult(internal_events=internal_events, external_events=external_events)
