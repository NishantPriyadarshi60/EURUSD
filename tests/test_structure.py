import pandas as pd

from primitives.swings import detect_swings
from primitives.structure import run_dual_resolution_structure, run_single_resolution_structure
from primitives.premium_discount import DealingRangeState


def _df(closes):
    n = len(closes)
    times = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    # high == low == close: swings become pure local-extrema-of-close, and "close beyond
    # a level" is exactly "close beyond a prior close" — makes every scenario in this
    # file hand-traceable without OHLC noise obscuring the structure logic under test.
    return pd.DataFrame({"time": times, "open": closes, "high": closes, "low": closes, "close": closes})


_WIDE_OPEN_RANGE = [DealingRangeState(
    as_of_index=0, range_low=-1e9, range_high=1e9, equilibrium=0.0, stale=False,
    low_swing_bar_index=0, high_swing_bar_index=0,
)]


def _run(closes, n, dealing_ranges=None):
    df = _df(closes)
    swings = detect_swings(df, n=n, role="external")
    # single-resolution check: feed the same swings as both internal and external so
    # the dual-resolution engine degenerates to one state machine we can hand-verify.
    # A wide-open dealing range (always containing every close) isolates whatever this
    # test is actually checking from the separate leg-containment condition -- see the
    # dedicated leg-containment tests further down for that half of entry_eligible.
    ranges = dealing_ranges if dealing_ranges is not None else _WIDE_OPEN_RANGE
    return run_dual_resolution_structure(df, swings, swings, ranges)


# ---------------------------------------------------------------------------
# Single-resolution mechanics (internal == external swings here, so only one state
# machine is really exercised; entry_eligible is ignored in these tests).
# ---------------------------------------------------------------------------

def test_first_break_is_initial_not_bos_or_choch():
    # Hand-traced at n=1: SL confirmed at idx2/conf3 (price 8), SH confirmed at idx5/conf6
    # (price 11). First break: close[9]=7 < last_SL(8) while direction is still
    # "undefined" -> must be logged as "initial", not BOS or CHoCH (neither trend exists yet).
    closes = [10, 9, 8, 9, 10, 11, 10, 9, 8, 7, 8, 9, 10, 12]
    result = _run(closes, n=1)
    events = result.external_events
    assert events[0].event_type == "initial"
    assert events[0].direction == "bearish"
    assert events[0].bar_index == 9
    assert events[0].broken_level == 8


def test_choch_flips_direction_after_initial_break():
    # Continuing the same series: close[13]=12 > last_SH(11) while direction is bearish
    # (set by the initial break) -> CHoCH, flips direction to bullish.
    closes = [10, 9, 8, 9, 10, 11, 10, 9, 8, 7, 8, 9, 10, 12]
    result = _run(closes, n=1)
    events = result.external_events
    assert len(events) == 2
    choch = events[1]
    assert choch.event_type == "CHoCH"
    assert choch.direction == "bullish"
    assert choch.bar_index == 13
    assert choch.broken_level == 11


def test_bos_fires_once_direction_already_established():
    # Hand-traced at n=1: SH@2(conf3,price7), SL@4(conf5,price5), SH@7(conf8,price8),
    # SL@9(conf10,price6). close[7]=8 > last_SH(7) while undefined -> "initial" bullish.
    # Later close[11]=9 > last_SH(8) while already bullish -> BOS (continuation).
    closes = [5, 6, 7, 6, 5, 6, 7, 8, 7, 6, 7, 9]
    result = _run(closes, n=1)
    events = result.external_events
    assert len(events) == 2
    initial, bos = events
    assert initial.event_type == "initial" and initial.direction == "bullish" and initial.bar_index == 7
    assert bos.event_type == "BOS" and bos.direction == "bullish" and bos.bar_index == 11
    assert bos.broken_level == 8  # the swing high confirmed at bar 8, not the stale bar-7 level


def test_broken_level_does_not_refire_every_bar():
    # Regression test for a real bug found while building this: a monotonic run past a
    # broken level must fire exactly ONE break event, not one per bar. The broken
    # reference must be cleared ("advances" per spec §2.3) until a new swing replaces it.
    closes = [5, 6, 7, 6, 5, 6, 7, 8, 7, 6, 7, 9, 11, 13, 15, 17]
    result = _run(closes, n=1)
    bos_events = [e for e in result.external_events if e.event_type == "BOS"]
    # Only bar 11 should fire (breaking the bar-8-confirmed high of 8); the subsequent
    # monotonic climb (13, 15, 17) must not re-fire since no new swing high re-confirms.
    assert [e.bar_index for e in bos_events] == [11]


def test_single_resolution_matches_the_degenerate_dual_resolution_case():
    # run_single_resolution_structure must reuse the identical BOS/CHoCH mechanics as
    # run_dual_resolution_structure -- feeding the same swings to both internal and
    # external slots of the dual-resolution engine must reproduce exactly the single-
    # resolution result, event for event, with entry-eligibility fields left None
    # (there's no separate external context in a standalone single-timeframe run).
    closes = [10, 9, 8, 9, 10, 11, 10, 9, 8, 7, 8, 9, 10, 12]
    df = _df(closes)
    swings = detect_swings(df, n=1, role="single")

    single_events = run_single_resolution_structure(df, swings, resolution="single")
    dual_result = run_dual_resolution_structure(df, swings, swings, _WIDE_OPEN_RANGE)

    assert len(single_events) == len(dual_result.external_events) == 2
    for se, de in zip(single_events, dual_result.external_events):
        assert se.bar_index == de.bar_index
        assert se.event_type == de.event_type
        assert se.direction == de.direction
        assert se.broken_level == de.broken_level
        assert se.direction_aligned is None
        assert se.inside_external_leg is None
        assert se.entry_eligible is None


def test_no_lookahead_event_not_detectable_before_confirmation_bar_exists():
    # Same series as the initial-break test; the bearish "initial" event at bar 9 depends
    # on the swing low confirmed at bar 3. Truncating the series before bar 9 must not
    # produce that event, proving the state machine can't see it before it's live-knowable.
    closes = [10, 9, 8, 9, 10, 11, 10, 9, 8, 7, 8, 9, 10, 12]
    truncated = closes[:9]  # indices 0..8, stops one bar short of the break at bar 9
    result = _run(truncated, n=1)
    assert result.external_events == []


# ---------------------------------------------------------------------------
# Direction-alignment half of the interaction rule (spec §2.2, and the required
# negative-case test from spec §10). Uses a wide-open dealing range throughout, so
# entry_eligible varies ONLY with direction alignment here -- leg containment is tested
# separately below. Scenario built and verified interactively, then transcribed here:
#   - a macro external downtrend forms (external "initial" bearish at bar 10)
#   - an internal-only up-wiggle causes an aligned internal BOS at bar 17 (still bearish)
#   - a genuine internal reversal (CHoCH to bullish) fires at bar 28 while external is
#     STILL bearish (external doesn't flip until bar 31) -> must be blocked
#   - external itself flips bullish (CHoCH) at bar 31
#   - a further internal bullish BOS at bar 36, now aligned with the (now-bullish)
#     external bias -> must be entry-eligible again
# ---------------------------------------------------------------------------

def _ramp(a, b, n):
    step = (b - a) / (n - 1)
    return [round(a + step * i, 2) for i in range(n)]


def _dual_resolution_scenario_closes():
    closes = []
    closes += _ramp(100, 90, 5)        # 0-4: decline
    closes += _ramp(90, 96, 5)[1:]       # 5-8: pullback (lower high than 100)
    closes += _ramp(96, 80, 6)[1:]        # 9-13: decline to new low
    closes += [79, 78, 80, 77]              # 14-17: internal-only wiggle (up then down)
    closes += _ramp(77, 65, 6)[1:]            # 18-22: decline to new low
    closes += [64, 63]                          # 23-24: further decline
    closes += _ramp(63, 110, 10)[1:]              # 25-33: big reversal, flips external too
    closes += [108, 106, 115]                       # 34-36: pullback then new high
    return closes


def test_internal_counter_trend_break_blocked_until_external_flips():
    closes = _dual_resolution_scenario_closes()
    df = _df(closes)
    internal = detect_swings(df, n=1, role="internal")
    external = detect_swings(df, n=3, role="external")
    result = run_dual_resolution_structure(df, internal, external, _WIDE_OPEN_RANGE)

    ext_by_bar = {e.bar_index: e for e in result.external_events}
    int_by_bar = {e.bar_index: e for e in result.internal_events}

    # External establishes bearish bias at bar 10 and does not flip until bar 31.
    assert ext_by_bar[10].event_type == "initial" and ext_by_bar[10].direction == "bearish"
    assert ext_by_bar[31].event_type == "CHoCH" and ext_by_bar[31].direction == "bullish"

    # THE REQUIRED NEGATIVE CASE (spec §10): an internal bullish CHoCH fires at bar 28,
    # while external bias is still bearish (bar 28 < bar 31) -> must NOT be entry-eligible,
    # specifically because direction is misaligned (leg containment is wide-open here).
    assert int_by_bar[28].event_type == "CHoCH"
    assert int_by_bar[28].direction == "bullish"
    assert int_by_bar[28].direction_aligned is False
    assert int_by_bar[28].entry_eligible is False

    # Happy-path contrast, so the negative case above isn't vacuously true: an internal
    # bearish BOS at bar 17, aligned with the (still-bearish) external bias, IS eligible.
    assert int_by_bar[17].direction == "bearish"
    assert int_by_bar[17].direction_aligned is True
    assert int_by_bar[17].entry_eligible is True

    # And once external genuinely flips (bar 31), a later aligned internal break (bar 36,
    # bullish BOS) becomes entry-eligible again -> proves the block is lifted by an actual
    # external CHoCH, not just by direction happening to match forever after.
    assert int_by_bar[36].direction == "bullish"
    assert int_by_bar[36].direction_aligned is True
    assert int_by_bar[36].entry_eligible is True


# ---------------------------------------------------------------------------
# Leg-containment half of the interaction rule (spec §2.2/§2.10). Reuses the simple
# single-resolution series from test_bos_fires_once_direction_already_established, where
# internal == external swings so direction_aligned is always True by construction --
# isolating inside_external_leg as the only variable that can block entry_eligible here.
# ---------------------------------------------------------------------------

def test_direction_aligned_but_outside_leg_is_not_entry_eligible():
    closes = [5, 6, 7, 6, 5, 6, 7, 8, 7, 6, 7, 9]  # "initial" bullish @ bar7, BOS @ bar11
    ranges = [DealingRangeState(
        as_of_index=0, range_low=20.0, range_high=30.0, equilibrium=25.0, stale=False,
        low_swing_bar_index=0, high_swing_bar_index=0,
    )]  # never contains any close in this series (5-9 range)
    result = _run(closes, n=1, dealing_ranges=ranges)
    initial = result.internal_events[0]
    assert initial.direction_aligned is True    # internal == external swings here
    assert initial.inside_external_leg is False   # close=8 is nowhere near [20, 30]
    assert initial.entry_eligible is False


def test_direction_aligned_and_inside_leg_is_entry_eligible():
    closes = [5, 6, 7, 6, 5, 6, 7, 8, 7, 6, 7, 9]
    ranges = [
        DealingRangeState(as_of_index=0, range_low=20.0, range_high=30.0, equilibrium=25.0,
                           stale=False, low_swing_bar_index=0, high_swing_bar_index=0),
        # a fresh range becomes active before bar 11 (BOS, close=9) and genuinely contains it
        DealingRangeState(as_of_index=9, range_low=5.0, range_high=10.0, equilibrium=7.5,
                           stale=False, low_swing_bar_index=0, high_swing_bar_index=0),
    ]
    result = _run(closes, n=1, dealing_ranges=ranges)
    bos = next(e for e in result.internal_events if e.event_type == "BOS")
    assert bos.bar_index == 11
    assert bos.direction_aligned is True
    assert bos.inside_external_leg is True   # close=9 is inside [5, 10], the range active by bar 11
    assert bos.entry_eligible is True
