"""Automated lookahead audit (spec §6), run against a real slice of M15 market data —
not synthetic bars — since a bug that only shows up on real price action's irregular
swing spacing would be invisible to hand-picked synthetic scenarios. This is in addition
to (not a replacement for) each primitive's own no-lookahead unit test.

Scope note on what's audited via strict prefix-equality vs. not, and why: Swing, ATR,
StructureEvent, SweepEvent, and DealingRangeState are all "write-once" facts that must
never change once knowable, so exact equality across truncation cutoffs is the right
test. OrderBlock's mitigation status and SRLevel/LiquidityPool's running-average
price/touch-count are DELIBERATELY NOT prefix-stable in that strict sense — they
legitimately keep evolving as more data arrives after their creation point (a later
mitigation touch, a later equal high joining a cluster). Comparing those fields for
strict equality across cutoffs would flag false positives, not real bugs. OrderBlock is
audited on a projection that excludes mitigation; liquidity clustering is instead
checked with a narrower, still-meaningful property (no touch is ever used before its own
confirm index) rather than whole-object equality.
"""
import dataclasses

import pandas as pd
import pytest

from primitives.atr import compute_atr
from primitives.swings import detect_swings, compute_dual_resolution_swings
from primitives.structure import run_dual_resolution_structure
from primitives.sweep import detect_sweeps
from primitives.displacement import is_displacement
from primitives.order_block import detect_order_blocks
from primitives.premium_discount import compute_dealing_ranges
from primitives.liquidity import compute_sr_levels, compute_equal_highs_lows
from backtest.lookahead_audit import audit_prefix_stability, LookaheadViolation

SNAPSHOT = "data/snapshots/EURUSD_M15_20220808_20260817_20260817T150321Z.parquet"
SLICE_SIZE = 3000
CUTOFFS = [300, 700, 1200, 1800, 2400, 2900]

N_INT, N_EXT = 2, 5
ATR_PERIOD = 14
MIN_LEG_ATR = 3.0
SWEEP_TOL_ATR = 0.1
SWEEP_WINDOW = 3


@pytest.fixture(scope="module")
def real_slice():
    df = pd.read_parquet(SNAPSHOT).reset_index(drop=True)
    return df.iloc[:SLICE_SIZE].reset_index(drop=True)


def _assert_clean(violations: list[LookaheadViolation]):
    if violations:
        sample = "\n".join(v.detail for v in violations[:5])
        pytest.fail(f"{len(violations)} lookahead violation(s) found. First few:\n{sample}")


def test_atr_is_prefix_stable(real_slice):
    def compute(df):
        atr = compute_atr(df, period=ATR_PERIOD)
        # wrap each valid value as a tiny comparable record so audit_prefix_stability's
        # equality-based diffing works the same way it does for the dataclass primitives
        return [(i, round(float(v), 10)) for i, v in enumerate(atr) if pd.notna(v)]

    violations = audit_prefix_stability(real_slice, compute, confirm_index_fn=lambda e: e[0],
                                         cutoffs=CUTOFFS, label="atr")
    _assert_clean(violations)


def test_internal_swings_are_prefix_stable(real_slice):
    def compute(df):
        return detect_swings(df, n=N_INT, role="internal")
    violations = audit_prefix_stability(real_slice, compute, confirm_index_fn=lambda s: s.confirmed_at_index,
                                         cutoffs=CUTOFFS, label="internal_swings")
    _assert_clean(violations)


def test_external_swings_are_prefix_stable(real_slice):
    def compute(df):
        return detect_swings(df, n=N_EXT, role="external")
    violations = audit_prefix_stability(real_slice, compute, confirm_index_fn=lambda s: s.confirmed_at_index,
                                         cutoffs=CUTOFFS, label="external_swings")
    _assert_clean(violations)


def _structure_pipeline(df):
    swings = compute_dual_resolution_swings(df, n_internal=N_INT, n_external=N_EXT)
    atr = compute_atr(df, period=ATR_PERIOD)
    ranges = compute_dealing_ranges(swings["external"], atr, min_leg_atr=MIN_LEG_ATR)
    result = run_dual_resolution_structure(df, swings["internal"], swings["external"], ranges)
    return result, ranges


def test_structure_events_are_prefix_stable(real_slice):
    def compute(df):
        result, _ = _structure_pipeline(df)
        return list(result.internal_events) + list(result.external_events)
    violations = audit_prefix_stability(real_slice, compute, confirm_index_fn=lambda e: e.bar_index,
                                         cutoffs=CUTOFFS, label="structure_events")
    _assert_clean(violations)


def test_dealing_ranges_are_prefix_stable(real_slice):
    def compute(df):
        _, ranges = _structure_pipeline(df)
        return ranges
    violations = audit_prefix_stability(real_slice, compute, confirm_index_fn=lambda r: r.as_of_index,
                                         cutoffs=CUTOFFS, label="dealing_ranges")
    _assert_clean(violations)


def test_sweep_events_are_prefix_stable(real_slice):
    def compute(df):
        external = detect_swings(df, n=N_EXT, role="external")
        atr = compute_atr(df, period=ATR_PERIOD)
        return detect_sweeps(df, external, atr, tol_atr=SWEEP_TOL_ATR, window_bars=SWEEP_WINDOW)
    violations = audit_prefix_stability(real_slice, compute, confirm_index_fn=lambda e: e.reclaim_bar_index,
                                         cutoffs=CUTOFFS, label="sweep_events")
    _assert_clean(violations)


def test_order_blocks_are_prefix_stable_on_identity_fields(real_slice):
    # Projected: excludes mitigated_at_index, which legitimately isn't known yet in a
    # truncated run if the mitigating touch hasn't happened within that window (see
    # module docstring) -- that's expected, not a lookahead bug.
    def compute(df):
        result, _ = _structure_pipeline(df)
        atr = compute_atr(df, period=ATR_PERIOD)
        disp = is_displacement(df, atr, body_ratio_min=0.6, atr_multiple_min=1.5)
        obs = detect_order_blocks(df, result.external_events, disp)
        return [(ob.candle_bar_index, ob.break_bar_index, ob.direction,
                 ob.zone_low, ob.zone_high, ob.body_low, ob.body_high) for ob in obs]

    violations = audit_prefix_stability(real_slice, compute, confirm_index_fn=lambda t: t[1],
                                         cutoffs=CUTOFFS, label="order_blocks_identity")
    _assert_clean(violations)


def test_liquidity_clustering_never_uses_a_touch_before_its_confirm_index(real_slice):
    # Narrower property than full prefix-equality (see module docstring: SRLevel/
    # LiquidityPool aggregates legitimately evolve as more data arrives). What must
    # never happen: a cluster computed from a truncated window includes a member swing
    # whose own confirmed_at_index is >= the truncation cutoff.
    external = detect_swings(real_slice, n=N_EXT, role="external")
    atr = compute_atr(real_slice, period=ATR_PERIOD)
    swing_confirm_by_bar = {s.bar_index: s.confirmed_at_index for s in external}

    violations = []
    for cutoff in CUTOFFS:
        truncated = real_slice.iloc[:cutoff].reset_index(drop=True)
        t_external = detect_swings(truncated, n=N_EXT, role="external")
        t_atr = compute_atr(truncated, period=ATR_PERIOD)

        sr_levels = compute_sr_levels(t_external, t_atr, tol_atr=0.15)
        pools = compute_equal_highs_lows(t_external, t_atr, tol_atr=0.1)

        for level in sr_levels:
            for bar_idx in level.touches:
                if swing_confirm_by_bar.get(bar_idx, -1) >= cutoff:
                    violations.append(f"SRLevel touch at bar {bar_idx} confirms at/after cutoff {cutoff}")
        for pool in pools:
            for bar_idx in pool.member_bar_indices:
                if swing_confirm_by_bar.get(bar_idx, -1) >= cutoff:
                    violations.append(f"LiquidityPool member at bar {bar_idx} confirms at/after cutoff {cutoff}")

    assert violations == []
