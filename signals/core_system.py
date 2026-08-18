"""Phase 2 core system v1 (spec §8): HTF bias + dual-resolution BOS/CHoCH structure +
displacement-gated OB + sweep + premium/discount + session gate + regime filter,
combined via weighted-score confluence (signals/scoring.py). No FVG, no inducement, no
breaker (spec's Phase 2 scope).

This is the composition point every analysis script in this project has been hand-
rolling ad hoc — crystallized here as the one reusable, canonical version so the
funnel diagnostic, weighted-score run, and walk-forward runner all build candidates and
score them identically. The OB-retest condition, in particular, is the canonical
definition settled after a four-attempt investigation (memory: validation-harness-
design) and must not be redefined ad hoc elsewhere.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import pandas as pd

from primitives.swings import compute_dual_resolution_swings
from primitives.atr import compute_atr
from primitives.premium_discount import compute_dealing_ranges
from primitives.structure import run_dual_resolution_structure, StructureEvent
from primitives.displacement import is_displacement
from primitives.order_block import detect_order_blocks, OrderBlock
from primitives.sweep import detect_sweeps
from primitives.liquidity import compute_equal_highs_lows
from signals.htf_bias import align_htf_direction_to_entry_tf, htf_gate_allows
from signals.session import in_any_session, SessionWindow
from signals.regime import compute_adx, classify_regime
from signals.scoring import WeightedCondition

OB_LOOKBACK_BARS = 200   # how far back to search for a still-relevant unmitigated OB
OB_RETEST_WINDOW_BARS = 20    # canonical OB-retest window (memory: validation-harness-design)
SWEEP_LOOKBACK_BARS = 10


@dataclass(frozen=True)
class SystemOutput:
    candidates: list[StructureEvent]     # direction_aligned internal BOS/CHoCH events
    conditions: list[WeightedCondition]    # the six weighted confluence conditions
    atr: pd.Series
    obs: list[OrderBlock]


def compute_system(
    df: pd.DataFrame,
    htf_timeline: pd.DataFrame,
    session_windows: list[SessionWindow],
    primitives_cfg: dict,
    signals_cfg: dict,
) -> SystemOutput:
    """df: entry-TF (M15) OHLC, positionally indexed. htf_timeline: from
    signals.htf_bias.compute_htf_direction_timeline on the H4 data covering (at least)
    df's range plus warmup. primitives_cfg/signals_cfg: loaded configs/primitives.yaml
    and configs/signals.yaml (or equivalents) — nothing hardcoded here that already has
    a config home.
    """
    sw = primitives_cfg["swings"]
    atr_period = primitives_cfg["atr"]["period"]
    min_leg_atr = primitives_cfg["premium_discount"]["min_leg_atr"]
    disp_cfg = primitives_cfg["displacement"]
    sweep_cfg = primitives_cfg["sweep"]
    eqh_tol = primitives_cfg["equal_highs_lows"]["tol_atr"]
    regime_cfg = signals_cfg["regime"]
    htf_cfg = signals_cfg["htf_bias"]

    swings = compute_dual_resolution_swings(df, n_internal=sw["n_internal"], n_external=sw["n_external"])
    atr = compute_atr(df, period=atr_period)
    ranges = compute_dealing_ranges(swings["external"], atr, min_leg_atr=min_leg_atr)
    structure = run_dual_resolution_structure(df, swings["internal"], swings["external"], ranges)
    disp = is_displacement(df, atr, body_ratio_min=disp_cfg["body_ratio_min"], atr_multiple_min=disp_cfg["atr_multiple_min"])
    obs = detect_order_blocks(df, structure.external_events, disp)
    eqh_eql = compute_equal_highs_lows(swings["external"], atr, tol_atr=eqh_tol)
    sweeps = detect_sweeps(df, swings["external"] + eqh_eql, atr,
                            tol_atr=sweep_cfg["tol_atr"], window_bars=sweep_cfg["window_bars"])
    adx = compute_adx(df, period=regime_cfg["adx_period"])
    htf_bias = align_htf_direction_to_entry_tf(df, htf_timeline, entry_timeframe_minutes=15)

    candidates = [e for e in structure.internal_events if e.direction_aligned is True]

    obs_by_dir = {"bullish": [ob for ob in obs if ob.direction == "bullish"],
                  "bearish": [ob for ob in obs if ob.direction == "bearish"]}
    obs_by_dir_bars = {d: [ob.break_bar_index for ob in obs_by_dir[d]] for d in obs_by_dir}

    def has_ob_retest(bar_index: int, direction: str) -> bool:
        lst, bars = obs_by_dir[direction], obs_by_dir_bars[direction]
        lo = bisect.bisect_left(bars, bar_index - OB_LOOKBACK_BARS)
        for ob in lst[lo:]:
            if ob.break_bar_index > bar_index:
                break
            if ob.is_unmitigated_at(bar_index) and ob.mitigated_at_index is not None \
                    and ob.mitigated_at_index <= bar_index + OB_RETEST_WINDOW_BARS:
                return True
        return False

    sweep_reclaim_bars = sorted(s.reclaim_bar_index for s in sweeps)

    def has_recent_sweep(bar_index: int) -> bool:
        lo = bisect.bisect_left(sweep_reclaim_bars, bar_index - SWEEP_LOOKBACK_BARS)
        hi = bisect.bisect_right(sweep_reclaim_bars, bar_index)
        return hi > lo

    # Displacement is deliberately NOT its own condition here -- see signals/scoring.py
    # guardrail 2 (0.74-of-ceiling correlation with has_ob_retest).
    conditions = [
        WeightedCondition("htf_pass", 1.0, lambda e: htf_gate_allows(
            htf_bias.iloc[e.bar_index], "long" if e.direction == "bullish" else "short",
            allow_neutral=False)),
        WeightedCondition("session_pass", 1.0, lambda e: in_any_session(e.time, session_windows)),
        WeightedCondition("regime_pass", 1.0, lambda e: classify_regime(
            adx["adx"].iloc[e.bar_index], regime_cfg["trending_threshold"], regime_cfg["ranging_threshold"]) == "trending"),
        WeightedCondition("premium_discount_pass", 1.0, lambda e: e.inside_external_leg is True),
        WeightedCondition("has_ob_retest", 1.0, lambda e: has_ob_retest(e.bar_index, e.direction)),
        WeightedCondition("has_recent_sweep", 1.0, lambda e: has_recent_sweep(e.bar_index)),
    ]

    return SystemOutput(candidates=candidates, conditions=conditions, atr=atr, obs=obs)
