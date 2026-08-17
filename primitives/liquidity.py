"""Support/resistance and equal-highs/equal-lows liquidity pools (spec §2.7, §2.8).

Both cluster confirmed swings by price proximity, processed sequentially in
confirmed_at_index order (not by sorting on price first) so a level's identity and
membership at any point in time reflect only swings that had actually confirmed by
then — the same no-lookahead discipline used throughout primitives/.

Tolerance is ATR-relative, sized using the ATR at the NEW swing's own confirming bar —
"is this new swing close enough to an existing level, as of right now" is the only
question a live signal could actually ask; using a later or earlier ATR value would be
lookahead or stale respectively.

Simplification worth flagging: a cluster's representative price is a running average of
its members, updated online as each new member joins. This means a swing's cluster
membership is decided against the average AS IT STOOD when that swing arrived, not
retroactively re-checked as the average drifts — no backtracking/re-clustering. This is
a reasonable first pass, not a spec requirement either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from primitives.swings import Swing


@dataclass
class _Cluster:
    price: float
    member_bar_indices: list[int] = field(default_factory=list)
    member_confirm_indices: list[int] = field(default_factory=list)

    def add(self, swing: Swing) -> None:
        n = len(self.member_bar_indices)
        self.price = (self.price * n + swing.price) / (n + 1)
        self.member_bar_indices.append(swing.bar_index)
        self.member_confirm_indices.append(swing.confirmed_at_index)


def _cluster_swings(swings: list[Swing], atr: pd.Series, tol_atr: float) -> list[_Cluster]:
    ordered = sorted(swings, key=lambda s: s.confirmed_at_index)
    clusters: list[_Cluster] = []
    for swing in ordered:
        idx = swing.confirmed_at_index
        tol = tol_atr * atr.iloc[idx] if idx < len(atr) else float("nan")
        if pd.isna(tol):
            continue  # can't size tolerance during ATR warm-up -- skip rather than guess

        best, best_dist = None, None
        for c in clusters:
            dist = abs(c.price - swing.price)
            if dist <= tol and (best_dist is None or dist < best_dist):
                best, best_dist = c, dist
        if best is not None:
            best.add(swing)
        else:
            fresh = _Cluster(price=swing.price)
            fresh.add(swing)
            clusters.append(fresh)
    return clusters


@dataclass(frozen=True)
class SRLevel:
    price: float
    touches: list[int]                     # swing bar_index per touch, time-ordered
    touch_confirmed_at_indices: list[int]    # parallel list of confirm indices
    first_confirmed_at_index: int

    @property
    def strength(self) -> int:
        return len(self.touches)

    def strength_as_of(self, bar_index: int) -> int:
        return sum(1 for c in self.touch_confirmed_at_indices if c <= bar_index)


def compute_sr_levels(swings: list[Swing], atr: pd.Series, tol_atr: float) -> list[SRLevel]:
    """Clusters BOTH swing highs and lows together (spec §2.7) — a level's significance
    comes from any confirmed swing reacting there, regardless of type."""
    clusters = _cluster_swings(swings, atr, tol_atr)
    return [
        SRLevel(
            price=c.price, touches=c.member_bar_indices,
            touch_confirmed_at_indices=c.member_confirm_indices,
            first_confirmed_at_index=c.member_confirm_indices[0],
        )
        for c in clusters
    ]


@dataclass(frozen=True)
class LiquidityPool:
    price: float
    kind: str            # "high" | "low" -- matches Swing's convention deliberately, so
                           # a LiquidityPool can be passed directly to sweep.detect_sweeps
                           # as a target wherever a Swing is expected (duck-typed: both
                           # expose .price/.kind/.confirmed_at_index/.bar_index).
    bar_index: int          # most recent contributing member (representative)
    confirmed_at_index: int  # the 2nd member's confirm index -- when "equal" became true
    member_bar_indices: list[int]


def compute_equal_highs_lows(swings: list[Swing], atr: pd.Series, tol_atr: float) -> list[LiquidityPool]:
    """Clusters swing highs and swing lows SEPARATELY (spec §2.8) — only same-kind
    clusters count as "equal." A cluster only becomes a liquidity pool once it has 2+
    members; its confirmed_at_index is the 2nd member's, since a single swing isn't
    "equal" to anything yet and must not be usable before that.
    """
    pools: list[LiquidityPool] = []
    for kind in ("high", "low"):
        same_kind = [s for s in swings if s.kind == kind]
        for c in _cluster_swings(same_kind, atr, tol_atr):
            if len(c.member_bar_indices) < 2:
                continue
            pools.append(LiquidityPool(
                price=c.price, kind=kind, bar_index=c.member_bar_indices[-1],
                confirmed_at_index=c.member_confirm_indices[1],
                member_bar_indices=list(c.member_bar_indices),
            ))
    return pools
