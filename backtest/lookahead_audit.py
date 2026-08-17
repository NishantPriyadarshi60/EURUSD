"""Automated lookahead-bias audit (spec §6): "automated check that no signal uses
same-bar close to trigger something that would have occurred intrabar, and no HTF value
from an unclosed HTF bar is used. Fail the build if violated."

Implemented as PREFIX STABILITY: every primitive in this codebase precomputes over the
full dataset and exposes an explicit confirmed_at_index/as_of_index field marking when
an event first becomes knowable (§2.1's confirmation-lag pattern, used consistently
throughout primitives/). If that promise is honest, truncating the data to a prefix and
recomputing must reproduce EXACTLY the same events (every field) for anything whose
confirm index falls before the truncation point — nothing about an already-knowable
event can depend on bars that, at that point in time, hadn't happened yet.

This generalizes the ad hoc technique already used in individual primitive unit tests
(e.g. test_swings.py's no-lookahead test) into a reusable tool that runs across many
cutoffs on real data — a materially stronger check than any one hand-picked example,
because a bug that only shows up at a specific cutoff (an edge case in how a primitive
handles a partial window) would be invisible to a single fixed scenario.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class LookaheadViolation:
    cutoff: int
    kind: str          # "missing_when_truncated" | "extra_or_differs_when_truncated"
    detail: str


def _list_diff(a: list, b: list) -> tuple[list, list]:
    """Elements of a with no equal (==) counterpart in b, and vice versa. Deliberately
    avoids Python's set() -- several event dataclasses here carry list fields (e.g.
    LiquidityPool.member_bar_indices), which makes them unhashable but still comparable
    via ==. O(n*m); fine at the audit's scale (bounded slices, not full history)."""
    only_in_a = [x for x in a if not any(x == y for y in b)]
    only_in_b = [y for y in b if not any(x == y for x in a)]
    return only_in_a, only_in_b


def audit_prefix_stability(
    df: pd.DataFrame,
    compute_fn: Callable[[pd.DataFrame], list],
    confirm_index_fn: Callable[[Any], int],
    cutoffs: list[int],
    label: str = "",
) -> list[LookaheadViolation]:
    """compute_fn(df) -> list of events (comparable dataclass instances) computed over
    the whole df. confirm_index_fn(event) -> bar index at which the event is first
    knowable. Returns [] if no violation is found across any cutoff.
    """
    full_result = compute_fn(df)
    violations: list[LookaheadViolation] = []

    for cutoff in cutoffs:
        if cutoff <= 0 or cutoff > len(df):
            continue
        truncated_df = df.iloc[:cutoff].reset_index(drop=True)
        truncated_result = compute_fn(truncated_df)

        full_knowable = [e for e in full_result if confirm_index_fn(e) < cutoff]
        missing, extra = _list_diff(full_knowable, truncated_result)

        for e in missing:
            violations.append(LookaheadViolation(
                cutoff, "missing_when_truncated",
                f"[{label}] confirm_index={confirm_index_fn(e)} present with full data "
                f"but MISSING when truncated to {cutoff} bars -- {e!r}",
            ))
        for e in extra:
            violations.append(LookaheadViolation(
                cutoff, "extra_or_differs_when_truncated",
                f"[{label}] present when truncated to {cutoff} bars but absent/different "
                f"with full data -- future bars appear to have changed a supposedly "
                f"already-knowable result -- {e!r}",
            ))

    return violations
