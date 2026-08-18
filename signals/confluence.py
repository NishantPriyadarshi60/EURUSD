"""Confluence assembly — redundancy guard (spec §4).

"Stacking correlated conditions is not confluence — it's counting one vote several
times and it silently shrinks your sample. Before trusting a combo, compute pairwise
correlation of the signal flags and flag high-overlap sets."

This module computes the pairwise correlation matrix over binary signal flags and
flags high-overlap pairs. It deliberately does NOT decide which gate to drop — spec
ties that judgment to human review of the matrix (§4/§7.0), same diagnose-don't-fix
stance as validation.feasibility.evaluate_feasibility and validation.sensitivity's
plateau classifier.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


def compute_signal_flags(candidates: list[Any], flag_fns: dict[str, Callable[[Any], bool]]) -> pd.DataFrame:
    """One row per candidate, one boolean column per named flag."""
    return pd.DataFrame({name: [bool(fn(c)) for c in candidates] for name, fn in flag_fns.items()})


def compute_correlation_matrix(flags: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation on 0/1-encoded columns == the phi coefficient for binary
    variables — the standard measure of association between two binary flags."""
    return flags.astype(float).corr()


@dataclass(frozen=True)
class CorrelationFinding:
    flag_a: str
    flag_b: str
    correlation: float


def high_overlap_pairs(matrix: pd.DataFrame, threshold: float) -> list[CorrelationFinding]:
    """All (a, b) pairs with |correlation| >= threshold, each pair reported once
    (a before b in column order), sorted by |correlation| descending."""
    findings: list[CorrelationFinding] = []
    cols = list(matrix.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            corr = matrix.loc[a, b]
            if pd.notna(corr) and abs(corr) >= threshold:
                findings.append(CorrelationFinding(a, b, float(corr)))
    findings.sort(key=lambda f: -abs(f.correlation))
    return findings


def max_possible_phi(p_a: float, p_b: float) -> float:
    """Theoretical ceiling on |phi| (Pearson correlation on 0/1 data) achievable
    between two binary flags with given marginal probabilities — reached only under
    perfect containment (the rarer flag implies the more common one). Two flags with
    very different base rates CANNOT show a high raw phi even if one perfectly implies
    the other, so a low raw reading is not, by itself, evidence of true independence
    when marginals are skewed and dissimilar. Spec's own point about the internal/
    external BOS pair (the "frequency trap") is a special case of this general fact —
    it applies to any pair of dissimilar-rate flags, not just the one spec named.
    """
    if not (0 < p_a < 1) or not (0 < p_b < 1):
        return float("nan")
    lo, hi = min(p_a, p_b), max(p_a, p_b)
    return math.sqrt((lo / (1 - lo)) * ((1 - hi) / hi))


@dataclass(frozen=True)
class NormalizedOverlap:
    flag_a: str
    flag_b: str
    correlation: float
    max_possible: float
    normalized: float   # |correlation| / max_possible -- how close to its OWN ceiling


def compute_normalized_overlap(matrix: pd.DataFrame, base_rates: dict[str, float]) -> list[NormalizedOverlap]:
    """For every pair, express the observed correlation as a fraction of the maximum
    that pair's OWN marginals could ever produce — surfaces pairs whose raw |phi| looks
    low only because dissimilar base rates cap it there, not because the flags are
    genuinely weakly associated. Sorted by normalized value descending, so a pair sitting
    near its own ceiling shows up first even if its raw correlation looked unremarkable.
    """
    results: list[NormalizedOverlap] = []
    cols = list(matrix.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            corr = matrix.loc[a, b]
            if pd.isna(corr) or a not in base_rates or b not in base_rates:
                continue
            cap = max_possible_phi(base_rates[a], base_rates[b])
            if pd.isna(cap) or cap == 0:
                continue
            results.append(NormalizedOverlap(a, b, float(corr), cap, float(abs(corr) / cap)))
    results.sort(key=lambda r: -r.normalized)
    return results
