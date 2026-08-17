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
