"""Weighted-score confluence assembly (spec §4, second combination mode) — "each concept
contributes a weight; enter above a threshold -> more trades, softer... lets you measure
which concepts actually add expectancy."

Two discipline guardrails, both load-bearing, not optional:

1. THE THRESHOLD MUST NOT BE SELECTED BY SEARCHING FOR A TRADE COUNT. Picking a
   threshold because it happens to clear the §7.0 sample-size bar is the §11
   anti-pattern ("select the window/params that hit a target") wearing a different hat.
   This module deliberately exposes no "find the threshold that produces N trades"
   helper — pick the threshold on stated, principled grounds BEFORE measuring what
   count falls out, the same discipline as every other parameter in this build.

2. DISPLACEMENT IS DELIBERATELY NOT A SEPARATE SCORED DIMENSION. The correlation
   analysis (memory: validation-harness-design) found displacement_at_break and the
   canonical OB-retest definition sit at ~74% of their own theoretical maximum
   association (base-rate-normalized phi) — a real, substantial relationship, not a
   base-rate artifact. Scoring both independently would double-count the same
   "structure broke with force" fact spec §4 explicitly warns the dual-resolution/
   displacement family produces. An OB-retest condition already carries displacement's
   information (an OB can only exist where its origin bar passed the displacement gate
   — see primitives/order_block.py), so it is the single scored dimension; a bare
   displacement flag gets no separate weight slot in the default condition set below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WeightedCondition:
    name: str
    weight: float
    predicate: Callable[[Any], bool]


@dataclass(frozen=True)
class ScoreResult:
    total_score: float
    max_score: float
    fraction: float                 # total_score / max_score, NaN if max_score == 0
    passed_conditions: list[str]
    failed_conditions: list[str]

    def passes(self, threshold_fraction: float) -> bool:
        return self.fraction >= threshold_fraction  # NaN >= anything is False -- correct: no conditions means nothing to score


def score_candidate(candidate: Any, conditions: list[WeightedCondition]) -> ScoreResult:
    total = 0.0
    max_total = sum(c.weight for c in conditions)
    passed, failed = [], []
    for c in conditions:
        if c.predicate(candidate):
            total += c.weight
            passed.append(c.name)
        else:
            failed.append(c.name)
    fraction = total / max_total if max_total > 0 else float("nan")
    return ScoreResult(total_score=total, max_score=max_total, fraction=fraction,
                        passed_conditions=passed, failed_conditions=failed)


def filter_by_threshold(
    candidates: list[Any], conditions: list[WeightedCondition], threshold_fraction: float,
) -> list[Any]:
    """Candidates whose weighted score clears threshold_fraction (0..1). threshold_fraction
    must already be justified on principled grounds by the caller — see guardrail 1 above."""
    return [c for c in candidates if score_candidate(c, conditions).passes(threshold_fraction)]
