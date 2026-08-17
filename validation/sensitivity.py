"""Parameter sensitivity sweeps (spec §7.2) — "a real edge degrades gracefully as
parameters shift... require broad, stable plateaus, not sharp peaks."

Phase 2 supplies the actual evaluate_fn (run the strategy with a given parameter value,
return a scalar performance metric — e.g. walk-forward expectancy or Sharpe). This
module only owns the sweep mechanics and the plateau-vs-peak classification, both
strategy-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SweepPoint:
    value: Any
    metric: float


@dataclass(frozen=True)
class SensitivityReport:
    points: list[SweepPoint]
    best_value: Any
    best_metric: float
    plateau_width_fraction: float   # fraction of the swept range covered by the
                                      # CONTIGUOUS near-best run containing the best point
    is_plateau: bool


def sweep_parameter(values: list[Any], evaluate_fn: Callable[[Any], float]) -> list[SweepPoint]:
    return [SweepPoint(v, evaluate_fn(v)) for v in values]


def classify_plateau(
    points: list[SweepPoint],
    relative_tolerance: float = 0.2,
    min_plateau_width_fraction: float = 0.3,
) -> SensitivityReport:
    """A point is "near-best" if its metric is within relative_tolerance of the best
    (e.g. 0.2 = within 20% of the peak). The plateau width is the fraction of the
    CONTIGUOUS near-best run containing the best point, relative to the full swept
    range — a narrow spike surrounded by collapse scores low even if a few scattered
    far-away points also happen to score well by chance (which a non-contiguous count
    would have wrongly credited as "stable").

    If the best metric is <= 0, there's no positive edge anywhere in the sweep and the
    plateau/peak framing doesn't meaningfully apply — flagged explicitly (is_plateau
    False, width 0.0) rather than reporting a number that could be misread as "stable."
    """
    if not points:
        raise ValueError("points must be non-empty")
    if len(points) == 1:
        return SensitivityReport(points, points[0].value, points[0].metric, 1.0, True)

    best_idx = max(range(len(points)), key=lambda i: points[i].metric)
    best_metric = points[best_idx].metric

    if best_metric <= 0:
        return SensitivityReport(points, points[best_idx].value, best_metric, 0.0, False)

    threshold = best_metric * (1 - relative_tolerance)

    lo = best_idx
    while lo - 1 >= 0 and points[lo - 1].metric >= threshold:
        lo -= 1
    hi = best_idx
    while hi + 1 < len(points) and points[hi + 1].metric >= threshold:
        hi += 1

    plateau_width_fraction = (hi - lo + 1) / len(points)
    is_plateau = plateau_width_fraction >= min_plateau_width_fraction

    return SensitivityReport(points, points[best_idx].value, best_metric, plateau_width_fraction, is_plateau)
