"""Sample-size feasibility gate (spec §7.0) — run EARLY, before tuning anything.

The full funnel spec §7.0 describes (raw structure events -> HTF gate -> session gate ->
regime gate -> premium/discount -> displacement-gated OB -> sweep) needs the HTF bias
gate (§3.1), session gate (§3.2), and regime filter (§3.3) to exist — those are Phase
2's strategy-layer deliverables, not built in this Phase 1 harness. What's built here is
the FUNNEL MACHINERY itself: a generic, ordered sequence of named filter stages applied
to a candidate set, producing the gate-survival counts spec §7.0 requires ("which gate
is killing the sample, not just that it died"). Phase 2 supplies the actual
HTF/session/regime stage predicates; nothing here assumes what a "candidate" or a
"stage" is beyond "a predicate over it."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence


@dataclass(frozen=True)
class FunnelStage:
    name: str
    count_before: int
    count_after: int

    @property
    def dropped(self) -> int:
        return self.count_before - self.count_after

    @property
    def survival_rate(self) -> float:
        return self.count_after / self.count_before if self.count_before > 0 else float("nan")


@dataclass(frozen=True)
class FunnelReport:
    stages: list[FunnelStage]

    @property
    def final_count(self) -> int:
        return self.stages[-1].count_after if self.stages else 0

    @property
    def initial_count(self) -> int:
        return self.stages[0].count_before if self.stages else 0

    def biggest_drop_stage(self) -> Optional[FunnelStage]:
        """The stage killing the most candidates — spec's explicit ask: identify WHICH
        gate is responsible, not just that the sample died."""
        if not self.stages:
            return None
        return max(self.stages, key=lambda s: s.dropped)

    def summary(self) -> str:
        lines = [f"{'stage':<30} {'before':>8} {'after':>8} {'dropped':>8} {'survival':>10}"]
        for s in self.stages:
            lines.append(f"{s.name:<30} {s.count_before:>8} {s.count_after:>8} {s.dropped:>8} {s.survival_rate:>9.1%}")
        return "\n".join(lines)


def compute_gate_funnel(
    candidates: Sequence[Any],
    stages: list[tuple[str, Callable[[Any], bool]]],
) -> FunnelReport:
    """Applies each (name, predicate) stage IN SEQUENCE, each operating only on
    candidates that survived every prior stage — exactly what a live AND-gated
    confluence stack does. Predicates should be pure (no state carried between calls)."""
    surviving = list(candidates)
    funnel_stages = []
    for name, predicate in stages:
        before = len(surviving)
        surviving = [c for c in surviving if predicate(c)]
        after = len(surviving)
        funnel_stages.append(FunnelStage(name=name, count_before=before, count_after=after))
    return FunnelReport(stages=funnel_stages)


@dataclass(frozen=True)
class FeasibilityVerdict:
    final_count: int
    status: str          # "ok" | "warn" | "reject"
    recommendation: str


def evaluate_feasibility(report: FunnelReport, warn_below: int, reject_below: int) -> FeasibilityVerdict:
    """Spec §7.0's BINDING decision rule: a thin sample means loosen the gate stack
    (AND-gate -> weighted-score, or drop the least-independent gate) — NEVER add more
    filters. This function only diagnoses; it does not auto-fix anything, since choosing
    WHICH gate to loosen/drop is a judgment call spec §4 ties to measured redundancy
    (the correlation matrix), not something to automate blindly here.
    """
    n = report.final_count
    if n >= warn_below:
        return FeasibilityVerdict(
            final_count=n, status="ok",
            recommendation="Trade count is within the acceptable range; proceed to full validation.",
        )

    worst = report.biggest_drop_stage()
    worst_desc = (
        f"'{worst.name}' (dropped {worst.dropped}, survival {worst.survival_rate:.1%})"
        if worst is not None else "unknown"
    )
    base = (
        f"Trade count ({n}) is below the warn threshold ({warn_below}). Per spec §7.0: "
        f"this is diagnostic information about the gate stack, not a signal to add "
        f"confidence. Do NOT add filters. Consider switching the AND-gate to "
        f"weighted-score mode, or dropping the least-independent gate (check §4's "
        f"correlation matrix first). Biggest single contributor to the drop: {worst_desc}."
    )
    if n < reject_below:
        return FeasibilityVerdict(
            final_count=n, status="reject",
            recommendation=base + f" Below the reject threshold ({reject_below}) — do not "
                                   f"proceed to tuning on this sample.",
        )
    return FeasibilityVerdict(final_count=n, status="warn", recommendation=base)
