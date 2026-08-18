import math

import pandas as pd
import pytest

from signals.confluence import (
    compute_correlation_matrix, max_possible_phi, compute_normalized_overlap,
)


def test_equal_base_rates_have_a_ceiling_of_one():
    assert round(max_possible_phi(0.5, 0.5), 10) == 1.0
    assert round(max_possible_phi(0.1, 0.1), 10) == 1.0   # any equal pair, not just 0.5/0.5
    assert round(max_possible_phi(0.9, 0.9), 10) == 1.0


def test_dissimilar_base_rates_have_a_low_ceiling():
    # Hand-derived: sqrt((0.05/0.95) * (0.1/0.9)) = sqrt(0.05263 * 0.11111) = 0.0765
    cap = max_possible_phi(0.05, 0.9)
    assert round(cap, 4) == 0.0765


def test_ceiling_is_symmetric_in_its_arguments():
    assert round(max_possible_phi(0.2, 0.6), 10) == round(max_possible_phi(0.6, 0.2), 10)


def test_degenerate_marginals_yield_nan():
    assert math.isnan(max_possible_phi(0.0, 0.5))
    assert math.isnan(max_possible_phi(1.0, 0.5))


def test_maximal_containment_actually_achieves_its_theoretical_ceiling():
    # Constructed so A implies B exactly: p_a=0.2, p_b=0.5, n=100.
    # 20 rows A=1,B=1; 30 rows A=0,B=1; 50 rows A=0,B=0.
    # Hand-derived: Cov=0.2-0.2*0.5=0.1, Var(A)=0.16, Var(B)=0.25 -> phi=0.1/0.2=0.5
    a = [True] * 20 + [False] * 30 + [False] * 50
    b = [True] * 20 + [True] * 30 + [False] * 50
    flags = pd.DataFrame({"a": a, "b": b})
    matrix = compute_correlation_matrix(flags)
    observed = matrix.loc["a", "b"]
    theoretical_cap = max_possible_phi(0.2, 0.5)
    assert round(observed, 6) == round(theoretical_cap, 6) == 0.5


def test_normalized_overlap_flags_a_pair_sitting_at_its_own_ceiling_despite_low_raw_corr():
    # Same maximal-containment construction as above, but with much more different
    # base rates (0.05 vs 0.8) so the RAW correlation reads low even though the pair is
    # at 100% of what it could possibly achieve given those marginals -- exactly the
    # "false reassurance" scenario the base-rate check exists to catch.
    n = 1000
    p_a, p_b = 0.05, 0.8
    n_both = int(p_a * n)          # 50: A=1,B=1
    n_b_only = int(p_b * n) - n_both  # 750: A=0,B=1
    n_neither = n - n_both - n_b_only  # 200: A=0,B=0
    a = [True] * n_both + [False] * n_b_only + [False] * n_neither
    b = [True] * n_both + [True] * n_b_only + [False] * n_neither
    flags = pd.DataFrame({"a": a, "b": b})
    matrix = compute_correlation_matrix(flags)

    raw_corr = matrix.loc["a", "b"]
    assert raw_corr < 0.3  # would NOT be flagged by a naive |corr| >= 0.3 threshold

    findings = compute_normalized_overlap(matrix, base_rates={"a": p_a, "b": p_b})
    assert len(findings) == 1
    assert round(findings[0].normalized, 2) == 1.0  # sitting exactly at its own ceiling


def test_normalized_overlap_does_not_flag_genuinely_independent_similar_rate_flags():
    # Similar base rates AND low raw correlation -> low normalized value too (no false
    # positive from the ceiling correction when there's no ceiling effect to correct for).
    import random
    rng = random.Random(11)
    n = 2000
    a = [rng.random() < 0.4 for _ in range(n)]
    b = [rng.random() < 0.45 for _ in range(n)]  # independently drawn, similar rate to a
    flags = pd.DataFrame({"a": a, "b": b})
    matrix = compute_correlation_matrix(flags)
    p_a, p_b = flags["a"].mean(), flags["b"].mean()

    findings = compute_normalized_overlap(matrix, base_rates={"a": p_a, "b": p_b})
    assert findings[0].normalized < 0.3


def test_compute_normalized_overlap_sorted_descending_and_skips_missing_base_rates():
    flags = pd.DataFrame({
        "a": [True, False, True, False],
        "b": [True, True, False, False],
        "c": [False, False, True, True],
    })
    matrix = compute_correlation_matrix(flags)
    # only provide base rates for a and b -> c must be excluded entirely
    findings = compute_normalized_overlap(matrix, base_rates={"a": 0.5, "b": 0.5})
    assert {(f.flag_a, f.flag_b) for f in findings} == {("a", "b")}
    normalized_values = [f.normalized for f in findings]
    assert normalized_values == sorted(normalized_values, reverse=True)
