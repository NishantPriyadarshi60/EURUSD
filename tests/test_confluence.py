import math

import pandas as pd

from signals.confluence import compute_signal_flags, compute_correlation_matrix, high_overlap_pairs


def test_compute_signal_flags_builds_one_column_per_predicate():
    candidates = [1, 2, 3, 4, 5]
    flags = compute_signal_flags(candidates, {
        "even": lambda x: x % 2 == 0,
        "gt_3": lambda x: x > 3,
    })
    assert list(flags["even"]) == [False, True, False, True, False]
    assert list(flags["gt_3"]) == [False, False, False, True, True]


def test_identical_flags_are_perfectly_correlated():
    flags = pd.DataFrame({"a": [True, True, False, False, True], "b": [True, True, False, False, True]})
    matrix = compute_correlation_matrix(flags)
    assert round(matrix.loc["a", "b"], 10) == 1.0


def test_inverted_flags_are_perfectly_anticorrelated():
    flags = pd.DataFrame({"a": [True, True, False, False], "b": [False, False, True, True]})
    matrix = compute_correlation_matrix(flags)
    assert round(matrix.loc["a", "b"], 10) == -1.0


def test_correlation_matches_hand_derived_phi_coefficient():
    # n11=2 (idx 0,1), n10=1 (idx 2), n01=1 (idx 3), n00=6 (idx 4-9)
    # phi = (n11*n00 - n10*n01) / sqrt((n11+n10)(n01+n00)(n11+n01)(n10+n00))
    #     = (2*6 - 1*1) / sqrt(3*7*3*7) = 11/21
    a = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    b = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    flags = pd.DataFrame({"a": a, "b": b}).astype(bool)
    matrix = compute_correlation_matrix(flags)
    expected_phi = 11 / 21
    assert round(matrix.loc["a", "b"], 10) == round(expected_phi, 10)


def test_high_overlap_pairs_filters_by_threshold_and_dedupes():
    flags = pd.DataFrame({
        "a": [True, True, False, False, True, True, False, False],
        "b": [True, True, False, False, True, True, False, False],   # identical to a -> corr 1.0
        "c": [True, False, True, False, True, False, True, False],    # unrelated
    })
    matrix = compute_correlation_matrix(flags)
    findings = high_overlap_pairs(matrix, threshold=0.9)
    assert len(findings) == 1
    assert {findings[0].flag_a, findings[0].flag_b} == {"a", "b"}
    assert round(findings[0].correlation, 6) == 1.0


def test_high_overlap_pairs_empty_when_nothing_clears_threshold():
    flags = pd.DataFrame({
        "a": [True, False, True, False, True, False],
        "b": [False, True, True, False, False, True],
    })
    matrix = compute_correlation_matrix(flags)
    findings = high_overlap_pairs(matrix, threshold=0.99)
    assert findings == []


def test_constant_flag_produces_nan_correlation_not_a_crash():
    # A flag that's always True (zero variance) makes Pearson correlation undefined
    # (0/0) -- must come through as NaN, and high_overlap_pairs must skip it cleanly
    # rather than raising or wrongly reporting it as high-overlap.
    flags = pd.DataFrame({
        "always_true": [True, True, True, True],
        "varies": [True, False, True, False],
    })
    matrix = compute_correlation_matrix(flags)
    assert math.isnan(matrix.loc["always_true", "varies"])
    findings = high_overlap_pairs(matrix, threshold=0.5)
    assert findings == []
