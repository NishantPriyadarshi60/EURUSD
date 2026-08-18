import math

from signals.scoring import WeightedCondition, score_candidate, filter_by_threshold


def _conditions(weights=(1.0, 1.0, 1.0), passes=(True, False, True)):
    return [
        WeightedCondition(f"c{i}", w, lambda _candidate, outcome=p: outcome)
        for i, (w, p) in enumerate(zip(weights, passes))
    ]


def test_equal_weight_score_matches_hand_calculation():
    conditions = _conditions(weights=(1.0, 1.0, 1.0), passes=(True, False, True))
    result = score_candidate(object(), conditions)
    assert result.total_score == 2.0
    assert result.max_score == 3.0
    assert round(result.fraction, 6) == round(2 / 3, 6)
    assert result.passed_conditions == ["c0", "c2"]
    assert result.failed_conditions == ["c1"]


def test_unequal_weight_score_matches_hand_calculation():
    conditions = _conditions(weights=(3.0, 1.0, 2.0), passes=(True, False, True))
    result = score_candidate(object(), conditions)
    # passes c0 (3) and c2 (2) = 5, out of max 3+1+2=6
    assert result.total_score == 5.0
    assert result.max_score == 6.0
    assert round(result.fraction, 6) == round(5 / 6, 6)


def test_threshold_boundary_is_inclusive():
    conditions = _conditions(weights=(1.0, 1.0), passes=(True, False))  # fraction = 0.5
    result = score_candidate(object(), conditions)
    assert result.passes(0.5) is True
    assert result.passes(0.50001) is False


def test_zero_conditions_never_passes_and_does_not_crash():
    result = score_candidate(object(), [])
    assert math.isnan(result.fraction)
    assert result.passes(0.0) is False  # NaN >= 0.0 is False -- no conditions means nothing scored


def test_filter_by_threshold_selects_correct_candidates():
    # candidate "value" determines which conditions pass, via simple predicates
    candidates = [0, 1, 2, 3]  # 0: passes none, 3: passes all (each condition checks >= its index)
    conditions = [
        WeightedCondition("ge0", 1.0, lambda c: c >= 0),
        WeightedCondition("ge1", 1.0, lambda c: c >= 1),
        WeightedCondition("ge2", 1.0, lambda c: c >= 2),
    ]
    # thresholds: candidate 0 -> 1/3, candidate 1 -> 2/3, candidate 2 -> 3/3, candidate 3 -> 3/3
    selected = filter_by_threshold(candidates, conditions, threshold_fraction=2 / 3)
    assert selected == [1, 2, 3]


def test_full_score_and_zero_score_extremes():
    all_pass = _conditions(weights=(1.0, 1.0), passes=(True, True))
    all_fail = _conditions(weights=(1.0, 1.0), passes=(False, False))
    assert score_candidate(object(), all_pass).fraction == 1.0
    assert score_candidate(object(), all_fail).fraction == 0.0
