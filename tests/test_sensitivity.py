import pytest

from validation.sensitivity import SweepPoint, sweep_parameter, classify_plateau


def test_sweep_parameter_evaluates_each_value():
    results = sweep_parameter([1, 2, 3], lambda v: v * 10)
    assert [p.metric for p in results] == [10, 20, 30]
    assert [p.value for p in results] == [1, 2, 3]


def test_broad_stable_region_is_classified_as_plateau():
    # Hand-verified: best=1.1 at index 3 (first occurrence of the tie with index 4),
    # threshold = 1.1*0.8 = 0.88, contiguous near-best run = indices 0-6 -> width 0.7
    values = list(range(10))
    metrics = [1.0, 1.0, 1.05, 1.1, 1.1, 1.05, 1.0, 0.3, 0.2, 0.1]
    points = [SweepPoint(v, m) for v, m in zip(values, metrics)]
    report = classify_plateau(points, relative_tolerance=0.2, min_plateau_width_fraction=0.3)
    assert report.best_metric == 1.1
    assert round(report.plateau_width_fraction, 4) == 0.7
    assert report.is_plateau is True


def test_narrow_spike_is_classified_as_peak_not_plateau():
    # Hand-verified: best=1.2 at index 4, threshold=0.96, only index 4 itself qualifies
    # (neighbors are 0.1) -> width = 1/10 = 0.1
    values = list(range(10))
    metrics = [0.1, 0.1, 0.1, 0.1, 1.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    points = [SweepPoint(v, m) for v, m in zip(values, metrics)]
    report = classify_plateau(points, relative_tolerance=0.2, min_plateau_width_fraction=0.3)
    assert round(report.plateau_width_fraction, 4) == 0.1
    assert report.is_plateau is False


def test_non_contiguous_far_away_good_points_do_not_count_toward_the_plateau():
    # A second isolated spike elsewhere must NOT inflate the plateau width around the
    # best point -- only the contiguous run touching the best index counts.
    values = list(range(10))
    metrics = [0.1, 0.1, 1.15, 0.1, 0.1, 1.2, 0.1, 0.1, 0.1, 0.1]  # spikes at 2 and 5
    points = [SweepPoint(v, m) for v, m in zip(values, metrics)]
    report = classify_plateau(points, relative_tolerance=0.2, min_plateau_width_fraction=0.3)
    assert report.best_metric == 1.2  # index 5
    # threshold = 1.2*0.8 = 0.96; index 2's 1.15 is isolated (neighbors 0.1) and must
    # not be stitched into the plateau around index 5 even though it individually clears
    # the threshold.
    assert round(report.plateau_width_fraction, 4) == 0.1
    assert report.is_plateau is False


def test_single_point_sweep_is_trivially_a_plateau():
    report = classify_plateau([SweepPoint(5, 0.7)])
    assert report.is_plateau is True
    assert report.plateau_width_fraction == 1.0
    assert report.best_value == 5


def test_non_positive_best_metric_is_flagged_not_misreported_as_stable():
    values = [1, 2, 3]
    metrics = [-0.5, -0.1, -0.3]
    points = [SweepPoint(v, m) for v, m in zip(values, metrics)]
    report = classify_plateau(points)
    assert report.is_plateau is False
    assert report.plateau_width_fraction == 0.0
    assert report.best_metric == -0.1


def test_empty_points_raises():
    with pytest.raises(ValueError):
        classify_plateau([])
