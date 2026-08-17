import math

import pytest
from scipy.stats import norm

from validation.deflated_sharpe import (
    sharpe_ratio, sharpe_ratio_std_error, expected_max_sharpe_under_n_trials,
    compute_deflated_sharpe, _sample_skewness, _sample_kurtosis, EULER_MASCHERONI,
)


def test_sharpe_ratio_matches_hand_calculation():
    # mean=3, sample variance=(4+1+0+1+4)/4=2.5, std=1.58113883 -> sharpe=3/1.58113883
    sr = sharpe_ratio([1, 2, 3, 4, 5])
    assert round(sr, 6) == round(3 / math.sqrt(2.5), 6)


def test_sharpe_ratio_requires_at_least_two_returns():
    with pytest.raises(ValueError):
        sharpe_ratio([1.0])


def test_skewness_and_kurtosis_hand_verified_symmetric_dataset():
    # [1,2,3,4,5]: symmetric around 3 -> skew=0 exactly. m2=2, m4=6.8 -> kurtosis=1.7
    returns = [1, 2, 3, 4, 5]
    assert round(_sample_skewness(returns), 10) == 0.0
    assert round(_sample_kurtosis(returns), 6) == 1.7


def test_sharpe_ratio_std_error_matches_hand_derivation():
    # Using the hand-verified skew=0, kurtosis=1.7 from the symmetric dataset above:
    # inside = 1 - 0*sr + ((1.7-1)/4)*sr^2 = 1 + 0.175*sr^2
    returns = [1, 2, 3, 4, 5]
    sr = sharpe_ratio(returns)
    expected_inside = 1 + 0.175 * sr ** 2
    expected = math.sqrt(expected_inside / (len(returns) - 1))
    assert round(sharpe_ratio_std_error(returns, sr), 8) == round(expected, 8)


def test_expected_max_sharpe_is_zero_for_a_single_trial():
    assert expected_max_sharpe_under_n_trials(sr_std_error=0.5, n_trials=1) == 0.0


def test_expected_max_sharpe_matches_formula_assembled_independently():
    sr_std_error = 0.4
    n_trials = 50
    z_a = norm.ppf(1 - 1 / n_trials)
    z_b = norm.ppf(1 - 1 / (n_trials * math.e))
    expected = sr_std_error * ((1 - EULER_MASCHERONI) * z_a + EULER_MASCHERONI * z_b)
    assert round(expected_max_sharpe_under_n_trials(sr_std_error, n_trials), 10) == round(expected, 10)


def test_expected_max_sharpe_increases_with_more_trials():
    # More trials tried -> a higher Sharpe is expected by chance alone -> a harder bar
    # for the observed SR to clear. This is the whole point of the deflation.
    low = expected_max_sharpe_under_n_trials(0.5, n_trials=5)
    high = expected_max_sharpe_under_n_trials(0.5, n_trials=500)
    assert high > low


def test_more_trials_lowers_the_deflated_sharpe_for_the_same_returns():
    returns = [2.0, 1.5, -1.0, 2.0, 1.0, -0.5, 1.8, -1.2, 2.2, 0.5]
    dsr_few = compute_deflated_sharpe(returns, n_trials=1)
    dsr_many = compute_deflated_sharpe(returns, n_trials=200)
    assert dsr_many.deflated_sharpe_ratio < dsr_few.deflated_sharpe_ratio
    assert dsr_many.expected_max_sr_by_chance > dsr_few.expected_max_sr_by_chance


def test_single_trial_deflated_sharpe_matches_a_plain_sr_significance_test():
    # With n_trials=1, expected_max_sr_by_chance=0, so DSR reduces to the standard
    # PSR-against-zero test: Phi(SR / sigma(SR) * sqrt(T-1)).
    returns = [2.0, 1.5, -1.0, 2.0, 1.0, -0.5, 1.8, -1.2, 2.2, 0.5]
    result = compute_deflated_sharpe(returns, n_trials=1)
    sr = sharpe_ratio(returns)
    std_err = sharpe_ratio_std_error(returns, sr)
    expected = norm.cdf(sr / std_err * math.sqrt(len(returns) - 1))
    assert round(result.deflated_sharpe_ratio, 8) == round(expected, 8)


def test_near_degenerate_two_value_sample_clamps_floating_point_noise_instead_of_raising():
    # Discovered while testing this module: kurtosis >= skewness^2 + 1 is a universal
    # inequality for any real distribution, which makes "inside the sqrt" a perfect
    # square (1 - gamma3*sr/2)^2 at its theoretical floor -- provably >= 0 always. A
    # two-distinct-value sample sits exactly at that boundary (kurtosis == skewness^2+1
    # mathematically), so floating-point rounding can push the computed value a hair
    # below zero (observed: ~-8.9e-16) even though the true value is exactly 0. This
    # must be clamped and return a valid (zero) result, not raise.
    returns = [100.0] + [-1.0] * 9
    skew = _sample_skewness(returns)
    kurt = _sample_kurtosis(returns)
    # At the true boundary the quadratic's discriminant is 0 (double root); probe there.
    sr_at_boundary = skew / (2 * (kurt - 1) / 4)
    result = sharpe_ratio_std_error(returns, sr_at_boundary)
    assert result == 0.0 or result >= 0.0  # never raises, never negative/NaN


def test_substantially_negative_inside_still_raises():
    # A hand-forced pathological case bypassing real skew/kurtosis entirely, to confirm
    # the raise path itself still works for a genuinely corrupted/impossible input.
    import validation.deflated_sharpe as ds

    original = ds._sample_skewness
    ds._sample_skewness = lambda returns: 1000.0  # not achievable by any real sample
    try:
        with pytest.raises(ValueError, match="substantially negative"):
            sharpe_ratio_std_error([1.0, 2.0, 3.0], sr=1.0)
    finally:
        ds._sample_skewness = original


def test_compute_deflated_sharpe_reports_moments_used():
    returns = [1, 2, 3, 4, 5]
    result = compute_deflated_sharpe(returns, n_trials=10)
    assert result.n_observations == 5
    assert result.n_trials == 10
    assert round(result.skewness, 10) == 0.0
    assert round(result.kurtosis, 6) == 1.7
