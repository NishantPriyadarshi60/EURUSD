"""Deflated Sharpe Ratio (spec §7.3) — degrees-of-freedom / multiple-testing correction.
Implements Bailey & López de Prado's DSR (2014), which answers: given the number of
independent trials (parameter/concept combinations) actually tried, and the
non-normality (skew, kurtosis) of the trade-return distribution, is the observed Sharpe
ratio still statistically significant — or is it plausibly the best of N noisy draws?

This is a pure statistical computation with no dependency on Phase 2's strategy layer —
it only needs a return/R-multiple series, the number of trials, and the annualization
factor already used elsewhere (backtest/metrics.py). "State per-trade vs annualized and
the annualization factor explicitly" (spec §7.4) applies here too: the Sharpe passed in
must be consistently on ONE basis, and the observation count T must match that basis
(e.g. trade count if SR is per-trade, not "annualized SR with T=trade_count").
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def _sample_skewness(returns: list[float]) -> float:
    n = len(returns)
    mean = sum(returns) / n
    m2 = sum((r - mean) ** 2 for r in returns) / n
    m3 = sum((r - mean) ** 3 for r in returns) / n
    if m2 == 0:
        return 0.0
    return m3 / (m2 ** 1.5)


def _sample_kurtosis(returns: list[float]) -> float:
    """Regular (not excess) kurtosis -- a normal distribution gives 3.0 here, which is
    what the Bailey/Lopez de Prado standard-error formula below expects (it reduces to
    the classical Lo (2002) asymptotic SR variance for a normal, zero-skew return series
    when kurtosis=3: (gamma4-1)/4 = 0.5, matching the textbook SR^2/2 term)."""
    n = len(returns)
    mean = sum(returns) / n
    m2 = sum((r - mean) ** 2 for r in returns) / n
    m4 = sum((r - mean) ** 4 for r in returns) / n
    if m2 == 0:
        return 3.0
    return m4 / (m2 ** 2)


def sharpe_ratio(returns: list[float]) -> float:
    n = len(returns)
    if n < 2:
        raise ValueError("need at least 2 returns to compute a Sharpe ratio")
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance)
    if std == 0:
        return float("nan")
    return mean / std


def sharpe_ratio_std_error(returns: list[float], sr: float) -> float:
    """sigma(SR_hat) per Bailey & Lopez de Prado -- the standard error of the Sharpe
    ratio ESTIMATOR itself, accounting for skew/kurtosis of the underlying returns."""
    n = len(returns)
    if n < 2:
        raise ValueError("need at least 2 returns")
    gamma3 = _sample_skewness(returns)
    gamma4 = _sample_kurtosis(returns)
    inside = 1 - gamma3 * sr + ((gamma4 - 1) / 4) * sr ** 2
    if inside < 0:
        # This can never be genuinely negative for a real distribution: the universal
        # inequality kurtosis >= skewness^2 + 1 (provable for any distribution) makes
        # "inside" equal to a perfect square, (1 - gamma3*sr/2)^2, at its theoretical
        # floor -- always >= 0. A tiny negative value here is floating-point rounding
        # noise at that boundary (most visible for near-degenerate/two-value samples,
        # where kurtosis sits exactly at skewness^2+1), not a real violation -- clamp it.
        # A SUBSTANTIALLY negative value would mean something is actually wrong (a bug
        # in the moment calculations, or corrupted input), and still raises.
        if inside > -1e-9:
            inside = 0.0
        else:
            raise ValueError(
                f"Value under the sqrt in the SR std-error formula is substantially "
                f"negative ({inside:.3e}), which should be mathematically impossible "
                f"(skew={gamma3:.3f}, kurtosis={gamma4:.3f}, SR={sr:.3f}) -- check the "
                f"input data for corruption rather than trusting this result."
            )
    return math.sqrt(inside / (n - 1))


def expected_max_sharpe_under_n_trials(sr_std_error: float, n_trials: int) -> float:
    """E[max SR_n over N independent trials] under the null of no genuine skill —
    the benchmark the DSR tests the observed SR against, instead of testing against 0.
    Uses sr_std_error as a proxy for the cross-sectional std of the N trial Sharpes
    (the standard practical approximation when the full distribution of all N trial
    Sharpes isn't available — only the selected/final strategy's own SR std error is).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0  # a single trial has no "best of N" inflation to correct for
    z_a = norm.ppf(1 - 1 / n_trials)
    z_b = norm.ppf(1 - 1 / (n_trials * math.e))
    return sr_std_error * ((1 - EULER_MASCHERONI) * z_a + EULER_MASCHERONI * z_b)


@dataclass(frozen=True)
class DeflatedSharpeResult:
    observed_sr: float
    n_trials: int
    n_observations: int
    sr_std_error: float
    expected_max_sr_by_chance: float
    deflated_sharpe_ratio: float   # probability the true SR exceeds what chance alone predicts
    skewness: float
    kurtosis: float


def compute_deflated_sharpe(returns: list[float], n_trials: int) -> DeflatedSharpeResult:
    """returns: the trade-return series (per-trade R-multiples or $ P&L — must match
    whatever basis produced `n_trials` worth of comparable variants) on ONE consistent
    basis (do not mix per-trade and annualized here). n_trials: total variant/parameter
    combinations tried during development (spec §7.3: "log every variant tried"), NOT
    just 1 for the final selected system — using n_trials=1 silently reports the naive,
    non-deflated figure and defeats the entire point of this check.
    """
    sr = sharpe_ratio(returns)
    sr_std_err = sharpe_ratio_std_error(returns, sr)
    benchmark = expected_max_sharpe_under_n_trials(sr_std_err, n_trials)
    n = len(returns)
    dsr = norm.cdf((sr - benchmark) / sr_std_err * math.sqrt(n - 1)) if sr_std_err > 0 else float("nan")

    return DeflatedSharpeResult(
        observed_sr=sr, n_trials=n_trials, n_observations=n, sr_std_error=sr_std_err,
        expected_max_sr_by_chance=benchmark, deflated_sharpe_ratio=dsr,
        skewness=_sample_skewness(returns), kurtosis=_sample_kurtosis(returns),
    )
