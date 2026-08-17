"""Monte Carlo robustness check (spec §7.5) — the FINAL robustness gate, run ON TOP OF
walk-forward + OOS + deflated Sharpe, never instead of them. Spec §0's explicit warning:
"MC on the final equity curve tells you the range of outcomes a given system could
produce — but if you data-mined many concepts and parameters to arrive at that system,
MC blesses an already-overfit artifact." Nothing here changes that; this module doesn't
and can't detect overfitting from the development process.

Resamples the trade-return sequence WITH REPLACEMENT (bootstrap — "resample", per spec
§7.5's wording, as distinct from a pure reshuffle/permutation) to build a distribution
of equity paths, max drawdowns, and Sharpe ratios. A pure reshuffle (permutation without
replacement) would leave the final-return distribution a single fixed value every time
— the sum of a fixed set of trades doesn't change by reordering them — so it wouldn't
answer "what final-return outcomes are plausible," only "how much does trade order
affect the drawdown path." Bootstrap resampling answers both, which is why it's the
primary technique implemented here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from validation.deflated_sharpe import sharpe_ratio


def _max_drawdown_pct(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        elif peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)
    return max_dd


@dataclass(frozen=True)
class MonteCarloResult:
    n_simulations: int
    n_trades_per_simulation: int
    seed: int
    final_return_pct_percentiles: dict[int, float]
    max_drawdown_pct_percentiles: dict[int, float]
    sharpe_percentiles: dict[int, float]


def run_monte_carlo(
    trade_returns: list[float],
    starting_equity: float,
    n_simulations: int,
    seed: int,
    percentiles: tuple[int, ...] = (5, 25, 50, 75, 95),
) -> MonteCarloResult:
    """trade_returns: net P&L per trade (account currency), in any order — bootstrap
    resampling discards the original sequence. starting_equity: used only to express
    drawdown/final-return as percentages, not to affect the resampled composition.
    seed: required, not optional — every run must be exactly reproducible (spec §0.6).
    """
    if len(trade_returns) < 2:
        raise ValueError("need at least 2 trade returns for a meaningful Monte Carlo resample")
    if starting_equity <= 0:
        raise ValueError("starting_equity must be positive")
    if n_simulations < 1:
        raise ValueError("n_simulations must be >= 1")

    rng = random.Random(seed)
    n = len(trade_returns)

    final_returns, max_drawdowns, sharpes = [], [], []
    for _ in range(n_simulations):
        resampled = [trade_returns[rng.randrange(n)] for _ in range(n)]

        equity = starting_equity
        equity_curve = [equity]
        for r in resampled:
            equity += r
            equity_curve.append(equity)

        final_returns.append((equity_curve[-1] - starting_equity) / starting_equity)
        max_drawdowns.append(_max_drawdown_pct(equity_curve))
        sharpes.append(sharpe_ratio(resampled))

    def _pct(values: list[float]) -> dict[int, float]:
        arr = np.array(values, dtype=float)
        return {p: float(np.percentile(arr, p)) for p in percentiles}

    return MonteCarloResult(
        n_simulations=n_simulations, n_trades_per_simulation=n, seed=seed,
        final_return_pct_percentiles=_pct(final_returns),
        max_drawdown_pct_percentiles=_pct(max_drawdowns),
        sharpe_percentiles=_pct(sharpes),
    )
