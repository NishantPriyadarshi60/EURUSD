import pytest

from validation.monte_carlo import run_monte_carlo, _max_drawdown_pct


def test_same_seed_is_exactly_reproducible():
    trade_returns = [100, -50, 200, -80, 150, -30, 75, -100, 60, 40]
    r1 = run_monte_carlo(trade_returns, starting_equity=10000, n_simulations=200, seed=42)
    r2 = run_monte_carlo(trade_returns, starting_equity=10000, n_simulations=200, seed=42)
    assert r1.final_return_pct_percentiles == r2.final_return_pct_percentiles
    assert r1.max_drawdown_pct_percentiles == r2.max_drawdown_pct_percentiles
    assert r1.sharpe_percentiles == r2.sharpe_percentiles


def test_different_seeds_generally_differ():
    trade_returns = [100, -50, 200, -80, 150, -30, 75, -100, 60, 40]
    r1 = run_monte_carlo(trade_returns, starting_equity=10000, n_simulations=200, seed=1)
    r2 = run_monte_carlo(trade_returns, starting_equity=10000, n_simulations=200, seed=2)
    assert r1.final_return_pct_percentiles != r2.final_return_pct_percentiles


def test_all_winning_trades_never_produces_a_losing_simulation():
    trade_returns = [50, 100, 75, 200, 30]  # all positive
    result = run_monte_carlo(trade_returns, starting_equity=10000, n_simulations=300, seed=7)
    assert all(v > 0 for v in result.final_return_pct_percentiles.values())
    assert all(v == 0.0 for v in result.max_drawdown_pct_percentiles.values())  # monotonic equity, never down


def test_percentiles_are_monotonically_ordered():
    trade_returns = [120, -90, 200, -150, 60, -40, 300, -200, 80, -60, 150, -100]
    result = run_monte_carlo(trade_returns, starting_equity=10000, n_simulations=500, seed=99,
                              percentiles=(5, 25, 50, 75, 95))
    fr = result.final_return_pct_percentiles
    assert fr[5] <= fr[25] <= fr[50] <= fr[75] <= fr[95]
    dd = result.max_drawdown_pct_percentiles
    assert dd[5] <= dd[25] <= dd[50] <= dd[75] <= dd[95]


def test_identical_trade_values_give_a_single_deterministic_outcome():
    # Every resample is composed of the SAME repeated value regardless of which indices
    # get drawn -> every simulation must produce the exact same final return and zero
    # drawdown, so every percentile collapses to that one value.
    trade_returns = [100.0] * 20
    result = run_monte_carlo(trade_returns, starting_equity=10000, n_simulations=50, seed=3)
    expected_final_return = (20 * 100.0) / 10000
    for v in result.final_return_pct_percentiles.values():
        assert round(v, 10) == round(expected_final_return, 10)
    for v in result.max_drawdown_pct_percentiles.values():
        assert v == 0.0


def test_max_drawdown_helper_hand_verified():
    # Same equity path as backtest/metrics.py's hand-verified drawdown test:
    # 10000 -> 10200 -> 10400 -> 10600 (peak) -> 10500 (drawdown of 100/10600)
    curve = [10000, 10200, 10400, 10600, 10500]
    assert round(_max_drawdown_pct(curve), 8) == round(100 / 10600, 8)


def test_requires_at_least_two_trade_returns():
    with pytest.raises(ValueError):
        run_monte_carlo([100.0], starting_equity=10000, n_simulations=10, seed=1)


def test_requires_positive_starting_equity():
    with pytest.raises(ValueError):
        run_monte_carlo([100.0, -50.0], starting_equity=0, n_simulations=10, seed=1)


def test_requires_at_least_one_simulation():
    with pytest.raises(ValueError):
        run_monte_carlo([100.0, -50.0], starting_equity=10000, n_simulations=0, seed=1)
