import pandas as pd
import pytest

from validation.walk_forward import generate_folds, assert_oos_vault_untouched, run_walk_forward, Fold


def test_generate_folds_reproduces_the_locked_8_fold_schedule():
    folds = generate_folds()
    assert len(folds) == 8

    expected = [
        (1, "2022-09-01", "2023-09-01", "2023-09-01", "2023-12-01"),
        (2, "2022-12-01", "2023-12-01", "2023-12-01", "2024-03-01"),
        (3, "2023-03-01", "2024-03-01", "2024-03-01", "2024-06-01"),
        (4, "2023-06-01", "2024-06-01", "2024-06-01", "2024-09-01"),
        (5, "2023-09-01", "2024-09-01", "2024-09-01", "2024-12-01"),
        (6, "2023-12-01", "2024-12-01", "2024-12-01", "2025-03-01"),
        (7, "2024-03-01", "2025-03-01", "2025-03-01", "2025-06-01"),
        (8, "2024-06-01", "2025-06-01", "2025-06-01", "2025-08-17"),  # truncated
    ]
    for fold, (idx, opt_s, opt_e, test_s, test_e) in zip(folds, expected):
        assert fold.index == idx
        assert fold.optimize_start == pd.Timestamp(opt_s, tz="UTC")
        assert fold.optimize_end == pd.Timestamp(opt_e, tz="UTC")
        assert fold.test_start == pd.Timestamp(test_s, tz="UTC")
        assert fold.test_end == pd.Timestamp(test_e, tz="UTC")


def test_oos_vault_guard_raises_on_data_crossing_the_boundary():
    times = pd.date_range("2025-08-01", "2025-09-01", freq="1D", tz="UTC")  # crosses 2025-08-17
    df = pd.DataFrame({"time": times})
    with pytest.raises(ValueError, match="sealed OOS vault"):
        assert_oos_vault_untouched(df)


def test_oos_vault_guard_passes_for_pool_only_data():
    times = pd.date_range("2023-01-01", "2023-06-01", freq="1D", tz="UTC")  # entirely inside the pool
    df = pd.DataFrame({"time": times})
    assert_oos_vault_untouched(df) is None  # does not raise


def test_run_walk_forward_slices_each_fold_correctly_and_concatenates_in_order():
    # Small custom fold schedule (not the locked one) purely to exercise the slicing
    # and concatenation mechanics with a controllable, hand-traceable date range.
    folds = [
        Fold(1, pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-03", tz="UTC"),
             pd.Timestamp("2023-01-03", tz="UTC"), pd.Timestamp("2023-01-04", tz="UTC")),
        Fold(2, pd.Timestamp("2023-01-02", tz="UTC"), pd.Timestamp("2023-01-04", tz="UTC"),
             pd.Timestamp("2023-01-04", tz="UTC"), pd.Timestamp("2023-01-05", tz="UTC")),
    ]
    times = pd.date_range("2023-01-01", "2023-01-05", freq="6h", tz="UTC", inclusive="left")
    df = pd.DataFrame({"time": times, "value": range(len(times))})

    calls = []

    def mock_runner(opt_df, test_df):
        calls.append((len(opt_df), len(test_df), list(test_df["time"])))
        # return one "trade" marker per test-window row, tagged with its own time,
        # so we can verify the aggregate preserves fold order and only test-window data
        return [(t, "trade") for t in test_df["time"]]

    result = run_walk_forward(df, folds, mock_runner)

    assert len(result.trades_by_fold) == 2
    # fold 1 test window: 2023-01-03 00:00 to 2023-01-04 00:00 (exclusive) at 6h steps -> 4 bars
    assert calls[0][1] == 4
    # fold 2 test window: 2023-01-04 00:00 to 2023-01-05 00:00 (exclusive) -> 4 bars
    assert calls[1][1] == 4

    all_times = [t for t, _ in result.all_out_of_window_trades]
    assert all_times == sorted(all_times)  # fold order preserved, no scrambling
    assert all(pd.Timestamp("2023-01-03", tz="UTC") <= t < pd.Timestamp("2023-01-05", tz="UTC") for t in all_times)


def test_run_walk_forward_refuses_data_touching_the_oos_vault():
    folds = generate_folds()
    times = pd.date_range("2025-08-10", "2025-08-20", freq="1D", tz="UTC")  # crosses the real vault boundary
    df = pd.DataFrame({"time": times})
    with pytest.raises(ValueError, match="sealed OOS vault"):
        run_walk_forward(df, folds, lambda opt, test: [])
