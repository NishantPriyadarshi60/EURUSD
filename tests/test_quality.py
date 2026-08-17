import pandas as pd

from data.quality import run_quality_gates


def _bars(rows):
    """rows: list of (time_str, o, h, l, c) -> UTC-aware DataFrame."""
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def test_clean_daily_series_passes():
    rows = [
        ("2023-01-02", 1.10, 1.11, 1.09, 1.105),
        ("2023-01-03", 1.105, 1.12, 1.10, 1.11),
        ("2023-01-04", 1.11, 1.115, 1.108, 1.112),
    ]
    report = run_quality_gates(_bars(rows), "EURUSD", "D1")
    assert report.passed
    assert report.n_rows == 3


def test_duplicate_timestamp_fails():
    rows = [
        ("2023-01-02", 1.10, 1.11, 1.09, 1.105),
        ("2023-01-02", 1.105, 1.12, 1.10, 1.11),
    ]
    report = run_quality_gates(_bars(rows), "EURUSD", "D1")
    assert not report.passed
    assert report.n_duplicate_timestamps == 1


def test_non_monotonic_fails():
    rows = [
        ("2023-01-03", 1.10, 1.11, 1.09, 1.105),
        ("2023-01-02", 1.105, 1.12, 1.10, 1.11),
    ]
    report = run_quality_gates(_bars(rows), "EURUSD", "D1")
    assert not report.passed
    assert report.is_monotonic is False


def test_non_positive_price_fails():
    rows = [
        ("2023-01-02", 1.10, 1.11, 1.09, 1.105),
        ("2023-01-03", 0.0, 1.12, 1.10, 1.11),
    ]
    report = run_quality_gates(_bars(rows), "EURUSD", "D1")
    assert not report.passed
    assert report.n_non_positive_prices == 1


def test_ohlc_violation_fails():
    # low > open is impossible on a real candle
    rows = [
        ("2023-01-02", 1.10, 1.11, 1.09, 1.105),
        ("2023-01-03", 1.09, 1.12, 1.15, 1.11),
    ]
    report = run_quality_gates(_bars(rows), "EURUSD", "D1")
    assert not report.passed
    assert report.n_ohlc_violations == 1


def test_weekend_gap_is_not_an_anomaly():
    # Friday -> Monday, skipping Sat/Sun, must NOT count as an anomalous gap
    rows = [
        ("2023-01-06", 1.10, 1.11, 1.09, 1.105),  # Friday
        ("2023-01-09", 1.105, 1.12, 1.10, 1.11),   # Monday
        ("2023-01-10", 1.11, 1.115, 1.108, 1.112),  # Tuesday
    ]
    report = run_quality_gates(_bars(rows), "EURUSD", "D1")
    assert report.passed
    assert report.n_weekend_gaps == 1
    assert report.n_anomalous_gaps == 0


def test_midweek_gap_is_flagged_but_not_blocking():
    # Tuesday -> Thursday with no weekend in between must be flagged (visible in the
    # report) but must NOT hard-fail the gate by itself — real multi-year EURUSD data
    # has legitimate non-weekend closures (Christmas, New Year's) that aren't corruption.
    rows = [
        ("2023-01-03", 1.10, 1.11, 1.09, 1.105),  # Tuesday
        ("2023-01-05", 1.105, 1.12, 1.10, 1.11),   # Thursday (missed Wednesday)
    ]
    report = run_quality_gates(_bars(rows), "EURUSD", "D1")
    assert report.passed  # gaps don't block; corruption checks do
    assert report.n_anomalous_gaps == 1
    assert report.n_weekend_gaps == 0
    assert len(report.anomalous_gap_rows) == 1
