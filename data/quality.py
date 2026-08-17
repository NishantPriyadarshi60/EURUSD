"""Data-quality gates that must pass before any backtest touches a snapshot (spec §1.5).

Checks: duplicate timestamps, monotonic time index, non-positive prices, OHLC sanity
(low <= open,close <= high), and gaps — with weekend gaps (Fri close -> Sun/Mon open)
classified separately from anomalous gaps during the trading week, since collapsing
those two into one bucket would hide real data holes behind the expected weekend close.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

TIMEFRAME_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}


@dataclass
class QualityReport:
    symbol: str
    timeframe: str
    n_rows: int
    n_duplicate_timestamps: int
    is_monotonic: bool
    n_non_positive_prices: int
    n_ohlc_violations: int
    n_weekend_gaps: int
    n_anomalous_gaps: int
    anomalous_gap_rows: pd.DataFrame = field(repr=False)
    passed: bool = field(init=False)

    def __post_init__(self):
        # Gaps (weekend or otherwise) are deliberately NOT part of the pass/fail gate.
        # Spec §1.5 says gaps must be *flagged*, not that zero gaps are required to pass —
        # real EURUSD data legitimately has non-weekend closures (Christmas, New Year's)
        # that would otherwise hard-fail every multi-year snapshot. Hardcoding a holiday
        # calendar to auto-excuse those would just swap one discretionary judgment call
        # for another. Instead every gap is counted and reported (see anomalous_gap_rows)
        # so a human reviews the snapshot sidecar before trusting it; only genuine
        # corruption (duplicates, non-monotonic index, bad prices, OHLC violations) blocks.
        self.passed = (
            self.n_duplicate_timestamps == 0
            and self.is_monotonic
            and self.n_non_positive_prices == 0
            and self.n_ohlc_violations == 0
        )

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.symbol} {self.timeframe}: {self.n_rows} rows | "
            f"dup_ts={self.n_duplicate_timestamps} monotonic={self.is_monotonic} "
            f"bad_prices={self.n_non_positive_prices} ohlc_violations={self.n_ohlc_violations} "
            f"weekend_gaps={self.n_weekend_gaps} anomalous_gaps={self.n_anomalous_gaps}"
        )


def run_quality_gates(df: pd.DataFrame, symbol: str, timeframe: str) -> QualityReport:
    """df must have a UTC-aware 'time' column (already converted via broker_time) and
    open/high/low/close columns. Does not mutate df.
    """
    if df["time"].dt.tz is None:
        raise ValueError("quality gates require a UTC-aware 'time' column — convert first")

    n_rows = len(df)
    n_dup = int(df["time"].duplicated().sum())
    is_monotonic = bool(df["time"].is_monotonic_increasing)

    non_positive = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    n_non_positive = int(non_positive.sum())

    ohlc_bad = (
        (df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])
        | (df["high"] < df["open"]) | (df["high"] < df["close"])
    )
    n_ohlc_bad = int(ohlc_bad.sum())

    weekend_gaps, anomalous_gaps = _classify_gaps(df, timeframe)

    return QualityReport(
        symbol=symbol,
        timeframe=timeframe,
        n_rows=n_rows,
        n_duplicate_timestamps=n_dup,
        is_monotonic=is_monotonic,
        n_non_positive_prices=n_non_positive,
        n_ohlc_violations=n_ohlc_bad,
        n_weekend_gaps=len(weekend_gaps),
        n_anomalous_gaps=len(anomalous_gaps),
        anomalous_gap_rows=anomalous_gaps,
    )


def _classify_gaps(df: pd.DataFrame, timeframe: str):
    expected_minutes = TIMEFRAME_MINUTES.get(timeframe)
    if expected_minutes is None:
        raise ValueError(f"Unknown timeframe {timeframe!r}; add it to TIMEFRAME_MINUTES")

    expected = pd.Timedelta(minutes=expected_minutes)
    deltas = df["time"].diff()

    is_gap = deltas > expected
    gap_rows = df.loc[is_gap, ["time"]].copy()
    gap_rows["prev_time"] = df["time"].shift(1).loc[is_gap]
    gap_rows["delta"] = deltas.loc[is_gap]

    # A gap "spans a weekend" if some calendar day strictly between prev_time and time is
    # a Saturday, and the gap is no longer than ~3.5 days total. We check for an actual
    # Saturday in the span rather than comparing endpoint weekdays directly, because a D1
    # (or H4) bar's own boundary shifts by the broker's UTC offset once converted (spec
    # §1.3) — e.g. a broker-local Friday-00:00 D1 bar can land on UTC Thursday 22:00,
    # which breaks any check that assumes the bar's own weekday still reads "Friday".
    def _is_weekend_gap(row) -> bool:
        if row["delta"] > pd.Timedelta(days=3, hours=12):
            return False
        day = row["prev_time"].normalize()
        end_day = row["time"].normalize()
        while day <= end_day:
            if day.weekday() == 5:  # Saturday
                return True
            day += pd.Timedelta(days=1)
        return False

    if len(gap_rows) == 0:
        return gap_rows, gap_rows

    is_weekend = gap_rows.apply(_is_weekend_gap, axis=1)
    return gap_rows.loc[is_weekend], gap_rows.loc[~is_weekend]
