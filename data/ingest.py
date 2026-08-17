"""MT5 -> UTC-converted, quality-gated, versioned Parquet snapshots (spec §1.1, §1.5).

Research code reads only from snapshots produced here — never live-pulls — so every run
is reproducible from a dated snapshot (spec §0.6, §9).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from configs.loader import load as load_config
from data.broker_time import broker_epoch_to_utc
from data.mt5_client import mt5_session
from data.quality import run_quality_gates
from data.symbol_info import fetch_symbol_meta

MT5_TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}

RATE_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]

# Approx bars/calendar-day per timeframe, used only to size safe chunk windows against
# the MT5 terminal's copy_rates_range cap (observed maxbars=100000 on this terminal).
_BARS_PER_DAY = {"M1": 1440, "M5": 288, "M15": 96, "M30": 48, "H1": 24, "H4": 6, "D1": 1}
_MAX_BARS_PER_CALL = 90_000  # buffer under the observed 100,000 cap


def _copy_rates_chunked(mt5, symbol: str, timeframe: str, tf_const, start_utc: datetime, end_utc: datetime):
    """copy_rates_range in date-bounded chunks so multi-year pulls on lower timeframes
    don't silently exceed the terminal's row cap (which fails with 'Invalid params',
    not an obvious size error) — see memory: this is a real MT5 gotcha, not a bug here.
    """
    bars_per_day = _BARS_PER_DAY[timeframe]
    chunk_days = max(1, _MAX_BARS_PER_CALL // bars_per_day)
    chunks = []
    cur = start_utc
    while cur < end_utc:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days), end_utc)
        raw = mt5.copy_rates_range(symbol, tf_const, cur, chunk_end)
        if raw is not None and len(raw) > 0:
            chunks.append(pd.DataFrame(raw))
        cur = chunk_end
    if not chunks:
        return None
    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return combined


def pull_and_snapshot(symbol: str, timeframe: str, start_utc: datetime, end_utc: datetime) -> Path:
    """Pull raw bars for [start_utc, end_utc), convert to true UTC, run quality gates,
    and write an immutable dated Parquet snapshot + JSON sidecar with full provenance.
    Raises if the quality gates fail — a failing snapshot is never written silently.
    """
    data_cfg = load_config("data")
    broker_tz = data_cfg["broker_timezone"]
    broker_meta = data_cfg["broker"]

    with mt5_session() as mt5:
        tf_const = getattr(mt5, MT5_TIMEFRAME_MAP[timeframe])
        raw = _copy_rates_chunked(mt5, symbol, timeframe, tf_const, start_utc, end_utc)
        if raw is None or len(raw) == 0:
            raise RuntimeError(
                f"No data returned for {symbol} {timeframe} [{start_utc}, {end_utc}): "
                f"{mt5.last_error()}"
            )
        sym_meta = fetch_symbol_meta(mt5, symbol)

    df = raw[RATE_COLUMNS].copy()
    df["time"] = broker_epoch_to_utc(df["time"], broker_tz)

    report = run_quality_gates(df, symbol, timeframe)
    print(report.summary())
    if not report.passed:
        raise ValueError(
            f"Data quality gates failed for {symbol} {timeframe} — refusing to snapshot. "
            f"{report.summary()}"
        )

    snapshot_dir = Path(data_cfg["snapshot"]["dir"])
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    pulled_at = datetime.now(timezone.utc)
    stamp = pulled_at.strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{symbol}_{timeframe}_{start_utc:%Y%m%d}_{end_utc:%Y%m%d}_{stamp}"
    parquet_path = snapshot_dir / f"{base_name}.parquet"
    sidecar_path = snapshot_dir / f"{base_name}.json"

    df.to_parquet(parquet_path, index=False)

    sidecar = {
        "symbol": symbol,
        "timeframe": timeframe,
        "requested_start_utc": start_utc.isoformat(),
        "requested_end_utc": end_utc.isoformat(),
        "actual_first_bar_utc": df["time"].iloc[0].isoformat(),
        "actual_last_bar_utc": df["time"].iloc[-1].isoformat(),
        "n_rows": len(df),
        "pulled_at_utc": pulled_at.isoformat(),
        "broker": broker_meta,
        "broker_timezone_used_for_conversion": broker_tz,
        "symbol_meta": asdict(sym_meta),
        "quality_report": {
            k: v for k, v in asdict_report(report).items() if k != "anomalous_gap_rows"
        },
        "parquet_file": parquet_path.name,
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, default=str)

    return parquet_path


def asdict_report(report) -> dict:
    d = report.__dict__.copy()
    return d
