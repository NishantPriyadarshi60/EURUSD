"""Broker-server-time -> UTC conversion (spec §1.3).

The trap: MT5's `copy_rates_range` returns a `time` field that is a Unix epoch integer,
but that integer encodes the broker server's *local wall-clock* reading, not true UTC.
Feeding it straight into `datetime.utcfromtimestamp()` silently produces a datetime whose
Y-M-D-H:M:S fields look plausible but are actually the server's local time mislabeled as
UTC — every session/kill-zone filter downstream would then be shifted by the broker's
UTC offset (2-3 hours for a EET/EEST broker), and that offset itself changes on DST
transition dates. Every bar in this project must go through this module before any
session-aware logic touches it.
"""
from __future__ import annotations

import pandas as pd
from zoneinfo import ZoneInfo


def broker_epoch_to_utc(epoch_seconds, broker_tz: str) -> pd.DatetimeIndex:
    """Convert MT5 raw epoch seconds (server-local wall clock, mislabeled as UTC) to
    genuine UTC timestamps, correctly handling the broker's own DST calendar.

    Accepts a list/array/Series of epoch seconds; always returns a DatetimeIndex
    (call site decides whether to assign it back into a Series/column).
    """
    naive_server_wallclock = pd.DatetimeIndex(pd.to_datetime(epoch_seconds, unit="s", utc=False))
    localized = naive_server_wallclock.tz_localize(ZoneInfo(broker_tz))
    return localized.tz_convert("UTC")


def offset_report(index_utc: pd.DatetimeIndex, broker_tz: str) -> pd.DataFrame:
    """Per-timestamp UTC offset that was applied, for the audit trail spec §1.3 requires
    ("store the offset used"). Also flags DST-transition days so gaps/jumps are explainable.
    """
    tz = ZoneInfo(broker_tz)
    local = index_utc.tz_convert(tz)
    offsets = local.map(lambda ts: ts.utcoffset())
    df = pd.DataFrame({"utc_time": index_utc, "broker_local_time": local, "utc_offset": offsets})
    df["dst_transition"] = df["utc_offset"].ne(df["utc_offset"].shift()).fillna(False)
    return df
