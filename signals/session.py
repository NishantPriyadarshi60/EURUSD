"""Session / kill-zone gate (spec §3.2) — timing filter, all windows UTC. EUR/USD
structure means different things in Asian chop vs London/overlap; this gates entries to
configured windows and exposes which session(s) were active for logging.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dt_time

import pandas as pd


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start: dt_time
    end: dt_time

    def contains(self, t: dt_time) -> bool:
        if self.start <= self.end:
            return self.start <= t < self.end
        return t >= self.start or t < self.end  # wraps past midnight


def _parse_hhmm(s: str) -> dt_time:
    hh, mm = s.split(":")
    return dt_time(int(hh), int(mm))


def load_session_windows(raw: dict) -> list[SessionWindow]:
    return [
        SessionWindow(name=w["name"], start=_parse_hhmm(w["start_utc"]), end=_parse_hhmm(w["end_utc"]))
        for w in raw["session"]["windows"]
    ]


def active_sessions(timestamp: pd.Timestamp, windows: list[SessionWindow]) -> list[str]:
    """A timestamp can be in MULTIPLE overlapping sessions at once (e.g. the London/NY
    overlap window plus the standalone NY window) — returns all matching names, not
    just one, so downstream logging/gating can see the full picture, not a collapsed one."""
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware — see data/broker_time.py")
    t = timestamp.tz_convert("UTC").time()
    return [w.name for w in windows if w.contains(t)]


def in_any_session(timestamp: pd.Timestamp, windows: list[SessionWindow]) -> bool:
    return len(active_sessions(timestamp, windows)) > 0
