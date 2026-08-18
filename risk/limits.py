"""Hard risk limits (spec §5) — coded, not discretionary. Breaching a limit halts new
entries; an already-open position is not force-closed by this module (that would be a
separate policy decision the spec doesn't mandate).

For a single-instrument (EUR/USD only) v1 system, "max correlated exposure" collapses to
"max concurrent positions" — there is only one instrument to be correlated with itself.
The check is still named separately so the seam exists if a second instrument is ever
added, without pretending correlation is meaningfully measured today.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class RiskLimitsConfig:
    max_daily_loss_fraction: float
    max_weekly_loss_fraction: float
    max_concurrent_positions: int


def load_risk_limits_config(raw: dict) -> RiskLimitsConfig:
    limits = raw["limits"]
    return RiskLimitsConfig(
        max_daily_loss_fraction=limits["max_daily_loss_fraction"],
        max_weekly_loss_fraction=limits["max_weekly_loss_fraction"],
        max_concurrent_positions=limits["max_concurrent_positions"],
    )


@dataclass
class RiskManager:
    """Reactive state machine: call `can_open` before opening a trade, `register_open`
    when it fills, `register_close` when it exits. Daily/weekly loss fractions are
    measured against equity AT THE START of that calendar day/week (a rolling
    reference), not the account's original starting balance — the standard convention
    for "you can't lose more than X% of today's starting balance" risk limits. A
    position's P&L is attributed to the day/week it CLOSED in (realized loss), not the
    day it opened — an open position doesn't count against the daily loss limit until
    it's actually realized.
    """
    config: RiskLimitsConfig
    starting_equity: float

    _equity: float = field(init=False, repr=False)
    _open_positions: int = field(default=0, init=False)
    _current_day: Optional[pd.Timestamp] = field(default=None, init=False, repr=False)
    _current_week: Optional[tuple] = field(default=None, init=False, repr=False)
    _day_start_equity: float = field(default=0.0, init=False, repr=False)
    _week_start_equity: float = field(default=0.0, init=False, repr=False)
    _daily_realized_pnl: float = field(default=0.0, init=False, repr=False)
    _weekly_realized_pnl: float = field(default=0.0, init=False, repr=False)
    _halted_today: bool = field(default=False, init=False)
    _halted_this_week: bool = field(default=False, init=False)

    def __post_init__(self):
        self._equity = self.starting_equity

    @property
    def equity(self) -> float:
        return self._equity

    def _roll_calendar(self, as_of: pd.Timestamp) -> None:
        day = as_of.normalize()
        if self._current_day is None or day != self._current_day:
            self._current_day = day
            self._day_start_equity = self._equity
            self._daily_realized_pnl = 0.0
            self._halted_today = False

        iso_year, iso_week, _ = as_of.isocalendar()
        week_key = (iso_year, iso_week)
        if self._current_week is None or week_key != self._current_week:
            self._current_week = week_key
            self._week_start_equity = self._equity
            self._weekly_realized_pnl = 0.0
            self._halted_this_week = False

    def can_open(self, as_of: pd.Timestamp) -> bool:
        self._roll_calendar(as_of)
        if self._halted_today or self._halted_this_week:
            return False
        if self._open_positions >= self.config.max_concurrent_positions:
            return False
        return True

    def register_open(self) -> None:
        self._open_positions += 1

    def register_close(self, exit_time: pd.Timestamp, net_pnl: float) -> None:
        self._roll_calendar(exit_time)
        self._open_positions = max(0, self._open_positions - 1)
        self._equity += net_pnl
        self._daily_realized_pnl += net_pnl
        self._weekly_realized_pnl += net_pnl

        if self._day_start_equity > 0 and \
                self._daily_realized_pnl <= -self.config.max_daily_loss_fraction * self._day_start_equity:
            self._halted_today = True
        if self._week_start_equity > 0 and \
                self._weekly_realized_pnl <= -self.config.max_weekly_loss_fraction * self._week_start_equity:
            self._halted_this_week = True
