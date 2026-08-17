"""Cost model (spec §6) — spread, commission, slippage, swap. All tagged to the broker
in configs/data.yaml; results are not transferable to another broker without re-validation.

Spread is charged as a single round-trip deduction rather than split across separate
bid/ask legs: the ingested OHLC series is a single (presumed bid) price stream with a
per-bar 'spread' column, not independent bid and ask series, so there's no honest way to
price two separate legs. This also means real historical spread widening (news,
rollover) is already captured automatically via the per-bar spread value — it is not a
synthetic assumption layered on top.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    commission_per_lot_round_trip: float
    slippage_base_pips: float
    slippage_spike_atr_multiple: float
    slippage_spike_extra_pips: float
    slippage_apply_to_target_exits: bool
    slippage_apply_to_market_entries: bool
    swap_rollover_hour_utc: int
    swap_triple_swap_weekday: int  # 0=Monday ... 6=Sunday


def load_cost_config(raw: dict) -> CostConfig:
    return CostConfig(
        commission_per_lot_round_trip=raw["commission"]["per_lot_round_trip"],
        slippage_base_pips=raw["slippage"]["base_pips"],
        slippage_spike_atr_multiple=raw["slippage"]["spike_atr_multiple"],
        slippage_spike_extra_pips=raw["slippage"]["spike_extra_pips"],
        slippage_apply_to_target_exits=raw["slippage"]["apply_to_target_exits"],
        slippage_apply_to_market_entries=raw["slippage"]["apply_to_market_entries"],
        swap_rollover_hour_utc=raw["swap"]["rollover_hour_utc"],
        swap_triple_swap_weekday=raw["swap"]["triple_swap_weekday"],
    )


@dataclass(frozen=True)
class TradeCosts:
    spread_cost: float        # account currency, all fields
    commission_cost: float
    slippage_cost: float
    swap_cost: float

    @property
    def total(self) -> float:
        return self.spread_cost + self.commission_cost + self.slippage_cost + self.swap_cost


def _price_amount_to_account_currency(price_amount: float, pip_size: float, pip_value_per_lot: float, lots: float) -> float:
    return (price_amount / pip_size) * pip_value_per_lot * lots


def spread_cost(spread_points: float, point_size: float, pip_size: float, pip_value_per_lot: float, lots: float) -> float:
    spread_price = spread_points * point_size
    return _price_amount_to_account_currency(spread_price, pip_size, pip_value_per_lot, lots)


def commission_cost(lots: float, cfg: CostConfig) -> float:
    return lots * cfg.commission_per_lot_round_trip


def slippage_cost(
    lots: float, pip_size: float, pip_value_per_lot: float, cfg: CostConfig,
    is_spike: bool, is_stop_exit: bool, is_target_exit: bool, is_market_entry: bool,
) -> float:
    """Only applies to fills spec §6 calls out as slippage-prone: stop exits always,
    market entries if configured, target exits only if explicitly configured on (limit-
    style fills are assumed better-executed by default)."""
    applies = is_stop_exit or (is_target_exit and cfg.slippage_apply_to_target_exits) \
        or (is_market_entry and cfg.slippage_apply_to_market_entries)
    if not applies:
        return 0.0
    pips = cfg.slippage_base_pips + (cfg.slippage_spike_extra_pips if is_spike else 0.0)
    return _price_amount_to_account_currency(pips * pip_size, pip_size, pip_value_per_lot, lots)


def is_spike_bar(bar_range_price: float, atr_value: float, cfg: CostConfig) -> bool:
    if pd.isna(atr_value) or atr_value <= 0:
        return False
    return bar_range_price >= cfg.slippage_spike_atr_multiple * atr_value


def count_swap_nights(entry_time: pd.Timestamp, exit_time: pd.Timestamp, cfg: CostConfig) -> int:
    """Number of rollover crossings the position was open through, with the configured
    triple-swap weekday counted 3x (standard FX weekend-settlement convention)."""
    if exit_time <= entry_time:
        return 0

    first_rollover = entry_time.normalize() + pd.Timedelta(hours=cfg.swap_rollover_hour_utc)
    if first_rollover < entry_time:
        first_rollover += pd.Timedelta(days=1)

    total = 0
    t = first_rollover
    while t < exit_time:
        total += 3 if t.weekday() == cfg.swap_triple_swap_weekday else 1
        t += pd.Timedelta(days=1)
    return total


def swap_cost(
    direction: str, nights: int, point_size: float, pip_size: float,
    pip_value_per_lot: float, lots: float, swap_long_points: float, swap_short_points: float,
) -> float:
    """MT5 swap_long/swap_short are broker-quoted points per lot per night (signed —
    usually negative, i.e. a cost, but can be positive for some pairs/directions)."""
    swap_points = swap_long_points if direction == "long" else swap_short_points
    swap_price_per_night = swap_points * point_size
    per_night_cost = -_price_amount_to_account_currency(swap_price_per_night, pip_size, pip_value_per_lot, lots)
    return per_night_cost * nights
