"""Position sizing (spec §5) — risk a fixed fraction of equity per trade, size derived
from ATR-based stop distance and computed pip value. Never a fixed lot size.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionSize:
    lots: float
    risk_amount: float          # account-currency amount actually at risk after rounding
    stop_distance_price: float
    clamped: bool                 # True if volume_min/max forced a size away from the
                                    # raw risk-target size (distinct from ordinary step
                                    # rounding, which always happens and isn't a problem)


def compute_position_size(
    equity: float,
    risk_fraction: float,
    stop_distance_price: float,
    pip_value_per_lot: float,
    pip_size: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
) -> PositionSize:
    if equity <= 0:
        raise ValueError("equity must be positive")
    if stop_distance_price <= 0:
        raise ValueError("stop_distance_price must be positive")
    if volume_step <= 0:
        raise ValueError("volume_step must be positive")

    risk_target = equity * risk_fraction
    stop_distance_pips = stop_distance_price / pip_size
    loss_per_lot = stop_distance_pips * pip_value_per_lot
    raw_lots = risk_target / loss_per_lot

    stepped = round(raw_lots / volume_step) * volume_step
    was_clamped = stepped < volume_min or stepped > volume_max
    lots = min(max(stepped, volume_min), volume_max)
    lots = round(round(lots / volume_step) * volume_step, 8)  # trim float step-arithmetic noise

    return PositionSize(
        lots=lots,
        risk_amount=lots * loss_per_lot,
        stop_distance_price=stop_distance_price,
        clamped=was_clamped,
    )
