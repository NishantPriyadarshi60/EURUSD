"""Order block detection (spec §2.6) — the last opposite-colored candle before a
structure-breaking move, gated by displacement quality (§2.9) so a real OB is
distinguished from a random candle that happened to precede a break.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from primitives.structure import StructureEvent


@dataclass(frozen=True)
class OrderBlock:
    candle_bar_index: int
    break_bar_index: int          # structure event bar that created/confirmed this OB
    direction: str                  # "bullish" | "bearish"
    zone_low: float
    zone_high: float
    body_low: float
    body_high: float
    mitigated_at_index: Optional[int]

    def is_unmitigated_at(self, bar_index: int) -> bool:
        """Usable from break_bar_index onward (when the OB is genuinely known live)."""
        if bar_index < self.break_bar_index:
            return False
        return self.mitigated_at_index is None or bar_index < self.mitigated_at_index


def detect_order_blocks(
    df: pd.DataFrame,
    structure_events: list[StructureEvent],
    displacement: pd.Series,
) -> list[OrderBlock]:
    """structure_events: BOS/CHoCH/initial events from structure.py (either resolution —
    "initial" is included since it's the same close-beyond-a-level break mechanic, just
    the bootstrap case). displacement: per-bar boolean Series from
    primitives.displacement.is_displacement, aligned to df's index — the §2.6 quality
    gate requires the breaking candle itself to qualify as displacement, or no OB is
    registered for that event.
    """
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n_bars = len(df)

    order_blocks: list[OrderBlock] = []

    for event in structure_events:
        break_idx = event.bar_index
        if break_idx >= n_bars or not bool(displacement.iloc[break_idx]):
            continue

        ob_idx = _find_ob_candle(opens, closes, break_idx, event.direction)
        if ob_idx is None:
            continue

        zone_low, zone_high = float(lows[ob_idx]), float(highs[ob_idx])
        body_low, body_high = sorted((float(opens[ob_idx]), float(closes[ob_idx])))

        mitigated_at = _find_mitigation(lows, highs, break_idx + 1, n_bars, zone_low, zone_high)

        order_blocks.append(OrderBlock(
            candle_bar_index=ob_idx,
            break_bar_index=break_idx,
            direction=event.direction,
            zone_low=zone_low,
            zone_high=zone_high,
            body_low=body_low,
            body_high=body_high,
            mitigated_at_index=mitigated_at,
        ))

    return order_blocks


def _find_ob_candle(opens, closes, break_idx: int, direction: str) -> Optional[int]:
    # Bullish break -> last DOWN-close candle before it. Bearish break -> last UP-close.
    for i in range(break_idx - 1, -1, -1):
        is_down = closes[i] < opens[i]
        is_up = closes[i] > opens[i]
        if direction == "bullish" and is_down:
            return i
        if direction == "bearish" and is_up:
            return i
    return None


def _find_mitigation(lows, highs, start: int, n_bars: int, zone_low: float, zone_high: float) -> Optional[int]:
    for j in range(start, n_bars):
        if lows[j] <= zone_high and highs[j] >= zone_low:
            return j
    return None
