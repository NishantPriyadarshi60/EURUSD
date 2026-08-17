"""Instrument metadata (spec §1.4). Pip value must be *computed*, never hardcoded —
contract size and tick value vary by broker and account currency.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolMeta:
    symbol: str
    point: float
    digits: int
    trade_tick_size: float
    trade_tick_value: float
    trade_contract_size: float
    pip_size: float          # 1 pip in price terms (0.0001 for a 5-digit EURUSD feed)
    pip_value_per_lot: float  # account-currency value of 1 pip move for 1.0 lot
    volume_min: float
    volume_max: float
    volume_step: float
    swap_long: float           # points per lot per night, long side (broker-quoted, signed)
    swap_short: float           # points per lot per night, short side


def fetch_symbol_meta(mt5_module, symbol: str) -> SymbolMeta:
    info = mt5_module.symbol_info(symbol)
    if info is None:
        raise ValueError(f"symbol_info returned None for {symbol}; is it selected/visible?")

    # 5-digit (or 3-digit JPY-style) feeds quote in fractional pips; a real pip is 10 points.
    pip_size = info.point * 10 if info.digits in (3, 5) else info.point

    # tick_value is the account-currency value of one trade_tick_size move, per lot.
    # Pip value per lot = tick_value * (pip_size / trade_tick_size).
    pip_value_per_lot = info.trade_tick_value * (pip_size / info.trade_tick_size)

    return SymbolMeta(
        symbol=symbol,
        point=info.point,
        digits=info.digits,
        trade_tick_size=info.trade_tick_size,
        trade_tick_value=info.trade_tick_value,
        trade_contract_size=info.trade_contract_size,
        pip_size=pip_size,
        pip_value_per_lot=pip_value_per_lot,
        volume_min=info.volume_min,
        volume_max=info.volume_max,
        volume_step=info.volume_step,
        swap_long=info.swap_long,
        swap_short=info.swap_short,
    )
