"""MT5 terminal connection lifecycle. Credentials come from .env only — never hardcoded,
never logged (spec §9: config-driven, nothing hardcoded).
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import MetaTrader5 as mt5
from dotenv import load_dotenv

load_dotenv()


class MT5ConnectionError(RuntimeError):
    pass


@contextmanager
def mt5_session():
    """Context manager: initialize + login on enter, shutdown on exit, always."""
    login_raw = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER", "").strip()

    if not login_raw or not password or not server:
        raise MT5ConnectionError(
            "MT5_LOGIN, MT5_PASSWORD, MT5_SERVER must all be set in .env"
        )

    if not mt5.initialize():
        raise MT5ConnectionError(f"mt5.initialize() failed: {mt5.last_error()}")

    try:
        ok = mt5.login(int(login_raw), password=password, server=server)
        if not ok:
            raise MT5ConnectionError(f"mt5.login() failed: {mt5.last_error()}")
        yield mt5
    finally:
        mt5.shutdown()
