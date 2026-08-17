import pandas as pd
import pytest

from signals.session import SessionWindow, load_session_windows, active_sessions, in_any_session
from configs.loader import load as load_config


@pytest.fixture(scope="module")
def windows():
    return load_session_windows(load_config("signals"))


def test_loads_all_configured_windows():
    ws = load_session_windows(load_config("signals"))
    names = {w.name for w in ws}
    assert names == {"asian", "london", "london_ny_overlap", "ny"}


def test_timestamp_in_single_window(windows):
    ts = pd.Timestamp("2024-01-02 03:00", tz="UTC")  # inside Asian only
    assert active_sessions(ts, windows) == ["asian"]


def test_timestamp_in_overlapping_windows(windows):
    ts = pd.Timestamp("2024-01-02 13:00", tz="UTC")  # inside both london_ny_overlap and ny
    sessions = active_sessions(ts, windows)
    assert set(sessions) == {"london_ny_overlap", "ny"}


def test_timestamp_outside_any_window(windows):
    ts = pd.Timestamp("2024-01-02 20:00", tz="UTC")
    assert active_sessions(ts, windows) == []
    assert in_any_session(ts, windows) is False


def test_start_boundary_is_inclusive_end_boundary_is_exclusive():
    w = SessionWindow("test", start=pd.Timestamp("00:00").time(), end=pd.Timestamp("06:00").time())
    assert w.contains(pd.Timestamp("2024-01-01 00:00").time()) is True
    assert w.contains(pd.Timestamp("2024-01-01 05:59").time()) is True
    assert w.contains(pd.Timestamp("2024-01-01 06:00").time()) is False


def test_window_wrapping_past_midnight():
    w = SessionWindow("overnight", start=pd.Timestamp("22:00").time(), end=pd.Timestamp("02:00").time())
    assert w.contains(pd.Timestamp("2024-01-01 23:00").time()) is True
    assert w.contains(pd.Timestamp("2024-01-01 01:00").time()) is True
    assert w.contains(pd.Timestamp("2024-01-01 12:00").time()) is False
    assert w.contains(pd.Timestamp("2024-01-01 22:00").time()) is True   # inclusive start
    assert w.contains(pd.Timestamp("2024-01-01 02:00").time()) is False  # exclusive end


def test_naive_timestamp_raises(windows):
    naive = pd.Timestamp("2024-01-02 03:00")
    with pytest.raises(ValueError, match="timezone-aware"):
        active_sessions(naive, windows)


def test_non_utc_timezone_is_converted_before_checking(windows):
    # 23:00 in UTC-5 (e.g. US Eastern standard) = 04:00 UTC -> inside Asian
    ts = pd.Timestamp("2024-01-01 23:00", tz="Etc/GMT+5")
    sessions = active_sessions(ts, windows)
    assert "asian" in sessions
