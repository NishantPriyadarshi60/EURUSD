import pandas as pd

from signals.htf_bias import (
    compute_htf_direction_timeline, align_htf_direction_to_entry_tf,
    compute_htf_bias, htf_gate_allows,
)


def _h4_df(closes):
    n = len(closes)
    times = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"time": times, "high": closes, "low": closes, "close": closes})


def test_direction_timeline_hand_verified():
    # Same series/trace as test_structure.py: initial bearish @bar9, CHoCH bullish @bar13
    closes = [10, 9, 8, 9, 10, 11, 10, 9, 8, 7, 8, 9, 10, 12]
    df = _h4_df(closes)
    timeline = compute_htf_direction_timeline(df, swing_n=1, htf_timeframe_hours=4)

    assert list(timeline["direction"].iloc[:9]) == ["undefined"] * 9
    assert list(timeline["direction"].iloc[9:13]) == ["bearish"] * 4
    assert timeline["direction"].iloc[13] == "bullish"

    assert timeline["close_time"].iloc[0] == df["time"].iloc[0] + pd.Timedelta(hours=4)
    assert timeline["close_time"].iloc[5] == df["time"].iloc[5] + pd.Timedelta(hours=4)


def test_entry_tf_never_sees_a_still_forming_htf_bar():
    # H4 bars close at 04:00, 08:00, 12:00 with directions bullish, bullish, bearish.
    # An M15 bar at 08:00 (closes 08:15) must see the 08:00-closing H4 bar (bullish),
    # NOT the 12:00-closing H4 bar even though it has already OPENED by then and is
    # mid-formation -- this is the critical no-lookahead property spec §3.1 requires.
    htf_timeline = pd.DataFrame({
        "close_time": pd.to_datetime(["2024-01-01 04:00", "2024-01-01 08:00", "2024-01-01 12:00"], utc=True),
        "direction": ["bullish", "bullish", "bearish"],
    })
    entry_times = pd.to_datetime([
        "2024-01-01 07:45",  # closes 08:00 -> exact tie with 2nd H4 bar's close -> bullish
        "2024-01-01 08:00",  # closes 08:15 -> 3rd H4 bar (bearish) not yet closed -> still bullish
        "2024-01-01 11:30",  # closes 11:45 -> still before 12:00 -> still bullish
        "2024-01-01 11:45",  # closes 12:00 -> exact tie with 3rd H4 bar's close -> bearish
        "2024-01-01 12:00",  # closes 12:15 -> 3rd H4 bar now closed -> bearish
    ], utc=True)
    entry_df = pd.DataFrame({"time": entry_times})

    result = align_htf_direction_to_entry_tf(entry_df, htf_timeline, entry_timeframe_minutes=15)
    assert list(result) == ["bullish", "bullish", "bullish", "bearish", "bearish"]


def test_entry_bars_before_any_htf_bar_closes_are_undefined():
    htf_timeline = pd.DataFrame({
        "close_time": pd.to_datetime(["2024-01-01 04:00"], utc=True),
        "direction": ["bullish"],
    })
    entry_df = pd.DataFrame({"time": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"], utc=True)})
    result = align_htf_direction_to_entry_tf(entry_df, htf_timeline, entry_timeframe_minutes=15)
    assert list(result) == ["undefined", "undefined"]


def test_compute_htf_bias_end_to_end():
    closes = [10, 9, 8, 9, 10, 11, 10, 9, 8, 7, 8, 9, 10, 12]
    htf_df = _h4_df(closes)
    # entry-TF bars every 15 minutes spanning the whole H4 range
    entry_times = pd.date_range("2023-01-01 00:00", periods=4 * 14 * 4, freq="15min", tz="UTC")
    entry_df = pd.DataFrame({"time": entry_times})

    bias = compute_htf_bias(htf_df, entry_df, swing_n=1, htf_timeframe_hours=4, entry_timeframe_minutes=15)
    assert len(bias) == len(entry_df)
    assert set(bias.unique()) <= {"undefined", "bearish", "bullish"}
    # by the very end of the range the HTF has long since flipped bullish (bar 13 close)
    assert bias.iloc[-1] == "bullish"


def test_htf_gate_allows_long_only_when_bullish_or_neutral_allowed():
    assert htf_gate_allows("bullish", "long", allow_neutral=False) is True
    assert htf_gate_allows("bearish", "long", allow_neutral=False) is False
    assert htf_gate_allows("undefined", "long", allow_neutral=True) is True
    assert htf_gate_allows("undefined", "long", allow_neutral=False) is False


def test_htf_gate_allows_short_only_when_bearish_or_neutral_allowed():
    assert htf_gate_allows("bearish", "short", allow_neutral=False) is True
    assert htf_gate_allows("bullish", "short", allow_neutral=False) is False
    assert htf_gate_allows("undefined", "short", allow_neutral=True) is True
    assert htf_gate_allows("undefined", "short", allow_neutral=False) is False


def test_htf_gate_rejects_unknown_trade_direction():
    import pytest
    with pytest.raises(ValueError):
        htf_gate_allows("bullish", "sideways", allow_neutral=True)
