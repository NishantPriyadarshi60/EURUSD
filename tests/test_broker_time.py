import calendar
from datetime import datetime

import pandas as pd

from data.broker_time import broker_epoch_to_utc, offset_report

BROKER_TZ = "Europe/Bucharest"  # EET/EEST, EU DST calendar


def _epoch_for_server_wallclock(y, m, d, h, mi=0, s=0) -> int:
    """Build the raw epoch MT5 would report for a bar whose *server-local* wall clock
    reads y-m-d h:mi:s. This is deliberately computed the same buggy way naive code
    does it (as if the wall-clock reading were UTC), because that's what MT5 hands us.
    """
    return calendar.timegm(datetime(y, m, d, h, mi, s).timetuple())


def test_winter_offset_is_utc_plus_2():
    # 2023-01-15 10:00 server-local time (EET, winter) -> 08:00 true UTC
    epoch = _epoch_for_server_wallclock(2023, 1, 15, 10, 0, 0)
    utc_index = broker_epoch_to_utc([epoch], BROKER_TZ)
    assert utc_index[0] == pd.Timestamp("2023-01-15 08:00:00", tz="UTC")


def test_summer_offset_is_utc_plus_3():
    # 2023-07-15 10:00 server-local time (EEST, summer) -> 07:00 true UTC
    epoch = _epoch_for_server_wallclock(2023, 7, 15, 10, 0, 0)
    utc_index = broker_epoch_to_utc([epoch], BROKER_TZ)
    assert utc_index[0] == pd.Timestamp("2023-07-15 07:00:00", tz="UTC")


def test_dst_transition_is_flagged():
    # EU DST 2023: spring-forward last Sunday of March = 2023-03-26.
    # Bars either side of the transition should show different utc_offset and the
    # transition itself must be flagged so downstream session logic can see it.
    epochs = [
        _epoch_for_server_wallclock(2023, 3, 25, 12, 0, 0),  # still EET (+2)
        _epoch_for_server_wallclock(2023, 3, 27, 12, 0, 0),  # now EEST (+3)
    ]
    utc_index = broker_epoch_to_utc(epochs, BROKER_TZ)
    report = offset_report(utc_index, BROKER_TZ)
    assert report["utc_offset"].iloc[0] != report["utc_offset"].iloc[1]
    assert bool(report["dst_transition"].iloc[1]) is True


def test_no_naive_utc_mislabeling():
    # The classic bug: treating the raw epoch as if it were already true UTC.
    # Assert our conversion actually differs from that naive (wrong) interpretation
    # during winter, proving the offset correction is really being applied.
    epoch = _epoch_for_server_wallclock(2023, 1, 15, 10, 0, 0)
    naive_wrong = pd.to_datetime([epoch], unit="s", utc=True)
    correct = broker_epoch_to_utc([epoch], BROKER_TZ)
    assert naive_wrong[0] != correct[0]
    assert (naive_wrong[0] - correct[0]).total_seconds() == 2 * 3600
