import pandas as pd

from risk.limits import RiskLimitsConfig, RiskManager


def test_max_concurrent_positions_blocks_until_close():
    config = RiskLimitsConfig(max_daily_loss_fraction=0.5, max_weekly_loss_fraction=0.5, max_concurrent_positions=1)
    manager = RiskManager(config=config, starting_equity=10000)
    t0 = pd.Timestamp("2024-01-02 10:00", tz="UTC")  # Tuesday

    assert manager.can_open(t0) is True
    manager.register_open()
    assert manager.can_open(t0) is False

    manager.register_close(t0 + pd.Timedelta(hours=1), net_pnl=-50)
    assert manager.can_open(t0 + pd.Timedelta(hours=2)) is True


def test_daily_loss_halt_resets_next_day():
    config = RiskLimitsConfig(max_daily_loss_fraction=0.02, max_weekly_loss_fraction=0.5, max_concurrent_positions=5)
    manager = RiskManager(config=config, starting_equity=10000)
    t0 = pd.Timestamp("2024-01-02 10:00", tz="UTC")

    manager.register_open()
    manager.register_close(t0, net_pnl=-250)  # -2.5% of 10000 > 2% daily cap

    assert manager.can_open(t0 + pd.Timedelta(hours=1)) is False  # halted, same day

    next_day = pd.Timestamp("2024-01-03 10:00", tz="UTC")
    assert manager.can_open(next_day) is True  # new day -> reset


def test_weekly_loss_halt_accumulates_across_days_and_resets_next_week():
    config = RiskLimitsConfig(max_daily_loss_fraction=0.5, max_weekly_loss_fraction=0.03, max_concurrent_positions=5)
    manager = RiskManager(config=config, starting_equity=10000)

    monday = pd.Timestamp("2024-01-01 10:00", tz="UTC")
    manager.register_open()
    manager.register_close(monday, net_pnl=-200)  # -2%, under the 50% daily cap
    assert manager.can_open(monday) is True  # weekly cumulative so far: -2%, under 3%

    tuesday = pd.Timestamp("2024-01-02 10:00", tz="UTC")
    manager.register_open()
    manager.register_close(tuesday, net_pnl=-150)  # cumulative -350 = -3.5% > 3% weekly cap
    assert manager.can_open(tuesday) is False

    wednesday_same_week = pd.Timestamp("2024-01-03 10:00", tz="UTC")
    assert manager.can_open(wednesday_same_week) is False  # still halted, same ISO week

    next_monday_new_week = pd.Timestamp("2024-01-08 10:00", tz="UTC")
    assert manager.can_open(next_monday_new_week) is True  # new week -> reset


def test_daily_loss_threshold_uses_day_start_equity_not_original_balance():
    # After a winning day1 raises equity to 11000, day2's 2% threshold must be measured
    # against 11000 (=220), not the original 10000 (=200) -- a -210 loss should NOT halt
    # under the correct (day-start-equity) rule, even though it WOULD have under the
    # original-balance rule. This is the discriminating case between the two conventions.
    config = RiskLimitsConfig(max_daily_loss_fraction=0.02, max_weekly_loss_fraction=0.5, max_concurrent_positions=5)
    manager = RiskManager(config=config, starting_equity=10000)

    monday = pd.Timestamp("2024-01-01 10:00", tz="UTC")
    manager.register_open()
    manager.register_close(monday, net_pnl=1000)  # equity -> 11000
    assert manager.equity == 11000

    tuesday = pd.Timestamp("2024-01-02 10:00", tz="UTC")
    manager.register_open()
    manager.register_close(tuesday, net_pnl=-210)  # -210 <= -220? no -> must NOT halt
    assert manager.can_open(tuesday) is True


def test_equity_tracks_realized_pnl():
    config = RiskLimitsConfig(max_daily_loss_fraction=0.5, max_weekly_loss_fraction=0.5, max_concurrent_positions=5)
    manager = RiskManager(config=config, starting_equity=10000)
    t0 = pd.Timestamp("2024-01-02 10:00", tz="UTC")

    manager.register_open()
    manager.register_close(t0, net_pnl=150)
    assert manager.equity == 10150

    manager.register_open()
    manager.register_close(t0, net_pnl=-60)
    assert manager.equity == 10090
