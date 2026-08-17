"""Walk-forward orchestration (spec §7.1) — rolls the LOCKED fold schedule
(configs/validation.yaml, decided 2026-08-17, see memory: validation-date-split) over
data, calling a caller-supplied strategy runner per fold and concatenating only the
out-of-window (test) results — "judge on the concatenated out-of-window results, never
the in-sample fit." The OOS vault is never touched here: an explicit guard raises if any
data crossing into the vault is ever handed to this module, since spec §7.1 requires it
sealed until every other check has passed.

The strategy runner itself is Phase 2's job (this Phase 1 harness has no strategy to
run) — `run_walk_forward` takes it as a callback: `(optimize_df, test_df) -> list of
trades from the test window only`. This module only owns the fold mechanics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from configs.loader import load as load_config


@dataclass(frozen=True)
class Fold:
    index: int
    optimize_start: pd.Timestamp
    optimize_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_folds(validation_cfg: Optional[dict] = None) -> list[Fold]:
    """Derives fold boundaries mechanically from configs/validation.yaml -- do not
    hand-edit folds (per that config's own comment). Reproduces the exact 8-fold
    schedule recorded in memory: validation-date-split."""
    cfg = validation_cfg if validation_cfg is not None else load_config("validation")
    wf = cfg["walk_forward"]
    pool_start = pd.Timestamp(wf["pool_start"], tz="UTC")
    pool_end = pd.Timestamp(wf["pool_end"], tz="UTC")
    opt_months = wf["optimize_months"]
    test_months = wf["test_months"]
    step_months = wf["step_months"]

    folds: list[Fold] = []
    opt_start = pool_start
    i = 1
    while True:
        opt_end = opt_start + pd.DateOffset(months=opt_months)
        test_start = opt_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_start >= pool_end:
            break
        if test_end > pool_end:
            test_end = pool_end
        folds.append(Fold(i, opt_start, opt_end, test_start, test_end))
        if test_end >= pool_end:
            break
        opt_start = opt_start + pd.DateOffset(months=step_months)
        i += 1
    return folds


def assert_oos_vault_untouched(df: pd.DataFrame, validation_cfg: Optional[dict] = None) -> None:
    cfg = validation_cfg if validation_cfg is not None else load_config("validation")
    oos_start = pd.Timestamp(cfg["oos_vault"]["start"], tz="UTC")
    if (df["time"] >= oos_start).any():
        raise ValueError(
            f"Data extends into the sealed OOS vault (starts {oos_start}). The vault "
            f"must not be touched until every other validation check has passed (spec "
            f"§7.1) -- pass only walk-forward-pool-range data to run_walk_forward."
        )


@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[Fold]
    trades_by_fold: list[list] = field(default_factory=list)   # test-window trades, per fold, in fold order
    all_out_of_window_trades: list = field(default_factory=list)  # concatenated -- the ONLY figure to judge


def run_walk_forward(
    df: pd.DataFrame,
    folds: list[Fold],
    strategy_runner: Callable[[pd.DataFrame, pd.DataFrame], list],
) -> WalkForwardResult:
    """df must be restricted to the walk-forward pool -- asserted, not assumed."""
    assert_oos_vault_untouched(df)

    trades_by_fold: list[list] = []
    for fold in folds:
        opt_df = df[(df["time"] >= fold.optimize_start) & (df["time"] < fold.optimize_end)].reset_index(drop=True)
        test_df = df[(df["time"] >= fold.test_start) & (df["time"] < fold.test_end)].reset_index(drop=True)
        test_trades = strategy_runner(opt_df, test_df)
        trades_by_fold.append(test_trades)

    all_trades = [t for fold_trades in trades_by_fold for t in fold_trades]
    return WalkForwardResult(folds=folds, trades_by_fold=trades_by_fold, all_out_of_window_trades=all_trades)
