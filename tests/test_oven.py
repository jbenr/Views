"""Oven engine: frozen-model trade arithmetic on the hedged portfolio, daily
marks that add up to the trades, and that it finds edge only where it exists."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl

from oven.engine import bake, config_grid, daily_marks, walk_model


def _panel(y, x, target="10s30s", feature="10y"):
    ts = [date(2000, 1, 3) + timedelta(days=i) for i in range(len(y))]
    data = pl.DataFrame({"ts": ts, target: y, feature: x})
    return SimpleNamespace(data=data, target=SimpleNamespace(name=target))


def _arrays(n, alpha=0.0, beta_l=0.5, beta_c=0.2, sigma=1.0, resid=None):
    resid = np.zeros(n) if resid is None else resid
    return {"alpha": np.full(n, alpha), "beta_levels": np.full(n, beta_l), "beta_changes": np.full(n, beta_c),
            "resid_sigma": np.full(n, sigma), "entry_z": resid / sigma, "resid": resid}


def test_frozen_trade_is_the_hedged_portfolio():
    # residual y - 0.5x: 0 on bar 1, +3 on bar 2 (the cross), back to 0 on bar 4
    x = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 4.0, 4.0])
    y = np.array([0.0, 0.0, 3.0, 4.0, 1.0, 2.0, 2.0])
    arrays = _arrays(7, resid=y - 0.5 * x)
    configs = pl.DataFrame(dict(scale=["raw"], entry=[2.0], revert=[1.0], max_hold=[10], stop=[0.0]))

    trades, stats = walk_model(y, x, arrays, configs, hedge=1.0)
    assert len(trades) == 1
    cfg, decided, filled, side, pnl, reason = trades[0]
    assert (decided, filled, side, reason) == (2, 5, -1.0, 0)  # decided bar 2, reverted bar 4, filled bar 5
    # short the portfolio y - 0.2x from the bar-3 fill to the bar-5 fill: y 4 -> 2, x 2 -> 4
    assert np.isclose(pnl, -((2 - 4) - 0.2 * (4 - 2)))
    assert np.isclose(stats[0, 0], pnl)  # daily marks sum to the trade

    unhedged, _ = walk_model(y, x, arrays, configs, hedge=0.0)
    assert np.isclose(unhedged[0, 4], -(2 - 4))  # an untradable input: y alone


def test_daily_marks_match_board_totals():
    rng = np.random.default_rng(3)
    n = 1500
    x = np.cumsum(rng.normal(0, 5, n))
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.9 * ou[i - 1] + rng.normal(0, 2)
    y = 0.5 * x + ou
    panel = _panel(y, x)
    configs = config_grid(z_entries=[2.0], raw_entries=[], reverts=[1.0], max_holds=[20], stops=[0.0])
    run = bake(panel, ["10y"], lookbacks=[126], configs=configs, cost=0.5)
    marks = daily_marks(panel.data["ts"], y, x, run["trades"], cost=0.5)
    assert np.isclose(marks.sum(), run["board"]["total_pnl"][0])
    assert np.isclose(marks.sum(), run["trades"]["pnl"].sum())


def test_reverting_residual_pays_and_random_walk_does_not():
    rng = np.random.default_rng(7)
    n = 3000
    x = np.cumsum(rng.normal(0, 5, n))
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.9 * ou[i - 1] + rng.normal(0, 2)
    configs = config_grid(z_entries=[2.0], raw_entries=[], reverts=[1.0], max_holds=[20], stops=[0.0])

    reverting = bake(_panel(0.5 * x + ou, x), ["10y"], lookbacks=[252], configs=configs)["board"]
    assert reverting["sharpe"][0] > 1 and reverting["trade_t"][0] > 3

    walk = np.cumsum(rng.normal(0, 2, n))
    placebo = bake(_panel(walk, x), ["10y"], lookbacks=[252], configs=configs)["board"]
    assert abs(placebo["trade_t"][0]) < 3


def test_untradable_input_trades_the_target_alone():
    rng = np.random.default_rng(1)
    n = 1200
    x, y = np.cumsum(rng.normal(size=n)), np.cumsum(rng.normal(size=n))
    configs = config_grid(z_entries=[1.0], raw_entries=[], reverts=[1.0], max_holds=[20], stops=[0.0])
    run = bake(_panel(y, x, feature="oil"), ["oil"], lookbacks=[63], configs=configs)
    assert (run["trades"]["hedge_ratio"] == 0).all()
    assert not run["board"]["hedged"][0]
