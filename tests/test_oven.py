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
            "resid_sigma": np.full(n, sigma), "entry_z": resid / sigma, "resid": resid, "half_life": np.full(n, 5.0)}


def test_frozen_trade_is_the_hedged_portfolio():
    # residual y - 0.5x: 0 on bar 1, +3 on bar 2 (the cross), back to 0 on bar 4
    x = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 4.0, 4.0])
    y = np.array([0.0, 0.0, 3.0, 4.0, 1.0, 2.0, 2.0])
    arrays = _arrays(7, resid=y - 0.5 * x)
    configs = pl.DataFrame(dict(scale=["raw"], entry=[2.0], revert=[1.0], max_hold=[10], stop=[0.0], hl_mult=[0.0]))

    trades, stats = walk_model(y, x, arrays, configs, hedge=1.0)
    assert len(trades) == 1
    cfg, decided, filled, side, pnl, reason = trades[0][:6]
    assert (decided, filled, side, reason) == (2, 5, -1.0, 0)  # decided bar 2, reverted bar 4, filled bar 5
    # short the portfolio y - 0.2x from the bar-3 fill to the bar-5 fill: y 4 -> 2, x 2 -> 4
    assert np.isclose(pnl, -((2 - 4) - 0.2 * (4 - 2)))
    assert np.isclose(stats[0, 0], pnl)  # daily marks sum to the trade

    unhedged, _ = walk_model(y, x, arrays, configs, hedge=0.0)
    assert np.isclose(unhedged[0, 4], -(2 - 4))  # an untradable input: y alone


def test_half_life_time_exit_and_reversion_off():
    # the residual crosses on bar 2 and never comes back: only a time stop can end the trade
    x = np.zeros(12)
    y = np.array([0.0, 0.0, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9])
    arrays = {**_arrays(12, resid=y - 0.5 * x), "half_life": np.full(12, 2.0)}
    base = dict(scale=["raw"], entry=[2.0], max_hold=[8], stop=[0.0])
    trades, _ = walk_model(y, x, arrays, pl.DataFrame({**base, "revert": [0.0], "hl_mult": [1.5]}))
    # 1.5 x half-life 2 = 3 bars: decided bar 2, timed out bar 5, filled bar 6, reason half_life
    assert (trades[0, 1], trades[0, 2], trades[0, 5]) == (2, 6, 3)
    trades, _ = walk_model(y, x, arrays, pl.DataFrame({**base, "revert": [0.0], "hl_mult": [0.0]}))
    assert (trades[0, 2], trades[0, 5]) == (11, 2)  # no half-life stop: the 8-bar max hold, reason time

    # a reversion exit still wins when it comes first
    y2 = np.array([0.0, 0.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    arrays2 = {**_arrays(12, resid=y2), "half_life": np.full(12, 4.0)}
    trades, _ = walk_model(y2, x, arrays2, pl.DataFrame({**base, "revert": [0.5], "hl_mult": [1.0]}))
    assert (trades[0, 2], trades[0, 5]) == (4, 0)


def test_daily_marks_match_board_totals():
    rng = np.random.default_rng(3)
    n = 1500
    x = np.cumsum(rng.normal(0, 5, n))
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.9 * ou[i - 1] + rng.normal(0, 2)
    y = 0.5 * x + ou
    panel = _panel(y, x)
    configs = config_grid(z_entries=[2.0], raw_entries=[], reverts=[1.0], max_holds=[20], stops=[0.0], hl_mults=[0.0])
    run = bake(panel, ["10y"], lookbacks=[126], hedge_lookbacks=[126], configs=configs, cost=0.5)
    for basis in ("levels", "portfolio"):
        # the board scores closed trades; a position still open at the end is kept out
        trades = run["trades"].filter((pl.col("basis") == basis) & (pl.col("reason") != "open"))
        marks = daily_marks(panel.data["ts"], y, x, trades, cost=0.5)
        assert np.isclose(marks.sum(), run["board"].filter(pl.col("basis") == basis)["total_pnl"][0])
        assert np.isclose(marks.sum(), trades["pnl"].sum())


def test_reverting_residual_pays_and_random_walk_does_not():
    rng = np.random.default_rng(7)
    n = 3000
    x = np.cumsum(rng.normal(0, 5, n))
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.9 * ou[i - 1] + rng.normal(0, 2)
    configs = config_grid(z_entries=[2.0], raw_entries=[], reverts=[1.0], max_holds=[20], stops=[0.0], hl_mults=[0.0])

    reverting = bake(_panel(0.5 * x + ou, x), ["10y"], lookbacks=[252], hedge_lookbacks=[126], configs=configs)["board"]
    assert reverting["sharpe"][0] > 1 and reverting["trade_t"][0] > 3

    walk = np.cumsum(rng.normal(0, 2, n))
    placebo = bake(_panel(walk, x), ["10y"], lookbacks=[252], hedge_lookbacks=[126], configs=configs)["board"]
    assert abs(placebo["trade_t"][0]) < 3


def test_portfolio_basis_signal_is_the_traded_portfolio():
    rng = np.random.default_rng(11)
    n = 3000
    x = np.cumsum(rng.normal(0, 5, n))
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.9 * ou[i - 1] + rng.normal(0, 2)
    configs = config_grid(z_entries=[2.0], raw_entries=[], reverts=[1.0], max_holds=[20], stops=[0.0], hl_mults=[0.0])
    run = bake(_panel(0.5 * x + ou, x), ["10y"], lookbacks=[252], hedge_lookbacks=[126], configs=configs)
    board = run["board"].sort("basis")
    assert board["basis"].to_list() == ["levels", "portfolio"]
    assert board.filter(pl.col("basis") == "portfolio")["sharpe"][0] > 1


def test_trades_record_every_horizon_at_entry():
    rng = np.random.default_rng(5)
    n = 2000
    x = np.cumsum(rng.normal(0, 5, n))
    y = 0.5 * x + np.cumsum(rng.normal(0, 1, n)) * 0.3
    configs = config_grid(z_entries=[1.5], raw_entries=[], reverts=[1.0], max_holds=[20], stops=[0.0], hl_mults=[0.0])
    trades = bake(_panel(y, x), ["10y"], lookbacks=[63, 252], hedge_lookbacks=[126], configs=configs)["trades"]
    assert {"z_lb63", "z_lb252"} <= set(trades.columns)
    for lb in (63, 252):  # a trade's own horizon is its entry z
        own = trades.filter(pl.col("lookback") == lb)
        assert np.allclose(own[f"z_lb{lb}"].to_numpy(), own["entry_z"].to_numpy(), equal_nan=True)


def test_untradable_input_trades_the_target_alone():
    rng = np.random.default_rng(1)
    n = 1200
    x, y = np.cumsum(rng.normal(size=n)), np.cumsum(rng.normal(size=n))
    configs = config_grid(z_entries=[1.0], raw_entries=[], reverts=[1.0], max_holds=[20], stops=[0.0], hl_mults=[0.0])
    run = bake(_panel(y, x, feature="oil"), ["oil"], lookbacks=[63], hedge_lookbacks=[126], configs=configs)
    assert (run["trades"]["hedge_ratio"] == 0).all()
    assert not run["board"]["hedged"][0]
    assert run["board"]["basis"].to_list() == ["levels"]  # no hedge, so no portfolio basis


def test_config_grid_keeps_fractional_entries_after_many_whole_ones():
    # the app sends whole numbers as ints; hundreds of int-entry rows must not truncate a later 2.5
    configs = config_grid(z_entries=[1, 1.5], raw_entries=[2.5, 5, 7.5], reverts=[0, 0.5, 1],
                          max_holds=[5, 10, 20, 40, 60, 120], stops=[0, 1, 2], hl_mults=[0, 1, 2, 3])
    assert configs["entry"].dtype == pl.Float64
    assert sorted(configs["entry"].unique().to_list()) == [1.0, 1.5, 2.5, 5.0, 7.5]


def test_position_still_held_at_the_end_is_an_open_trade_outside_the_stats():
    # the residual crosses on bar 2 and never reverts inside the sample
    x = np.zeros(7)
    y = np.array([0.0, 0.0, 3.0, 3.5, 4.0, 4.5, 5.0])
    arrays = _arrays(7, resid=y - 0.5 * x)
    configs = pl.DataFrame(dict(scale=["raw"], entry=[2.0], revert=[1.0], max_hold=[20], stop=[0.0], hl_mult=[0.0]))

    trades, stats = walk_model(y, x, arrays, configs, hedge=1.0, cost=0.5)
    assert len(trades) == 1
    cfg, decided, filled, side, pnl, reason = trades[0][:6]
    assert (decided, filled, side, reason) == (2, 6, -1.0, 4)  # "open", marked to the last bar
    assert np.isclose(pnl, -(5.0 - 3.5))  # short from the bar-3 fill, no exit cost yet
    assert stats[0, 0] == 0.0 and stats[0, 4] == 0  # no closed trade: nothing scored


def test_stop_loss_fires_on_the_close_and_fills_a_bar_later():
    # short from the bar-3 fill at 3.0; on bar 5's close the trade is down 3 bps, past the 2 bp stop loss
    x = np.zeros(10)
    y = np.array([0.0, 0.0, 3.0, 3.0, 4.0, 6.0, 7.0, 7.0, 7.0, 7.0])
    arrays = _arrays(10, resid=y - 0.5 * x)
    configs = pl.DataFrame(dict(scale=["raw"] * 2, entry=[2.0] * 2, revert=[1.0] * 2, max_hold=[20] * 2,
                                stop=[0.0] * 2, hl_mult=[0.0] * 2, pnl_stop=[2.0, 0.0]))

    trades, _ = walk_model(y, x, arrays, configs, hedge=1.0)
    stopped = trades[trades[:, 0] == 0][0]
    assert (stopped[1], stopped[2], stopped[5]) == (2, 6, 5)  # decided bar 5, filled bar 6, "pnl_stop"
    assert np.isclose(stopped[4], -(7.0 - 3.0))  # the fill is a bar after the close that tripped it
    assert trades[trades[:, 0] == 1][0][5] == 4  # without the stop loss it is still open


def test_expected_return_and_time_follow_the_ou_half_life():
    # entry residual 3 with a 2-bar half-life: half of it (revert 0.5) is due in one half-life
    x = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 4.0, 4.0])
    y = np.array([0.0, 0.0, 3.0, 4.0, 1.0, 2.0, 2.0])
    arrays = {**_arrays(7, resid=y - 0.5 * x), "half_life": np.full(7, 2.0)}
    configs = pl.DataFrame(dict(scale=["raw"], entry=[2.0], revert=[0.5], max_hold=[10], stop=[0.0], hl_mult=[0.0]))

    trades, _ = walk_model(y, x, arrays, configs, hedge=1.0, cost=0.25)
    size, exp_return, exp_bars = trades[0][6:]
    assert np.isclose(exp_bars, 2.0)
    assert np.isclose(exp_return, 3.0 * 0.5 - 0.25)  # net of cost
    assert size == 1.0  # flat


def test_sizing_by_expected_return_scales_each_trade_and_averages_one():
    # two trades, entry residuals 3 and 6, each fully reverting: flat P&L 3 and 6
    x = np.zeros(12)
    y = np.array([0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 6.0, 6.0, 0.0, 0.0, 0.0])
    arrays = {**_arrays(12, resid=y), "half_life": np.full(12, 2.0)}
    base = dict(scale=["raw"] * 2, entry=[2.0] * 2, revert=[1.0] * 2, max_hold=[20] * 2, stop=[0.0] * 2,
                hl_mult=[0.0] * 2)
    configs = pl.DataFrame({**base, "sizing": ["flat", "exp_return"]})

    trades, stats = walk_model(y, x, arrays, configs, hedge=1.0)
    flat, sized = trades[trades[:, 0] == 0], trades[trades[:, 0] == 1]
    assert np.allclose(flat[:, 4], [3.0, 6.0])
    assert np.isclose(sized[:, 6].mean(), 1.0)  # sizes scaled to average 1
    assert np.allclose(sized[:, 6], [2 / 3, 4 / 3], atol=1e-3)  # in proportion to expected return 3 : 6
    assert np.allclose(sized[:, 4], flat[:, 4] * sized[:, 6])
    assert np.isclose(stats[1, 0], sized[:, 4].sum())  # the stats score the sized marks


def test_full_reversion_expects_ninety_percent_by_three_point_three_half_lives():
    x = np.zeros(12)
    y = np.array([0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 6.0, 6.0, 0.0, 0.0, 0.0])
    arrays = {**_arrays(12, resid=y), "half_life": np.full(12, 2.0)}
    configs = pl.DataFrame(dict(scale=["raw"], entry=[2.0], revert=[1.0], max_hold=[20], stop=[0.0], hl_mult=[0.0]))
    trades, _ = walk_model(y, x, arrays, configs, hedge=1.0)
    assert np.allclose(trades[:, 8], 2.0 * np.log2(10))  # well inside the 20-bar max hold
    assert np.allclose(trades[:, 7], [0.9 * 3.0, 0.9 * 6.0])


def test_a_promoted_setup_runs_on_the_dashboard_as_it_did_in_the_bake(tmp_path):
    # bake a synthetic 10s30s vs 10y, promote its best setup, re-walk it the way the dashboard does
    from dashboard.registry import LiveRegistry
    from oven import app
    from oven.engine import model_frame, save
    from oven.live import OvenStrategy

    rng = np.random.default_rng(7)
    n = 1500
    x = np.cumsum(rng.normal(0, 5, n))
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.9 * ou[i - 1] + rng.normal(0, 2)
    panel = _panel(0.5 * x + ou, x)
    configs = config_grid(z_entries=[2.0], raw_entries=[], reverts=[1.0], max_holds=[20], stops=[0.0], hl_mults=[0.0])
    run = bake(panel, ["10y"], lookbacks=[126], hedge_lookbacks=[126], configs=configs, cost=0.5)
    run_dir = save(run, "10s30s", ["10y"], {"cost": 0.5}, root=tmp_path)
    best = run["board"].sort("sharpe", descending=True).row(0, named=True)

    registry = LiveRegistry(path=tmp_path / "live_signals.parquet")
    entry = registry.promote_oven(str(run_dir), "10y", app.setup_id(best))
    assert entry["kind"] == "oven" and registry.get(entry["signal_id"])["oven_setup"] == entry["oven_setup"]

    strategy = OvenStrategy(registry.get(entry["signal_id"]))
    log, _open, equity = strategy.history(model_frame(panel.data, "10s30s", "10y", 126, 126))
    assert len(log) == best["n_trades"]
    assert np.isclose(log["pnl_bps"].sum(), best["total_pnl"])
    assert set(equity.columns) == {"ts", "pnl_bps", "cumulative_pnl"}
