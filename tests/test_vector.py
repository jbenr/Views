"""The vectorised backtester must reproduce Engine bar for bar."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from backtest.engine import BacktestConfig, Engine, TradeDef, trade_log
from backtest.vector import run_vector
from research.dislocation_backtest import _entry_gate, make_pipeline, signal_frame

GATES = [("(none)", None, None), ("resid_vol20", "high_75", 126), ("r2", "below_50", 252),
         ("feature_move20", "tails_25_75", 126)]
STYLES = {"time": [1, 3, 10, 25], "band": [0.0, 0.25, 0.5], "revert_frac": [0.25, 0.5, 1.0],
          "half_life_frac": [0.5, 1.0, 2.0]}


def market(n=900, seed=5):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n))
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = 0.9 * resid[i - 1] + rng.normal()
    left = 300 + np.cumsum(rng.normal(size=n))
    right = left + 40 + 0.5 * x + resid
    return pl.DataFrame({
        "ts": [date(2020, 1, 1) + timedelta(days=i) for i in range(n)],
        "left": left, "right": right, "y": right - left, "x": x,
        # frozen-at-entry hedge weights, with a hole the engine must refuse
        "wl": -1 + 0.2 * np.sin(np.arange(n) / 30), "wr": np.where(np.arange(n) % 97 == 0, np.nan, 1.0),
    })


def engine_pnl(data, trade, cand, state, entry, style, param, stop, cost, lag, cap=None, signal_stop=None):
    pipe = make_pipeline(data, trade, cand, entry_z=entry, exit_style=style, exit_param=param,
                         stop_loss_bps=stop or None, state=state, execution_lag=lag,
                         half_life_cap=cap or None, signal_stop=signal_stop or None)
    result = Engine(BacktestConfig(transaction_cost_bps=cost, max_total_positions=1)).add_signal(pipe).run(data)
    return result.equity_curve["pnl_bps"].to_numpy(), trade_log(result.closed_trades)


@pytest.mark.parametrize("seed", range(6))
def test_vector_matches_engine_on_random_configs(seed):
    rng = np.random.default_rng(seed)
    data = market(seed=seed)
    trade = TradeDef("y", {"left": -1.0, "right": 1.0})
    weighted = bool(seed % 2)
    fit_on, kind = ["levels", "changes"][seed % 2], ["normalized", "ou_z", "raw"][seed % 3]
    cand = dict(target="y", feature="x", fit_on=fit_on, beta_lb=60, residual_lb=None if fit_on == "levels" else 10,
                norm_lb=80, signal_kind=kind)
    if weighted:
        cand["weight_columns"] = {"left": "wl", "right": "wr"}
    state = signal_frame(data, target="y", feature="x", fit_on=fit_on, beta_lb=60,
                         residual_lb=cand["residual_lb"], norm_lb=80, signal_kind=kind)
    engine_data = data
    if seed >= 3:  # a missing leg print: the engine skips the bar without ageing the trade
        engine_data = data.with_columns(pl.when(pl.int_range(pl.len()) % 173 == 50)
                                        .then(None).otherwise(pl.col("right")).alias("right"))

    masks, rows = [], []
    for name, bucket, window in GATES:
        if name != "(none)":
            masks.append(_entry_gate(state, dict(gate=name, gate_bucket=bucket, gate_window=window)))
    scale = 3.0 if kind == "raw" else 1.0
    for _ in range(25):
        style = str(rng.choice(list(STYLES)))
        gate = int(rng.integers(len(GATES)))
        rows.append(dict(model=0, gate=gate - 1, entry=scale * float(rng.choice([0.5, 1.0, 1.5, 2.0])),
                         exit_style=style, exit_param=float(rng.choice(STYLES[style])),
                         stop=float(rng.choice([0.0, 2.0, 5.0])),
                         signal_stop=scale * float(rng.choice([0.0, 0.25, 1.0])),
                         cap=float(rng.choice([0.0, 0.5, 2.0])) if style in ("band", "revert_frac") else 0.0))
    configs = pl.DataFrame(rows).filter((pl.col("exit_style") != "band") | (pl.col("exit_param") < pl.col("entry")))
    lag, cost = seed % 2, 0.3

    got = run_vector(
        engine_data.select("left", "right").to_numpy(), np.array([-1.0, 1.0]),
        state["signal"].to_numpy()[None, :], configs,
        half_life=state["half_life"].to_numpy()[None, :], gates=np.array(masks),
        entry_weights=data.select("wl", "wr").to_numpy() if weighted else None,
        cost=cost, lag=lag, folds=data["ts"].dt.month().to_numpy(), keep=range(len(configs)),
    )
    assert len(configs) > 15
    for k, row in enumerate(configs.iter_rows(named=True)):
        name, bucket, window = GATES[row["gate"] + 1]
        cand_k = {**cand, "gate": name, "gate_bucket": bucket, "gate_window": window}
        pnl, trades = engine_pnl(engine_data, trade, cand_k, state, row["entry"], row["exit_style"],
                                 row["exit_param"], row["stop"], cost, lag, row["cap"], row["signal_stop"])
        np.testing.assert_allclose(got.pnl[:, k], pnl, atol=1e-9, err_msg=str(row))
        metrics = got.metrics.row(k, named=True)
        assert metrics["n_trades"] == len(trades), row
        assert metrics["closed_pnl_bps"] == pytest.approx(float(trades["pnl_bps"].sum()) if len(trades) else 0.0, abs=1e-9)
        assert metrics["sharpe"] == pytest.approx(pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0.0, abs=1e-9)


def test_fold_statistics_rebuild_full_sample_sharpe():
    data = market()
    state = signal_frame(data, target="y", feature="x", fit_on="levels", beta_lb=60, residual_lb=None, norm_lb=80)
    configs = pl.DataFrame(dict(model=[0, 0], entry=[1.0, 1.5], exit_style=["time", "band"], exit_param=[5.0, 0.0]))
    got = run_vector(data.select("left", "right").to_numpy(), np.array([-1.0, 1.0]),
                     state["signal"].to_numpy()[None, :], configs, folds=data["ts"].dt.year().to_numpy(), keep=[0, 1])
    np.testing.assert_allclose(got.sharpe(), got.metrics["sharpe"].to_numpy())
    first = got.folds == got.folds[0]
    part = got.pnl[data["ts"].dt.year().to_numpy() == got.folds[0]]
    np.testing.assert_allclose(got.sharpe(first), part.mean(0) / part.std(0) * np.sqrt(252))
    assert got.fold_trades.sum(axis=1).tolist() == got.metrics["n_trades"].to_list()


def test_unknown_exit_style_and_missing_inputs_are_rejected():
    legs, sig = np.zeros((10, 1)), np.zeros((1, 10))
    with pytest.raises(ValueError, match="unknown exit styles"):
        run_vector(legs, np.ones(1), sig, pl.DataFrame(dict(model=[0], entry=[1.0], exit_style=["nope"], exit_param=[1.0])))
    with pytest.raises(ValueError, match="need half_life"):
        run_vector(legs, np.ones(1), sig, pl.DataFrame(dict(model=[0], entry=[1.0], exit_style=["half_life_frac"], exit_param=[1.0])))
    with pytest.raises(ValueError, match="no gate masks"):
        run_vector(legs, np.ones(1), sig, pl.DataFrame(dict(model=[0], gate=[0], entry=[1.0], exit_style=["time"], exit_param=[1.0])))


def test_stop_needs_a_loss_strictly_beyond_the_stop_like_engine():
    # long entry at 0 (fills bar 2); loss hits exactly -2 on bar 3, beyond it on bar 5
    level = np.array([0.0, 0.0, 0.0, -2.0, -2.0, -3.0, -3.0, -3.0])
    signal = np.array([0.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    configs = pl.DataFrame(dict(model=[0], entry=[1.5], exit_style=["time"], exit_param=[0.0], stop=[2.0]))
    got = run_vector(level[:, None], np.ones(1), signal[None, :], configs, lag=1, keep=[0])
    data = pl.DataFrame({"ts": [date(2020, 1, 1) + timedelta(days=i) for i in range(len(level))], "y": level})
    state = pl.DataFrame({"signal": signal, "resid": 0.0 * signal, "beta": 1.0 + 0 * signal,
                          "r2": 0.5 + 0 * signal, "half_life": 1.0 + 0 * signal})
    cand = dict(target="y", feature="x", fit_on="levels", beta_lb=30, residual_lb=None, norm_lb=40, gate="(none)")
    pnl, trades = engine_pnl(data, TradeDef.outright("y", "y"), cand, state, 1.5, "time", 0.0, 2.0, 0.0, 1)
    assert trades["exit_date"].to_list() == [data["ts"][5]]
    np.testing.assert_allclose(got.pnl[:, 0], pnl)
    assert got.metrics["n_trades"][0] == 1


@pytest.mark.parametrize("variant", ["plain", "gated_split", "weighted_raw"])
def test_vector_grid_reproduces_run_grid_table(variant):
    from research.dislocation_backtest import run_grid, run_vector_grid
    data = market()
    trade = TradeDef("y", {"left": -1.0, "right": 1.0})
    cand = dict(target="y", feature="x", fit_on="changes", beta_lb=60, residual_lb=10, norm_lb=80,
                signal_kind="normalized", gate="(none)")
    if variant == "gated_split":
        cand.update(gate="resid_vol20", gate_bucket="high_75", gate_window=126, split_date="2021-06-01")
    if variant == "weighted_raw":
        cand.update(signal_kind="raw", weight_columns={"left": "wl", "right": "wr"}, split_date="2021-01-15")
    entries = [3.0, 5.0] if variant == "weighted_raw" else [1.0, 1.5]
    kwargs = dict(entry_zs=entries, exit_params={"time": [5, 20], "band": [0.0, 0.5], "revert_frac": [0.5],
                                                 "half_life_frac": [1.0]},
                  stop_losses=[None, 3.0], round_trip_cost_bps=0.25, execution_lag=1,
                  half_life_caps=[None, 1.0], signal_stops=[None, 0.5])
    exact, exact_sel = run_grid(data, trade, cand, **kwargs)
    fast, fast_sel, yearly, _ = run_vector_grid(data, trade, cand, **kwargs)
    cols = [c for c in fast.columns if c in exact.columns]
    assert set(exact.columns) - set(fast.columns) == {"median_pnl_bps"}
    for col in cols:
        a, b = exact[col].to_list(), fast[col].to_list()
        if isinstance(a[0], float) or isinstance(b[0], float):
            np.testing.assert_allclose(np.array(b, dtype=float), np.array(a, dtype=float), atol=1e-9, err_msg=col)
        else:
            assert a == b, col
    assert fast_sel["metrics"]["config_id"] == exact_sel["metrics"]["config_id"]
    assert fast_sel["trades"] == exact_sel["trades"]
    per_year = yearly.group_by("config_id").agg(pl.col("pnl_bps").sum())
    total = fast.join(per_year, on="config_id")
    np.testing.assert_allclose(total["pnl_bps"].to_numpy(), total["total_pnl_bps"].to_numpy(), atol=1e-9)



def test_signal_stop_exits_when_the_dislocation_extends_and_cap_bounds_band_trades():
    # long entry at -2 (fills bar 2); the signal extends to -3.2 on bar 4 -> signal stop
    signal = np.array([0.0, -2.0, -2.5, -3.2, -3.0, -1.0, 0.5, 0.5])
    level = np.arange(8, dtype=float)
    configs = pl.DataFrame(dict(model=[0, 0, 0], entry=[1.5] * 3, exit_style=["band"] * 3, exit_param=[0.0] * 3,
                                signal_stop=[1.0, 0.0, 0.0], cap=[0.0, 0.0, 1.0]))
    half_life = np.full(8, 2.0)
    got = run_vector(level[:, None], np.ones(1), signal[None, :], configs, half_life=half_life[None, :], lag=1, keep=[0, 1, 2])
    # signal stop: -3.2 < -2.0 - 1.0 on signal bar 3, filled bar 4 -> held bars 3..4 = 2 bars
    assert got.metrics["avg_holding_days"].to_list() == [2.0, 5.0, 2.0]
    # no stop: band exit once the signal crosses back above 0 (signal bar 6 -> fill bar 7);
    # cap: ceil(2 x 1) = 2 bars
    assert got.metrics["n_trades"].to_list() == [1, 1, 1]
