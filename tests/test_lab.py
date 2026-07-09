"""Parameter lab: grids, fast scan, exact sweep, metric store, gates."""

import numpy as np
import polars as pl
import pytest

from backtest.lab import (
    MetricStore,
    ParamGrid,
    _ffill_positions,
    fast_scan,
    gate_scan,
    signal_matrix,
    sweep_strategy,
)
from book.rate_vol.template import synthetic_data


# ── ParamGrid ────────────────────────────────────────────────────────────────

def test_param_grid_combos():
    grid = ParamGrid({"a": [1, 2, 3], "b": [10, 20]})
    combos = grid.combos()
    assert len(grid) == 6
    assert len(combos) == 6
    assert {"a": 1, "b": 10} in combos
    assert {"a": 3, "b": 20} in combos


# ── fast scan ────────────────────────────────────────────────────────────────

def test_ffill_positions_hand_example():
    nan = np.nan
    events = np.array([[nan], [1.0], [nan], [0.0], [nan], [-1.0], [nan]])
    pos = _ffill_positions(events, np)
    assert pos[:, 0].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0]


def test_fast_scan_huge_entry_never_trades():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 500)
    level = rng.normal(0, 1, 500).cumsum()
    out = fast_scan(z, level, entries=[50.0])
    assert out["n_trades"][0] == 0
    assert out["total_pnl_bps"][0] == 0.0


def test_fast_scan_profits_on_true_ou():
    # residual IS the traded level and it's OU -> fading must be profitable
    rng = np.random.default_rng(2)
    n = 3000
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = resid[i - 1] * 0.95 + rng.normal(0, 1)
    mu, sd = resid.mean(), resid.std()
    z = (resid - mu) / sd

    out = fast_scan(z, resid, entries=[1.5, 2.0])
    assert len(out) == 2
    assert (out["total_pnl_bps"] > 0).all()
    assert (out["hit_rate"] > 0.5).all()
    assert (out["max_drawdown_bps"] <= 0).all()


def test_fast_scan_costs_reduce_pnl():
    rng = np.random.default_rng(3)
    n = 2000
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = resid[i - 1] * 0.95 + rng.normal(0, 1)
    z = (resid - resid.mean()) / resid.std()

    free = fast_scan(z, resid, entries=[1.5], cost_bps=0.0)
    costly = fast_scan(z, resid, entries=[1.5], cost_bps=0.5)
    assert costly["total_pnl_bps"][0] < free["total_pnl_bps"][0]


def test_signal_matrix_shape_and_combos():
    data = synthetic_data(n=800)
    z, combos = signal_matrix(data["anchor"], data["target"], [42, 63], [42, 63])
    assert z.shape == (800, 4)
    assert combos[0] == {"beta_lb": 42, "z_lb": 42}
    assert len(combos) == 4
    # warmup rows are NaN, later rows populated
    assert np.isnan(z[0]).all()
    assert np.isfinite(z[-1]).all()

    out = fast_scan(z, data["target"].to_numpy(), entries=[2.0], combos=combos)
    assert len(out) == 4
    assert {"beta_lb", "z_lb", "entry_z", "sharpe"} <= set(out.columns)


# ── exact sweep ──────────────────────────────────────────────────────────────

def test_sweep_strategy_serial():
    data = synthetic_data(n=800)
    grid = {"beta_lb": [42, 63], "entry_z": [1.5, 2.0]}
    out = sweep_strategy("book.rate_vol.template", data, grid, n_jobs=1)
    assert len(out) == 4
    assert {"beta_lb", "entry_z", "sharpe", "n_trades"} <= set(out.columns)
    assert "error" not in out.columns
    # sorted best-first
    assert out["sharpe"][0] == out["sharpe"].max()


def test_sweep_strategy_surfaces_errors():
    data = synthetic_data(n=800)
    out = sweep_strategy(
        "book.rate_vol.template", data, {"beta_lb": [-5]}, n_jobs=1, sort_by="sharpe"
    )
    assert "error" in out.columns
    assert out["error"][0] is not None


# ── metric store ─────────────────────────────────────────────────────────────

def test_metric_store_roundtrip(tmp_path):
    store = MetricStore(tmp_path / "runs.parquet")
    assert store.load().is_empty()
    assert store.leaderboard().is_empty()

    r1 = pl.DataFrame({"beta_lb": [42, 63], "z_lb": [42, 42], "sharpe": [0.5, 1.5]})
    store.log("strat_a", r1, meta={"source": "synthetic"})
    r2 = pl.DataFrame({"beta_lb": [42], "z_lb": [63], "sharpe": [2.5]})
    store.log("strat_b", r2)

    df = store.load()
    assert len(df) == 3
    assert {"strategy", "run_ts", "sharpe"} <= set(df.columns)

    lead = store.leaderboard(top=2)
    assert lead["strategy"].to_list() == ["strat_b", "strat_a"]
    assert store.leaderboard(strategy="strat_a")["sharpe"].max() == 1.5

    mat = store.matrix(x="beta_lb", y="z_lb", metric="sharpe")
    assert len(mat) == 2  # z_lb 42 and 63 rows


# ── gates ────────────────────────────────────────────────────────────────────

def test_gate_scan_finds_informative_condition():
    # OU residual whose reversion only exists in persistent (block) regimes;
    # the regime indicator should show hit/pnl lift, the noise column ~none.
    rng = np.random.default_rng(5)
    n = 6000
    regime = (np.arange(n) // 250 % 2).astype(float)  # 250-bar blocks, 1 = mean-reverting
    resid = np.zeros(n)
    for i in range(1, n):
        # reverting regime is high-vol so it actually produces |z| extremes
        theta, sigma = (0.15, 4.0) if regime[i] else (0.0, 1.0)
        resid[i] = resid[i - 1] * (1 - theta) + rng.normal(0, sigma)
    z = (resid - resid.mean()) / resid.std()

    conditions = pl.DataFrame({
        "regime": regime,   # binary -> bucketed by value ("=0" / "=1")
        "noise": rng.normal(size=n),
    })
    out = gate_scan(z, resid, conditions, entry_z=1.0, horizon=10, n_buckets=2)

    assert out.filter(pl.col("condition") == "(all)")["n"][0] > 100
    regime_on = out.filter((pl.col("condition") == "regime") & (pl.col("bucket") == "=1"))
    regime_off = out.filter((pl.col("condition") == "regime") & (pl.col("bucket") == "=0"))
    # edge concentrates where reversion exists
    assert regime_on["hit"][0] > regime_off["hit"][0]
    assert regime_on["hit_lift"][0] > 0
    # noise buckets shouldn't beat the informative regime bucket
    noise_lift = out.filter(pl.col("condition") == "noise")["hit_lift"].abs().max()
    assert regime_on["hit_lift"][0] > noise_lift
