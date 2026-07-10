"""Parameter lab: grids, fast scan, exact sweep, metric store, gates."""

import numpy as np
import polars as pl
import pytest

from backtest.lab import (
    CONDITION_NAMES,
    MetricStore,
    ParamGrid,
    _ffill_positions,
    _import_strategy,
    _max_drawdown,
    add_gate_lift,
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


def test_ffill_positions_gpu_matches_cpu():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
        cupy.add(cupy.asarray([0.0]), 1.0)
    except Exception as exc:
        pytest.skip(f"CUDA unavailable: {exc}")

    nan = np.nan
    events = np.array([
        [nan, nan],
        [1.0, -1.0],
        [nan, nan],
        [0.0, nan],
        [nan, 1.0],
        [-1.0, nan],
        [nan, 0.0],
    ])
    expected = _ffill_positions(events, np)
    actual = _ffill_positions(cupy.asarray(events), cupy).get()
    np.testing.assert_array_equal(actual, expected)

    cumulative = np.array([
        [1.0, -1.0],
        [3.0, 2.0],
        [2.0, 1.5],
        [-1.0, 4.0],
    ])
    expected_dd = _max_drawdown(cumulative, np)
    actual_dd = _max_drawdown(cupy.asarray(cumulative), cupy).get()
    np.testing.assert_array_equal(actual_dd, expected_dd)


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


def test_signal_matrix_returns_conditions():
    data = synthetic_data(n=800)
    z, combos, conditions = signal_matrix(
        data["anchor"], data["target"], [42, 63], [42, 63], return_conditions=True
    )
    assert set(conditions) == set(CONDITION_NAMES)
    for name, mat in conditions.items():
        assert mat.shape == z.shape, name
    # conditions vary by beta_lb only: columns 0/1 share beta_lb=42, 2/3 share 63
    r2 = conditions["r2"]
    np.testing.assert_array_equal(r2[:, 0], r2[:, 1])
    np.testing.assert_array_equal(r2[:, 2], r2[:, 3])
    tail = r2[-50:]
    assert not np.allclose(tail[:, 0], tail[:, 2], equal_nan=True)


# ── gated fast scan ──────────────────────────────────────────────────────────

def _regime_ou(n=6000, seed=5):
    """Block-regime OU: reversion (and vol) only exist when regime=1."""
    rng = np.random.default_rng(seed)
    regime = (np.arange(n) // 250 % 2).astype(float)
    resid = np.zeros(n)
    for i in range(1, n):
        theta, sigma = (0.15, 4.0) if regime[i] else (0.0, 1.0)
        resid[i] = resid[i - 1] * (1 - theta) + rng.normal(0, sigma)
    z = (resid - resid.mean()) / resid.std()
    return z, resid, regime


def test_fast_scan_gated_row_count_and_baseline():
    z, resid, regime = _regime_ou()
    gates = {"regime": regime, "noise": np.random.default_rng(9).normal(size=len(z))}
    out = fast_scan(z, resid, entries=[1.0, 1.5], gates=gates, gate_buckets=2)

    # rows = K × E × (1 + G × B) = 1 × 2 × (1 + 2×2)
    assert len(out) == 10
    assert {"gate", "gate_bucket", "sharpe"} <= set(out.columns)
    assert len(out.filter(pl.col("gate") == "(none)")) == 2

    # gating only removes entries — can never trade more than ungated
    for e in (1.0, 1.5):
        base_n = out.filter(
            (pl.col("gate") == "(none)") & (pl.col("entry_z") == e)
        )["n_trades"][0]
        gated_n = out.filter(
            (pl.col("gate") != "(none)") & (pl.col("entry_z") == e)
        )["n_trades"]
        assert (gated_n <= base_n).all()


def test_fast_scan_gate_concentrates_edge():
    z, resid, regime = _regime_ou()
    out = fast_scan(z, resid, entries=[1.0], gates={"regime": regime}, gate_buckets=2)

    base = out.filter(pl.col("gate") == "(none)")["sharpe"][0]
    on = out.filter(pl.col("gate_bucket") == "q2/2")["sharpe"][0]   # regime=1
    off = out.filter(pl.col("gate_bucket") == "q1/2")["sharpe"][0]  # regime=0
    # trading only the reverting regime beats ungated beats the dead regime
    assert on > base > off


def test_add_gate_lift():
    z, resid, regime = _regime_ou()
    out = fast_scan(z, resid, entries=[1.0], gates={"regime": regime}, gate_buckets=2)
    lifted = add_gate_lift(out)

    assert {"base_sharpe", "sharpe_lift", "hit_lift"} <= set(lifted.columns)
    base_rows = lifted.filter(pl.col("gate") == "(none)")
    assert (base_rows["sharpe_lift"].abs() < 1e-12).all()

    on = lifted.filter(pl.col("gate_bucket") == "q2/2")
    assert on["sharpe_lift"][0] == pytest.approx(
        on["sharpe"][0] - on["base_sharpe"][0]
    )
    assert on["sharpe_lift"][0] > 0


# ── exact sweep ──────────────────────────────────────────────────────────────

def test_strategy_import_falls_back_to_source_file(monkeypatch):
    def missing_top_package(_module_name):
        raise ModuleNotFoundError("No module named 'book'", name="book")

    monkeypatch.setattr("backtest.lab.importlib.import_module", missing_top_package)
    module = _import_strategy("book.rate_vol.template")
    assert callable(module.make_pipeline)


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
