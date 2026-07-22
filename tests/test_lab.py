"""Parameter lab: grids, fast scan, exact sweep, metric store, gates."""

import numpy as np
import polars as pl
import pytest

from backtest.lab import (
    CONDITION_NAMES,
    MetricStore,
    ParamGrid,
    _ffill_positions,
    _get_xp,
    _import_strategy,
    _max_drawdown,
    add_gate_lift,
    add_predict_lift,
    fast_scan,
    gate_allow_mask,
    gate_scan,
    gate_variant_count,
    parse_gate,
    predict_scan,
    stateful_exit_scan,
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


def test_get_xp_auto_falls_back_or_uses_gpu():
    xp = _get_xp("auto")
    assert xp.__name__ in {"numpy", "cupy"}


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


def test_fast_scan_scans_exit_bands():
    rng = np.random.default_rng(33)
    n = 2000
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = resid[i - 1] * 0.95 + rng.normal(0, 1)
    z = (resid - resid.mean()) / resid.std()

    out = fast_scan(z, resid, entries=[1.5, 2.0], exit_band=[0.25, 0.75])

    assert len(out) == 4
    assert set(out["exit_band_bps"].to_list()) == {0.25, 0.75}
    assert {"entry_z", "exit_band_bps", "sharpe"} <= set(out.columns)


def test_fast_scan_entry_allow_restricts_opening_bars():
    z = np.array([0.0, -3.0, -3.0, 0.0, 0.0])
    level = np.array([0.0, 3.0, 2.0, 0.0, 0.0])

    blocked = fast_scan(z, level, entries=[2.0], entry_allow=np.zeros(5, dtype=bool))
    allowed = fast_scan(
        z,
        level,
        entries=[2.0],
        entry_allow=np.array([False, True, False, False, False]),
    )

    assert blocked["n_trades"][0] == 0
    assert allowed["n_trades"][0] == 1


def test_predict_scan_detects_true_ou():
    rng = np.random.default_rng(34)
    n = 3000
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = resid[i - 1] * 0.95 + rng.normal(0, 1)
    z = (resid - resid.mean()) / resid.std()

    out = predict_scan(z, resid, entries=[1.5], horizons=[5, 20])

    assert len(out) == 2
    assert {"horizon", "ic", "hit_rate", "fire_rate", "n_obs"} <= set(out.columns)
    assert {"avg_fwd_pnl", "fwd_sharpe", "avg_abs_signal"}.isdisjoint(out.columns)
    assert (out["ic"] > 0).all()
    assert (out["hit_rate"] > 0.5).all()
    assert ((out["fire_rate"] > 0) & (out["fire_rate"] <= 1)).all()


def test_predict_scan_counts_threshold_crossings_not_active_bars():
    z = np.zeros(35)
    z[2:22] = 2.0   # one long positive excursion
    z[25:30] = -2.0  # one long negative excursion
    level = np.arange(len(z), dtype=float)

    out = predict_scan(z, level, entries=[1.5], horizons=[1])

    assert out["n_obs"][0] == 2


def test_predict_scan_each_entry_threshold_fires_when_first_crossed():
    z = np.array([0.0, 26.0, 28.0, 31.0, 29.0, 24.0, 26.0, 0.0])
    level = np.arange(len(z), dtype=float)

    out = predict_scan(z, level, entries=[25.0, 30.0], horizons=[1])

    n_by_entry = dict(zip(out["entry_threshold"], out["n_obs"]))
    assert n_by_entry == {25.0: 2, 30.0: 1}


def test_predict_scan_gates_and_lift():
    z, resid, regime = _regime_ou()
    out = predict_scan(z, resid, entries=[1.0], horizons=[10], gates={"regime": regime}, gate_buckets=2)
    lifted = add_predict_lift(out)

    assert len(out) == 3
    assert {"base_ic", "ic_lift", "base_hit_rate", "hit_lift", "base_fire_rate", "fire_rate_lift"} <= set(lifted.columns)
    assert {"base_fwd_sharpe", "fwd_sharpe_lift"}.isdisjoint(lifted.columns)
    assert len(lifted.filter(pl.col("gate") == "(none)")) == 1


def test_predict_lift_matches_null_param_keys():
    results = pl.DataFrame({
        "entry_signal": ["residual", "residual"],
        "beta_lb": [60, 60],
        "ou_lb": [None, None],
        "entry_threshold": [20.0, 20.0],
        "horizon": [20, 20],
        "gate": ["(none)", "resid_phi"],
        "gate_bucket": ["all", "low_10"],
        "n_obs": [100, 30],
        "ic": [0.10, 0.25],
        "hit_rate": [0.52, 0.60],
        "fire_rate": [0.20, 0.08],
    })

    lifted = add_predict_lift(results)
    gated = lifted.filter(pl.col("gate") == "resid_phi").row(0, named=True)
    assert gated["base_ic"] == pytest.approx(0.10)
    assert gated["ic_lift"] == pytest.approx(0.15)


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


def test_fast_scan_regime_gate_labels_and_count():
    z, resid, regime = _regime_ou()
    out = fast_scan(z, resid, entries=[1.0], gates={"regime": regime}, gate_buckets="regime")

    expected_buckets = {
        "low_10", "high_90", "mid_10_90", "tails_10_90",
        "low_25", "high_75", "mid_25_75", "tails_25_75",
        "mid_40_60", "tails_40_60",
        "below_50", "above_50",
    }
    assert gate_variant_count("regime") == len(expected_buckets)
    assert len(out) == 1 + len(expected_buckets)
    assert set(out.filter(pl.col("gate") != "(none)")["gate_bucket"]) == expected_buckets


def test_predict_scan_regime_gate_labels():
    z, resid, regime = _regime_ou()
    out = predict_scan(
        z, resid, entries=[1.0], horizons=[10], gates={"regime": regime}, gate_buckets="regime"
    )

    assert "tails_10_90" in set(out["gate_bucket"])
    assert "mid_25_75" in set(out["gate_bucket"])
    assert "mid_40_60" in set(out["gate_bucket"])
    assert "tails_40_60" in set(out["gate_bucket"])
    assert {"ic", "fire_rate", "n_obs"} <= set(out.columns)


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


def test_gate_lift_matches_null_param_keys():
    results = pl.DataFrame({
        "entry_signal": ["residual", "residual"],
        "beta_lb": [60, 60],
        "ou_lb": [None, None],
        "entry_threshold": [20.0, 20.0],
        "exit_threshold": [5.0, 5.0],
        "gate": ["(none)", "resid_phi"],
        "gate_bucket": ["all", "low_10"],
        "total_pnl_bps": [10.0, 15.0],
        "sharpe": [0.20, 0.50],
        "hit_rate": [0.51, 0.56],
        "max_drawdown_bps": [-5.0, -4.0],
        "n_trades": [20, 8],
        "n_bars_active": [100, 40],
    })

    lifted = add_gate_lift(results)
    gated = lifted.filter(pl.col("gate") == "resid_phi").row(0, named=True)
    assert gated["base_sharpe"] == pytest.approx(0.20)
    assert gated["sharpe_lift"] == pytest.approx(0.30)


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


# ── stateful exit scan ───────────────────────────────────────────────────────

def test_stateful_exit_scan_revert_frac_hand_example():
    # enter long at z=-3 (t1); 50% reversion target = -1.5; z hits -1.4 at t3
    z = np.array([0.0, -3.0, -2.0, -1.4, 0.0, 0.0])
    level = np.array([0.0, 3.0, 2.0, 1.4, 0.0, 0.0])
    out = stateful_exit_scan(
        z, level, entries=[2.0], exit_style="revert_frac", exit_params=[0.5]
    )
    row = out.row(0, named=True)
    assert row["n_trades"] == 1
    assert row["n_bars_active"] == 2  # position live t1->t3, earns on t2, t3
    assert row["total_pnl_bps"] == pytest.approx(-1.6)  # (2-3) + (1.4-2)


def test_stateful_exit_scan_half_life_frac_times_out():
    # constant dislocation: only the time stop can exit, clock = frac x hl
    z = np.array([0.0, -3.0, -3.0, -3.0, -3.0, -3.0])
    level = np.zeros(6)
    hl = np.full(6, 2.0)
    out = stateful_exit_scan(
        z, level, entries=[2.0], exit_style="half_life_frac",
        exit_params=[1.0, 2.0], half_life=hl,
    )
    # frac=1.0: exit after 2 bars -> trade t1-t3, re-enter t4 -> 2 trades
    fast = out.filter(pl.col("exit_threshold") == 1.0).row(0, named=True)
    assert fast["n_trades"] == 2
    # frac=2.0: exit after 4 bars -> single trade t1-t5
    slow = out.filter(pl.col("exit_threshold") == 2.0).row(0, named=True)
    assert slow["n_trades"] == 1
    # longer clock holds longer per trade
    assert (
        slow["n_bars_active"] / slow["n_trades"]
        > fast["n_bars_active"] / fast["n_trades"]
    )


def test_stateful_exit_scan_profits_on_true_ou():
    rng = np.random.default_rng(7)
    n = 3000
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = resid[i - 1] * 0.95 + rng.normal(0, 1)
    z = (resid - resid.mean()) / resid.std()

    out = stateful_exit_scan(
        z, resid, entries=[1.5], exit_style="revert_frac", exit_params=[0.5, 1.0]
    )
    assert len(out) == 2
    assert (out["n_trades"] > 0).all()
    assert (out["total_pnl_bps"] > 0).all()  # fading true OU must pay


def test_stateful_exit_scan_entry_allow_restricts_opening_bars():
    z = np.array([0.0, -3.0, -3.0, 0.0, 0.0])
    level = np.array([0.0, 3.0, 2.0, 0.0, 0.0])

    blocked = stateful_exit_scan(
        z,
        level,
        entries=[2.0],
        exit_style="revert_frac",
        exit_params=[1.0],
        entry_allow=np.zeros(5, dtype=bool),
    )
    allowed = stateful_exit_scan(
        z,
        level,
        entries=[2.0],
        exit_style="revert_frac",
        exit_params=[1.0],
        entry_allow=np.array([False, True, False, False, False]),
    )

    assert blocked["n_trades"][0] == 0
    assert allowed["n_trades"][0] == 1


def test_stateful_exit_scan_validates_inputs():
    z = np.zeros(10)
    level = np.zeros(10)
    with pytest.raises(ValueError):
        stateful_exit_scan(z, level, [1.0], "nope", [0.5])
    with pytest.raises(ValueError):  # half_life_frac needs the half-life matrix
        stateful_exit_scan(z, level, [1.0], "half_life_frac", [1.0])


def test_stateful_exit_scan_combo_annotation():
    z = np.zeros((50, 2))
    z[10:, 0] = -3.0
    combos = [{"beta_lb": 42}, {"beta_lb": 63}]
    out = stateful_exit_scan(
        np.asarray(z), np.zeros(50), entries=[2.0, 2.5],
        exit_style="revert_frac", exit_params=[0.5], combos=combos,
    )
    # rows = K x entries x params, combos repeated per (entry, param)
    assert len(out) == 4
    assert set(out["beta_lb"]) == {42, 63}
    assert set(out["entry_threshold"]) == {2.0, 2.5}


# ── gate specs ───────────────────────────────────────────────────────────────

def test_parse_gate_tuple_and_dict_forms():
    assert parse_gate(("r2", "high_75")) == ("r2", "above", (0.75,))
    assert parse_gate(("beta_cv", "between", 0.25, 0.75)) == ("beta_cv", "between", (0.25, 0.75))
    assert parse_gate({"condition": "r2", "bucket": "low_10"}) == ("r2", "below", (0.10,))
    assert parse_gate({"condition": "r2", "kind": "above", "q": 0.75}) == ("r2", "above", (0.75,))
    assert parse_gate(
        {"condition": "resid_phi", "kind": "outside", "q": (0.1, 0.9)}
    ) == ("resid_phi", "outside", (0.1, 0.9))


def test_parse_gate_rejects_bad_specs():
    with pytest.raises(ValueError):
        parse_gate(("r2", "nope_99"))            # unknown named bucket
    with pytest.raises(ValueError):
        parse_gate(("r2", "between", 0.25))      # wrong quantile arity
    with pytest.raises(ValueError):
        parse_gate({"condition": "r2"})          # no bucket or kind+q
    with pytest.raises(ValueError):
        parse_gate("r2")                         # not a tuple/dict


def test_gate_allow_mask_bucket_semantics():
    # Expanding ranks for the finite values are 1, .5, .667, .5, .6, .167, 1.
    c = np.array([np.nan, 5, 1, 4, 2, 3, 0, 6], dtype=float)
    below = gate_allow_mask(c, ("c", "below", 0.5), min_history=1)
    above = gate_allow_mask(c, ("c", "above", 0.5), min_history=1)
    between = gate_allow_mask(
        c, ("c", "between", 0.25, 0.75), min_history=1
    )
    outside = gate_allow_mask(
        c, ("c", "outside", 0.25, 0.75), min_history=1
    )
    assert below.sum() == 3
    assert above.sum() == 6
    assert between.sum() == 4
    assert outside.sum() == 3
    assert not (between & outside).any()
    # NaN condition bars are never allowed
    for mask in (below, above, between, outside):
        assert not mask[0]
    # named bucket matches its explicit form
    np.testing.assert_array_equal(
        gate_allow_mask(c, ("c", "high_75"), min_history=1),
        gate_allow_mask(c, ("c", "above", 0.75), min_history=1),
    )


def test_gate_allow_mask_is_prefix_stable_and_honors_warmup():
    prefix = np.array([3.0, 1.0, 4.0, 2.0, 5.0, 0.0])
    full = np.concatenate([prefix, [1000.0, -1000.0, 7.0]])
    spec = ("c", "low_25")

    prefix_mask = gate_allow_mask(prefix, spec, min_history=3)
    full_mask = gate_allow_mask(full, spec, min_history=3)

    np.testing.assert_array_equal(full_mask[: len(prefix)], prefix_mask)
    assert not full_mask[:2].any()


def test_gate_allow_mask_accepts_polars_series():
    s = pl.Series([None, 1.0, 2.0, 3.0, 4.0])
    mask = gate_allow_mask(s, ("c", "above_50"), min_history=1)
    assert mask.dtype == bool
    assert not mask[0]
    assert mask.sum() > 0


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
