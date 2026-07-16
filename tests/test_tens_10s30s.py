"""10s vs 10s30s module: standard scriptable pattern, runnable without DB."""

import importlib

import polars as pl
import pytest

from backtest import sweep_strategy

view = importlib.import_module("book.curve.tens_10s30s")
compute = view.compute
main = view.main
pipeline = view.pipeline
synthetic_data = view.synthetic_data
TARGET = view.TARGET
FEATURES = view.FEATURES


def test_synthetic_data_shape():
    data = synthetic_data(n=900)
    assert data.columns == ["ts", "10y", "10s30s"]
    assert len(data) == 900
    assert data["10y"].equals(data["10y"].round(2))
    assert data["10s30s"].equals(data["10s30s"].round(2))


def test_compute_schema_matches_input_length():
    data = synthetic_data(n=900)
    sig = compute(data)
    assert "signal" in sig.columns
    assert {"resid", "beta", "r2"} <= set(sig.columns)
    assert len(sig) == len(data)


def test_end_to_end_synthetic():
    state = main(use_db=False)
    assert set(state) == {"raw_data", "coverage", "data", "signals", "diag", "result"}
    assert isinstance(state["diag"], pl.DataFrame)
    assert len(state["result"].closed_trades) > 0
    # residual is OU by construction -> fading it must show positive IC
    assert (state["diag"]["ic"] > 0).all()


def test_pipeline_wiring():
    assert pipeline.compute_fn is compute
    assert TARGET == "10s30s"
    assert FEATURES == ["10y"]
    assert pipeline.trade_def.legs == {TARGET: 1.0}
    assert pipeline.name == "tens_10s30s"


def test_compute_gate_param_adds_allow_column():
    data = synthetic_data(n=900)
    gated = compute(data, params={"gate": ("r2", "high_75")})
    assert "gate_allow" in gated.columns
    assert gated["gate_allow"].dtype == pl.Boolean
    assert gated["gate_allow"].sum() > 0          # some bars allowed
    assert not gated["gate_allow"].all()          # ...but not all
    assert "gate_allow" not in compute(data).columns


def test_entry_filter_z_gate_none_skips_ou_confirmation():
    bar = {"half_life": 10.0, "ou_z": 0.0}
    ungated = view._entry_filter(None, 3.0, 120.0)
    assert ungated(1, bar) and ungated(-1, bar)
    confirmed = view._entry_filter(0.5, 3.0, 120.0)
    assert not confirmed(1, bar) and not confirmed(-1, bar)
    # the quantile gate and half-life bounds still apply with z_gate=None
    assert not ungated(1, {**bar, "gate_allow": 0.0})
    assert not ungated(1, {"half_life": 500.0, "ou_z": 0.0})


def test_sweep_grid_gates_are_buildable():
    frame = compute(synthetic_data(n=900))
    p = view._params({})
    for spec in view.SWEEP_GRID["gate"]:
        if spec is None:
            continue
        cond = view._gate_condition(frame, {**p, "gate": spec})
        assert len(cond) == len(frame)


def test_exits_mode_reports_trading_stats(monkeypatch, tmp_path):
    monkeypatch.setenv("VIEWS_STORE_DIR", str(tmp_path))
    monkeypatch.setattr(view, "EXIT_ENTRY_SIGNALS", ["ou_z"])
    monkeypatch.setattr(view, "EXIT_BETA_LBS", [120])
    monkeypatch.setattr(view, "EXIT_OU_LBS", [60])
    monkeypatch.setattr(view, "EXIT_OU_Z_ENTRIES", [1.0])
    monkeypatch.setattr(view, "EXIT_OU_Z_BANDS", [0.25, 0.5])

    state = view.exits(use_db=False, device="cpu")
    results = state["results"]
    assert {
        "sharpe", "hit_rate", "total_pnl_bps", "pnl_per_trade_bps",
        "entry_threshold", "exit_threshold",
    } <= set(results.columns)
    summary = state["exit_summary"]
    assert set(summary["exit_threshold"].to_list()) == {0.25, 0.5}
    assert {"med_sharpe", "med_hit_rate", "med_pnl_per_trade_bps"} <= set(summary.columns)


def test_gates_mode_ring_fences_to_gate_setup(monkeypatch):
    monkeypatch.setattr(view, "GATE_SETUP", {"beta_lb": 120, "entry_resid_bps": 10.0})
    monkeypatch.setattr(view, "GATE_HORIZON", 15)
    state = view.gates(use_db=False)
    table = state["gates"]
    assert isinstance(table, pl.DataFrame)
    assert "(all)" in table["condition"].to_list()  # baseline row present
    # explicit horizon override beats the constant
    state2 = view.gates(use_db=False, horizon=5)
    assert isinstance(state2["gates"], pl.DataFrame)


def test_sweep_gate_specs_serialize_and_block_entries():
    data = view.model_frame(synthetic_data())
    grid = {
        "gate": [
            None,
            ("r2", "high_75"),
            {"condition": "r2", "kind": "above", "q": 0.75},
        ],
    }
    out = sweep_strategy("book.curve.tens_10s30s", data, grid, n_jobs=1)

    assert len(out) == 3
    assert "error" not in out.columns
    ungated = out.filter(pl.col("gate").is_null())
    gated = out.filter(pl.col("gate").is_not_null())
    assert len(ungated) == 1 and len(gated) == 2
    # gating only removes entries — never trades more than ungated
    assert (gated["n_trades"] <= ungated["n_trades"][0]).all()
    # tuple and dict spec for the same bucket are the same gate
    assert gated["n_trades"].n_unique() == 1
    sharpes = gated["sharpe"].to_list()
    assert sharpes[0] == pytest.approx(sharpes[1], nan_ok=True)


def test_sweep_z_gate_none_lets_gates_filter_alone():
    data = view.model_frame(synthetic_data())
    grid = {"z_gate": [None, 0.5], "gate": [None, ("r2", "high_75")]}
    out = sweep_strategy("book.curve.tens_10s30s", data, grid, n_jobs=1)

    assert "error" not in out.columns
    assert len(out) == 4
    no_z = out.filter(pl.col("z_gate").is_null() & pl.col("gate").is_null())
    with_z = out.filter((pl.col("z_gate") == 0.5) & pl.col("gate").is_null())
    # dropping the OU-z confirmation can only open up entries
    assert no_z["n_trades"][0] >= with_z["n_trades"][0] > 0
