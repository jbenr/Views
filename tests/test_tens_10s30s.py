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


def test_gate_conditions_are_buildable():
    frame = compute(synthetic_data(n=900))
    p = view._params({})
    names = [
        "r2", "beta_cv", "beta", "beta_vol20", "beta_mom10",
        "r2_vol20", "r2_mom10", "resid_phi", "resid_half_life",
        "resid_vol20", "resid_mom10",
    ]
    for name in names:
        cond = view._gate_condition(frame, {**p, "gate": (name, "high_75")})
        assert len(cond) == len(frame)


def test_sweep_grids_build_one_grid_per_saved_winner(monkeypatch, tmp_path):
    exits = pl.DataFrame({
        "setup": ["ou120/60/e1 h20 r2:tails_10_90", "res100/e21 h40"],
        "entry_signal": ["ou_z", "residual"],
        "beta_lb": [120, 100],
        "ou_lb": [60, None],
        "entry_threshold": [1.0, 21.0],
        "predict_horizon": [20, 40],
        "gate": ["r2", "(none)"],
        "gate_bucket": ["tails_10_90", "all"],
        "exit_style": ["revert_frac", "band"],
        "exit_threshold": [0.5, 5.0],
    })
    path = tmp_path / "exits.parquet"
    exits.write_parquet(path)
    monkeypatch.setattr(view, "EXITS_FILE", path)

    grids = view._sweep_grids()
    assert len(grids) == 2
    ou, res = grids
    assert ou["entry_signal"] == ["ou_z"] and ou["ou_lb"] == [60]
    assert ou["gate"] == [("r2", "tails_10_90")]
    assert ou["exit_style"] == ["revert_frac"] and ou["exit_param"] == [0.5]
    assert res["gate"] == [None] and "ou_lb" not in res
    for grid in grids:
        assert grid["stop_loss_bps"] == view.SWEEP_STOP_LOSS_BPS
        assert grid["z_gate"] == [None]
        assert grid["half_life_min"] == [None]
        assert grid["half_life_max"] == [None]


def test_sweep_mode_saves_results_and_trade_log(monkeypatch, tmp_path):
    exits = pl.DataFrame([{
        "setup": "res100/e15 h40",
        "entry_signal": "residual",
        "beta_lb": 100,
        "ou_lb": None,
        "entry_threshold": 15.0,
        "predict_horizon": 40,
        "gate": "(none)",
        "gate_bucket": "all",
        "exit_style": "revert_frac",
        "exit_threshold": 0.5,
    }])
    exits_path = tmp_path / "exits.parquet"
    exits.write_parquet(exits_path)
    monkeypatch.setattr(view, "EXITS_FILE", exits_path)
    monkeypatch.setattr(view, "SWEEP_RESULTS_FILE", tmp_path / "sweep.parquet")
    monkeypatch.setattr(view, "TRADES_FILE", tmp_path / "trades.parquet")
    monkeypatch.setattr(view, "SWEEP_STOP_LOSS_BPS", [25.0])

    state = view.sweep(use_db=False, n_jobs=1)

    assert (tmp_path / "sweep.parquet").exists()
    trades = pl.read_parquet(tmp_path / "trades.parquet")
    assert trades.equals(state["trades"])
    assert len(trades) > 0
    assert {
        "setup", "entry_date", "exit_date", "direction", "entry_level",
        "exit_level", "pnl_bps", "bars_held", "exit_reason", "stop_loss_bps",
    } <= set(trades.columns)
    assert set(trades["setup"]) == {"res100/e15 h40"}
    assert set(trades["direction"]) <= {"long", "short"}
    # trade log comes from the same engine the sweep scored: counts must agree
    sweep_row = pl.read_parquet(tmp_path / "sweep.parquet").row(0, named=True)
    assert len(trades) == sweep_row["n_trades"]

    robustness = state["robustness"]
    assert {
        "setup", "pnl_ex_best", "best_trade_share", "median_trade_bps",
        "sharpe", "sharpe_ex_best", "eras_pos",
    } <= set(robustness.columns)
    assert len(robustness) == 1


def test_robustness_flags_one_trade_wonders():
    data = view.model_frame(synthetic_data(n=3000))
    dates = data["ts"].to_list()

    def trade(setup, i0, i1, pnl):
        return {
            "setup": setup, "entry_date": dates[i0], "exit_date": dates[i1],
            "direction": "long", "pnl_bps": pnl,
        }

    # "lucky": one 50bps trade carries it; "steady": same total in 4 trades
    # spread across the sample
    trades = pl.DataFrame([
        trade("lucky", 100, 110, 50.0),
        trade("lucky", 300, 310, 1.0),
        trade("lucky", 500, 510, 1.0),
        trade("steady", 200, 260, 13.0),
        trade("steady", 900, 960, 13.0),
        trade("steady", 1700, 1760, 13.0),
        trade("steady", 2500, 2560, 13.0),
    ])
    rob = view._robustness(trades, data)

    lucky = rob.filter(pl.col("setup") == "lucky").row(0, named=True)
    steady = rob.filter(pl.col("setup") == "steady").row(0, named=True)
    assert lucky["best_trade_share"] == pytest.approx(0.96, abs=0.01)
    assert lucky["pnl_ex_best"] == pytest.approx(2.0)
    assert steady["best_trade_share"] == pytest.approx(0.25)
    assert steady["pnl_ex_best"] == pytest.approx(39.0)
    # ~11.5y of weekdays (2010 -> mid-2021) -> 3 four-year eras
    assert steady["eras_pos"] == "3/3"
    assert lucky["eras_pos"] == "1/3"
    # ranked by sharpe_ex_best, descending
    ex = rob["sharpe_ex_best"].to_list()
    assert ex == sorted(ex, reverse=True)


def test_sweep_half_life_bounds_none_disable_filter():
    data = view.model_frame(synthetic_data())
    bounded = sweep_strategy(
        "book.curve.tens_10s30s",
        data,
        {"half_life_min": [3.0], "half_life_max": [120.0]},
        n_jobs=1,
    )
    unbounded = sweep_strategy(
        "book.curve.tens_10s30s",
        data,
        {"half_life_min": [None], "half_life_max": [None]},
        n_jobs=1,
    )
    assert "error" not in bounded.columns and "error" not in unbounded.columns
    # dropping the half-life sanity bounds can only open up entries
    assert unbounded["n_trades"][0] >= bounded["n_trades"][0] > 0


def test_predict_setups_select_save_load_roundtrip(tmp_path):
    # a cluster of adjacent grid cells (+-1 step in beta_lb/ou_lb/threshold)
    # and one lone high-IC spike with no neighbors
    cluster = [
        {"beta_lb": 330, "ou_lb": 250, "entry_threshold": 2.7, "ic": 0.80},
        {"beta_lb": 320, "ou_lb": 250, "entry_threshold": 2.7, "ic": 0.60},
        {"beta_lb": 340, "ou_lb": 250, "entry_threshold": 2.7, "ic": 0.79},
        {"beta_lb": 330, "ou_lb": 240, "entry_threshold": 2.7, "ic": 0.77},
        {"beta_lb": 330, "ou_lb": 250, "entry_threshold": 2.5, "ic": 0.76},
    ]
    spike = [{"beta_lb": 100, "ou_lb": 100, "entry_threshold": 1.5, "ic": 0.95}]
    valid = pl.DataFrame([
        {
            "entry_signal": "ou_z", "horizon": 40,
            "gate": "r2_mom10", "gate_bucket": "high_75",
            "hit_rate": 0.6, "fire_rate": 0.03, "n_obs": 35, **row,
        }
        for row in cluster + spike
    ])
    setups = view._select_setups(valid)

    # the lone 0.95-IC spike has no corroborating neighbors -> dropped
    assert "ou100/100/e1.5 h40 r2_mom10:high_75" not in setups["name"].to_list()
    # ranked by neighborhood IC, one row per (signal, lookbacks, gate) cell
    assert len(setups) == 4
    assert setups["name"][0] == "ou330/250/e2.5 h40 r2_mom10:high_75"
    assert setups["nbr_ic"][0] == pytest.approx(0.78)

    path = tmp_path / "setups.parquet"
    setups.write_parquet(path)
    loaded = view.load_setups(path)
    assert loaded[0]["gate"] == ("r2_mom10", "high_75")
    assert loaded[0]["beta_lb"] == 330 and loaded[0]["ou_lb"] == 250
    assert loaded[0]["entry_threshold"] == 2.5


def test_load_setups_and_exits_require_prior_modes(tmp_path):
    with pytest.raises(FileNotFoundError, match="--predict"):
        view.load_setups(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError, match="--exit"):
        view.load_exits(tmp_path / "nope.parquet")


def test_compute_ou_z_entry_signal():
    data = synthetic_data(n=900)
    resid_frame = compute(data)
    z_frame = compute(data, params={"entry_signal": "ou_z"})
    assert resid_frame["signal"].equals(resid_frame["resid"])
    assert z_frame["signal"].equals(z_frame["ou_z"])
    with pytest.raises(ValueError, match="entry_signal"):
        compute(data, params={"entry_signal": "nope"})


def test_predict_signal_names_normalize_ou_and_reject_unknown():
    assert view._normalize_predict_signals(["residual", "ou"]) == [
        "residual",
        "ou_z",
    ]
    assert view._normalize_predict_signals(["ou", "ou_z"]) == ["ou_z"]
    with pytest.raises(ValueError, match="PREDICT_ENTRY_SIGNALS"):
        view._normalize_predict_signals(["residual", "nope"])


def test_sweep_engine_runs_every_signal_and_exit_style():
    data = view.model_frame(synthetic_data())
    cases = [
        ("residual", 20.0, "band", 5.0),
        ("residual", 20.0, "revert_frac", 0.5),
        ("residual", 20.0, "half_life_frac", 1.0),
        ("ou_z", 1.0, "band", 0.25),
        ("ou_z", 1.0, "revert_frac", 0.5),
        ("ou_z", 1.0, "half_life_frac", 1.0),
    ]
    rows = []
    for sig, thr, style, param in cases:
        out = sweep_strategy(
            "book.curve.tens_10s30s",
            data,
            {
                "entry_signal": [sig],
                "entry_threshold": [thr],
                "exit_style": [style],
                "exit_param": [param],
            },
            n_jobs=1,
        )
        assert "error" not in out.columns, f"{sig}/{style} errored"
        rows.append(out)
    combined = pl.concat(rows, how="diagonal_relaxed")
    assert (combined["n_trades"] > 0).all()
    assert combined["sharpe"].is_finite().all()


def test_exit_mode_reports_exit_stats(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(view, "EXIT_RESULTS_FILE", tmp_path / "exit_results.parquet")
    setups_path = tmp_path / "setups.parquet"
    pl.DataFrame([{
        "name": "test_ou_setup",
        "entry_signal": "ou_z",
        "beta_lb": 120,
        "ou_lb": 60,
        "entry_threshold": 1.0,
        "predict_horizon": 20,
        "gate": "r2",
        "gate_bucket": "tails_10_90",
    }]).write_parquet(setups_path)
    monkeypatch.setattr(view, "SETUPS_FILE", setups_path)
    monkeypatch.setattr(view, "EXITS_FILE", tmp_path / "exits.parquet")
    monkeypatch.setattr(view, "EXIT_OU_Z_BANDS", [0.25, 0.5])
    monkeypatch.setattr(view, "EXIT_REVERT_FRACS", [0.5])
    monkeypatch.setattr(view, "EXIT_HALF_LIFE_FRACS", [1.0])

    state = view.exit_scan(use_db=False, device="cpu")
    results = state["results"]
    assert {
        "sharpe", "hit_rate", "total_pnl_bps", "pnl_per_trade_bps",
        "setup", "avg_hold_bars", "entry_threshold", "predict_horizon", "gate",
        "gate_bucket", "exit_style", "exit_threshold",
    } <= set(results.columns)
    assert set(results["gate"]) == {"r2"}
    assert set(results["gate_bucket"]) == {"tails_10_90"}
    assert set(results["setup"]) == {"test_ou_setup"}
    assert set(results["exit_style"]) == {"band", "revert_frac", "half_life_frac"}
    # 1 combo x 1 entry x (2 bands + 1 revert frac + 1 hl frac)
    assert len(results) == 4

    summary = state["exit_summary"]
    bands = summary.filter(pl.col("exit_style") == "band")
    assert set(bands["exit_threshold"].to_list()) == {0.25, 0.5}
    assert {
        "med_sharpe", "med_hit_rate", "med_pnl_per_trade_bps", "med_hold_bars",
    } <= set(summary.columns)

    # full results overwrite the per-mode results file
    assert (tmp_path / "exit_results.parquet").exists()

    # best exit per setup is saved for --sweep
    winners = pl.read_parquet(tmp_path / "exits.parquet")
    assert len(winners) == 1
    assert winners["setup"][0] == "test_ou_setup"
    assert winners["exit_style"][0] in {"band", "revert_frac", "half_life_frac"}
    assert winners.row(0, named=True) == state["winners"].row(0, named=True)

    output = capsys.readouterr().out.lower()
    assert "which exits pay" not in output
    assert "sharpe by entry" not in output
    assert "pnl/trd" in output
    assert "threshold_units" not in output
    assert "pnl_per_trade_bps" not in output


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
