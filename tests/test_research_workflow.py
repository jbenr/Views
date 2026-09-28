"""Research handoff, discovery/engine parity, and complete exit comparisons."""

import json
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from backtest.engine import BacktestConfig, Engine, TradeDef
from research.dislocation import dislocation_scan
from research.dislocation_backtest import make_pipeline, run_grid, signal_frame


def panel(n=420):
    rng = np.random.default_rng(42)
    x = np.cumsum(rng.normal(size=n))
    residual = np.zeros(n)
    for i in range(1, n):
        residual[i] = 0.8 * residual[i-1] + rng.normal()
    return pl.DataFrame({"ts": [date(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                         "x": x, "y": 0.6*x + residual})


def candidate(**kwargs):
    return dict(target="y", feature="x", fit_on="levels", beta_lb=30,
                residual_lb=None, norm_lb=40, gate="(none)", signal_kind="ou_z", **kwargs)


@pytest.mark.parametrize("kind", ["normalized", "ou_z", "raw"])
@pytest.mark.parametrize("basis", ["levels", "changes"])
def test_discovery_and_execution_use_identical_signals(monkeypatch, kind, basis):
    import research.dislocation as module
    original = module.predict_scan
    captured = []
    def capture(z, *args, **kwargs):
        captured.append(z.copy())
        return original(z, *args, **kwargs)
    monkeypatch.setattr(module, "predict_scan", capture)
    data = panel()
    dislocation_scan(data, target="y", feature="x", beta_lookbacks=[30],
        residual_lookbacks=[10], normalization_lookbacks=[40], thresholds=[0.5],
        horizons=[5], gate_names=[], fit_on=[basis], signal_kind=kind, device="cpu")
    state = signal_frame(data, target="y", feature="x", fit_on=basis, beta_lb=30,
                         residual_lb=10 if basis == "changes" else None,
                         norm_lb=40, signal_kind=kind)
    np.testing.assert_allclose(captured[0][:, 0], state["signal"].to_numpy(), equal_nan=True)


def test_short_gate_window_is_rejected_before_regression(monkeypatch):
    import research.dislocation as module

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Invalid gate settings must be rejected before fitting")

    monkeypatch.setattr(module, "roll_lr", unexpected_fit)
    with pytest.raises(ValueError, match="Choose lookbacks >= 126"):
        dislocation_scan(panel(), target="y", feature="x", beta_lookbacks=[30],
            residual_lookbacks=[10], normalization_lookbacks=[40], thresholds=[0.5],
            horizons=[5], gate_names=["resid_phi"], gate_windows=[63, 126],
            fit_on=["levels"], device="cpu")


def test_failed_progress_does_not_claim_completion():
    from research.app import progress_view

    view = progress_view(dict(message="Failed: invalid settings", done=1, total=1,
                             elapsed=30, history=["Failed: invalid settings"]), "SCANNING")
    assert view.children[1].children == "Stopped · 0:30 elapsed"
    assert all(getattr(child, "role", None) != "progressbar" for child in view.children)


def test_progress_log_retains_bounded_scroll_history():
    from research import progress
    from research.app import progress_view
    progress.start('scroll-test', 'dis', 'Starting')
    for i in range(205):
        progress.update('scroll-test', 'dis', f'Model {i}', i, 300)
    state = progress.snapshot('scroll-test', 'dis')
    assert len(state['history']) == 200
    assert state['history'][0] == 'Model 5'
    view = progress_view(state, 'SEARCHING')
    log = next(child for child in view.children if getattr(child, 'className', '') == 'research-work-log')
    assert len(log.children) == 200
    assert log.children[0].children == 'Model 5'
    assert log.children[-1].children == 'Model 204'
    assert log.tabIndex == 0
    progress.start('scroll-test', 'dis', 'New run')
    assert progress.snapshot('scroll-test', 'dis')['history'] == ['New run']


def test_chart_inversion_redraws_cached_data_without_reload(monkeypatch):
    from types import SimpleNamespace
    from copy import deepcopy
    from research import app as ui
    app = ui.build_app()
    callback = next(v['callback'].__wrapped__ for v in app.callback_map.values()
                    if v['callback'].__wrapped__.__name__ == '_resize_level')
    stored = dict(target='y', features=['x'], invert_features=False,
                  weight_cols=[], rows=panel(10).with_columns(pl.col('ts').cast(pl.Utf8)).to_dicts())
    original = deepcopy(stored)
    renders = []
    monkeypatch.setattr(ui, 'level_chart', lambda data, target, **kwargs: renders.append(kwargs) or 'png')
    def unexpected_load(*args, **kwargs):
        pytest.fail('Inversion must not reload the panel')
    monkeypatch.setattr(ui, 'build_panel', unexpected_load)
    monkeypatch.setattr(ui, 'ctx', SimpleNamespace(triggered_id='invert-feature'))
    result = callback(*([0]*len(ui.WINDOW_PRESETS)), ['invert'], stored, '6M')
    assert result[0] == 'data:image/png;base64,png'
    assert result[1] is ui.no_update
    assert result[2] == '6M'
    assert renders[-1]['invert_features'] is True
    # Changing chart windows must use the live checkbox, not the load-time flag.
    monkeypatch.setattr(ui, 'ctx', SimpleNamespace(triggered_id='research-level-window-1M'))
    result = callback(*([1]*len(ui.WINDOW_PRESETS)), ['invert'], stored, '6M')
    assert result[2] == '1M'
    assert renders[-1]['invert_features'] is True
    monkeypatch.setattr(ui, 'ctx', SimpleNamespace(triggered_id='invert-feature'))
    callback(*([0]*len(ui.WINDOW_PRESETS)), [], stored, '1M')
    assert renders[-1]['invert_features'] is False
    assert stored == original
    result = callback(*([0]*len(ui.WINDOW_PRESETS)), ['invert'], None, '6M')
    assert result[0] is ui.no_update


def test_discovery_training_scores_cannot_see_later_outcomes():
    data = panel()
    kwargs = dict(target="y", feature="x", beta_lookbacks=[30], residual_lookbacks=[10],
        normalization_lookbacks=[40], thresholds=[0.5], horizons=[5, 20], gate_names=[],
        fit_on=["levels"], train_fraction=0.7, device="cpu")
    _, original = dislocation_scan(data, **kwargs)
    changed = data.with_row_index().with_columns(
        pl.when(pl.col("index") >= int(len(data)*0.7)).then(pl.col("y")*100).otherwise(pl.col("y")).alias("y"))
    _, rerun = dislocation_scan(changed, **kwargs)
    assert original.filter(pl.col("sample") == "train").equals(rerun.filter(pl.col("sample") == "train"))
    assert set(original["sample"]) == {"train", "test"}


def test_all_exits_and_stops_retain_closed_trade_win_rates_and_logs():
    progress = []
    grid, selected = run_grid(panel(), TradeDef.outright("y", "y"), candidate(split_date="2020-10-01"),
        entry_zs=[0.5], exit_params={"time": [5], "band": [0, 0.25],
        "revert_frac": [0.5], "half_life_frac": [1, 2]}, stop_losses=[15, 25],
        execution_lag=1, round_trip_cost_bps=0.1,
        progress=lambda done, total: progress.append((done, total)))
    assert len(grid) == 12
    assert len(selected["runs"]) == 12
    assert progress[-1] == (12, 12)
    for row in grid.iter_rows(named=True):
        trades = selected["runs"][row["config_id"]]["trades"]
        assert trades
        assert row["trade_win_rate"] == pytest.approx(sum(t["pnl_bps"] > 0 for t in trades)/len(trades))
        assert row["avg_pnl_per_trade_bps"] == pytest.approx(np.mean([t["pnl_bps"] for t in trades]))
        assert "later_sharpe" in row
        assert all(t["mae_bps"] <= 0 <= t["mfe_bps"] for t in trades)
        assert "hit_rate" not in row


def test_entry_half_life_is_frozen_and_execution_can_be_delayed():
    data = panel(32)
    state = pl.DataFrame({"signal": [0.0, -2.0] + [-1.0]*30,
        "resid": [0.0]*32, "beta": [1.0]*32, "r2": [0.5]*32,
        "half_life": [10.5, 10.5] + [1.0]*30})
    pipe = make_pipeline(data, TradeDef.outright("y", "y"), candidate(), entry_z=1.5,
        exit_style="half_life_frac", exit_param=2, stop_loss_bps=None, state=state, execution_lag=1)
    result = Engine(BacktestConfig()).add_signal(pipe).run(data)
    trade = result.closed_trades[0]
    assert trade.entry_date == data["ts"][2]
    assert trade.bars_held == 21
    assert trade.exit_reason == "time_stop"


def test_beta_weights_are_held_at_entry_not_remarked_daily():
    data = pl.DataFrame({"ts": [date(2020, 1, 1)+timedelta(days=i) for i in range(8)],
        "left": [100.0]*8, "right": [50.0]*8, "y": [0.0]*8,
        "wl": [1.0]*8, "wr": [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]})
    state = pl.DataFrame({"signal": [0.0, -2.0]+[-1.0]*6, "resid": [0.0]*8,
        "beta": [1.0]*8, "r2": [0.5]*8, "half_life": [1.0]*8})
    trade = TradeDef("y", {"left": 1, "right": -1})
    pipe = make_pipeline(data, trade, candidate(weight_columns={"left": "wl", "right": "wr"}),
        entry_z=1.5, exit_style="time", exit_param=3, stop_loss_bps=None, state=state)
    result = Engine().add_signal(pipe).run(data)
    assert result.closed_trades[0].trade_def.legs == {"left": 1, "right": -2}
    assert result.closed_trades[0].pnl_bps == 0
    assert result.daily_pnl["pnl_bps"].sum() == 0


def test_app_controls_fill_from_loaded_panel_and_hide_irrelevant_inputs():
    from research.app import build_app
    app = build_app()
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in app.callback_map.values()}
    assert callbacks["_weight_controls"]("10s30s", "fixed") == ({"display": "none"},)*3
    assert callbacks["_weight_controls"]("custom", "beta")[0].get("display") != "none"
    opts, selected, *_ = callbacks["_fill"]({"target": "10s30s", "features": ["10y"], "rows": [{}]}, "oil")
    assert selected == "10y"
    assert [o["value"] for o in opts] == ["10y"]
    client = app.server.test_client()
    assert client.get("/_dash-layout").status_code == 200
    assert client.get("/_dash-dependencies").status_code == 200


def test_artifacts_save_every_configuration_without_overwriting(tmp_path, monkeypatch):
    from pathlib import Path
    from research import artifacts
    monkeypatch.setattr(artifacts, "RUNS", tmp_path)
    grid, selected = run_grid(panel(), TradeDef.outright("y", "y"), candidate(),
                             entry_zs=[0.5], exit_params={"time": [5, 10]})
    first = Path(artifacts.save_run("exits", panel(), grid, candidate(), selected["runs"]))
    second = Path(artifacts.save_run("exits", panel(), grid, candidate(), selected["runs"]))
    assert first != second
    assert set(pl.read_parquet(first / "trades.parquet")["config_id"]) == {"1", "2"}
    assert json.loads((first / "metadata.json").read_text())["target"] == "y"


def test_app_discovery_to_exit_inspection_callback_flow(tmp_path, monkeypatch):
    from pathlib import Path
    from research import app as ui, artifacts
    from plotly.utils import PlotlyJSONEncoder
    monkeypatch.setattr(artifacts, "RUNS", tmp_path)
    app = ui.build_app()
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in app.callback_map.values()}
    stored = {"target": "y", "legs": {"y": 1}, "features": ["x"], "panel_id": "test",
              "weighting": "fixed", "rows": panel().with_columns(pl.col("ts").cast(pl.Utf8)).to_dicts()}
    view, status, board = callbacks["_run_dislocation"](1, stored, "x", ["levels"], [30], [10], [40],
        [0.5], [5], [], [126], "ou_z", 0.7, 5, "session")
    assert isinstance(board, dict) and board["rows"]
    row = board["rows"][0]
    assert row["n_non_overlapping"] <= row["n_obs"]
    frozen = {**row, "target": "y", "feature": "x", "panel_id": "test", "split_date": board["split_date"]}
    bt_view, _, saved = callbacks["_run_backtest_grid"](1, frozen, stored, "ignored", None,
        [0.5], ["time", "half_life_frac"], [5], [0], [0.5], [15], 0.1, [2], 1, "session")
    # Every config's trades/equity/periods live on disk (see save_run), not
    # duplicated into the Store -- that duplication was what made this
    # callback's response balloon and appear to hang on a big grid.
    assert isinstance(saved, dict) and "runs" not in saved and len(saved["rows"]) == 2
    assert set(pl.read_parquet(Path(saved["run_path"]) / "trades.parquet")["config_id"]) == {"1", "2"}
    detail = callbacks["_inspect"]("1", saved)
    # All callback results must survive Dash's JSON serialization.
    json.dumps([view, status, bt_view, detail, saved], cls=PlotlyJSONEncoder)
    assert len(list(tmp_path.iterdir())) == 2
    options, run_id, query = callbacks['_find_saved'](stored, 'x', ['levels'], [30], [10], [40],
        [.5], [5], [], [126], 'ou_z', .7, board, None)
    assert options and run_id
    summary = callbacks['_saved_summary'](run_id, query)
    opened, opened_status, archived = callbacks['_open_saved'](1, run_id, 5, 'session')
    assert archived['rows'] and archived['archive_id'] == run_id
    assert archived['panel_id'] != stored['panel_id']
    assert archived['rows'] == board['rows']
    historical_candidate = dict(frozen, archive_id=run_id, panel_id=archived['panel_id'])
    _, _, archived_exits = callbacks['_run_backtest_grid'](1, historical_candidate, None, 'ignored', None,
        [.5], ['time'], [5], [0], [.5], [15], .1, [2], 1, 'session')
    assert len(archived_exits['rows']) == 1
    json.dumps([summary, opened, opened_status], cls=PlotlyJSONEncoder)
