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
    opts, selected, *_ = callbacks["_fill"]({"target": "10s30s", "features": ["10y"], "rows": [{"ts": "2020-01-02"}]}, "oil")
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


@pytest.fixture
def job_dir(tmp_path, monkeypatch):
    from research import jobs
    path = tmp_path.parent / f"{tmp_path.name}_jobs"
    monkeypatch.setattr(jobs, "JOBS", path)
    return path


def run_as_job(callbacks, name, *args, **kwargs):
    """Click Run as the page does, wait for the background job, return what the page then shows."""
    from research import jobs
    kind = "dis" if name == "_run_dislocation" else "bt"
    before = jobs.latest(kind)
    info = callbacks[name](*args, **kwargs)
    record = jobs.latest(kind)
    if record is None or (before and record["id"] == before["id"]):
        return info, None, None  # refused before a job started
    record = jobs.wait(record["id"])
    # The page's tick names the finished job; the area's own callback then draws it once.
    status = callbacks["_job_status"](1, None, None, None, None, False, False)
    ready = status[0] if kind == "dis" else status[1]
    assert ready == record["id"]
    out = (callbacks["_show_discovery_job"](ready, 5) if kind == "dis" else callbacks["_show_grid_job"](ready))
    return out[0:3]


def test_app_discovery_to_exit_inspection_callback_flow(tmp_path, monkeypatch, job_dir):
    from pathlib import Path
    from research import app as ui, artifacts
    from plotly.utils import PlotlyJSONEncoder
    monkeypatch.setattr(artifacts, "RUNS", tmp_path)
    app = ui.build_app()
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in app.callback_map.values()}
    stored = {"target": "y", "legs": {"y": 1}, "features": ["x"], "panel_id": "test",
              "weighting": "fixed", "rows": panel().with_columns(pl.col("ts").cast(pl.Utf8)).to_dicts()}
    view, status, board = run_as_job(callbacks, "_run_dislocation", 1, stored, "x", ["levels"], [30], [10], [40],
        [0.5], [5], [], [126], "ou_z", 0.7, 5, "session")
    assert isinstance(board, dict) and board["rows"]
    row = board["rows"][0]
    # Discovery rows are real backtests now, ranked on the discovery period.
    assert row["n_trades"] >= 5 and "sharpe" in row and "later_sharpe" in row
    assert board["cost_bps"] == 0.1 and board["execution_lag"] == 1
    frozen = {**row, "target": "y", "feature": "x", "panel_id": "test", "split_date": board["split_date"]}
    bt_view, _, saved = run_as_job(callbacks, "_run_backtest_grid", 1, frozen, stored, "ignored", None,
        [0.5], ["time", "half_life_frac"], [5], [0], [0.5], [15], 0.1, [2], 1, "session")
    # Trades/equity/periods live on disk, not in the Store. The vectorised
    # grid saves exact Engine detail for its best cell; inspecting another
    # cell reruns it through Engine, checks parity and adds it to the run.
    # The page keeps where the grid is, not its rows (a big grid's rows froze the browser).
    assert isinstance(saved, dict) and "rows" not in saved and saved["cells"] == 2
    grid_rows = pl.read_parquet(Path(saved["run_path"]) / "results.parquet").to_dicts()
    run_path = Path(saved["run_path"])
    best = str(max(grid_rows, key=lambda r: r["earlier_sharpe"])["config_id"])
    other = ({"1", "2"} - {best}).pop()
    assert set(pl.read_parquet(run_path / "equity.parquet")["config_id"]) == {best}
    assert (run_path / "yearly.parquet").is_file()
    detail = callbacks["_inspect"](other, saved)
    assert "matches the vectorised grid" in json.dumps(detail, cls=PlotlyJSONEncoder)
    assert set(pl.read_parquet(run_path / "equity.parquet")["config_id"]) == {"1", "2"}
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
    _, _, archived_exits = run_as_job(callbacks, '_run_backtest_grid', 1, historical_candidate, None, 'ignored', None,
        [.5], ['time'], [5], [0], [.5], [15], .1, [2], 1, 'session')
    assert archived_exits['cells'] == 1
    json.dumps([summary, opened, opened_status], cls=PlotlyJSONEncoder)


def test_backtest_discovery_validation_pick_and_legacy_ic_runs(tmp_path, monkeypatch, job_dir):
    from pathlib import Path
    from research import app as ui, artifacts
    from research.saved_runs import grid_spec
    from plotly.utils import PlotlyJSONEncoder
    monkeypatch.setattr(artifacts, "RUNS", tmp_path)
    app = ui.build_app()
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in app.callback_map.values()}
    stored = {"target": "y", "legs": {"y": 1}, "features": ["x"], "panel_id": "test",
              "weighting": "fixed", "rows": panel().with_columns(pl.col("ts").cast(pl.Utf8)).to_dicts()}
    args = (1, stored, "x", ["levels"], [30], [10], [40], [0.5], [5, 10], [], [126], "ou_z", 0.7, 5, "session", None)

    refused = callbacks["_run_dislocation"](*args, 0.25, 1, 0, "cv_mean", 0)
    assert "needs cross-validation blocks" in json.dumps(refused, cls=PlotlyJSONEncoder)

    view, _, board = run_as_job(callbacks, "_run_dislocation", *args, 0.25, 0, 3, "cv_mean", 2)
    text = json.dumps(view, cls=PlotlyJSONEncoder)
    assert "SELECTION CHECKS" in text and "PLACEBO" in text
    meta = json.loads((Path(board["run_path"]) / "metadata.json").read_text(encoding="utf-8"))
    assert meta["scoring"] == "backtest" and meta["cv_folds"] == 3 and len(meta["placebo"]) == 2
    assert meta["selection_checks"] and meta["grid"]["cost_bps"] == 0.25

    untouched = callbacks["_pick_dislocation_candidate"]([0] * len(board["rows"]), board, [0.5], *[None] * 8)
    assert untouched == (no_update_marker(),) * (5 + len(ui.MECHANICS_EXITS))
    candidate, label, entries, *exits, cost, lag = ui.freeze_candidate(
        board, 1, [0.5], {"bt-time-stops": [20], "bt-exit-styles": ["band"], "bt-stop": [15.0]})
    exits = dict(zip(ui.MECHANICS_EXITS, exits))
    row = board["rows"][1]
    assert candidate["beta_lb"] == row["beta_lb"] and candidate["gate"] == row["gate"]
    # the row's own exit is added to what was selected, so the grid contains the discovery trade
    assert exits["bt-exit-styles"] == ["band", row["exit_style"]]
    assert exits["bt-time-stops"] == sorted({20, int(row["exit_param"])})
    assert exits["bt-stop"] == [0.0, 15.0] and exits["bt-caps"] == [0.0] and exits["bt-signal-stops"] == [0.0]
    assert cost == 0.25 and lag == 0

    run_id = Path(board["run_path"]).name
    reopened, _, archived = callbacks["_open_saved"](1, run_id, 5, "session")
    assert "SELECTION CHECKS" in json.dumps(reopened, cls=PlotlyJSONEncoder)
    assert archived["rows"] == board["rows"] and archived["cost_bps"] == 0.25

    # Every rankable column saved its own board; a header click on one shows it, best first.
    assert ui.board_rules(run_id) == set(ui.RANK_RULES)
    resorted_view, resorted = callbacks["_sort_board"]({"rule": "pnl", "at": 1}, archived, 5, None)
    pnls = [r["pnl_bps"] for r in resorted["rows"]]
    assert pnls == sorted(pnls, reverse=True) and resorted["run_path"] == archived["run_path"]
    text = json.dumps(resorted_view, cls=PlotlyJSONEncoder)
    assert '"data-board-rule": "pnl"' in text and '"data-sort-dir": "desc"' in text
    assert "PLACEBO" in text  # the placebo test still compares the selection rule's top score
    # A candidate frozen from this run stays frozen when its board is re-sorted.
    frozen_here = ui.freeze_candidate(archived, 0)[0]
    assert callbacks["_invalidate_candidate_for_board"](resorted, None, frozen_here) == (no_update_marker(),) * 2

    # A discovery run saved before backtest scoring still opens, on the IC board.
    frame, results = dislocation_scan(panel(), target="y", feature="x", beta_lookbacks=[30], residual_lookbacks=[10],
                                      normalization_lookbacks=[40], thresholds=[0.5], horizons=[5], gate_names=[],
                                      fit_on=["levels"], signal_kind="ou_z", train_fraction=0.7, device="cpu")
    legacy = artifacts.save_run("discovery", frame, results, dict(
        feature="x", target="y", legs={"y": 1}, weighting="fixed", weight_columns={}, train_fraction=0.7,
        grid=grid_spec(["levels"], [30], [10], [40], [0.5], [5], [], [126], "ou_z", 0.7)))
    opened, status, legacy_board = callbacks["_open_saved"](1, Path(legacy).name, 5, "session")
    assert "legacy IC board" in json.dumps(opened, cls=PlotlyJSONEncoder)
    assert legacy_board["rows"] and "ic" in legacy_board["rows"][0]


def no_update_marker():
    from dash import no_update
    return no_update


def test_pnl_heatmap_sums_months_colours_by_sign_and_blanks_missing_months():
    from research.app import pnl_heatmap
    days = [date(2020, 11, 2) + timedelta(days=i) for i in range(90)]  # Nov 2020 .. Jan 2021
    pnl = [1.0 if d.month == 11 else -2.0 if d.month == 12 else 0.5 for d in days]
    equity = pl.DataFrame({"ts": days, "pnl_bps": pnl})
    periods = pl.DataFrame({"year": [2020, 2021], "pnl_bps": [29.0 - 62.0, 15.0], "active_days": [60, 30]})
    view = pnl_heatmap(equity, periods).to_plotly_json()
    table_el = view["props"]["children"][1].to_plotly_json()
    header = [th.children[0] if isinstance(th.children, list) else th.children
              for th in table_el["props"]["children"][0].children.children]
    assert header[:2] == ["year", "Jan"] and header[-2:] == ["full year", "active days"]
    rows = table_el["props"]["children"][1].children
    cells = {r.children[0].children: {header[i]: c for i, c in enumerate(r.children)} for r in rows}
    assert cells["2020"]["Nov"].children == "29.0" and cells["2020"]["Dec"].children == "-62.0"
    assert cells["2020"]["Jan"].children == "" and "background" not in cells["2020"]["Jan"].style
    assert cells["2021"]["Jan"].children == "15.0"
    blue, red = cells["2020"]["Nov"].style["background"], cells["2020"]["Dec"].style["background"]
    assert blue.startswith("rgb(") and red.startswith("rgb(")
    r, g, b = map(int, blue[4:-1].split(","))
    assert b > r  # gains lean blue
    r, g, b = map(int, red[4:-1].split(","))
    assert r > b  # losses lean red; the largest loss is full strength with white ink
    assert cells["2020"]["Dec"].style["color"] == "#FFFFFF"
    assert cells["2020"]["full year"].style["fontWeight"] == "bold"


def test_trade_mechanics_select_all_fills_entries_exits_and_stops_only():
    from research import app as ui
    app = ui.build_app()
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in app.callback_map.values()}
    options = [[{"label": str(v), "value": v} for v in values] for values in
               ([0.5, 1.0], ["time", "band"], [5, 10], [0.0], [0.5], [1.0, 2.0], [0.0, 15.0])]
    assert callbacks["_select_all_mechanics"](1, *options) == [
        [0.5, 1.0], ["time", "band"], [5, 10], [0.0], [0.5], [1.0, 2.0], [0.0, 15.0]]
    outputs = next(k for k, v in app.callback_map.items() if v["callback"].__wrapped__.__name__ == "_select_all_mechanics")
    assert "bt-cost" not in outputs and "bt-lag" not in outputs


def test_single_series_targets_load_with_fixed_weighting_and_disable_beta(monkeypatch):
    from research import app as ui
    from plotly.utils import PlotlyJSONEncoder
    app = ui.build_app()
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in app.callback_map.values()}
    options, value = callbacks["_beta_only_for_packages"]("10y", "beta")  # an outright: nothing to hedge
    assert value == "fixed" and next(o for o in options if o["value"] == "beta")["disabled"] is True
    options, value = callbacks["_beta_only_for_packages"]("swsp10", "beta")  # swap vs Treasury legs
    beta = next(o for o in options if o["value"] == "beta")
    assert value is no_update_marker() and not beta["disabled"] and "sofr10" in beta["label"]
    options, value = callbacks["_beta_only_for_packages"]("10s30s", "beta")
    assert value is no_update_marker() and not next(o for o in options if o["value"] == "beta")["disabled"]

    seen = {}
    def fake_panel(trade, features, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after the weighting decision")
    monkeypatch.setattr(ui, "build_panel", fake_panel)
    view, _, _ = callbacks["_load"](1, 0, "10y", None, "beta", 126, None, ["2y"], "2000-01-01", [], None, "6M", "s")
    assert seen["weighting"] == "fixed"
    trades = []
    monkeypatch.setattr(ui, "build_panel", lambda trade, features, **kw: trades.append((trade, kw)) or (_ for _ in ()).throw(RuntimeError("stop")))
    callbacks["_load"](1, 0, "swsp10", None, "beta", 126, None, ["10y"], "2000-01-01", [], None, "6M", "s")
    assert trades[0][0].legs == {"sofr10": 1.0, "10y": -1.0} and trades[0][1]["weighting"] == "beta"
