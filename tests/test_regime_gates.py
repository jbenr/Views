"""Macro regimes as discovery gates: which gates exist, what they allow, and Engine parity."""

import numpy as np
import polars as pl
import pytest

from backtest.engine import TradeDef
from research import regimes as rg
from research.dislocation import backtest_scan
from research.dislocation_backtest import _entry_gate, run_grid, signal_frame
from test_backtest_discovery import SCAN, market


def _with_policy(data: pl.DataFrame) -> pl.DataFrame:
    """Alternating 60-day hiking / cutting blocks, one lone on-hold block, and no state for the first 50 days."""
    n = len(data)
    states = [None if i < 50 else ("on hold" if 290 <= i < 330 else ("hiking" if (i // 60) % 2 else "cutting"))
              for i in range(n)]
    return data.with_columns(pl.Series(rg.regime_column("policy"), states, dtype=pl.Utf8))


def test_regime_gates_are_one_per_state_with_enough_episodes():
    data = _with_policy(market())
    legs = {"left": -1.0, "right": 1.0}
    frame, results, extras = backtest_scan(data, legs=legs, regime_gates=["policy"], min_regime_episodes=3, **SCAN)
    assert len(frame) == len(data)  # the regime's missing first 50 days do not trim the sample
    regime = results.filter(pl.col("gate") == "regime:policy")
    assert set(regime["gate_bucket"].unique()) == {"hiking", "cutting"}  # "on hold" happened once: no gate
    cut = extras["cut"]
    for state in ("hiking", "cutting"):
        expected = rg.count_episodes(data[rg.regime_column("policy")].head(cut).to_list(), state)
        assert set(regime.filter(pl.col("gate_bucket") == state)["regime_episodes"].unique()) == {expected}
    assert results.filter(pl.col("gate") != "regime:policy")["regime_episodes"].null_count() == len(
        results.filter(pl.col("gate") != "regime:policy"))
    # a regime gate trades no more than ungated, and the two states together cover what the regime allows
    ungated = results.filter(pl.col("gate") == "(none)").select("signal_kind", "fit_on", "entry_z", "exit_style",
                                                                "exit_param", "n_trades")
    assert regime["n_trades"].max() <= ungated["n_trades"].max()


def test_regime_gate_opens_exactly_on_its_state_and_matches_the_engine():
    data = _with_policy(market())
    legs = {"left": -1.0, "right": 1.0}
    frame, results, _ = backtest_scan(data, legs=legs, regime_gates=["policy"], min_regime_episodes=3, **SCAN)
    state = signal_frame(data, target="y", feature="x", fit_on="changes", beta_lb=40, residual_lb=10, norm_lb=60)
    allowed = _entry_gate(state, {"gate": "regime:policy", "gate_bucket": "hiking", "gate_window": None})
    regime = state["gate_regime:policy"].to_list()
    assert allowed.tolist() == [s == "hiking" for s in regime]
    split = str(frame["ts"][int(len(frame) * 0.7)])
    rows = results.filter((pl.col("gate") == "regime:policy") & (pl.col("n_trades") > 0))
    rng = np.random.default_rng(3)
    for i in rng.choice(len(rows), 8, replace=False):
        row = rows.row(int(i), named=True)
        cand = {k: row[k] for k in ("fit_on", "beta_lb", "residual_lb", "norm_lb", "signal_kind",
                                    "gate", "gate_bucket", "gate_window")}
        cand.update(target="y", feature="x", split_date=split)
        exact, selected = run_grid(data, TradeDef("y", legs), cand, entry_zs=[row["entry_z"]],
                                   exit_params={row["exit_style"]: [row["exit_param"]]},
                                   stop_losses=[row["stop_loss_bps"]], half_life_caps=[row["half_life_cap"]],
                                   signal_stops=[row["signal_stop"]], round_trip_cost_bps=0.25, execution_lag=1)
        e = exact.row(0, named=True)
        assert row["pnl_bps"] == pytest.approx(e["earlier_pnl_bps"], abs=1e-9), row
        assert row["n_trades"] == len([t for t in selected["trades"] if str(t["exit_date"]) < split]), row


def test_with_regimes_joins_states_by_date(monkeypatch):
    from test_regimes import _inputs
    inputs = _inputs()
    monkeypatch.setattr(rg, "load_inputs", lambda start: inputs)
    frame = inputs.select("ts").tail(300).with_columns(pl.lit(1.0).alias("y"))
    out = rg.with_regimes(frame, ["policy", "inflation", "nonsense"], rg.RegimeParams(confirm=1))
    expected = rg.policy_cycle(inputs, rg.RegimeParams(confirm=1)).tail(300)["state"].to_list()
    assert out[rg.regime_column("policy")].to_list() == expected
    assert rg.regime_column("inflation") in out.columns and "regime_nonsense" not in out.columns


def test_discovery_job_saves_regime_gates_and_trade_mechanics_reproduces_a_regime_row(tmp_path, monkeypatch):
    from dataclasses import asdict
    from pathlib import Path
    import research.app as ui
    from research import artifacts
    from test_research_workflow import panel

    monkeypatch.setattr(artifacts, "RUNS", tmp_path)
    data = panel()
    states = pl.DataFrame({"ts": data["ts"], rg.regime_column("policy"): [
        None if i < 30 else ("hiking" if (i // 40) % 2 else "cutting") for i in range(len(data))]})

    def fake_regimes(frame, names, params=None):  # no database: a fixed alternating policy regime
        assert list(names) == ["policy"]
        return frame.join(states.with_columns(pl.col("ts").cast(frame.schema["ts"])), on="ts", how="left")
    monkeypatch.setattr(ui, "with_regimes", fake_regimes)

    stored = {"target": "y", "legs": {"y": 1}, "features": ["x"], "panel_id": "test", "weighting": "fixed",
              "rows": data.with_columns(pl.col("ts").cast(pl.Utf8)).to_dicts()}
    settings = dict(fit_on=["levels"], beta_lbs=[30], residual_lbs=[10], norm_lbs=[40], thresholds=[0.5],
                    raw_entries=[1.0], horizons=[5], gates=[], gate_windows=[126], signal_kind="ou_z",
                    train_fraction=0.7, min_trades=5, cost=0.1, lag=1, cv=0, rank_by="sharpe", placebos=0,
                    exit_params={"time": [5]}, exits={}, caps=None, signal_stops=None, stops=None,
                    regimes=["policy"], min_regime_episodes=2, regime_placebos=2,
                    regime_params=asdict(rg.RegimeParams()))
    out = ui.discovery_job(stored, "x", settings)
    view, _status, board = ui.render_discovery(Path(out["run_path"]).name, 5, fresh=True)
    assert board["regime_params"] == settings["regime_params"]
    import json
    from plotly.utils import PlotlyJSONEncoder
    assert "REGIME PLACEBO" in json.dumps(view, cls=PlotlyJSONEncoder)  # tested against shifted calendars
    regime_rows = [i for i, r in enumerate(board["rows"]) if r["gate"] == "regime:policy"]
    assert {board["rows"][i]["gate_bucket"] for i in regime_rows} == {"hiking", "cutting"}
    row = board["rows"][regime_rows[0]]
    assert row["regime_episodes"] >= 2 and "episodes" in ui.gate_label(row)
    # The switch shows the same rankings over regime-gated cells only, with readable regime columns.
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in ui.build_app().callback_map.values()}
    text = json.dumps(view, cls=PlotlyJSONEncoder)
    assert '"data-board-scope": "regimes"' in text
    regimes_view, regimes_board = callbacks["_sort_board"]({"scope": "regimes", "at": 1}, board, 5, None)
    assert regimes_board["scope"] == "regimes" and regimes_board["rows"]
    assert all(r["gate"] == "regime:policy" for r in regimes_board["rows"])
    shown = json.dumps(regimes_view, cls=PlotlyJSONEncoder)
    assert "macro regime gates only" in shown and ("policy = hiking" in shown or "policy = cutting" in shown)
    # a header click keeps the regime scope; the switch back keeps the sort
    by_pnl = callbacks["_sort_board"]({"rule": "pnl", "at": 2}, regimes_board, 5, None)[1]
    assert by_pnl["scope"] == "regimes" and by_pnl["sort_by"] == "pnl"
    back = callbacks["_sort_board"]({"scope": "all", "at": 3}, by_pnl, 5, None)[1]
    assert back["scope"] == "all" and back["sort_by"] == "pnl"

    candidate = ui.freeze_candidate(board, regime_rows[0])[0]
    assert candidate["gate"] == "regime:policy" and candidate["regime_params"] == settings["regime_params"]
    mech = ui.mechanics_job(stored, candidate, dict(entry_zs=[row["entry_z"]], exit_params={"time": [5.0]},
                                                    cost=0.1, stops=None, lag=1, caps=None, signal_stops=None))
    grid = pl.read_parquet(Path(mech["run_path"]) / "results.parquet").row(0, named=True)
    assert grid["earlier_pnl_bps"] == pytest.approx(row["pnl_bps"], abs=1e-9)  # the same regime-gated trade
    # Inspecting it shows the gate chart, with time-frame buttons; open exactly on the regime calendar's days.
    detail = json.dumps(callbacks["_inspect"](str(grid["config_id"]), {"run_path": mech["run_path"]}),
                        cls=PlotlyJSONEncoder)
    assert '"id": "bt-gate-img"' in detail and '"id": "bt-gate-window-1Y"' in detail
    assert f"regime policy = {row['gate_bucket']}" in detail and "Signal and gate" not in detail
    gate = ui.gate_state(Path(mech["run_path"]))
    calendar = dict(zip(states["ts"].to_list(), states[rg.regime_column("policy")].to_list()))
    assert gate["allowed"].tolist() == [calendar[d] == row["gate_bucket"] for d in gate["state"]["ts"].to_list()]
    png, summary = ui.gate_png(Path(mech["run_path"]), "1Y")
    assert png and "spells" in summary


def test_trade_histogram_hover_names_each_bucket_range():
    from research.app import trade_distribution
    bars = trade_distribution(np.array([-3.0, -1.0, 0.5, 1.5, 2.5, 4.0])).figure["data"][0]
    assert "%{customdata[0]:.1f} to %{customdata[1]:.1f} bp" in bars["hovertemplate"]
    for (low, high), centre in zip(bars["customdata"], bars["x"]):
        assert low < centre < high
        assert not (low < 0 < high)  # zero is always a bucket edge, so no bucket mixes winners and losers


def test_a_missing_day_does_not_split_an_episode():
    assert rg.count_episodes(["hiking", None, "hiking", "on hold", "hiking"], "hiking") == 2
    assert rg.count_episodes([None, None, "cutting"], "cutting") == 1


def test_shifting_a_regime_calendar_keeps_its_episodes_but_moves_their_dates():
    from research.dislocation import _shift_regime
    frame = pl.DataFrame({"r": [None, None, "a", "a", "b", "b", "b", "a"]})
    shifted = _shift_regime(frame, "r", 2)["r"].to_list()
    assert shifted[:2] == [None, None] and sorted(shifted[2:]) == sorted(frame["r"].to_list()[2:])
    assert shifted[2:] == ["b", "a", "a", "a", "b", "b"]
    assert _shift_regime(frame, "r", 0)["r"].to_list() == frame["r"].to_list()


def test_regime_placebo_scores_the_real_regime_gates_against_shifted_calendars():
    from research.dislocation import discovery_compact, rank_board
    data = _with_policy(market())
    kwargs = dict(SCAN, legs={"left": -1.0, "right": 1.0}, regime_gates=["policy"], min_regime_episodes=3)
    out = discovery_compact(data, min_trade_levels=(5,), top=20, selection_rule="sharpe", selection_min_trades=5,
                            regime_placebos=3, **kwargs)
    [test] = out["regime_placebo"]
    assert test["regime"] == "policy" and set(test["states"]) == {"hiking", "cutting"}
    assert len(test["placebo_scores"]) == 3 and 0 < test["p_value"] <= 1
    _, results, _ = backtest_scan(data, **kwargs)
    best = rank_board(results.filter(pl.col("gate") == "regime:policy"), "sharpe", 5)
    assert test["real_score"] == pytest.approx(best["rank_score"][0])
    assert out["skipped_regime_gates"] == [("policy", "on hold", 1)]


def test_saved_runs_match_on_regime_gates_and_their_definitions():
    from research.saved_runs import compare_run, grid_spec
    base = (["changes"], [60], [20], [126], [1.0], [10], [], [126], "normalized", 0.7)
    params = asdict_params = {"cycle_threshold_bp": 25.0, "confirm": 5}
    saved = grid_spec(*base, scoring="backtest", regimes=["policy"], min_regime_episodes=3, regime_params=params)
    assert saved["regime_gates"] == ["policy"] and saved["regime_params"] == asdict_params
    same = compare_run({"grid": saved}, saved, None)
    assert "covers all requested settings" in same
    more = grid_spec(*base, scoring="backtest", regimes=["policy", "vol"], min_regime_episodes=3, regime_params=params)
    assert "regime_gates: ['vol']" in compare_run({"grid": saved}, more, None)
    redefined = grid_spec(*base, scoring="backtest", regimes=["policy"], min_regime_episodes=3,
                          regime_params=dict(params, cycle_threshold_bp=50.0))
    assert "regime definitions differ" in compare_run({"grid": saved}, redefined, None)
    plain = grid_spec(*base, scoring="backtest")
    assert plain["regime_gates"] == [] and "covers all requested settings" in compare_run({"grid": saved}, plain, None)


def test_regime_only_boards_rank_just_the_regime_gated_cells():
    from research.dislocation import REGIME_SCOPE, discovery_compact, rank_board
    data = _with_policy(market())
    kwargs = dict(SCAN, legs={"left": -1.0, "right": 1.0}, regime_gates=["policy"], min_regime_episodes=3)
    out = discovery_compact(data, min_trade_levels=(5,), top=15, **kwargs)
    _, results, _ = backtest_scan(data, **kwargs)
    regime_rows = results.with_row_index("board_index").filter(pl.col("gate") == "regime:policy")
    for rule in ("sharpe", "pnl", "win_rate"):
        board = out["boards"][(REGIME_SCOPE + rule, 5)]
        expected = rank_board(results, rule, 5).filter(pl.col("gate") == "regime:policy").head(15)
        assert board["board_index"].to_list() == expected["board_index"].to_list()
        assert set(board["board_index"].to_list()) <= set(regime_rows["board_index"].to_list())
    assert out["eligible"][5] == len(rank_board(results, "sharpe", 5))  # counted over the whole grid


def test_an_empty_board_says_so_instead_of_failing():
    import json
    from plotly.utils import PlotlyJSONEncoder
    from research.app import backtest_discovery_view
    view, records = backtest_discovery_view(pl.DataFrame(), "x", {}, 30, "sharpe", scope="regimes", regime_boards=True)
    text = json.dumps(view, cls=PlotlyJSONEncoder)
    assert records == [] and "No macro-regime-gated cell has at least 30" in text and "data-board-scope" in text
