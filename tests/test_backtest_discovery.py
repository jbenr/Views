"""Discovery by real backtests: parity with Engine, CV folds and selection checks."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from backtest.engine import TradeDef
from backtest.validation import cv_fold_labels, cv_scores, pooled_sharpe, selection_checks
from research.dislocation import RANK_RULES, backtest_scan, placebo_scan, rank_board
from research.dislocation_backtest import run_grid


def market(n=900, seed=11):
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
        "wl": -1 + 0.2 * np.sin(np.arange(n) / 30), "wr": np.ones(n),
    })


SCAN = dict(target="y", feature="x", beta_lookbacks=[40], residual_lookbacks=[10], normalization_lookbacks=[60],
            thresholds=[1.0, 1.5], exit_params={"time": [5, 15], "band": [0.0], "revert_frac": [0.5], "half_life_frac": [1.0]},
            half_life_caps=[None, 2.0], signal_stops=[None, 1.0], stop_losses=[None, 3.0],
            fit_on=["changes", "levels"], signal_kind=["normalized", "ou_z"],
            gate_names=["resid_vol20", "feature_level"], gate_windows=[126], cost_bps=0.25, execution_lag=1,
            train_fraction=0.7)


@pytest.mark.parametrize("weighted", [False, True])
def test_every_discovery_cell_is_the_engine_trade_it_describes(weighted):
    data = market()
    legs = {"left": -1.0, "right": 1.0}
    weights = {"left": "wl", "right": "wr"} if weighted else None
    frame, results, _ = backtest_scan(data, legs=legs, weight_columns=weights, **SCAN)
    split = str(frame["ts"][int(len(frame) * 0.7)])
    rng = np.random.default_rng(0)
    gated = results.filter(pl.col("gate") != "(none)")
    rows = [results.row(int(i), named=True) for i in rng.choice(len(results), 6, replace=False)]
    rows += [gated.row(int(i), named=True) for i in rng.choice(len(gated), 6, replace=False)]
    # and one cell of every exit style, capped and signal-stopped where they apply
    for style in ("time", "band", "revert_frac", "half_life_frac"):
        pick = results.filter((pl.col("exit_style") == style) & pl.col("signal_stop").is_not_null()
                              & (pl.col("half_life_cap").is_not_null() | ~pl.col("exit_style").is_in(["band", "revert_frac"])))
        rows.append(pick.row(int(rng.integers(len(pick))), named=True))
    for row in rows:
        cand = {k: row[k] for k in ("fit_on", "beta_lb", "residual_lb", "norm_lb", "signal_kind",
                                    "gate", "gate_bucket", "gate_window")}
        cand.update(target="y", feature="x", split_date=split, **({"weight_columns": weights} if weights else {}))
        exact, selected = run_grid(data, TradeDef("y", legs), cand, entry_zs=[row["entry_z"]],
                                   exit_params={row["exit_style"]: [row["exit_param"]]}, stop_losses=[row["stop_loss_bps"]],
                                   half_life_caps=[row["half_life_cap"]], signal_stops=[row["signal_stop"]],
                                   round_trip_cost_bps=0.25, execution_lag=1)
        e = exact.row(0, named=True)
        assert row["pnl_bps"] == pytest.approx(e["earlier_pnl_bps"], abs=1e-9), row
        assert row["later_pnl_bps"] == pytest.approx(e["later_pnl_bps"], abs=1e-9), row
        daily = np.array([r["pnl_bps"] for r in selected["equity"]])
        early = daily[: int(len(frame) * 0.7)]
        expected = early.mean() / early.std() * np.sqrt(252) if early.std() > 0 else 0.0
        assert row["sharpe"] == pytest.approx(expected, abs=1e-9), row
        closed = [t for t in selected["trades"] if str(t["exit_date"]) < split]
        assert row["n_trades"] == len(closed), row


def test_cv_labels_embargo_each_internal_boundary():
    labels = cv_fold_labels(100, 80, folds=4, embargo=3)
    assert labels[:20].tolist() == [0] * 20
    assert labels[20:23].tolist() == [-1] * 3 and labels[23:40].tolist() == [1] * 17
    assert labels[60:63].tolist() == [-1] * 3 and labels[79] == 3
    assert (labels[80:] == 4).all()


def test_cv_scores_are_block_sharpes_of_the_underlying_pnl():
    rng = np.random.default_rng(1)
    pnl = rng.normal(0.1, 1, size=(3, 300))
    labels = np.repeat([0, 1, 2], 100)
    days = np.bincount(labels)
    sums = np.stack([[p[labels == k].sum() for k in range(3)] for p in pnl])
    sumsq = np.stack([[(p[labels == k] ** 2).sum() for k in range(3)] for p in pnl])
    scores = cv_scores(days, sums, sumsq)
    expected = np.array([[p[labels == k].mean() / p[labels == k].std() * np.sqrt(252) for k in range(3)] for p in pnl])
    np.testing.assert_allclose(scores["fold_sharpe"], expected)
    np.testing.assert_allclose(scores["worst"], expected.min(axis=1))
    np.testing.assert_allclose(pooled_sharpe(300, pnl.sum(1), (pnl ** 2).sum(1)), pnl.mean(1) / pnl.std(1) * np.sqrt(252))


def _fold_stats(pnl, folds):
    labels = np.repeat(np.arange(folds), pnl.shape[1] // folds)
    days = np.bincount(labels)
    sums = np.stack([pnl[:, labels == k].sum(axis=1) for k in range(folds)], axis=1)
    sumsq = np.stack([(pnl[:, labels == k] ** 2).sum(axis=1) for k in range(folds)], axis=1)
    return days, sums, sumsq


def test_selection_checks_find_real_edge_and_not_noise():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 1, size=(200, 2000))
    edge = noise.copy()
    edge[7] += 0.15  # one config with genuine, persistent edge
    for score in ("sharpe", "mean", "worst"):
        rows = selection_checks(*_fold_stats(edge, 5), score=score)
        assert {r["method"] for r in rows} == {"out-of-fold", "walk-forward"}
        assert all(r["config_index"] == 7 for r in rows)
        assert np.mean([r["chosen_held_out_percentile"] for r in rows]) > 0.9
    null = selection_checks(*_fold_stats(noise, 5))
    assert np.mean([r["chosen_held_out_percentile"] for r in null]) < 0.8
    assert selection_checks(*_fold_stats(edge, 5), eligible=np.arange(200) != 7)[0]["config_index"] != 7


def test_cv_columns_ranking_and_placebo():
    data = market()
    kwargs = dict(SCAN, legs={"left": -1.0, "right": 1.0}, cv_folds=4)
    _, results, extras = backtest_scan(data, **kwargs)
    assert {"cv_mean_sharpe", "cv_worst_sharpe", "cv_positive_share"} <= set(results.columns)
    assert extras["cv_sums"].shape == (len(results), 4)
    np.testing.assert_allclose(cv_scores(extras["cv_days"], extras["cv_sums"], extras["cv_sumsq"])["mean"],
                               results["cv_mean_sharpe"].to_numpy())
    board = rank_board(results, "cv_worst", min_trades=5)
    assert board["rank_score"].to_list() == sorted(board["rank_score"].to_list(), reverse=True)
    assert (board["n_trades"] >= 5).all()
    assert results.row(board["board_index"][0], named=True)["cv_worst_sharpe"] == board["rank_score"][0]
    no_cv = backtest_scan(data, **dict(SCAN, legs={"left": -1.0, "right": 1.0}))[1]
    with pytest.raises(ValueError, match="cross-validated"):
        rank_board(no_cv, "cv_mean")
    placebo = placebo_scan(data, feature="x", placebos=2, rank_by="sharpe", min_trades=5,
                           **{k: v for k, v in SCAN.items() if k != "feature"}, legs={"left": -1.0, "right": 1.0})
    assert len(placebo) == 2 and all(np.isfinite(p["best_score"]) for p in placebo)
    assert placebo[0]["shift_bars"] != placebo[1]["shift_bars"]



def test_exit_cells_skip_bands_at_entry_and_cap_only_signal_exits():
    from research.dislocation_backtest import exit_cells
    cells = exit_cells([1.0], {"time": [5], "band": [0.0, 1.0], "half_life_frac": [1.0]},
                       stop_losses=[0, 10.0], half_life_caps=[None, 2.0], signal_stops=[0.0])
    styles = [(c[1], c[2], c[4]) for c in cells]
    assert ("band", 1.0, None) not in styles and ("band", 1.0, 2.0) not in styles
    assert {c[4] for c in cells if c[1] != "band"} == {None}
    assert {c[4] for c in cells if c[1] == "band"} == {None, 2.0}
    assert {c[3] for c in cells} == {None, 10.0} and {c[5] for c in cells} == {None}
    assert len(cells) == (1 + 2 + 1) * 2  # time, band x 2 caps, half-life; x 2 stops


def test_placebo_progress_counts_models_across_every_placebo():
    seen = []
    placebo_scan(market(), feature="x", placebos=2, rank_by="sharpe", min_trades=5,
                 progress=lambda done, total, message: seen.append((done, total, message)),
                 **{k: v for k, v in SCAN.items() if k != "feature"}, legs={"left": -1.0, "right": 1.0})
    totals = {t for _, t, _ in seen}
    assert totals == {2 * 4}  # 2 placebos x 4 models
    assert [d for d, _, _ in seen] == sorted(d for d, _, _ in seen)
    assert seen[-1][0] == 8 and "Placebo 2/2" in seen[-1][2] and "model" in seen[1][2]


@pytest.mark.parametrize("rank_by", ["family", "sharpe", "cv_mean", "cv_worst"])
def test_ranking_from_disk_matches_ranking_in_memory(tmp_path, rank_by):
    from polars.testing import assert_frame_equal
    from research.dislocation import rank_board_file
    _, results, _ = backtest_scan(market(), **dict(SCAN, legs={"left": -1.0, "right": 1.0}, cv_folds=3))
    path = tmp_path / "results.parquet"
    results.write_parquet(path)
    expected = rank_board(results, rank_by, min_trades=5)
    board, cells, eligible = rank_board_file(path, rank_by, min_trades=5, top=25)
    assert (cells, eligible) == (len(results), len(expected))
    assert_frame_equal(board, expected.head(25), check_dtypes=False)


def test_ranking_from_disk_reads_runs_saved_before_exits_joined_discovery(tmp_path):
    from research.dislocation import rank_board_file
    _, results, _ = backtest_scan(market(), **dict(SCAN, legs={"left": -1.0, "right": 1.0},
                                                    exit_params={"time": [5, 15]}, half_life_caps=[None],
                                                    signal_stops=[None], stop_losses=[None]))
    old = results.drop("exit_style", "half_life_cap", "signal_stop", "stop_loss_bps").rename({"exit_param": "horizon"})
    old.with_columns(pl.col("horizon").cast(pl.Int64)).write_parquet(tmp_path / "old.parquet")
    board, cells, _ = rank_board_file(tmp_path / "old.parquet", "sharpe", min_trades=5, top=10)
    assert cells == len(old) and set(board["exit_style"]) == {"time"}
    assert board["exit_param"].to_list() == rank_board(results, "sharpe", 5).head(10)["exit_param"].to_list()


def test_compact_discovery_boards_equal_full_ranking_row_for_row():
    from polars.testing import assert_frame_equal
    from research.dislocation import discovery_compact
    kwargs = dict(SCAN, legs={"left": -1.0, "right": 1.0}, cv_folds=3)
    _, results, extras = backtest_scan(market(), **kwargs)
    out = discovery_compact(market(), min_trade_levels=(5, 20), top=30, selection_rule="cv_mean",
                            selection_min_trades=5, **kwargs)
    assert out["cells"] == len(results)
    assert {rule for rule, _ in out["boards"]} == set(RANK_RULES)  # every angle has its own board
    for rule in RANK_RULES:
        for m in (5, 20):
            expected = rank_board(results, rule, m)
            assert out["eligible"][m] == len(expected)
            assert_frame_equal(out["boards"][(rule, m)], expected.head(30), check_dtypes=False,
                               check_column_order=False)
    expected_checks = selection_checks(extras["cv_days"], extras["cv_sums"], extras["cv_sumsq"], "mean",
                                       eligible=(results["n_trades"] >= 5).to_numpy())
    assert out["checks"] == expected_checks


def test_old_results_files_rank_to_the_same_boards_when_streamed(tmp_path):
    import pyarrow.parquet as pq
    from polars.testing import assert_frame_equal
    from research.dislocation import rank_results_file
    _, results, _ = backtest_scan(market(), **dict(SCAN, legs={"left": -1.0, "right": 1.0}, cv_folds=3))
    path = tmp_path / "results.parquet"
    pq.write_table(results.to_arrow(), path, row_group_size=700)  # many row groups, models spanning them
    out = rank_results_file(path, min_trade_levels=(5, 20), top=30)
    assert out["cells"] == len(results)
    for rule in ("family", "sharpe", "cv_mean", "cv_worst"):
        for m in (5, 20):
            expected = rank_board(results, rule, m)
            assert out["eligible"][m] == len(expected)
            assert_frame_equal(out["boards"][(rule, m)], expected.head(30), check_dtypes=False,
                               check_column_order=False)


def test_an_interrupted_discovery_resumes_from_its_checkpoint(tmp_path, monkeypatch):
    from polars.testing import assert_frame_equal
    import research.dislocation as dis
    kwargs = dict(SCAN, legs={"left": -1.0, "right": 1.0}, cv_folds=3)
    reference = dis.discovery_compact(market(), min_trade_levels=(5,), top=20, **kwargs)
    # Crash partway: the third model blows up after two were checkpointed.
    original, calls = dis._ScanPlan.model_rows, []
    def flaky(self, i, cells=None):
        if cells is None:
            calls.append(i)
            if len(calls) == 3:
                raise RuntimeError("worker died")
        return original(self, i, cells)
    monkeypatch.setattr(dis._ScanPlan, "model_rows", flaky)
    with pytest.raises(RuntimeError, match="worker died"):
        dis.discovery_compact(market(), min_trade_levels=(5,), top=20, checkpoints=tmp_path, **kwargs)
    saved = list(tmp_path.glob("*/model_*.npz"))
    assert len(saved) == 2 and not list(tmp_path.glob("*/*.partial.npz"))
    # The rerun loads the two saved models and only backtests the rest.
    monkeypatch.setattr(dis._ScanPlan, "model_rows", original)
    rerun_models = []
    monkeypatch.setattr(dis._ScanPlan, "model_rows",
                        lambda self, i, cells=None: (rerun_models.append(i) if cells is None else None) or original(self, i, cells))
    resumed = dis.discovery_compact(market(), min_trade_levels=(5,), top=20, checkpoints=tmp_path, **kwargs)
    assert sorted(rerun_models) == [2, 3]
    for key in reference["boards"]:
        assert_frame_equal(resumed["boards"][key], reference["boards"][key])
    assert resumed["checks"] == reference["checks"]
    assert not any(tmp_path.iterdir())  # finished runs leave no checkpoint behind


def test_backtest_loops_pause_the_garbage_collector_and_restore_it():
    import gc
    import research.dislocation as dis
    seen = []
    original = dis._ScanPlan.model_rows
    def spy(self, i, cells=None):
        seen.append(gc.isenabled())
        return original(self, i, cells)
    try:
        dis._ScanPlan.model_rows = spy
        dis.discovery_compact(market(), min_trade_levels=(5,), top=5, **dict(SCAN, legs={"left": -1.0, "right": 1.0}))
    finally:
        dis._ScanPlan.model_rows = original
    assert seen and not any(seen[:4]) and gc.isenabled()


def test_max_entry_half_life_only_enters_reverting_residuals_and_matches_the_engine():
    data = market()
    legs = {"left": -1.0, "right": 1.0}
    scan = dict(SCAN, gate_names=[], max_half_lives=[None, 6.0])
    frame, results, _ = backtest_scan(data, legs=legs, **scan)
    keys = ["signal_kind", "fit_on", "beta_lb", "residual_lb", "norm_lb", "entry_z", "exit_style", "exit_param",
            "half_life_cap", "signal_stop", "stop_loss_bps", "gate", "gate_bucket"]
    pairs = (results.filter(pl.col("max_entry_half_life").is_null())
             .join(results.filter(pl.col("max_entry_half_life") == 6.0), on=keys, suffix="_capped", nulls_equal=True))
    # Skipping a slow-reverting entry can free the book for a later one, so counts can move either way;
    # what must hold is that the filter changes trading, and (below) every entry it takes is reverting.
    assert len(pairs) and (pairs["n_trades_capped"] != pairs["n_trades"]).any()
    split = str(frame["ts"][int(len(frame) * 0.7)])
    capped = results.filter((pl.col("max_entry_half_life") == 6.0) & (pl.col("n_trades") > 0))
    rng = np.random.default_rng(5)
    for i in rng.choice(len(capped), 8, replace=False):
        row = capped.row(int(i), named=True)
        cand = {k: row[k] for k in ("fit_on", "beta_lb", "residual_lb", "norm_lb", "signal_kind",
                                    "gate", "gate_bucket", "gate_window")}
        cand.update(target="y", feature="x", split_date=split)
        exact, selected = run_grid(data, TradeDef("y", legs), cand, entry_zs=[row["entry_z"]],
                                   exit_params={row["exit_style"]: [row["exit_param"]]},
                                   stop_losses=[row["stop_loss_bps"]], half_life_caps=[row["half_life_cap"]],
                                   signal_stops=[row["signal_stop"]], max_half_lives=[6.0],
                                   round_trip_cost_bps=0.25, execution_lag=1)
        e = exact.row(0, named=True)
        assert row["pnl_bps"] == pytest.approx(e["earlier_pnl_bps"], abs=1e-9), row
        assert row["n_trades"] == len([t for t in selected["trades"] if str(t["exit_date"]) < split]), row
        assert all(0 < t["entry_half_life"] <= 6.0 for t in selected["trades"])  # every entry was reverting
