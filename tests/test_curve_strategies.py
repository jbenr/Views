"""Graduated curve strategies on the Strategy template - no DB required."""

import importlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from backtest.engine import BacktestConfig, Engine

MODULES = [
    "book.curve.twos10s_real10y",
    "book.curve.tens30s_pc1",
    "book.curve.twos_10s30s",
]

LIVE_MODULES = [
    "book.curve.tens_10s30s",
    "book.curve.tens30s_pc1",
    "book.curve.twos_10s30s",
    "book.curve.twos10s_real10y",
]


def test_twos_10s30s_defaults_to_live_ungated_research_candidate():
    mod = importlib.import_module("book.curve.twos_10s30s")

    assert mod.STRATEGY.default_params == {
        **mod.DEFAULT_PARAMS,
        **mod.RESEARCH_PARAMS,
    }
    assert mod.STRATEGY.default_params["gate"] is None


@pytest.mark.parametrize("module_name", MODULES)
def test_end_to_end_synthetic(module_name):
    mod = importlib.import_module(module_name)
    state = mod.main(use_db=False)
    assert set(state) == {"raw_data", "coverage", "data", "signals", "diag", "result"}
    assert len(state["result"].closed_trades) > 0
    # residual is OU by construction -> fading it must show positive IC
    assert (state["diag"]["ic"] > 0).all()


@pytest.mark.parametrize("module_name", MODULES)
def test_pipeline_wiring(module_name):
    mod = importlib.import_module(module_name)
    assert mod.pipeline.compute_fn is mod.compute
    assert mod.pipeline.name == mod.SIGNAL_NAME
    assert mod.pipeline.trade_def.legs == {mod.TARGET: 1.0}
    assert mod.STRATEGY.module == module_name
    assert mod.STRATEGY.data_dir == (
        mod.STRATEGY.path.parent / "data" / mod.SIGNAL_NAME
    )
    assert mod.STRATEGY.setups_file.parent == mod.STRATEGY.data_dir


def test_xy_scan_uses_shared_curve_data_directory():
    mod = importlib.import_module("book.curve.xy_scan")
    assert mod.XY_DATA_DIR == Path(mod.__file__).parent / "data"
    assert mod.XY_SETUPS_FILE.parent == mod.XY_DATA_DIR


def test_pc1_feature_hook_adds_point_in_time_score():
    mod = importlib.import_module("book.curve.tens30s_pc1")
    data = mod.add_pc1(mod.synthetic_data(n=900))
    assert mod.FEATURE in data.columns
    # warmup nulls, then populated
    assert data[mod.FEATURE][: mod.PC1_LB - 1].is_null().all()
    assert data[mod.FEATURE].slice(mod.PC1_LB).null_count() == 0
    # PC1 is the level factor: its changes track the average yield change
    frame = mod.model_frame(data)
    assert len(frame) == 900 - (mod.PC1_LB - 1)


@pytest.mark.parametrize("module_name", LIVE_MODULES)
@pytest.mark.parametrize("gate", [None, ("beta_vol20", "high_75")])
def test_live_signal_is_prefix_invariant(module_name, gate):
    """Appending future observations cannot alter any earlier signal state."""
    mod = importlib.import_module(module_name)
    strategy = mod.STRATEGY
    raw = mod.synthetic_data(n=1600)
    cutoff = 1200

    def prepare(frame):
        enriched = (
            strategy.feature_fn(frame)
            if strategy.feature_fn is not None
            else frame
        )
        return strategy.model_frame(enriched)

    full_data = prepare(raw)
    prefix_data = prepare(raw.head(cutoff))
    assert full_data["ts"].head(len(prefix_data)).equals(prefix_data["ts"])

    params = {**strategy.default_params, "gate": gate}
    full = strategy.compute(full_data, params=params).head(len(prefix_data))
    prefix = strategy.compute(prefix_data, params=params)

    assert full.columns == prefix.columns
    for column in full.columns:
        left, right = full[column], prefix[column]
        if left.dtype == pl.Boolean:
            assert left.to_list() == right.to_list(), column
        else:
            np.testing.assert_allclose(
                left.to_numpy(),
                right.to_numpy(),
                equal_nan=True,
                rtol=0.0,
                atol=1e-12,
                err_msg=column,
            )


@pytest.mark.parametrize("module_name", LIVE_MODULES)
def test_live_backtest_history_is_prefix_invariant(module_name):
    """Future rows cannot rewrite earlier positions, PnL, or closed trades."""
    mod = importlib.import_module(module_name)
    strategy = mod.STRATEGY
    raw = mod.synthetic_data(n=1600)
    cutoff = 1200

    def prepare(frame):
        enriched = (
            strategy.feature_fn(frame)
            if strategy.feature_fn is not None
            else frame
        )
        return strategy.model_frame(enriched)

    full_data = prepare(raw)
    prefix_data = prepare(raw.head(cutoff))
    config = BacktestConfig(transaction_cost_bps=strategy.transaction_cost_bps)
    full = Engine(config).add_signal(strategy.make_pipeline()).run(full_data)
    prefix = Engine(config).add_signal(strategy.make_pipeline()).run(prefix_data)

    np.testing.assert_allclose(
        full.equity_curve["pnl_bps"].head(len(prefix.equity_curve)).to_numpy(),
        prefix.equity_curve["pnl_bps"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
    cutoff_date = prefix_data["ts"][-1]
    full_closed = [
        (t.entry_date, t.exit_date, t.direction, t.pnl_bps, t.exit_reason)
        for t in full.closed_trades
        if t.exit_date <= cutoff_date
    ]
    prefix_closed = [
        (t.entry_date, t.exit_date, t.direction, t.pnl_bps, t.exit_reason)
        for t in prefix.closed_trades
    ]
    assert full_closed == prefix_closed
