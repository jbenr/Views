"""Graduated curve strategies on the Strategy template - no DB required."""

import importlib

import polars as pl
import pytest

MODULES = ["book.curve.twos10s_real10y", "book.curve.tens30s_pc1"]


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
    assert mod.STRATEGY.setups_file.parent.name == "curve"


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
