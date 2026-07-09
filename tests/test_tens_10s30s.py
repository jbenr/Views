"""10s vs 10s30s module: standard scriptable pattern, runnable without DB."""

import importlib

import polars as pl

view = importlib.import_module("book.curve.tens_10s30s")
compute = view.compute
main = view.main
pipeline = view.pipeline
synthetic_data = view.synthetic_data


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
    assert set(state) == {"data", "signals", "diag", "result"}
    assert isinstance(state["diag"], pl.DataFrame)
    assert len(state["result"].closed_trades) > 0
    # residual is OU by construction -> fading it must show positive IC
    assert (state["diag"]["ic"] > 0).all()


def test_pipeline_wiring():
    assert pipeline.compute_fn is compute
    assert set(pipeline.trade_def.legs) == {"10y", "10s30s"}
    assert pipeline.name == "tens_10s30s"
