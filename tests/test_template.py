"""The rate_vol strategy template must run end-to-end on synthetic data."""

import polars as pl

from book.rate_vol.template import compute, main, pipeline, synthetic_data


def test_synthetic_data_shape():
    data = synthetic_data(n=800)
    assert data.columns == ["ts", "target", "anchor"]
    assert len(data) == 800


def test_compute_schema_matches_input_length():
    data = synthetic_data(n=800)
    sig = compute(data)
    assert "signal" in sig.columns
    assert {"resid", "beta", "r2"} <= set(sig.columns)
    assert len(sig) == len(data)


def test_template_end_to_end():
    state = main(use_db=False)
    result = state["result"]
    assert len(result.closed_trades) > 0
    assert isinstance(state["diag"], pl.DataFrame)
    assert set(state) == {"data", "signals", "diag", "result"}


def test_pipeline_wiring():
    assert pipeline.compute_fn is compute
    assert set(pipeline.trade_def.legs) == {"target", "anchor"}
