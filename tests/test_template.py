"""The rate_vol strategy template must run end-to-end on synthetic data."""

import polars as pl

from book.rate_vol.template import compute, main, pipeline
from tests.synthetic import template_panel


def test_template_panel_shape():
    data = template_panel(n=800)
    assert data.columns == ["ts", "target", "anchor"]
    assert len(data) == 800


def test_compute_schema_matches_input_length():
    data = template_panel(n=800)
    sig = compute(data)
    assert "signal" in sig.columns
    assert {"resid", "beta", "r2"} <= set(sig.columns)
    assert len(sig) == len(data)


def test_template_end_to_end(monkeypatch):
    monkeypatch.setattr("book.rate_vol.template.load_data", lambda *a, **k: template_panel())
    state = main()
    result = state["result"]
    assert len(result.closed_trades) > 0
    assert isinstance(state["diag"], pl.DataFrame)
    assert set(state) == {"data", "signals", "diag", "result"}


def test_pipeline_wiring():
    assert pipeline.compute_fn is compute
    assert set(pipeline.trade_def.legs) == {"target", "anchor"}
