"""The three concrete 10Y / 10s30s research reports run without market data."""

import importlib

import polars as pl


MODULES = [
    "book.curve.dislocation_tens_10s30s",
    "book.curve.rv_tens_10s30s",
    "book.curve.fair_value_tens_10s30s",
]


def test_concrete_10s30s_research_reports_run_on_the_shared_synthetic_panel():
    for name in MODULES:
        module = importlib.import_module(name)
        state = module.run(module.synthetic_data(n=900))
        assert "signals" in state and isinstance(state["signals"], pl.DataFrame)
        assert not state["signals"].is_empty()
        assert "coverage" in state


def test_dislocation_exposes_raw_and_ou_views_of_the_same_state():
    module = importlib.import_module("book.curve.dislocation_tens_10s30s")
    data = module.synthetic_data(n=900)
    raw = module.run(data, metric="dislocation")
    ou = module.run(data, metric="dislocation_ou_z")
    assert raw["events"]["threshold"].max() == 20.0
    assert ou["events"]["threshold"].max() == 2.5
    assert {"dislocation", "dislocation_ou_z", "dislocation_half_life"} <= set(raw["signals"].columns)


def test_rv_and_fair_value_reports_include_their_method_specific_evidence():
    rv = importlib.import_module("book.curve.rv_tens_10s30s")
    fair = importlib.import_module("book.curve.fair_value_tens_10s30s")
    rv_state = rv.run(rv.synthetic_data(n=900))
    fair_state = fair.run(fair.synthetic_data(n=900))
    assert {"diagnostics", "hedge", "horizons"} <= set(rv_state)
    assert {"hedge_weight", "rv_value"} <= set(rv_state["signals"].columns)
    assert {"diagnostics", "factor_search", "horizons"} <= set(fair_state)
    assert {"fair_value", "residual", "error_correction"} <= set(fair_state["signals"].columns)
