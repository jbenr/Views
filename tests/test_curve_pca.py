"""PCA curve fair value: point-in-time fits, structures, and the Fair Value tab's PCA view."""

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from stats import roll_pca_fair_value


def _curve(n=400, seed=1):
    """A curve driven by level, slope and curvature, plus a little tenor-specific noise."""
    rng = np.random.default_rng(seed)
    years = np.array([2, 3, 5, 7, 10, 30], dtype=float)
    level = 400 + np.cumsum(rng.normal(scale=5, size=n))
    slope = 50 + np.cumsum(rng.normal(scale=2, size=n))
    curve = 30 + np.cumsum(rng.normal(scale=1, size=n))
    shape = (years - years.mean()) / years.std()
    belly = -((years - 7) / 10) ** 2
    data = {f"{int(y)}y": level + slope * shape[i] + curve * belly[i] + rng.normal(scale=0.3, size=n)
            for i, y in enumerate(years)}
    return pl.DataFrame({"ts": [dt.date(2015, 1, 1) + dt.timedelta(days=i) for i in range(n)], **data})


def test_fair_value_is_point_in_time_and_three_factors_explain_a_three_factor_curve():
    curve = _curve()
    out = roll_pca_fair_value(curve.drop("ts"), lookback=100, n_components=3)
    resid = out["resid"].to_numpy()
    assert np.isnan(resid[:99]).all() and np.isfinite(resid[99:]).all()
    assert np.nanstd(resid) < 0.5  # only the tenor noise is left
    # changing the future never changes a past fair value
    shocked = curve.with_columns(pl.when(pl.int_range(pl.len()) >= 300).then(pl.col("10y") + 50)
                                 .otherwise(pl.col("10y")).alias("10y"))
    again = roll_pca_fair_value(shocked.drop("ts"), lookback=100, n_components=3)
    np.testing.assert_allclose(again["fair"].to_numpy()[:300], out["fair"].to_numpy()[:300])
    # one factor leaves slope and curvature in the residual
    assert np.nanstd(roll_pca_fair_value(curve.drop("ts"), lookback=100, n_components=1)["resid"].to_numpy()) > 2


def test_pca_curve_gives_todays_curve_its_implied_curve_and_residuals(monkeypatch):
    import research.curve_pca as cp
    curve = _curve()
    monkeypatch.setattr(cp, "load_wide", lambda tickers, **kw: curve.select("ts", *tickers))
    out = cp.pca_curve(["30y", "2y", "10y", "5y"], "2015-01-01", 100, 3)
    table = out["table"]
    assert table["tenor"].to_list() == ["2y", "5y", "10y", "30y"] and table["years"].to_list() == [2, 5, 10, 30]
    assert out["as_of"] == curve["ts"][-1]
    row = table.filter(pl.col("tenor") == "10y").row(0, named=True)
    assert row["actual_bp"] == pytest.approx(curve["10y"][-1])
    assert row["residual_bp"] == pytest.approx(row["actual_bp"] - row["fair_bp"])
    history = roll_pca_fair_value(curve.select("2y", "5y", "10y", "30y"), lookback=100)["resid"]["10y"].drop_nulls()
    assert row["z"] == pytest.approx(row["residual_bp"] / history.std())
    assert out["cumulative"][-1] > 0.99


def test_fair_value_tab_runs_pca_on_open(monkeypatch):
    import research.curve_pca as cp
    from plotly.utils import PlotlyJSONEncoder
    from research.app import build_app
    curve = _curve()
    monkeypatch.setattr(cp, "load_wide", lambda tickers, **kw: curve.select("ts", *tickers))
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in build_app().callback_map.values()}
    assert callbacks["_run_pca"](0, "setup", ["2y", "5y", "10y"], "2015-01-01", 100, 3, None) == (
        callbacks["_run_pca"](0, "setup", [], "", 0, 3, None))  # nothing until the tab is opened
    view, settings = callbacks["_run_pca"](0, "fv", ["2y", "5y", "10y", "30y"], "2015-01-01", 100, 3, None)
    text = json.dumps(view, cls=PlotlyJSONEncoder)
    assert text.count("data:image/png;base64,") == 3 and settings["window"] == 100  # curve, bp bars, z bars
    refused = callbacks["_run_pca"](1, "fv", ["2y"], "2015-01-01", 100, 3, None)[0]
    assert "Choose at least three" in json.dumps(refused, cls=PlotlyJSONEncoder)
