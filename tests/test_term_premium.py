"""Term premium: the 2s10s30s residual against 2s10s, the ACM loader, and the Fair Value tab."""

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from utils.market_data import _acm_monthly


def _yields(n=900, seed=4):
    rng = np.random.default_rng(seed)
    two = 300 + np.cumsum(rng.normal(size=n))
    ten = two + 80 + np.cumsum(rng.normal(scale=0.8, size=n))
    thirty = ten + 30 + 0.4 * (ten - two - 80) + rng.normal(scale=0.5, size=n)
    return pl.DataFrame({"ts": [dt.date(2015, 1, 1) + dt.timedelta(days=i) for i in range(n)],
                         "2y": two, "10y": ten, "30y": thirty})


def _acm(days):
    month_ends = [d for d in days if (d + dt.timedelta(days=1)).day == 1]
    tp = np.linspace(-20, 60, len(month_ends))
    return pl.DataFrame({"ts": month_ends, "tp10": tp, "fitted10": 400 + tp, "expected10": [400.0] * len(month_ends)})


def test_acm_chart_feed_parses_to_bp_with_the_expected_path():
    text = "RunDates,TERMYld,ACMFITYld,GSWYld\n31-Aug-2026,0.759,4.817,4.826\n30-Sep-2026,0.885,5.259,5.284\n"
    frame = _acm_monthly(text)
    assert frame["ts"].to_list() == [dt.date(2026, 8, 31), dt.date(2026, 9, 30)]
    assert frame["tp10"].to_list() == pytest.approx([75.9, 88.5])
    assert frame["expected10"].to_list() == pytest.approx([481.7 - 75.9, 525.9 - 88.5])


def test_fly_residual_is_point_in_time_with_a_valid_r2(monkeypatch):
    import research.term_premium as tp
    data = _yields()
    monkeypatch.setattr(tp, "load_wide", lambda tickers, **kw: data.select("ts", *tickers))
    curve = tp.load_curve("2015-01-01")
    assert curve["2s10s30s"].to_list() == pytest.approx((2 * data["10y"] - data["2y"] - data["30y"]).to_list())
    out = tp.fly_residual(curve, 252)["signals"].drop_nulls("residual")
    assert out["r2"].min() >= 0 and out["r2"].max() <= 1
    assert out["residual"].to_list() == pytest.approx((out["2s10s30s"] - out["fair_value"]).to_list())
    # changing the future never changes a past residual
    later = tp.fly_residual(curve.with_columns(
        pl.when(pl.int_range(pl.len()) >= 700).then(pl.col("2s10s30s") + 25).otherwise(pl.col("2s10s30s"))
        .alias("2s10s30s")), 252)["signals"]
    full = tp.fly_residual(curve, 252)["signals"]
    np.testing.assert_allclose(later["residual"].head(700).fill_null(0).to_numpy(),
                               full["residual"].head(700).fill_null(0).to_numpy())


def test_term_premium_tab_runs_when_opened(monkeypatch):
    import research.term_premium as tp
    from plotly.utils import PlotlyJSONEncoder
    from research.app import build_app
    data = _yields()
    monkeypatch.setattr(tp, "load_wide", lambda tickers, **kw: data.select("ts", *tickers))
    monkeypatch.setattr(tp, "load_acm_term_premium", lambda: (_acm(data["ts"].to_list()), "monthly"))
    result = tp.term_premium("2015-01-01", 252)
    assert result["comparison"]["level_corr"] is not None and len(result["comparison"]["matched"]) > 10
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in build_app().callback_map.values()}
    idle = callbacks["_term_premium"](0, "fv", "pca", "5Y", "2015-01-01", 252, None)
    assert idle[0] is idle[1]  # not until the Term Premium sub-tab is open
    view, shown = callbacks["_term_premium"](0, "fv", "tp", "1Y", "2015-01-01", 252, None)
    text = json.dumps(view, cls=PlotlyJSONEncoder)
    assert shown == "1Y" and text.count("data:image/png;base64,") == 4
    assert "ACM 10y term premium" in text and "residual" in text
