"""Macro regimes: causal definitions, confirmation, episodes, and the Regimes tab."""

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from research import regimes as rg


def _inputs(n=900, seed=3):
    rng = np.random.default_rng(seed)
    days = [dt.date(2015, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    i = np.arange(n)
    # fed funds: cuts priced early on; then a hiking cycle (300 -> 600bp over days 450-600, more priced);
    # then on hold once delivered
    rate = np.clip(300 + 2.0 * (i - 450), 300, 600)
    priced = np.where(i < 300, -60.0, np.where((i >= 400) & (i < 600), 60.0, 0.0))
    ff_now = 100 - rate / 100
    oil_moves = rng.normal(scale=1, size=n)
    tsy_moves = np.where(np.arange(n) < n // 2, 0.0, 3.0) * oil_moves + rng.normal(scale=3, size=n)
    implied = 80 + np.cumsum(rng.normal(scale=1.5, size=n))
    infl5 = np.where(i < 300, 200.0, np.where(i < 600, 250.0, 300.0))
    return pl.DataFrame({"ts": days, "ff_now": ff_now, "ff_6m": ff_now - priced / 100,
                         "10y": 400 + np.cumsum(tsy_moves), "oil": 60 + np.cumsum(oil_moves), "infl5": infl5,
                         rg.IMPLIED: implied})


def test_confirmation_ignores_short_flickers():
    raw = ["a", "a", "b", "a", "b", "b", "b", None, "b"]
    assert rg.confirm_states(raw, 3) == ["a", "a", "a", "a", "a", "a", "b", "b", "b"]
    assert rg.confirm_states(raw, 1) == ["a", "a", "b", "a", "b", "b", "b", "b", "b"]


def test_regimes_read_what_they_should_and_never_use_the_future():
    frame, p = _inputs(), rg.RegimeParams(confirm=1)
    policy = rg.policy_cycle(frame, p)
    assert policy["priced"][200] == pytest.approx(-60.0) and policy["realized"][200] == pytest.approx(0.0)
    assert policy["state"][200] == "cutting"
    assert policy["realized"][550] == pytest.approx(200.0) and policy["state"][550] == "hiking"  # moved + priced
    assert policy["state"][-1] == "on hold"  # the cycle has been delivered and nothing more is priced
    inflation = rg.inflation_regime(frame, p)
    assert [inflation["state"][k] for k in (100, 450, 800)] == ["low inflation", "anchored", "high inflation"]
    oil = rg.oil_sensitivity(frame, p)
    assert oil["state"][300] in ("weak", "none / inverse") and oil["state"][-1] == "trades with oil"
    vol = rg.rate_vol(frame, p)
    assert set(vol["state"].drop_nulls().unique()) <= set(rg.STATES["vol"])
    implied = rg.implied_rate_vol(frame, p)
    assert implied["value"].to_list() == pytest.approx(frame[rg.IMPLIED].to_list())  # the 1m10y level itself
    assert implied["state"].drop_nulls().len() > 0
    # appending data never changes a state already emitted
    for build in (rg.policy_cycle, rg.rate_vol, rg.implied_rate_vol, rg.oil_sensitivity, rg.inflation_regime):
        full, part = build(frame, rg.RegimeParams()), build(frame.head(600), rg.RegimeParams())
        assert full["state"].head(600).to_list() == part["state"].to_list()


def test_episode_table_counts_runs():
    series = pl.DataFrame({"ts": [dt.date(2020, 1, d) for d in range(1, 8)],
                           "value": [0.0] * 7, "state": ["a", "a", "b", "b", "a", "a", "a"]})
    table = rg.episodes(series, ["a", "b"])
    a = table.filter(pl.col("state") == "a").row(0, named=True)
    assert a["episodes"] == 2 and a["longest_days"] == 3 and a["share_of_days"] == pytest.approx(5 / 7)
    assert rg.current_state(series)["since"] == dt.date(2020, 1, 5)


def test_regimes_tab_runs_on_open(monkeypatch):
    from plotly.utils import PlotlyJSONEncoder
    from research.app import build_app
    frame = _inputs()
    monkeypatch.setattr(rg, "load_wide", lambda tickers, **kw: frame.select("ts", *tickers))
    monkeypatch.setattr(rg, "load_futures_wide", lambda generics, **kw: frame.select("ts", *generics))
    monkeypatch.setattr(rg, "load_swaption_wide", lambda points, **kw: frame.select("ts", rg.IMPLIED))
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in build_app().callback_map.values()}
    settings = ("2015-01-01", 5, 25.0, 20, 252, "25-75", 126, "0.3-0")
    idle = callbacks["_regimes"](0, "setup", "All", *settings, None)
    assert idle[0] is idle[1]  # no_update until the tab is opened
    view, shown = callbacks["_regimes"](0, "reg", "All", *settings, None)
    text = json.dumps(view, cls=PlotlyJSONEncoder, ensure_ascii=False)
    assert shown == "All" and text.count("data:image/png;base64,") == 5
    for title in ("Policy cycle", "moved, last 6m", "priced, next 6m", "Rate vol", "Implied rate vol", "1m10y",
                  "Oil sensitivity", "Inflation"):
        assert title in text
    assert "Rates vol" not in text
    # back on the tab with the same window: nothing to redraw; a new window zooms the charts
    assert callbacks["_regimes"](0, "reg", "All", *settings, "All")[0] is idle[0]
    zoomed, shown = callbacks["_regimes"](0, "reg", "1Y", *settings, "All")
    assert shown == "1Y" and json.dumps(zoomed, cls=PlotlyJSONEncoder, ensure_ascii=False) != text
