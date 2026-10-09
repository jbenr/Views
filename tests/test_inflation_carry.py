"""Breakeven carry: accrual and roll-down arithmetic, spike cleaning, and the Inflation tab."""

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from research import inflation_carry as ic


def _market(n=400):
    days = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    curve = {f"is{t}": np.full(n, 200.0 + 2.0 * t) for t in ic.SWAP_TENORS}  # upward-sloping swap curve
    curve["is1"] = np.full(n, 300.0)
    return pl.DataFrame({"ts": days, "be5": np.full(n, 250.0), "be10": np.full(n, 240.0), "be30": np.full(n, 230.0),
                         **curve})


def test_carry_is_accrual_plus_roll_down():
    series = ic.carry(_market(), "3m")
    now = series.row(-1, named=True)
    # accrual: (1y swap - breakeven) x h / N
    assert now["accrual10"] == pytest.approx((300 - 240) * 0.25 / 10)
    # roll: swap(10) - swap(9.75) on the interpolated curve (2bp per year of tenor between 7y and 10y)
    assert now["roll10"] == pytest.approx(2.0 * 0.25)
    assert now["carry10"] == pytest.approx(now["accrual10"] + now["roll10"])
    table = ic.carry_table(series)
    assert table["breakeven"].to_list() == ["5y", "10y", "30y"]


def test_isolated_spikes_go_but_real_moves_stay():
    spike = np.array([250.0, 251.0, -212.0, 252.0, 253.0])
    assert ic.drop_spikes(spike, 40.0).tolist() == [250.0, 251.0, 251.0, 252.0, 253.0]
    collapse = np.array([250.0, 150.0, 60.0, -40.0, -120.0])  # a real fall does not come back
    assert ic.drop_spikes(collapse, 40.0).tolist() == collapse.tolist()
    with_gap = np.array([250.0, np.nan, 100.0, 251.0])  # missing days are skipped, not compared
    assert ic.drop_spikes(with_gap, 40.0)[2] == 250.0


def test_inflation_tab_runs_when_opened(monkeypatch):
    from plotly.utils import PlotlyJSONEncoder
    from research.app import build_app
    market = _market(600)
    monkeypatch.setattr(ic, "load_wide", lambda tickers, **kw: market.select("ts", *tickers))
    monkeypatch.setattr("research.app.load_inflation", lambda start: ic.load_inflation(start))
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in build_app().callback_map.values()}
    idle = callbacks["_inflation"](0, "fv", "tp", "5Y", "3m", "2020-01-01", None)
    assert idle[0] is idle[1]  # not until the Inflation sub-tab is open
    view, shown = callbacks["_inflation"](0, "fv", "infl", "1Y", "1y", "2020-01-01", None)
    text = json.dumps(view, cls=PlotlyJSONEncoder)
    assert shown == "1Y" and text.count("data:image/png;base64,") == 3 and "breakeven carry over 1y" in text.lower()
