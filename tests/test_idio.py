"""Idio tab: SOFR / FF basis construction, seasonal and backtest timing; CTA trend replication and triggers."""

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from research import cta
from research import sofr_ff as sf


def _quotes(days, basis_fn):
    """FF and SER settlements for months 0-7 ahead; SER is priced so (FF - SER) x 100 = basis_fn(day, year, month)."""
    rows = []
    for day in days:
        for k in range(8):
            mi = day.year * 12 + day.month - 1 + k
            year, month = mi // 12, mi % 12 + 1
            yy = f"{year % 100:02d}" if year < 2025 else f"{year % 10}"
            code = sf.MONTH_CODES[month - 1]
            ff = 98.0
            rows.append((day, f"FF{code}{yy}", ff, 1000.0))
            rows.append((day, f"SER{code}{yy}", ff - basis_fn(day, year, month) / 100, 500.0))
    return pl.DataFrame(rows, schema=["ts", "contract", "px", "oi"], orient="row")


def _weekdays(start, n):
    out, day = [], start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def test_pairs_match_delivery_months_across_year_formats():
    days = [dt.date(2024, 11, 29), dt.date(2025, 6, 2)]
    pr = sf.pairs(_quotes(days, lambda d, y, m: 3.0 if m == 12 else 1.0))
    assert set(pr["ahead"].to_list()) == set(range(8))
    dec24 = pr.filter((pl.col("year") == 2024) & (pl.col("month") == 12))  # FFZ24 / SERZ24
    dec25 = pr.filter((pl.col("year") == 2025) & (pl.col("month") == 12))  # FFZ5 / SERZ5
    assert dec24["basis"].to_list() == pytest.approx([3.0])
    assert dec25["basis"].to_list() == pytest.approx([3.0])  # FF above SER in price = SOFR above EFFR
    assert dec24["ahead"][0] == 1


def test_seasonal_norm_uses_only_earlier_days():
    days = _weekdays(dt.date(2019, 1, 1), 400)
    pr = sf.with_seasonal(sf.pairs(_quotes(days, lambda d, y, m: 2.0 if m == 12 else 0.0)), min_obs=20)
    dec = pr.filter(pl.col("month") == 12).drop_nulls("turn")
    assert dec["turn"].to_list() == pytest.approx([2.0] * len(dec))  # December over its neighbours
    first = pr.filter(pl.col("month") == 12)["ts"].min()
    assert pr.filter((pl.col("month") == 12) & (pl.col("ts") == first))["season"].null_count() == 1
    later = pr.filter((pl.col("month") == 12) & pl.col("season").is_not_null())
    assert later["season"].to_list() == pytest.approx([2.0] * len(later))
    assert later["adj"].to_list() == pytest.approx([0.0] * len(later))


def test_backtest_fades_a_spike_at_the_next_close():
    days = _weekdays(dt.date(2019, 1, 1), 300)
    spike = {days[200]: 4.0, days[201]: 4.0, days[202]: 1.0}
    rng = np.random.default_rng(0)
    noise = {d: float(rng.normal(0, 0.1)) for d in days}
    pr = sf.with_seasonal(sf.pairs(_quotes(days, lambda d, y, m: spike.get(d, 0.0) + noise[d])))
    sig = sf.signal(pr, ahead=3, lookback=63, adjust=False)
    out = sf.backtest(pr, sig, adjust=False, entry=3.0, exit=0.5, max_hold=20, cost=0.25)
    trade = out["trades"].row(0, named=True)
    assert trade["entry"] == days[201]  # signal at day 200's close, filled at day 201's
    assert trade["side"] == "short basis"
    assert trade["pnl_bp"] == pytest.approx(trade["entry_bp"] - trade["exit_bp"] - 0.25)
    assert trade["pnl_bp"] > 2.0
    assert out["equity"]["equity"][-1] == pytest.approx(out["trades"]["pnl_bp"].sum())


def _yields(path):
    days = _weekdays(dt.date(2015, 1, 1), len(path))
    return pl.DataFrame({"ts": days, "y": np.asarray(path, dtype=float)})


def test_cta_rising_yields_mean_max_short_and_flip_levels_are_past_yields():
    rng = np.random.default_rng(1)
    path = 300 + np.cumsum(2.0 + rng.normal(0, 1, 600))  # steady rise in yields
    pos = cta.positions(_yields(path)).drop_nulls("position")
    assert pos["position"][-1] == pytest.approx(-1.0)
    trig = cta.triggers(pos)
    y = pos["y"].to_numpy()
    for L, row in zip(cta.LOOKBACKS, trig.iter_rows(named=True)):
        assert row["flip yield"] == pytest.approx(y[-L] / 100)
    flow = cta.flow_curve(pos)
    assert flow.filter(pl.col("move bp") == 0)["flow"][0] == pytest.approx(0.0, abs=0.05)
    assert flow["position"].is_sorted(descending=True)  # higher yields never mean longer bonds


def test_idio_tabs_run_when_opened(monkeypatch):
    from plotly.utils import PlotlyJSONEncoder
    from research.app import build_app
    days = _weekdays(dt.date(2019, 1, 1), 500)
    rng = np.random.default_rng(2)
    noise = {d: float(rng.normal(0, 0.5)) for d in days}
    quotes = _quotes(days, lambda d, y, m: noise[d] + (1.0 if m == 12 else 0.0))
    monkeypatch.setattr(sf, "load_pairs", lambda start: sf.pairs(quotes))
    spx = pl.DataFrame({"ts": days, "spx": 3000 + np.cumsum(rng.normal(0, 30, len(days)))})
    monkeypatch.setattr(sf, "load_wide", lambda tickers, **kw: spx)
    monkeypatch.setattr(cta, "load_yield", lambda tenor, start: _yields(400 + np.cumsum(rng.normal(0, 5, 800))))
    monkeypatch.setattr(cta, "load_cftc", lambda tenor, start: pl.DataFrame(
        schema={"ts": pl.Date, "lev": pl.Float64, "lev_chg": pl.Float64}))
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in build_app().callback_map.values()}
    idle = callbacks["_sofr_ff"](0, "fv", "sf", "5Y", 2, 63, "adj", 2.0, 0.5, 40, 0.5, None)
    assert idle[0] is idle[1]  # not until the Idio tab is open
    view, shown = callbacks["_sofr_ff"](0, "idio", "sf", "1Y", 2, 63, "raw", 1.5, 0.5, 40, 0.5, None)
    text = json.dumps(view, cls=PlotlyJSONEncoder)
    assert shown == "1Y" and text.count("data:image/png;base64,") == 3 and "failed" not in text
    view, shown = callbacks["_cta"](0, "idio", "cta", "5Y", "10y", "2015-01-01", None)
    text = json.dumps(view, cls=PlotlyJSONEncoder)
    assert shown == "5Y" and text.count("data:image/png;base64,") == 2 and "failed" not in text
