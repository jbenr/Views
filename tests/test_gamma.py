"""Options gamma: Black-76 inversion, open-interest gamma profile and flip, CFTC gamma estimate, option tickers."""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from research import gamma as gm


def test_implied_vol_inverts_black76_and_rejects_prices_below_intrinsic():
    F, K, T = np.array([107.0, 107.0, 107.0]), np.array([108.0, 105.0, 110.0]), np.array([0.1, 0.25, 0.05])
    call = np.array([True, False, True])
    sigma = np.array([0.06, 0.08, 0.05])
    price = gm.black76(F, K, T, sigma, call)
    assert gm.implied_vol(price, F, K, T, call) == pytest.approx(sigma, abs=1e-6)
    assert np.isnan(gm.implied_vol(np.array([1.0]), np.array([107.0]), np.array([105.0]), np.array([0.1]),
                                   np.array([True]))[0])  # a call worth less than F - K


def _day(F=107.5):
    """Big call open interest above the future, big put open interest below it."""
    rows = [("C", 110.0, 100_000.0), ("P", 105.0, 100_000.0), ("C", 107.5, 1_000.0), ("P", 107.5, 1_000.0)]
    return pl.DataFrame({"put_call": [r[0] for r in rows], "strike": [r[1] for r in rows], "oi": [r[2] for r in rows],
                         "px": [F] * 4, "T": [0.1] * 4, "iv": [0.06] * 4,
                         "expiry": [dt.date(2026, 11, 20)] * 4})


def test_profile_flips_between_put_and_call_strikes_under_the_split_convention():
    prof = gm.profile(_day())
    at = lambda m: prof.filter(pl.col("move") == m).row(0, named=True)
    assert at(2.5)["dealer_split"] > 0 > at(-2.5)["dealer_split"]  # long call gamma above, short put gamma below
    assert (prof["dealer_short"] < 0).all()  # short everything: never long gamma, no flip
    assert -1.0 < gm.flip(prof, "split") < 1.0  # between the put and call strikes, near today's price
    assert gm.flip(prof, "short") is None
    pins = gm.pins(_day().with_columns(pl.Series("gamma", gm.gamma76(107.5, _day()["strike"].to_numpy(), 0.1, 0.06))))
    assert set(pins["strike"].head(2).to_list()) == {105.0, 110.0}


def test_cftc_gamma_recovers_a_known_slope():
    rng = np.random.default_rng(3)
    n = 200
    days = [dt.date(2015, 1, 6) + dt.timedelta(weeks=i) for i in range(n)]
    dy = rng.normal(0, 8, n)
    y = 300 + np.cumsum(dy)
    dealer = np.cumsum(-1500.0 * np.r_[0, np.diff(y)] + rng.normal(0, 500, n))  # long gamma: delta up as yields fall
    cftc = pl.DataFrame({"ts": days, "dealer": dealer})
    yields = pl.DataFrame({"ts": days, "y": y})
    est = gm.cftc_gamma(cftc, yields, lookback=52)
    assert est["gamma_est"].drop_nulls()[-1] == pytest.approx(1500.0, rel=0.05)
    assert est["available"][0] == est["ts"][0] + dt.timedelta(days=3)


def test_option_ticker_helpers():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pull"))
    pytest.importorskip("blpapi")
    import pull_options as po
    assert po.future_key("TYU5", dt.date(2025, 6, 1)) == "TYU25"
    assert po.future_key("TYZ24", dt.date(2024, 9, 1)) == "TYZ24"
    info = po.parse_ticker("TYF7C 107.5 Comdty", dt.date(2026, 12, 24))  # Jan serial expires in December
    assert info == {"root": "TY", "pc": "C", "strike": 107.5, "underlying": "TYH27"}
    assert po.parse_ticker("TYZ6P  112 Comdty", dt.date(2016, 11, 25))["underlying"] == "TYZ16"
    assert po.parse_ticker("TYZ6P 112 Comdty", dt.date(2025, 11, 21)) is None  # year digit disagrees with expiry
    grid = po.strike_grid(108.3, 111.1)
    assert 108.25 in grid and 111.25 in grid and 106.0 in grid and 106.25 not in grid


def test_gamma_tab_runs_when_opened(monkeypatch):
    from plotly.utils import PlotlyJSONEncoder
    from research.app import build_app
    rng = np.random.default_rng(4)
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(120) if (dt.date(2026, 1, 1) + dt.timedelta(days=i)).weekday() < 5]
    px = 107 + np.cumsum(rng.normal(0, 0.3, len(days)))
    futs = pl.DataFrame({"ts": days, "generic": "TY1", "contract": "TYM6", "px": px, "volume": 1e6,
                         "underlying": "TYM26"})
    rows = []
    for day, f in zip(days, px):
        for k in np.arange(104, 111, 0.5):
            for pc in "CP":
                price = float(gm.black76(f, k, (dt.date(2026, 5, 22) - day).days / 365, 0.06, pc == "C"))
                rows.append((day, "TYM26", dt.date(2026, 5, 22), float(k), pc, price, float(rng.integers(1000, 50000))))
    opts = pl.DataFrame(rows, schema=["ts", "underlying", "expiry", "strike", "put_call", "settle", "oi"], orient="row")
    weeks = [dt.date(2015, 1, 6) + dt.timedelta(weeks=i) for i in range(300)]
    cftc = pl.DataFrame({"ts": weeks, "dealer": np.cumsum(rng.normal(0, 1000, 300)),
                         "asset_mgr": np.zeros(300), "lev_money": np.zeros(300), "options_oi": np.ones(300)})
    yields = pl.DataFrame({"ts": [dt.date(2015, 1, 1) + dt.timedelta(days=i) for i in range(4300)],
                           "y": 300 + np.cumsum(rng.normal(0, 5, 4300))})
    monkeypatch.setattr(gm, "load_futures", lambda start: futs)
    monkeypatch.setattr(gm, "load_options", lambda start: opts)
    monkeypatch.setattr(gm, "load_cftc_options", lambda start: cftc)
    monkeypatch.setattr(gm, "load_wide", lambda tickers, **kw: yields)
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in build_app().callback_map.values()}
    idle = callbacks["_gamma"](0, "idio", "sf", "5Y", "2025-01-01", "split", 52, None)
    assert idle[0] is idle[1]
    view, shown = callbacks["_gamma"](0, "idio", "gamma", "1Y", "2025-01-01", "split", 52, None)
    text = json.dumps(view, cls=PlotlyJSONEncoder)
    assert shown == "1Y" and "failed" not in text and text.count("data:image/png;base64,") == 4
