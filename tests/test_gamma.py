"""Options greek exposure engine (utils/options.py), its build script, the weekly tickers, and the Options gamma tab."""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from research import gamma as gm
from utils import options as op

DATA_PULL = Path(__file__).resolve().parents[1] / "data_pull"


def test_implied_vol_inverts_black76_and_rejects_prices_below_intrinsic():
    F, K, T = np.array([107.0, 107.0, 107.0]), np.array([108.0, 105.0, 110.0]), np.array([0.1, 0.25, 0.05])
    call = np.array([True, False, True])
    sigma = np.array([0.06, 0.08, 0.05])
    price = op.black76(F, K, T, sigma, call)
    assert op.implied_vol(price, F, K, T, call) == pytest.approx(sigma, abs=1e-6)
    assert np.isnan(op.implied_vol(np.array([1.0]), np.array([107.0]), np.array([105.0]), np.array([0.1]),
                                   np.array([True]))[0])  # a call worth less than F - K


@pytest.mark.parametrize("call", [True, False])
def test_greeks_match_finite_differences(call):
    F, K, T, s = 107.0, 108.0, 20 / 365, 0.065
    g = op.greeks(F, K, T, s, call)
    price = lambda f=F, t=T, v=s: float(op.black76(f, K, t, v, call))
    delta = lambda f=F, t=T, v=s: float(op.greeks(f, K, t, v, call)["delta"])
    h = 1e-3
    assert g["delta"] == pytest.approx((price(F + h) - price(F - h)) / (2 * h), rel=1e-4)
    assert g["gamma"] == pytest.approx((delta(F + h) - delta(F - h)) / (2 * h), rel=1e-4)
    assert g["vega"] == pytest.approx((price(v=s + 1e-4) - price(v=s - 1e-4)) / 2e-4 / 100, rel=1e-4)
    assert g["vanna"] == pytest.approx((delta(v=s + 1e-4) - delta(v=s - 1e-4)) / 2e-4 / 100, rel=1e-4)
    assert g["charm"] == pytest.approx((delta(t=T - 1e-5) - delta(t=T + 1e-5)) / 2e-5 / 365, rel=1e-4)  # per day


def _day(F=107.5, ts=dt.date(2026, 11, 2)):
    """Big call open interest above the future, big put open interest below it."""
    rows = [("C", 110.0, 100_000.0), ("P", 105.0, 100_000.0), ("C", 107.5, 1_000.0), ("P", 107.5, 1_000.0)]
    day = pl.DataFrame({"ts": [ts] * 4, "put_call": [r[0] for r in rows], "strike": [r[1] for r in rows],
                        "oi": [r[2] for r in rows], "px": [F] * 4, "T": [0.1] * 4, "iv": [0.06] * 4,
                        "expiry": [dt.date(2026, 11, 20)] * 4, "underlying": ["TYZ26"] * 4,
                        "kind": ["standard", "standard", "weekly", "weekly"]})
    g = op.greeks(day["px"].to_numpy(), day["strike"].to_numpy(), day["T"].to_numpy(), day["iv"].to_numpy(),
                  (day["put_call"] == "C").to_numpy())
    return day.with_columns(*[pl.Series(k, v) for k, v in g.items()])


def test_profile_levels_and_hedge_flows():
    day = _day()
    prof = op.profile(day)
    at = lambda m: prof.filter(pl.col("move") == m).row(0, named=True)
    assert at(2.5)["net_gamma"] > 0 > at(-2.5)["net_gamma"]  # call gamma above, put gamma below
    lv = op.levels(day, prof)
    assert lv["px"] - 3 < lv["flip"] < lv["px"] + 3 and lv["flip"] == pytest.approx(lv["px"] + lv["flip_move"])
    assert (lv["call_wall"], lv["put_wall"]) == (110.0, 105.0)
    assert 105.0 <= lv["max_pain"] <= 110.0
    assert 0 < lv["weekly_gamma_share"] < 0.5
    # long calls up there: a +2 rally adds delta, so the holder sells futures; a -2 fall below the flip, puts short:
    assert lv["flow_up2"] < 0 and lv["flow_dn2"] < 0
    assert op.flip(prof.with_columns(pl.col("call_gamma").alias("net_gamma"))) is None  # never changes sign


def test_strike_exposure_splits_sides_and_counts_open_interest_changes():
    days = [dt.date(2026, 10, 30), dt.date(2026, 11, 2)]
    priced = pl.concat([_day(ts=d).with_columns(pl.lit(None, pl.String).alias("ticker")) for d in days])
    priced = priced.with_columns((pl.col("put_call") + pl.col("strike").cast(pl.String)).alias("ticker"),
                                 pl.when(pl.col("ts") == days[1]).then(pl.col("oi") + 500).otherwise(pl.col("oi"))
                                 .alias("oi"))
    ladder = op.strike_exposure(priced)
    last = ladder.filter((pl.col("ts") == days[1]) & (pl.col("strike") == 107.5)).row(0, named=True)
    assert (last["call_oi"], last["put_oi"], last["d_call_oi"], last["d_put_oi"]) == (1500.0, 1500.0, 500.0, 500.0)
    first = ladder.filter((pl.col("ts") == days[0]) & (pl.col("strike") == 110.0)).row(0, named=True)
    assert first["d_call_oi"] == 100_000.0 and first["put_oi"] == 0.0  # a new listing counts in full


def test_build_script_columns_match_the_engine():
    sys.path.insert(0, str(DATA_PULL))
    pytest.importorskip("psycopg")
    import build_opt_exposure as b
    day = _day()
    lv, prof = op.daily_levels(day)
    assert set(b.columns("opt_levels")) == set(lv.columns) | {"root"}
    assert set(b.columns("opt_profile")) == set(prof.columns) | {"root"}
    ladder = op.strike_exposure(day.with_columns(pl.lit("x").alias("ticker")))
    assert set(b.columns("opt_strike_exposure")) == set(ladder.columns) | {"root"}


def test_option_ticker_helpers():
    sys.path.insert(0, str(DATA_PULL))
    pytest.importorskip("blpapi")
    import pull_options as po
    assert po.future_key("TYU5", dt.date(2025, 6, 1)) == "TYU25"
    info = po.parse_ticker("TYF7C 107.5 Comdty", dt.date(2026, 12, 24))  # Jan serial expires in December
    assert info == {"root": "TY", "pc": "C", "strike": 107.5, "underlying": "TYH27"}
    assert po.parse_ticker("TYZ6P 112 Comdty", dt.date(2025, 11, 21)) is None  # year digit disagrees with expiry
    assert po.weekly_tickers("TY", dt.date(2026, 11, 6), [107.0])[0] == "1MX6C 107 Comdty"      # 1st Friday
    assert po.weekly_tickers("TY", dt.date(2026, 10, 30), [107.5])[1] == "5MV6P 107.5 Comdty"   # 5th Friday
    assert po.weekly_tickers("TY", dt.date(2026, 11, 4), [107.0])[0] == "TYWX26C1 107 Comdty"   # 1st Wednesday
    assert po.weekly_tickers("TY", dt.date(2026, 11, 9), [107.0])[0] == "VBYX26C2 107 Comdty"   # 2nd Monday
    assert po.weekly_tickers("TY", dt.date(2026, 11, 7), [107.0]) == []                        # Saturday
    quarterlies = [(dt.date(2026, 11, 20), "TYZ26"), (dt.date(2027, 2, 19), "TYH27")]
    assert po.weekly_underlying(dt.date(2026, 11, 13), quarterlies) == "TYZ26"
    assert po.weekly_underlying(dt.date(2026, 11, 27), quarterlies) == "TYH27"  # after the Dec options expire
    assert po.weekly_underlying(dt.date(2027, 3, 1), quarterlies) is None


def _synthetic_engine(n_days=160, seed=4):
    """A front future and a strip of options around it, run through the engine as the build script does."""
    rng = np.random.default_rng(seed)
    days = [d for d in (dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(int(n_days * 1.45)))
            if d.weekday() < 5][:n_days]
    px = 107 + np.cumsum(rng.normal(0, 0.3, len(days)))
    futs = pl.DataFrame({"ts": days, "generic": "TY1", "contract": "TYU6", "px": px, "volume": 1e6,
                         "underlying": "TYU26"})
    expiry = dt.date(2026, 8, 21)
    rows = []
    for day, f in zip(days, px):
        for k in np.arange(103, 112, 0.5):
            for pc in "CP":
                price = float(op.black76(f, k, (expiry - day).days / 365, 0.06, pc == "C"))
                rows.append((day, f"TYU6{pc} {k:g}", "standard", "TYU26", expiry, float(k), pc, price,
                             float(rng.integers(1000, 50000)), 100.0))
    opts = pl.DataFrame(rows, schema=["ts", "ticker", "kind", "underlying", "expiry", "strike", "put_call", "settle",
                                      "oi", "volume"], orient="row")
    priced = op.price_options(opts, futs)
    levels, profiles = op.daily_levels(priced)
    return futs, priced, levels, profiles, op.strike_exposure(priced)


def test_level_tests_run_on_engine_output():
    futs, _, levels, _, _ = _synthetic_engine()
    front = futs.select("ts", "px", pl.col("px").diff().alias("ret"), pl.lit(1e6).alias("avg_volume"))
    regime = gm.regime_tests(levels, front, horizon=5)
    assert regime["days"].sum() > 100 and regime["avg |move| / sd"].is_not_null().all()
    walls = gm.wall_tests(levels, front, horizon=5)
    assert walls["level"].to_list()[1] == "call wall · placebo +0.5" and walls["level"].to_list()[3].endswith("-0.5")
    assert walls["crossed within horizon"].drop_nulls().is_between(0, 1).all()
    flows = gm.hedge_flows(levels.row(-1, named=True), 2e6)
    assert len(flows) == len(op.FLOW_MOVES) + 1


def test_gamma_tab_runs_when_opened(monkeypatch):
    from plotly.utils import PlotlyJSONEncoder
    from research.app import build_app
    futs, _, levels, profiles, ladder = _synthetic_engine()
    rng = np.random.default_rng(5)
    front = futs.select("ts", "px", pl.col("px").diff().alias("ret"), pl.lit(1e6).alias("avg_volume"))
    weeks = [dt.date(2015, 1, 6) + dt.timedelta(weeks=i) for i in range(300)]
    cftc = pl.DataFrame({"ts": weeks, "dealer": np.cumsum(rng.normal(0, 1000, 300)),
                         "asset_mgr": np.zeros(300), "lev_money": np.zeros(300), "options_oi": np.ones(300)})
    yields = pl.DataFrame({"ts": [dt.date(2015, 1, 1) + dt.timedelta(days=i) for i in range(4300)],
                           "y": 300 + np.cumsum(rng.normal(0, 5, 4300))})
    monkeypatch.setattr(gm, "load_levels", lambda start: levels)
    monkeypatch.setattr(gm, "load_front", lambda start: front)
    monkeypatch.setattr(gm, "load_profile", lambda day: profiles.filter(pl.col("ts") == day))
    monkeypatch.setattr(gm, "load_ladder", lambda day: ladder.filter(pl.col("ts") == day))
    monkeypatch.setattr(gm, "load_cftc_options", lambda start: cftc)
    monkeypatch.setattr(gm, "load_wide", lambda tickers, **kw: yields)
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in build_app().callback_map.values()}
    idle = callbacks["_gamma"](0, "idio", "sf", "5Y", "2025-01-01", 52, None)
    assert idle[0] is idle[1]
    view, shown = callbacks["_gamma"](0, "idio", "gamma", "1Y", "2025-01-01", 52, None)
    text = json.dumps(view, cls=PlotlyJSONEncoder)
    assert shown == "1Y" and "failed" not in text and text.count("data:image/png;base64,") == 6
    assert "TODAY&#x27;S POSITIONING" in text or "TODAY'S POSITIONING" in text
