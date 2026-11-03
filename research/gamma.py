"""Options gamma: where TY options hedging sits, where it flips, and whether price respects those levels.

Reads the greek exposure engine's tables (utils/options.py, stored by
data_pull/build_opt_exposure.py from every listed TY option, weeklies
included): the daily levels, the profile across futures moves and the
strike-by-strike exposure. Positioning is raw open interest: calls and puts
are kept apart, and "net" is calls - puts, the book of a holder long the
calls and short the puts. Walls, gamma peaks and max pain do not depend on
that sign; the flip and the hedge flow do.

Open interest for day t is published the next morning, so a level dated t
is tested on what the front future does from the close of t+1.

The level tests ask two questions. Regime: when the future sits where net
gamma is positive (above the flip), are subsequent moves smaller and do
they reverse more than when it sits below? Walls: from below the call wall
(above the put wall), how often does the future close through it within
the horizon, against the same count for a placebo level half a point
further out? A wall that holds should be crossed less often than its
placebo.

The CFTC section is the weekly view back to 2006: futures-and-options
positions minus futures-only positions is each trader group's
delta-equivalent options position; minus the rolling slope of the dealers'
weekly change on the 10y yield change estimates their gamma.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from stats import roll_lr
from utils.helpers import query_db
from utils.market_data import load_wide
from utils.options import FLOW_MOVES, _tag, load_future_px

ROOT = "TY"
CFTC_CODE = "043602"  # UST 10Y note
CFTC_LOOKBACKS = [26, 52, 104]
RELEASE_LAG_DAYS = 3  # Tuesday positions, Friday release
VOL_WINDOW = 20
HORIZON = 5           # trading days, level tests
LADDER_WIDTH = 3.0    # points either side of the future, strike ladder
PLACEBO = 0.5         # points beyond a wall, placebo level


# ---- stored engine output ------------------------------------------------------


def load_levels(start: str, root: str = ROOT) -> pl.DataFrame:
    """md.opt_levels from ``start``."""
    df = pl.from_pandas(query_db("SELECT * FROM md.opt_levels WHERE root = %s AND ts >= %s ORDER BY ts",
                                 params=[root, start]))
    return df.with_columns(pl.col("ts").cast(pl.Date), pl.col("front_expiry").cast(pl.Date)) if not df.is_empty() else df


def load_profile(day: dt.date, root: str = ROOT) -> pl.DataFrame:
    """md.opt_profile for one day."""
    return pl.from_pandas(query_db("SELECT * FROM md.opt_profile WHERE root = %s AND ts = %s ORDER BY move",
                                   params=[root, day]))


def load_ladder(day: dt.date, root: str = ROOT) -> pl.DataFrame:
    """md.opt_strike_exposure for one day."""
    return pl.from_pandas(query_db("SELECT * FROM md.opt_strike_exposure WHERE root = %s AND ts = %s",
                                   params=[root, day])).with_columns(pl.col("expiry").cast(pl.Date))


def load_front(start: str, root: str = ROOT) -> pl.DataFrame:
    """Front future by day: px, daily move within one contract (ret), and 20-day average volume of the first two."""
    futs = load_future_px(root, start)
    front = (futs.filter(pl.col("generic") == f"{root}1").sort("ts")
             .with_columns(pl.when(pl.col("underlying") == pl.col("underlying").shift(1))
                           .then(pl.col("px") - pl.col("px").shift(1)).alias("ret")))
    volume = (futs.group_by("ts").agg(pl.col("volume").sum()).sort("ts")
              .with_columns(pl.col("volume").rolling_mean(VOL_WINDOW, min_samples=5).alias("avg_volume")))
    return front.select("ts", "px", "ret").join(volume.select("ts", "avg_volume"), on="ts", how="left")


# ---- today's views ---------------------------------------------------------------


def strike_ladder(ladder: pl.DataFrame, px: float, width: float = LADDER_WIDTH) -> pl.DataFrame:
    """Gamma x open interest by strike near the future, all expiries, split weekly / standard by side."""
    near = ladder.filter((pl.col("strike") - px).abs() <= width)
    cols = []
    for s in ("call", "put"):
        for kind in ("weekly", "standard"):
            cols.append(pl.col(f"{s}_gamma").filter(pl.col("kind") == kind).sum().alias(f"{s}_{kind}"))
    return near.group_by("strike").agg(*cols).sort("strike")


def positioning(ladder: pl.DataFrame, top: int = 15) -> pl.DataFrame:
    """The strikes where open interest changed most on the day: new positions (+) and closed ones (-)."""
    return (ladder.with_columns((pl.col("d_call_oi").abs() + pl.col("d_put_oi").abs()).alias("size"))
            .sort("size", descending=True).head(top)
            .select(pl.col("expiry").cast(pl.String), "kind", "strike",
                    pl.col("call_oi").alias("call OI"), pl.col("d_call_oi").alias("call OI chg"),
                    pl.col("put_oi").alias("put OI"), pl.col("d_put_oi").alias("put OI chg"),
                    (pl.col("call_gamma") + pl.col("put_gamma")).alias("gamma x OI")))


def hedge_flows(level: dict, avg_volume: float | None) -> pl.DataFrame:
    """Futures the net (calls - puts) holder trades to stay hedged after each move, and as a share of volume."""
    rows = [{"futures move (pts)": m, "hedge flow (futures)": level[f"flow_{_tag(m)}"],
             "share of daily volume": level[f"flow_{_tag(m)}"] / avg_volume if avg_volume else None}
            for m in FLOW_MOVES]
    rows.append({"futures move (pts)": None, "hedge flow (futures)": -level["net_charm"],
                 "share of daily volume": -level["net_charm"] / avg_volume if avg_volume else None})
    return pl.DataFrame(rows).with_columns(pl.Series("move", [f"{m:+g} pts" for m in FLOW_MOVES] + ["1 day (charm)"])
                                           ).select("move", "hedge flow (futures)", "share of daily volume")


# ---- level tests ------------------------------------------------------------------


def _with_outcomes(levels: pl.DataFrame, front: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Levels joined to the front future's path over the ``horizon`` days after the next close."""
    f = front.sort("ts").with_columns(
        pl.col("px").shift(-1).alias("px_from"),  # close of t+1: levels dated t are known that morning
        *[pl.col("ret").shift(-1 - k).alias(f"r{k}") for k in range(1, horizon + 1)],
        *[pl.col("px").shift(-1 - k).alias(f"p{k}") for k in range(1, horizon + 1)])
    rets = [f"r{k}" for k in range(1, horizon + 1)]
    f = f.with_columns(pl.concat_list(rets).list.eval(pl.element().abs()).list.mean().alias("abs_move"),
                       pl.concat_list(rets).list.std().alias("fwd_sd"),
                       pl.col("ret").rolling_std(VOL_WINDOW, min_samples=VOL_WINDOW - 5).alias("sd"))
    return levels.join(f.drop("px"), on="ts", how="inner").drop_nulls(rets)


def regime_tests(levels: pl.DataFrame, front: pl.DataFrame, horizon: int = HORIZON) -> pl.DataFrame:
    """Next-``horizon``-day moves by where the future sits against the flip (in 20-day sd units)."""
    d = _with_outcomes(levels, front, horizon).drop_nulls("sd")
    if len(d) < 30:
        return pl.DataFrame()
    d = d.with_columns(((pl.col("px") - pl.col("flip")) / pl.col("sd")).alias("dist"))
    bucket = (pl.when(pl.col("flip").is_null() & (pl.col("net_gamma") > 0)).then(pl.lit("4 · long, no flip"))
              .when(pl.col("flip").is_null()).then(pl.lit("0 · short, no flip"))
              .when(pl.col("dist") < 0).then(pl.lit("1 · below flip"))
              .when(pl.col("dist") < 2).then(pl.lit("2 · 0-2 sd above"))
              .otherwise(pl.lit("3 · 2+ sd above")))
    d = d.with_columns(bucket.alias("where the future sits"))
    rows = []
    for (name,), g in sorted(d.group_by("where the future sits"), key=lambda kv: kv[0][0]):
        pair = g.select("r1", "r2").drop_nulls()
        rows.append({"where the future sits": name, "days": len(g), "avg net gamma": g["net_gamma"].mean(),
                     "avg |move| / sd": (g["abs_move"] / g["sd"]).mean(),
                     f"next {horizon}d sd / prior 20d sd": (g["fwd_sd"] / g["sd"]).mean(),
                     "follow-through corr": float(np.corrcoef(pair["r1"], pair["r2"])[0, 1]) if len(pair) > 10
                     else None})
    return pl.DataFrame(rows)


def wall_tests(levels: pl.DataFrame, front: pl.DataFrame, horizon: int = HORIZON, placebo: float = PLACEBO) -> pl.DataFrame:
    """How often the future closes through each wall within ``horizon`` days, against a placebo further out."""
    d = _with_outcomes(levels, front, horizon)
    path = pl.concat_list([f"p{k}" for k in range(1, horizon + 1)])
    rows = []
    for wall, above in (("call_wall", True), ("put_wall", False)):
        for label, shift in (("wall", 0.0), (f"placebo {placebo if above else -placebo:+g}",
                                              placebo if above else -placebo)):
            level = pl.col(wall) + shift
            start_side = (pl.col("px_from") < level) if above else (pl.col("px_from") > level)
            crossed = (path.list.max() > level) if above else (path.list.min() < level)
            dist = ((level - pl.col("px_from")).abs() / pl.col("sd"))
            g = d.drop_nulls("sd").filter(start_side).with_columns(crossed.alias("crossed"), dist.alias("dist"))
            near = g.filter(pl.col("dist") <= 1.5)
            rows.append({"level": f"{wall.replace('_', ' ')} · {label}", "days starting inside": len(g),
                         "crossed within horizon": g["crossed"].mean() if len(g) else None,
                         "avg distance (sd)": g["dist"].mean() if len(g) else None,
                         "days within 1.5 sd": len(near),
                         "crossed, within 1.5 sd": near["crossed"].mean() if len(near) else None})
    return pl.DataFrame(rows)


# ---- CFTC ------------------------------------------------------------------------


def _forward_vol(ret: pl.Series, start: int, length: int) -> pl.Series:
    """sd of returns over [t + start, t + start + length - 1], aligned to t."""
    return pl.Series(ret).rolling_std(length, min_samples=length - 1).shift(-(start + length - 1))


def load_cftc_options(start: str, code: str = CFTC_CODE) -> pl.DataFrame:
    """Weekly delta-equivalent options positions by trader group: combined report minus futures-only."""
    groups = {"dealer": "dealer", "asset_mgr": "asset managers", "lev_money": "leveraged funds"}
    nets = ", ".join(f"(c.{g}_positions_long_all - c.{g}_positions_short_all) - "
                     f"(f.{g}_positions_long_all - f.{g}_positions_short_all) AS {g}" for g in groups)
    df = pl.from_pandas(query_db(
        f"SELECT f.ts::date AS ts, {nets}, (c.open_interest_all - f.open_interest_all)::float AS options_oi "
        "FROM md.cftc f JOIN md.cftc_combined c USING (cftc_contract_market_code, ts) "
        "WHERE f.cftc_contract_market_code = %s AND f.ts >= %s ORDER BY f.ts", params=[code, start]))
    return df.with_columns(pl.col("ts").cast(pl.Date), *[pl.col(g).cast(pl.Float64) for g in groups])


def cftc_gamma(cftc: pl.DataFrame, yields: pl.DataFrame, lookback: int = 52) -> pl.DataFrame:
    """Rolling estimate of dealer gamma from the CFTC: minus the slope of weekly options-delta change on yield change.

    Contracts per bp; positive = dealers long gamma. ``available`` is the
    Friday release, when the estimate can first be used. Weeks after a gap
    in the reports are dropped rather than differenced across it.
    """
    w = (cftc.sort("ts").join_asof(yields.select("ts", "y").sort("ts"), on="ts")
         .with_columns(pl.col("dealer").diff().alias("d_delta"), pl.col("y").diff().alias("dy"),
                       pl.col("ts").diff().dt.total_days().alias("gap"))
         .filter(pl.col("gap") <= 8).drop_nulls(["d_delta", "dy"]))
    fit = roll_lr(w["dy"], w["d_delta"], lookback)
    w = w.tail(len(fit)).with_columns((-fit["beta"]).alias("gamma_est"), fit["r2"].alias("r2"))
    return w.with_columns((pl.col("ts") + pl.duration(days=RELEASE_LAG_DAYS)).alias("available"))


def cftc_tests(est: pl.DataFrame, yields: pl.DataFrame, horizon: int = 20) -> pl.DataFrame:
    """Realized 10y yield vol over the next ``horizon`` days after each release, by sign of estimated dealer gamma."""
    y = yields.sort("ts").with_columns(pl.col("y").diff().alias("dy"))
    y = y.with_columns(_forward_vol(y["dy"], 1, horizon).alias("fwd_vol"),
                       pl.col("dy").rolling_std(horizon, min_samples=horizon - 1).alias("past_vol"))
    d = est.sort("available").join_asof(y.select("ts", "fwd_vol", "past_vol").sort("ts"), left_on="available",
                                        right_on="ts", strategy="forward").drop_nulls(["fwd_vol", "gamma_est"])
    rows = []
    for name, cond in (("dealers short gamma", pl.col("gamma_est") < 0), ("dealers long gamma", pl.col("gamma_est") >= 0)):
        g = d.filter(cond)
        rows.append({"estimate": name, "weeks": len(g), f"next {horizon}d vol (bp/day)": g["fwd_vol"].mean(),
                     "prior vol (bp/day)": g["past_vol"].mean(),
                     "vol change": (g["fwd_vol"] - g["past_vol"]).mean()})
    return pl.DataFrame(rows)


# ---- the tab -------------------------------------------------------------------


def study(start: str = "2024-06-01", cftc_lookback: int = 52, horizon: int = HORIZON) -> dict:
    """Everything the Options gamma tab shows."""
    levels = load_levels(start)
    front = load_front((dt.date.fromisoformat(start) - dt.timedelta(days=60)).isoformat())
    out = dict(levels=levels, front=front, horizon=horizon, cftc_lookback=cftc_lookback)
    if not levels.is_empty():
        today = levels.row(-1, named=True)
        ladder = load_ladder(today["ts"])
        vol = front.filter(pl.col("ts") <= today["ts"]).drop_nulls("avg_volume")
        avg_volume = vol["avg_volume"][-1] if not vol.is_empty() else None
        out.update(today=today, profile=load_profile(today["ts"]), ladder=strike_ladder(ladder, today["px"]),
                   positioning=positioning(ladder), flows=hedge_flows(today, avg_volume), avg_volume=avg_volume,
                   regime=regime_tests(levels, front, horizon), walls=wall_tests(levels, front, horizon))
    yields = load_wide({"y": "USGG10YR Index"}, start="2006-01-01", bps_cols="all").drop_nulls().sort("ts")
    est = cftc_gamma(load_cftc_options("2006-01-01"), yields, cftc_lookback)
    out.update(cftc=est, cftc_tests=cftc_tests(est, yields))
    return out
