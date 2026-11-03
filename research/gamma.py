"""Dealer gamma in Treasury futures options: where hedging flows push the market, and whether that shows up.

Two measures of the same thing.

From open interest (daily, md.opt_eod): every listed TY option's implied
vol comes from its settlement (Black-76 on its quarterly future; the
out-of-the-money side prices each strike, gaps are interpolated across
strikes), and its gamma follows. Open interest x gamma, summed, is how many
futures the holders' hedges change per point the futures move. Who is
short is not in open interest, so it is signed by a convention: "split"
(dealers long calls and short puts, the usual equity convention) or
"short" (customers own every option, dealers short all of it). Under
either, a dealer short gamma buys rallies and sells sell-offs -- adding to
moves -- and long gamma does the reverse. The profile recomputes gamma as
the futures move (same vols, same expiry), giving the move where dealer
gamma changes sign (the flip) and the strikes carrying the most gamma (pins).
Open interest for day t is published the next morning, so tests start the
day after.

From the CFTC (weekly, back to 2006): futures-and-options positions minus
futures-only positions is each trader group's delta-equivalent options
position. If dealers are long gamma, their options delta rises as yields
fall. A rolling regression of the weekly change in dealers' options delta
on the weekly 10y yield change estimates dealer gamma (minus the slope,
contracts per bp; positive = long gamma), usable from the Friday release.

The tests ask the question the trade depends on: when dealers are short
gamma, is subsequent realized vol higher and do moves follow through?
"""

from __future__ import annotations

import datetime as dt
import re

import numpy as np
import polars as pl
from scipy.stats import norm

from stats import roll_lr
from utils.helpers import query_db
from utils.market_data import load_wide

ROOT = "TY"
CFTC_CODE = "043602"  # UST 10Y note
CONVENTIONS = {"split": "dealers long calls, short puts", "short": "dealers short all options"}
MOVES = np.round(np.arange(-4.0, 4.0001, 0.125), 3)  # futures points
VOL_WINDOW = 20
CFTC_LOOKBACKS = [26, 52, 104]
RELEASE_LAG_DAYS = 3  # Tuesday positions, Friday release


# ---- Black-76 ----------------------------------------------------------------


def _d1(F, K, T, sigma):
    return (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))


def black76(F, K, T, sigma, call):
    """Undiscounted Black-76 price (TY options are short-dated; discounting is a rounding error here)."""
    d1 = _d1(F, K, T, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return np.where(call, F * norm.cdf(d1) - K * norm.cdf(d2), K * norm.cdf(-d2) - F * norm.cdf(-d1))


def gamma76(F, K, T, sigma):
    """Change in delta (futures per option) per 1.0 point move in the future."""
    return norm.pdf(_d1(F, K, T, sigma)) / (F * sigma * np.sqrt(T))


def implied_vol(price, F, K, T, call, lo=0.002, hi=1.0, iters=60):
    """Vectorized bisection; NaN where the price is at or below intrinsic value."""
    price, F, K, T, call = map(np.asarray, (price, F, K, T, call))
    intrinsic = np.where(call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    ok = np.isfinite(price) & (price > intrinsic + 1e-9) & (T > 0)
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    for _ in range(iters):
        mid = (a + b) / 2
        too_high = black76(F, K, np.where(ok, T, 1.0), mid, call) > price
        b = np.where(too_high, mid, b)
        a = np.where(too_high, a, mid)
    return np.where(ok, (a + b) / 2, np.nan)


# ---- data ------------------------------------------------------------------


def future_key(contract: str, ts: dt.date) -> str:
    """md.fut_eod names recent contracts with one year digit (TYU5) and older ones with two (TYU24)."""
    root, code, yy = re.match(r"^([A-Z]+?)([HMUZ])(\d{1,2})$", contract).groups()
    if len(yy) == 1:
        year = ts.year - ts.year % 10 + int(yy)
        if year < ts.year - 1:
            year += 10
        yy = f"{year % 100:02d}"
    return f"{root}{code}{yy}"


def load_futures(start: str, root: str = ROOT) -> pl.DataFrame:
    """Quarterly futures settlements: ts, generic (TY1/TY2), underlying (TYU25), px, volume."""
    df = pl.from_pandas(query_db(
        "SELECT ts, generic_ticker AS generic, contract, px_last::float AS px, volume::float AS volume "
        "FROM md.fut_eod WHERE contract ~ %s AND ts >= %s AND px_last IS NOT NULL",
        params=[f"^{root}[HMUZ][0-9]{{1,2}}$", start])).with_columns(pl.col("ts").cast(pl.Date))
    return df.with_columns(pl.struct("contract", "ts").map_elements(
        lambda r: future_key(r["contract"], r["ts"]), return_dtype=pl.String).alias("underlying")).sort("ts")


def load_options(start: str, root: str = ROOT) -> pl.DataFrame:
    """Option settlements with open interest: ts, underlying, expiry, strike, put_call, settle, oi."""
    return pl.from_pandas(query_db(
        "SELECT ts, underlying, expiry, strike::float AS strike, put_call, settle::float AS settle, "
        "open_interest::float AS oi FROM md.opt_eod WHERE root = %s AND ts >= %s AND open_interest > 0",
        params=[root, start])).with_columns(pl.col("ts").cast(pl.Date), pl.col("expiry").cast(pl.Date))


def with_greeks(opts: pl.DataFrame, futs: pl.DataFrame) -> pl.DataFrame:
    """Each option row with its future's price, years to expiry, implied vol and gamma.

    Each strike is priced off its out-of-the-money side (the in-the-money
    settlement is mostly intrinsic); strikes without a usable price take a
    vol interpolated across the strikes of the same expiry that day.
    """
    px = futs.group_by("ts", "underlying").agg(pl.col("px").last())
    df = (opts.join(px, on=["ts", "underlying"])
          .with_columns(((pl.col("expiry") - pl.col("ts")).dt.total_days() / 365.0).alias("T"))
          .filter(pl.col("T") > 0))
    call = (df["put_call"] == "C").to_numpy()
    F, K = df["px"].to_numpy(), df["strike"].to_numpy()
    otm = np.where(call, K >= F, K < F)
    iv = implied_vol(df["settle"].to_numpy(), F, K, df["T"].to_numpy(), call)
    df = df.with_columns(pl.Series("iv_raw", np.where(otm, iv, np.nan)).fill_nan(None))
    smile = df.group_by("ts", "expiry", "strike").agg(pl.col("iv_raw").mean().alias("iv_k"))
    filled = []
    for (_, _), g in smile.sort("strike").group_by("ts", "expiry", maintain_order=True):
        k, v = g["strike"].to_numpy(), g["iv_k"].to_numpy().astype(float)
        good = np.isfinite(v)
        filled.append(g.with_columns(pl.Series("iv", np.interp(k, k[good], v[good]) if good.any()
                                               else np.full(len(k), np.nan))))
    df = df.join(pl.concat(filled).select("ts", "expiry", "strike", "iv"), on=["ts", "expiry", "strike"])
    df = df.filter(pl.col("iv").is_not_nan())
    return df.with_columns(pl.Series("gamma", gamma76(df["px"].to_numpy(), df["strike"].to_numpy(),
                                                      df["T"].to_numpy(), df["iv"].to_numpy())))


def _signs(put_call: np.ndarray, convention: str) -> np.ndarray:
    """Dealer side per option: +1 long, -1 short."""
    if convention == "split":
        return np.where(put_call == "C", 1.0, -1.0)
    return np.full(len(put_call), -1.0)


def profile(day: pl.DataFrame, moves=MOVES) -> pl.DataFrame:
    """One day's open-interest gamma if every future moved by each amount (points): calls, puts, dealer net.

    Vols and time to expiry are held; only the futures price moves.
    """
    F = day["px"].to_numpy()[None, :] + np.asarray(moves)[:, None]
    g = gamma76(F, day["strike"].to_numpy()[None, :], day["T"].to_numpy()[None, :], day["iv"].to_numpy()[None, :])
    weighted = g * day["oi"].to_numpy()[None, :]
    pc = day["put_call"].to_numpy()
    out = {"move": np.asarray(moves), "calls": weighted[:, pc == "C"].sum(1), "puts": weighted[:, pc == "P"].sum(1)}
    for name in CONVENTIONS:
        out[f"dealer_{name}"] = (weighted * _signs(pc, name)[None, :]).sum(1)
    return pl.DataFrame(out)


def flip(prof: pl.DataFrame, convention: str) -> float | None:
    """The move (points) nearest zero where dealer gamma changes sign, linearly interpolated; None if it never does."""
    x, y = prof["move"].to_numpy(), prof[f"dealer_{convention}"].to_numpy()
    crossings = [x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
                 for i in range(len(x) - 1) if np.sign(y[i]) != np.sign(y[i + 1])]
    return min(crossings, key=abs) if crossings else None


def daily_exposure(greeks: pl.DataFrame, futs: pl.DataFrame) -> pl.DataFrame:
    """Per day: call / put / dealer gamma at today's prices (futures per point), flips, front price and its vol.

    Hedge flow for a 1-sd move is minus dealer gamma x the front future's
    20-day daily sd (points): futures dealers buy (+) or sell (-) to stay
    hedged after a 1-sd rally. It is also shown as a share of the 20-day
    average futures volume (TY1 + TY2).
    """
    front = (futs.filter(pl.col("generic") == f"{ROOT}1").sort("ts")
             .with_columns(pl.when(pl.col("underlying") == pl.col("underlying").shift(1))
                           .then(pl.col("px") - pl.col("px").shift(1)).alias("ret"))
             .with_columns(pl.col("ret").rolling_std(VOL_WINDOW, min_samples=VOL_WINDOW - 5).alias("sd")))
    volume = (futs.group_by("ts").agg(pl.col("volume").sum()).sort("ts")
              .with_columns(pl.col("volume").rolling_mean(VOL_WINDOW, min_samples=5).alias("avg_volume")))
    rows = []
    for (day,), g in greeks.group_by("ts"):
        prof = profile(g)
        now = prof.filter(pl.col("move") == 0).row(0, named=True)
        rows.append({"ts": day, "calls": now["calls"], "puts": now["puts"],
                     **{f"dealer_{c}": now[f"dealer_{c}"] for c in CONVENTIONS},
                     **{f"flip_{c}": flip(prof, c) for c in CONVENTIONS},
                     "options_oi": g["oi"].sum(), "next_expiry": g["expiry"].min()})
    out = (pl.DataFrame(rows).sort("ts")
           .join(front.select("ts", "underlying", "px", "ret", "sd"), on="ts", how="left")
           .join(volume.select("ts", "avg_volume"), on="ts", how="left"))
    return out.with_columns(*[(-pl.col(f"dealer_{c}") * pl.col("sd")).alias(f"flow_{c}") for c in CONVENTIONS]
                            ).with_columns(*[(pl.col(f"flow_{c}") / pl.col("avg_volume")).alias(f"flow_share_{c}")
                                             for c in CONVENTIONS])


def pins(day: pl.DataFrame, expiries: int = 2, top: int = 6) -> pl.DataFrame:
    """For the next expiries: the strikes carrying the most open-interest gamma, with call and put open interest."""
    out = []
    for expiry in sorted(day["expiry"].unique().to_list())[:expiries]:
        g = day.filter(pl.col("expiry") == expiry)
        by_strike = (g.group_by("strike").agg(
            (pl.col("oi") * pl.col("gamma")).sum().alias("gamma x OI"),
            pl.col("oi").filter(pl.col("put_call") == "C").sum().alias("call OI"),
            pl.col("oi").filter(pl.col("put_call") == "P").sum().alias("put OI"))
            .sort("gamma x OI", descending=True).head(top))
        out.append(by_strike.with_columns(pl.lit(str(expiry)).alias("expiry"),
                                          (pl.col("strike") - g["px"][0]).alias("vs future")))
    return pl.concat(out).select("expiry", "strike", "vs future", "gamma x OI", "call OI", "put OI")


def _forward_vol(ret: pl.Series, start: int, length: int) -> pl.Series:
    """sd of returns over [t + start, t + start + length - 1], aligned to t."""
    return pl.Series(ret).rolling_std(length, min_samples=length - 1).shift(-(start + length - 1))


def gamma_tests(daily: pl.DataFrame, convention: str, horizon: int = 5) -> pl.DataFrame:
    """Does dealer gamma (known the morning of t+1) line up with what the future does next?

    Buckets days into thirds of dealer gamma. Realized vol is the sd of
    daily futures moves over t+2..t+1+horizon, next to the sd over the
    ``horizon`` days to t: compare the change, since vol regimes persist
    whatever dealers hold. ``horizon`` is at least 2. Follow-through is the
    correlation of day t+1's move with day t+2's: positive means moves
    extend (what short gamma should do), negative means they reverse.
    """
    col = f"dealer_{convention}"
    d = daily.sort("ts").with_columns(_forward_vol(daily.sort("ts")["ret"], 2, horizon).alias("fwd_vol"),
                                      pl.col("ret").rolling_std(horizon, min_samples=horizon - 1).alias("past_vol"),
                                      pl.col("ret").shift(-1).alias("r1"), pl.col("ret").shift(-2).alias("r2"))
    d = d.drop_nulls([col, "fwd_vol"])
    if len(d) < 30:
        return pl.DataFrame()
    lo, hi = d[col].quantile(1 / 3), d[col].quantile(2 / 3)
    d = d.with_columns(pl.when(pl.col(col) <= lo).then(pl.lit("1 · most short"))
                       .when(pl.col(col) <= hi).then(pl.lit("2 · middle")).otherwise(pl.lit("3 · most long"))
                       .alias("dealer gamma third"))
    rows = []
    for name, g in sorted(d.group_by("dealer gamma third"), key=lambda kv: kv[0][0]):
        pair = g.drop_nulls(["r1", "r2"])
        rows.append({"dealer gamma third": name[0], "days": len(g), "avg dealer gamma": g[col].mean(),
                     f"prior {horizon}d vol (pts/day)": g["past_vol"].mean(),
                     f"next {horizon}d vol (pts/day)": g["fwd_vol"].mean(),
                     "vol change": (g["fwd_vol"] - g["past_vol"]).mean(),
                     "follow-through corr": float(np.corrcoef(pair["r1"], pair["r2"])[0, 1]) if len(pair) > 10
                     else None})
    return pl.DataFrame(rows)


# ---- CFTC --------------------------------------------------------------------


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


def study(start: str = "2025-01-01", convention: str = "split", cftc_lookback: int = 52) -> dict:
    """Everything the Options gamma tab shows."""
    futs = load_futures(start)
    greeks = with_greeks(load_options(start), futs)
    daily = daily_exposure(greeks, futs) if not greeks.is_empty() else pl.DataFrame()
    last = greeks["ts"].max() if not greeks.is_empty() else None
    today = greeks.filter(pl.col("ts") == last) if last else greeks
    yields = load_wide({"y": "USGG10YR Index"}, start="2006-01-01", bps_cols="all").drop_nulls().sort("ts")
    est = cftc_gamma(load_cftc_options("2006-01-01"), yields, cftc_lookback)
    return dict(convention=convention, daily=daily, today=today,
                profile=profile(today) if not today.is_empty() else pl.DataFrame(),
                pins=pins(today) if not today.is_empty() else pl.DataFrame(),
                tests=gamma_tests(daily, convention) if not daily.is_empty() else pl.DataFrame(),
                cftc=est, cftc_tests=cftc_tests(est, yields), cftc_lookback=cftc_lookback)
