"""Greek exposure engine for options on futures: greeks per option, open-interest exposure per strike, hedging levels.

Turns `md.opt_eod` (settlement, open interest, volume per listed option,
weeklies included) + `md.fut_eod` into three things, each stored by
`data_pull/build_opt_exposure.py`:

    priced = price_options(load_option_eod("TY", start), load_future_px("TY", start))
    ladder = strike_exposure(priced)                  # md.opt_strike_exposure
    prof   = profile(priced_day)                      # md.opt_profile
    lv     = levels(priced_day, prof)                 # md.opt_levels

Greeks. Black-76 on each option's own future, undiscounted (the options are
short-dated). Implied vol comes from the settlement on the out-of-the-money
side of each strike (the in-the-money settlement is mostly intrinsic);
strikes without a usable price take a vol interpolated across the expiry's
strikes that day. Units, per option, in futures (one option on one future):

    delta   futures per option
    gamma   change in delta per 1.0 point move in the future
    vega    price change (points) per 1 vol point
    vanna   change in delta per 1 vol point
    charm   change in delta from one calendar day passing, price unchanged

Exposure is the greek x open interest, kept separately for calls and puts,
so it is raw positioning: open interest does not say who is long. Anything
signed as a hedger's position uses NET = calls - puts (the holder of the
hedge is long calls and short puts: customers sell calls and buy puts).
Flip the sign of a level's inputs to read it the other way; walls, gamma
peaks and max pain do not depend on the sign.

Levels, per day: net and gross gamma, the flip (the futures move where net
gamma changes sign, revaluing every option with vols held), the call and put
walls (strikes with the most call / put gamma x open interest), the gamma
peak, the front expiry's max pain, and the hedge flow for moves of the
future: the futures the NET holder trades to stay delta-neutral (minus the
change in its delta, revalued), plus a day's charm and a vol point's vanna.
Open interest for day t is published the next morning.
"""

from __future__ import annotations

import datetime as dt
import re

import numpy as np
import polars as pl
from scipy.stats import norm

from utils.helpers import query_db

MOVES = np.round(np.arange(-4.0, 4.0001, 0.125), 3)   # futures points, the profile grid
FLOW_MOVES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)          # futures points, hedge flow
SIDES = ("call", "put")
GREEKS = ("delta", "gamma", "vega", "vanna", "charm")


# ---- Black-76 ----------------------------------------------------------------


def _d1(F, K, T, sigma):
    return (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))


def black76(F, K, T, sigma, call):
    """Undiscounted Black-76 price."""
    d1 = _d1(F, K, T, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return np.where(call, F * norm.cdf(d1) - K * norm.cdf(d2), K * norm.cdf(-d2) - F * norm.cdf(-d1))


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


def greeks(F, K, T, sigma, call) -> dict[str, np.ndarray]:
    """Delta, gamma, vega, vanna and charm per option, in the units of the module docstring."""
    d1 = _d1(F, K, T, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    pdf = norm.pdf(d1)
    return {"delta": np.where(call, norm.cdf(d1), norm.cdf(d1) - 1.0),
            "gamma": pdf / (F * sigma * np.sqrt(T)),
            "vega": F * pdf * np.sqrt(T) / 100.0,
            "vanna": -pdf * d2 / sigma / 100.0,
            "charm": pdf * d2 / (2.0 * T) / 365.0}


# ---- data --------------------------------------------------------------------


def future_key(contract: str, ts: dt.date) -> str:
    """md.fut_eod names recent contracts with one year digit (TYU5) and older ones with two (TYU24)."""
    root, code, yy = re.match(r"^([A-Z]+?)([HMUZ])(\d{1,2})$", contract).groups()
    if len(yy) == 1:
        year = ts.year - ts.year % 10 + int(yy)
        if year < ts.year - 1:
            year += 10
        yy = f"{year % 100:02d}"
    return f"{root}{code}{yy}"


def load_future_px(root: str, start: str, end: str | None = None) -> pl.DataFrame:
    """Quarterly futures settlements: ts, generic (TY1/TY2), underlying (TYU25), px, volume."""
    df = pl.from_pandas(query_db(
        "SELECT ts, generic_ticker AS generic, contract, px_last::float AS px, volume::float AS volume "
        "FROM md.fut_eod WHERE contract ~ %s AND ts >= %s AND ts <= %s AND px_last IS NOT NULL",
        params=[f"^{root}[HMUZ][0-9]{{1,2}}$", start, end or "2100-01-01"])).with_columns(pl.col("ts").cast(pl.Date))
    return df.with_columns(pl.struct("contract", "ts").map_elements(
        lambda r: future_key(r["contract"], r["ts"]), return_dtype=pl.String).alias("underlying")).sort("ts")


def load_option_eod(root: str, start: str, end: str | None = None) -> pl.DataFrame:
    """Every option with open interest: ts, ticker, kind, underlying, expiry, strike, put_call, settle, oi, volume."""
    return pl.from_pandas(query_db(
        "SELECT ts, ticker, kind, underlying, expiry, strike::float AS strike, put_call, settle::float AS settle, "
        "open_interest::float AS oi, volume::float AS volume FROM md.opt_eod "
        "WHERE root = %s AND ts >= %s AND ts <= %s AND open_interest > 0",
        params=[root, start, end or "2100-01-01"])).with_columns(pl.col("ts").cast(pl.Date),
                                                                  pl.col("expiry").cast(pl.Date))


# ---- per option ----------------------------------------------------------------


def price_options(opts: pl.DataFrame, futs: pl.DataFrame) -> pl.DataFrame:
    """Each option with its future's price (px), years to expiry (T), implied vol and greeks."""
    px = futs.group_by("ts", "underlying").agg(pl.col("px").last())
    df = (opts.join(px, on=["ts", "underlying"])
          .with_columns(((pl.col("expiry") - pl.col("ts")).dt.total_days() / 365.0).alias("T"))
          .filter(pl.col("T") > 0))
    if df.is_empty():
        return df
    call = (df["put_call"] == "C").to_numpy()
    F, K = df["px"].to_numpy(), df["strike"].to_numpy()
    otm = np.where(call, K >= F, K < F)
    iv = implied_vol(df["settle"].to_numpy(), F, K, df["T"].to_numpy(), call)
    df = df.with_columns(pl.Series("iv_raw", np.where(otm, iv, np.nan)).fill_nan(None))
    smile = df.group_by("ts", "expiry", "strike").agg(pl.col("iv_raw").mean().alias("iv_k"))
    filled = []
    for _, g in smile.sort("strike").group_by("ts", "expiry", maintain_order=True):
        k, v = g["strike"].to_numpy(), g["iv_k"].to_numpy().astype(float)
        good = np.isfinite(v)
        filled.append(g.with_columns(pl.Series("iv", np.interp(k, k[good], v[good]) if good.any()
                                               else np.full(len(k), np.nan))))
    df = (df.join(pl.concat(filled).select("ts", "expiry", "strike", "iv"), on=["ts", "expiry", "strike"])
          .filter(pl.col("iv").is_not_nan()).drop("iv_raw"))
    g = greeks(df["px"].to_numpy(), df["strike"].to_numpy(), df["T"].to_numpy(), df["iv"].to_numpy(),
               (df["put_call"] == "C").to_numpy())
    return df.with_columns(*[pl.Series(name, values) for name, values in g.items()])


# ---- per strike ----------------------------------------------------------------


def strike_exposure(priced: pl.DataFrame) -> pl.DataFrame:
    """Per day, expiry and strike: call / put open interest, its change from the prior day, and greek x OI.

    The change is against the same option's previous stored day, so the
    first day of a listing counts its whole open interest as new.
    """
    side = pl.when(pl.col("put_call") == "C").then(pl.lit("call")).otherwise(pl.lit("put"))
    rows = priced.sort("ts").with_columns(side.alias("side"),
                                          (pl.col("oi") - pl.col("oi").shift(1).over("ticker")).alias("d_oi"))
    rows = rows.with_columns(pl.col("d_oi").fill_null(pl.col("oi")),
                             *[(pl.col(g) * pl.col("oi")).alias(f"{g}_x") for g in GREEKS])
    out = None
    for s in SIDES:
        part = (rows.filter(pl.col("side") == s)
                .group_by("ts", "expiry", "strike")
                .agg(pl.col("oi").sum().alias(f"{s}_oi"), pl.col("d_oi").sum().alias(f"d_{s}_oi"),
                     *[pl.col(f"{g}_x").sum().alias(f"{s}_{g}") for g in GREEKS],
                     pl.col("kind").first().alias(f"kind_{s}"), pl.col("underlying").first().alias(f"und_{s}"),
                     pl.col("px").first().alias(f"px_{s}")))
        out = part if out is None else out.join(part, on=["ts", "expiry", "strike"], how="full", coalesce=True)
    out = out.with_columns(pl.coalesce("kind_call", "kind_put").alias("kind"),
                           pl.coalesce("und_call", "und_put").alias("underlying"),
                           pl.coalesce("px_call", "px_put").alias("fut_px"))
    num = [c for c in out.columns if c.startswith(("call_", "put_", "d_"))]
    return (out.with_columns(pl.col(num).fill_null(0.0))
            .select("ts", "expiry", "strike", "kind", "underlying", "fut_px", *sorted(num)).sort("ts", "expiry", "strike"))


# ---- per day ---------------------------------------------------------------------


def front_px(day: pl.DataFrame) -> tuple[str, float]:
    """The front quarterly future among the day's underlyings and its price."""
    und = day.select("underlying", "px").unique("underlying")
    order = und.with_columns((pl.col("underlying").str.slice(-2).cast(pl.Int32) * 4
                              + pl.col("underlying").str.slice(-3, 1).replace({"H": "0", "M": "1", "U": "2", "Z": "3"})
                              .cast(pl.Int32)).alias("k")).sort("k")
    return order["underlying"][0], order["px"][0]


def profile(day: pl.DataFrame, moves=MOVES) -> pl.DataFrame:
    """One day's call and put delta and gamma x open interest if every future moved by each amount (points).

    Vols and expiries held; each option revalued on its own future plus the move.
    """
    m = np.asarray(moves)[:, None]
    F = day["px"].to_numpy()[None, :] + m
    K, T, iv = (day[c].to_numpy()[None, :] for c in ("strike", "T", "iv"))
    call = (day["put_call"] == "C").to_numpy()
    g = greeks(F, K, T, iv, call[None, :])
    oi = day["oi"].to_numpy()[None, :]
    out = {"move": np.asarray(moves)}
    for s, mask in (("call", call), ("put", ~call)):
        out[f"{s}_gamma"] = (g["gamma"] * oi)[:, mask].sum(1)
        out[f"{s}_delta"] = (g["delta"] * oi)[:, mask].sum(1)
    out = pl.DataFrame(out)
    return out.with_columns((pl.col("call_gamma") - pl.col("put_gamma")).alias("net_gamma"),
                            (pl.col("call_delta") - pl.col("put_delta")).alias("net_delta"))


def flip(prof: pl.DataFrame, col: str = "net_gamma") -> float | None:
    """The move (points) nearest zero where ``col`` changes sign, linearly interpolated; None if it never does."""
    x, y = prof["move"].to_numpy(), prof[col].to_numpy()
    crossings = [x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
                 for i in range(len(x) - 1) if np.sign(y[i]) != np.sign(y[i + 1])]
    return float(min(crossings, key=abs)) if crossings else None


def max_pain(day: pl.DataFrame) -> float | None:
    """The front expiry's settlement price, among its strikes, that pays option holders least."""
    front = day.filter(pl.col("expiry") == day["expiry"].min())
    if front.is_empty():
        return None
    K, oi, call = front["strike"].to_numpy(), front["oi"].to_numpy(), (front["put_call"] == "C").to_numpy()
    S = np.unique(K)[:, None]
    paid = (oi * np.where(call, np.maximum(S - K, 0), np.maximum(K - S, 0))).sum(1)
    return float(S[np.argmin(paid), 0])


def levels(day: pl.DataFrame, prof: pl.DataFrame) -> dict:
    """The day's hedging levels (see the module docstring). Strikes and flip are in front-future prices."""
    und, px = front_px(day)
    at = prof.filter(pl.col("move") == 0).row(0, named=True)
    by_strike = (day.with_columns((pl.col("gamma") * pl.col("oi")).alias("gx"))
                 .group_by("strike").agg(pl.col("gx").filter(pl.col("put_call") == "C").sum().alias("cg"),
                                         pl.col("gx").filter(pl.col("put_call") == "P").sum().alias("pg"))
                 .with_columns((pl.col("cg") + pl.col("pg")).alias("gg")))
    top = lambda col: float(by_strike.sort(col, descending=True)["strike"][0])
    f = flip(prof)
    sign = np.where(day["put_call"] == "C", 1.0, -1.0)
    out = {"underlying": und, "px": px, "options_oi": float(day["oi"].sum()),
           "call_gamma": at["call_gamma"], "put_gamma": at["put_gamma"], "net_gamma": at["net_gamma"],
           "gross_gamma": at["call_gamma"] + at["put_gamma"],
           "weekly_gamma_share": float((day.filter(pl.col("kind") == "weekly")["gamma"]
                                        * day.filter(pl.col("kind") == "weekly")["oi"]).sum())
                                 / (at["call_gamma"] + at["put_gamma"]) if "kind" in day.columns else None,
           "flip_move": f, "flip": px + f if f is not None else None,
           "call_wall": top("cg"), "put_wall": top("pg"), "gamma_peak": top("gg"),
           "front_expiry": day["expiry"].min(), "max_pain": max_pain(day),
           "net_charm": float((sign * day["charm"] * day["oi"]).sum()),
           "net_vanna": float((sign * day["vanna"] * day["oi"]).sum())}
    base = at["net_delta"]
    for m in FLOW_MOVES:
        row = prof.filter(pl.col("move") == m)
        out[f"flow_{_tag(m)}"] = -(row["net_delta"][0] - base) if not row.is_empty() else None
    return out


def _tag(m: float) -> str:
    """Column-safe name for a move: -0.5 -> dn0p5, 1.0 -> up1."""
    return ("up" if m > 0 else "dn") + f"{abs(m):g}".replace(".", "p")


def daily_levels(priced: pl.DataFrame, moves=MOVES) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Levels and profiles for every day in ``priced``: (levels, profiles)."""
    lv, profs = [], []
    for (day,), g in priced.sort("ts").group_by("ts", maintain_order=True):
        prof = profile(g, moves)
        lv.append({"ts": day, **levels(g, prof)})
        profs.append(prof.with_columns(pl.lit(day).alias("ts")))
    if not lv:
        return pl.DataFrame(), pl.DataFrame()
    return pl.DataFrame(lv, infer_schema_length=None), pl.concat(profs).select("ts", pl.exclude("ts"))
