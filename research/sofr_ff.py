"""SOFR / Fed Funds futures basis: SOFR minus EFFR for the same month, traded against its own range.

One-month SOFR futures (SER) and 30-day Fed Funds futures (FF) both settle
to 100 minus the simple average of a daily overnight fixing over the calendar
month, so for the same delivery month (FF price - SER price) x 100 is the
market's expected average SOFR - EFFR for that month, in bp. Positive means
repo trades above fed funds: collateral heavy, reserves scarce, quarter- and
year-ends, tax dates draining reserves into the TGA. Negative is a cash glut
(2021-22's reverse-repo era). The level drifts with the reserve regime, so
the signal is the basis for a fixed number of months ahead measured against
its own rolling range (z-score), optionally after removing the usual premium
of its delivery month (December's year-end, May's post-tax bill paydowns).
That seasonal premium is point-in-time: each day uses only earlier days.

A trade holds one delivery month's pair to the end: long basis = long FF,
short SER, which gains when SOFR rises against EFFR. FF is $41.67 a bp per
contract and SER $25, so 3 FF : 5 SER is bp-neutral; all P&L here is in bp
of basis per unit.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from utils.helpers import query_db
from utils.market_data import load_wide

MONTH_CODES = "FGHJKMNQUVXZ"
FF_BP_VALUE, SER_BP_VALUE = 41.67, 25.0  # $ per bp per contract; 3 FF : 5 SER is bp-neutral
AHEADS = list(range(1, 8))  # months ahead with both contracts listed (0 = the month being fixed)
LOOKBACKS = [63, 126, 252]
HORIZONS = [5, 10, 20]
SEASON_MIN_OBS = 126  # days of turn-premium history before a delivery month's seasonal is used


def pairs(quotes: pl.DataFrame) -> pl.DataFrame:
    """FF and SER settlements (ts, contract, px, oi) matched by delivery month, one row per (ts, month).

    Contract names carry a 1- or 2-digit year (FFZ6, FFZ19). ``mi`` is the
    delivery month as a single integer (year x 12 + month - 1) so months can
    be stepped and compared; ``ahead`` is how many months out it is on ts.
    """
    parsed = quotes.with_columns(
        pl.col("contract").str.extract(r"^(FF|SER)").alias("root"),
        pl.col("contract").str.extract(r"^(?:FF|SER)([A-Z])").alias("code"),
        pl.col("contract").str.extract(r"([0-9]+)$").alias("yy"),
    ).with_columns(
        pl.when(pl.col("yy").str.len_chars() == 1).then(2020 + pl.col("yy").cast(pl.Int32))
        .otherwise(2000 + pl.col("yy").cast(pl.Int32)).alias("year"),
        pl.col("code").replace_strict({c: i + 1 for i, c in enumerate(MONTH_CODES)}, return_dtype=pl.Int32)
        .alias("month"),
    ).filter(pl.col("px").is_not_null())
    keep = ["ts", "year", "month"]
    ff = parsed.filter(pl.col("root") == "FF").select(*keep, pl.col("px").alias("ff"), pl.col("oi").alias("ff_oi"))
    ser = parsed.filter(pl.col("root") == "SER").select(*keep, pl.col("px").alias("ser"),
                                                        pl.col("oi").alias("ser_oi"))
    return (ff.join(ser, on=keep)
            .with_columns(((pl.col("ff") - pl.col("ser")) * 100).alias("basis"),
                          (pl.col("year") * 12 + pl.col("month") - 1).alias("mi"),
                          ((pl.col("year") - pl.col("ts").dt.year()) * 12
                           + pl.col("month") - pl.col("ts").dt.month()).alias("ahead"))
            .filter(pl.col("ahead") >= 0)
            .sort("ts", "mi"))


def load_pairs(start: str = "2018-01-01") -> pl.DataFrame:
    """Matched FF / SER delivery months from md.fut_eod (SER starts in 2018)."""
    quotes = pl.from_pandas(query_db(
        "SELECT ts, contract, px_last::float AS px, open_interest::float AS oi FROM md.fut_eod "
        "WHERE contract ~ '^(FF|SER)[FGHJKMNQUVXZ][0-9]{1,2}$' AND ts >= %s", params=[start]))
    return pairs(quotes.with_columns(pl.col("ts").cast(pl.Date)))


def with_seasonal(pr: pl.DataFrame, min_obs: int = SEASON_MIN_OBS) -> pl.DataFrame:
    """Add each delivery month's turn premium, its point-in-time seasonal norm, and the adjusted basis.

    turn   = basis - average of the two neighbouring months' basis, same day:
             what this month pays over the months around it.
    season = mean of turn for the same calendar month (all Decembers, ...)
             over strictly earlier days, once ``min_obs`` days exist.
    adj    = basis - season. Null until a month's seasonal exists.
    """
    nb = pr.select("ts", "mi", "basis")
    turn = (pr.select("ts", "mi", "month", "basis")
            .join(nb.rename({"mi": "lo", "basis": "b_lo"}).with_columns(pl.col("lo") + 1),
                  left_on=["ts", "mi"], right_on=["ts", "lo"])
            .join(nb.rename({"mi": "hi", "basis": "b_hi"}).with_columns(pl.col("hi") - 1),
                  left_on=["ts", "mi"], right_on=["ts", "hi"])
            .with_columns((pl.col("basis") - (pl.col("b_lo") + pl.col("b_hi")) / 2).alias("turn")))
    daily = (turn.group_by("ts", "month").agg(pl.col("turn").mean()).sort("ts")
             .with_columns(pl.col("turn").cum_sum().over("month").alias("_sum"),
                           pl.col("turn").cum_count().over("month").alias("_n"))
             # shift one observation within the month: only strictly earlier days count
             .with_columns(pl.col("_sum").shift(1).over("month"), pl.col("_n").shift(1).over("month"))
             .with_columns(pl.when(pl.col("_n") >= min_obs).then(pl.col("_sum") / pl.col("_n")).alias("season"))
             .select("ts", "month", "season"))
    # a month's seasonal on day t is the latest one known by t (it may not have had a fresh turn reading that day)
    season = (pr.select("ts", "month").unique().sort("ts")
              .join_asof(daily.sort("ts"), on="ts", by="month", strategy="backward", check_sortedness=False))
    return (pr.join(turn.select("ts", "mi", "turn"), on=["ts", "mi"], how="left")
            .join(season, on=["ts", "month"], how="left")
            .with_columns((pl.col("basis") - pl.col("season")).alias("adj"))
            .sort("ts", "mi"))


def signal(pr: pl.DataFrame, ahead: int, lookback: int, adjust: bool = True) -> pl.DataFrame:
    """The basis ``ahead`` months out, stitched across month rolls, against its own rolling range.

    value is the seasonally adjusted basis (``adjust``) or the raw basis. z =
    (value - rolling mean) / rolling sd over ``lookback`` days, both known by
    the close. ``mi`` names the delivery month that is ``ahead`` out that day.
    """
    col = "adj" if adjust else "basis"
    s = (pr.filter(pl.col("ahead") == ahead).select("ts", "mi", "basis", pl.col(col).alias("value"))
         .drop_nulls("value").sort("ts"))
    return s.with_columns(pl.col("value").rolling_mean(lookback, min_samples=lookback).alias("mean"),
                          pl.col("value").rolling_std(lookback, min_samples=lookback).alias("sd")
                          ).with_columns(((pl.col("value") - pl.col("mean")) / pl.col("sd")).alias("z"))


def forward_changes(pr: pl.DataFrame, horizons=HORIZONS) -> pl.DataFrame:
    """Each (ts, mi) row's basis change over the next h days of the same contract pair (a label, not a signal)."""
    return pr.sort("mi", "ts").with_columns(
        *[(pl.col("basis").shift(-h).over("mi") - pl.col("basis")).alias(f"fwd{h}") for h in horizons]
    ).sort("ts", "mi")


def forward_table(sig: pl.DataFrame, fwd: pl.DataFrame, horizons=HORIZONS) -> pl.DataFrame:
    """Average forward change of the signalled contract by z bucket: does a wide basis narrow?"""
    joined = sig.drop_nulls("z").join(fwd.select("ts", "mi", *[f"fwd{h}" for h in horizons]), on=["ts", "mi"])
    labels = ["z < -2", "-2 to -1", "-1 to 1", "1 to 2", "z > 2"]
    z = pl.col("z")
    bucket = (pl.when(z < -2).then(pl.lit(labels[0])).when(z < -1).then(pl.lit(labels[1]))
              .when(z <= 1).then(pl.lit(labels[2])).when(z <= 2).then(pl.lit(labels[3])).otherwise(pl.lit(labels[4])))
    out = joined.with_columns(bucket.alias("bucket")).group_by("bucket").agg(
        pl.len().alias("days"), *[pl.col(f"fwd{h}").mean().alias(f"fwd {h}d bp") for h in horizons])
    order = {b: i for i, b in enumerate(labels)}
    return out.sort(pl.col("bucket").replace_strict(order))


def backtest(pr: pl.DataFrame, sig: pl.DataFrame, adjust: bool = True, entry: float = 2.0, exit: float = 0.5, max_hold: int = 40,
             cost: float = 0.5) -> dict:
    """Fade the z-score, holding the signalled delivery month's pair until it reverts, times out or is fixing.

    Decide at the close, trade at the next close. Short basis when z > entry,
    long when z < -entry. The held contract is scored each day against the
    signal's rolling mean and sd (it rolls one month closer while held; the
    basis term structure is shallow, about 0.1bp a month). Exit when |z| is
    back inside ``exit`` or has crossed zero, after ``max_hold`` days, or on
    the first day of the contract's own delivery month. ``cost`` is bp per
    round trip. P&L is in bp of basis per unit.
    """
    stats = sig.select("ts", "mean", "sd", "z", "mi")
    book = pr.select("ts", "mi", "basis", "ahead", (pl.col("season").fill_null(0.0) if adjust else pl.lit(0.0)).alias("season"))
    dates = stats["ts"].to_list()
    z_sig, mi_sig = stats["z"].to_list(), stats["mi"].to_list()
    mean, sd = stats["mean"].to_list(), stats["sd"].to_list()
    lookup = {(r["ts"], r["mi"]): r for r in book.iter_rows(named=True)}

    trades, pnl = [], [0.0] * len(dates)
    pos = None  # dict(mi, side, entry_i, entry_basis, entry_z, last_basis)
    pending = None  # ("enter", mi, side, z) or ("exit", reason)
    for i, day in enumerate(dates):
        if pos is not None:
            row = lookup.get((day, pos["mi"]))
            if row is not None:
                pnl[i] += pos["side"] * (row["basis"] - pos["last"])
                pos["last"] = row["basis"]
            if pending and pending[0] == "exit" and row is not None:
                pnl[i] -= cost
                trades.append(dict(entry=dates[pos["i"]], exit=day, contract=_label(pos["mi"]),
                                   side="long basis" if pos["side"] > 0 else "short basis", entry_z=pos["z"],
                                   entry_bp=pos["entry"], exit_bp=row["basis"], days=i - pos["i"],
                                   pnl_bp=pos["side"] * (row["basis"] - pos["entry"]) - cost, reason=pending[1]))
                pos, pending = None, None
        elif pending and pending[0] == "enter":
            row = lookup.get((day, pending[1]))
            if row is not None and row["ahead"] > 0:
                pos = dict(mi=pending[1], side=pending[2], z=pending[3], i=i, entry=row["basis"], last=row["basis"])
            pending = None

        if pos is None and pending is None:
            z = z_sig[i]
            if z is not None and abs(z) > entry:
                pending = ("enter", mi_sig[i], -1 if z > 0 else 1, z)
        elif pos is not None and pending is None:
            row = lookup.get((day, pos["mi"]))
            held_z = None
            if row is not None and sd[i]:
                value = row["basis"] - row["season"]
                held_z = (value - mean[i]) / sd[i]
            if row is not None and row["ahead"] == 0:
                pending = ("exit", "delivery month")
            elif held_z is not None and (abs(held_z) < exit or np.sign(held_z) == pos["side"]):
                pending = ("exit", "reverted")
            elif i - pos["i"] >= max_hold:
                pending = ("exit", "time stop")

    trades = pl.DataFrame(trades) if trades else pl.DataFrame(schema={
        "entry": pl.Date, "exit": pl.Date, "contract": pl.String, "side": pl.String, "entry_z": pl.Float64,
        "entry_bp": pl.Float64, "exit_bp": pl.Float64, "days": pl.Int64, "pnl_bp": pl.Float64, "reason": pl.String})
    equity = pl.DataFrame({"ts": dates, "pnl": pnl}).with_columns(pl.col("pnl").cum_sum().alias("equity"))
    return dict(trades=trades, equity=equity, summary=summarize(trades))


def summarize(trades: pl.DataFrame) -> dict:
    """Trade count, win rate, average and total bp, worst trade, average days held."""
    if trades.is_empty():
        return dict(trades=0, win_rate=None, avg_bp=None, total_bp=0.0, worst_bp=None, avg_days=None)
    return dict(trades=len(trades), win_rate=float((trades["pnl_bp"] > 0).mean()),
                avg_bp=float(trades["pnl_bp"].mean()), total_bp=float(trades["pnl_bp"].sum()),
                worst_bp=float(trades["pnl_bp"].min()), avg_days=float(trades["days"].mean()))


def risk_shocks(pr: pl.DataFrame, spx: pl.DataFrame, ahead: int, horizons=HORIZONS, z_cut: float = -2.0,
                gap: int = 10) -> pl.DataFrame:
    """What the basis ``ahead`` months out does after an equity sell-off, against all days.

    A shock is a 5-day S&P 500 return below ``z_cut`` standard deviations of
    the trailing year's 5-day returns, at least ``gap`` days after the last
    one. The forward change is of the same contract pair, from the shock's
    close.
    """
    eq = spx.sort("ts").with_columns((pl.col("spx") / pl.col("spx").shift(5) - 1).alias("r5"))
    eq = eq.with_columns((pl.col("r5") / pl.col("r5").rolling_std(252, min_samples=200).shift(1)).alias("rz"))
    raw = eq.filter(pl.col("rz") < z_cut)["ts"].to_list()
    shocks, last = [], None
    for day in raw:
        if last is None or (day - last).days > gap * 7 / 5:
            shocks.append(day)
            last = day
    fwd = forward_changes(pr.select("ts", "mi", "basis"), horizons).join(
        pr.filter(pl.col("ahead") == ahead).select("ts", "mi"), on=["ts", "mi"])
    after = fwd.filter(pl.col("ts").is_in(shocks))
    rows = []
    for h in horizons:
        col = f"fwd{h}"
        base, hit = fwd[col].drop_nulls(), after[col].drop_nulls()
        rows.append({"horizon": f"{h}d", "shocks": len(hit), "after shock bp": hit.mean() if len(hit) else None,
                     "median bp": hit.median() if len(hit) else None,
                     "share wider": float((hit > 0).mean()) if len(hit) else None,
                     "all days bp": base.mean(), "all days sd": base.std()})
    return pl.DataFrame(rows), after.select("ts", "mi", "basis", *[f"fwd{h}" for h in horizons])


def seasonal_table(pr: pl.DataFrame) -> pl.DataFrame:
    """Average turn premium by delivery calendar month over the whole sample (a description, not a signal)."""
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return (pr.drop_nulls("turn").group_by("month")
            .agg(pl.col("turn").mean().alias("turn premium bp"), pl.col("turn").std().alias("sd bp"),
                 pl.col("basis").mean().alias("avg basis bp"), pl.len().alias("obs"))
            .sort("month").with_columns(pl.col("month").replace_strict(dict(enumerate(names, 1))).alias("month")))


def curve_today(pr: pl.DataFrame) -> pl.DataFrame:
    """Today's basis strip: each listed delivery month's basis, turn premium and seasonal norm."""
    last = pr["ts"].max()
    return (pr.filter(pl.col("ts") == last).sort("mi")
            .select(pl.col("mi").map_elements(_label, return_dtype=pl.String).alias("month"), "ahead",
                    pl.col("ff"), pl.col("ser"), pl.col("basis").alias("basis bp"), pl.col("turn").alias("turn bp"),
                    pl.col("season").alias("seasonal bp"), pl.col("adj").alias("adjusted bp")))


def _label(mi: int) -> str:
    return f"{MONTH_CODES[mi % 12]}{mi // 12 % 100:02d}"


def study(start: str = "2018-01-01", ahead: int = 2, lookback: int = 126, adjust: bool = True,
          entry: float = 2.0, exit: float = 0.5, max_hold: int = 40, cost: float = 0.5) -> dict:
    """Everything the SOFR / FF tab shows."""
    pr = with_seasonal(load_pairs(start))
    sig = signal(pr, ahead, lookback, adjust)
    fwd = forward_changes(pr)
    spx = load_wide({"spx": "SPX Index"}, start=start)
    shocks, shock_days = risk_shocks(pr, spx, ahead)
    return dict(pairs=pr, signal=sig, forward=forward_table(sig, fwd), backtest=backtest(pr, sig, adjust, entry, exit,
                max_hold, cost), shocks=shocks, shock_days=shock_days, seasonal=seasonal_table(pr),
                today=curve_today(pr), ahead=ahead, lookback=lookback, adjust=adjust)
