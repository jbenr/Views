"""Trend-follower (CTA) positioning in Treasuries, replicated from yields: how long they are, where they flip, what they trade next.

Most CTA rate exposure comes from trend rules over a few lookbacks. This
replication scores each lookback L (1, 3, 6 and 12 months) as the yield move
over L days divided by the move ordinary volatility would make over L days
(daily sd x sqrt L), clipped to +-1: a strong trend is a full position, a
weak one a partial. Position is minus the average score -- falling yields
mean long bonds -- from -1 (max short) to +1 (max long). Every input is
known at the close.

Each lookback compares tomorrow's yield with a fixed past day, so the yield
that flips it is known a day ahead: the level L-1 days back. Flow for a
hypothetical move is tomorrow's position at today's yield plus the move,
minus today's position; where flow per bp is large, trend followers add to
the move. The predictive test asks whether that flow, measured as the
5-day change in position, is followed by more of the same move over the next
h days. CFTC leveraged-fund positions are shown as weekly changes only:
their level is dominated by the cash-futures basis trade, not trend.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from utils.helpers import query_db
from utils.market_data import load_wide

TENORS = {"2y": "USGG2YR Index", "5y": "USGG5YR Index", "10y": "USGG10YR Index", "30y": "USGG30YR Index"}
CFTC_CODES = {"2y": "042601", "5y": "044601", "10y": "043602", "30y": "020601"}
LOOKBACKS = (21, 63, 126, 252)
VOL_WINDOW = 63
SHIFTS = (-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30)
HORIZONS = (5, 10, 20)


def load_yield(tenor: str, start: str) -> pl.DataFrame:
    """One generic yield in bp: ts, y."""
    return load_wide({"y": TENORS[tenor]}, start=start, bps_cols="all").drop_nulls().sort("ts")


def positions(frame: pl.DataFrame, lookbacks=LOOKBACKS, vol_window: int = VOL_WINDOW) -> pl.DataFrame:
    """Per-lookback trend scores, the replicated position (+1 = max long bonds) and its 5-day change (flow)."""
    out = frame.with_columns(pl.col("y").diff().rolling_std(vol_window, min_samples=vol_window).alias("vol"))
    out = out.with_columns(*[((pl.col("y") - pl.col("y").shift(L)) / (pl.col("vol") * np.sqrt(L)))
                             .clip(-1, 1).alias(f"s{L}") for L in lookbacks])
    return out.with_columns((-pl.mean_horizontal(*[f"s{L}" for L in lookbacks])).alias("position")
                            ).with_columns(pl.col("position").diff(5).alias("flow5"))


def triggers(frame: pl.DataFrame, lookbacks=LOOKBACKS) -> pl.DataFrame:
    """Today's per-lookback picture: score, and the yields tomorrow that flip it or make it a full position.

    Tomorrow's score for L compares tomorrow's yield with the yield L-1 days
    before today, so that level is the flip; the full-position levels sit
    one sd x sqrt L either side of it.
    """
    y, vol = frame["y"].to_numpy(), float(frame["vol"][-1])
    now = y[-1]
    rows = []
    for L in lookbacks:
        anchor = float(y[-L])  # tomorrow's L-day move is measured from here
        band = vol * np.sqrt(L)
        rows.append({"lookback": f"{L}d", "score": -float(frame[f"s{L}"][-1]), "flip yield": anchor / 100,
                     "distance bp": anchor - now, "distance in daily sd": (anchor - now) / vol,
                     "full long below": (anchor - band) / 100, "full short above": (anchor + band) / 100})
    return pl.DataFrame(rows)


def flow_curve(frame: pl.DataFrame, lookbacks=LOOKBACKS, shifts=SHIFTS) -> pl.DataFrame:
    """Replicated position tomorrow, and the flow from today, if the yield moves by each shift (bp)."""
    y, vol, now = frame["y"].to_numpy(), float(frame["vol"][-1]), float(frame["position"][-1])
    rows = []
    for shift in shifts:
        tomorrow = y[-1] + shift
        scores = [np.clip((tomorrow - y[-L]) / (vol * np.sqrt(L)), -1, 1) for L in lookbacks]
        position = -float(np.mean(scores))
        rows.append({"move bp": shift, "yield": tomorrow / 100, "position": position, "flow": position - now})
    return pl.DataFrame(rows)


def predictive(frame: pl.DataFrame, horizons=HORIZONS) -> pl.DataFrame:
    """Does flow or position precede more of the same move? Correlation with the next h days' bond return.

    Bond return is minus the yield change, so a positive correlation means
    buying by trend followers came before yields fell (follow-through) and a
    negative one means the move reversed. Overlapping windows: treat the
    numbers as descriptive, not as t-stats. Also the move after the top and
    bottom fifth of flow days.
    """
    rows = []
    for h in horizons:
        f = frame.with_columns((pl.col("y") - pl.col("y").shift(-h)).alias("ret")).drop_nulls(["flow5", "ret"])
        hi, lo = f["flow5"].quantile(0.8), f["flow5"].quantile(0.2)
        rows.append({"horizon": f"{h}d", "days": len(f),
                     "corr flow": float(np.corrcoef(f["flow5"], f["ret"])[0, 1]),
                     "corr position": float(np.corrcoef(f["position"], f["ret"])[0, 1]),
                     "bond return after top-fifth buying bp": float(f.filter(pl.col("flow5") >= hi)["ret"].mean()),
                     "after bottom-fifth selling bp": float(f.filter(pl.col("flow5") <= lo)["ret"].mean()),
                     "all days bp": float(f["ret"].mean())})
    return pl.DataFrame(rows)


def load_cftc(tenor: str, start: str) -> pl.DataFrame:
    """Leveraged-fund and asset-manager net futures positions as a share of open interest, and their weekly change."""
    df = pl.from_pandas(query_db(
        "SELECT ts::date AS ts, open_interest_all::float AS oi, "
        "(lev_money_positions_long_all - lev_money_positions_short_all)::float AS lev, "
        "(asset_mgr_positions_long_all - asset_mgr_positions_short_all)::float AS am "
        "FROM md.cftc WHERE cftc_contract_market_code = %s AND futonly_or_combined = 'FutOnly' AND ts >= %s "
        "ORDER BY ts", params=[CFTC_CODES[tenor], start]))
    if df.is_empty():
        return pl.DataFrame(schema={"ts": pl.Date, "lev": pl.Float64, "lev_chg": pl.Float64})
    return df.with_columns(pl.col("ts").cast(pl.Date), (pl.col("lev") / pl.col("oi") * 100).alias("lev_pct"),
                           (pl.col("am") / pl.col("oi") * 100).alias("am_pct"),
                           # a change only between consecutive weekly reports, not across the 2013-25 gap
                           pl.when(pl.col("ts").diff().dt.total_days() <= 8).then(pl.col("lev").diff())
                           .alias("lev_chg"))


def cftc_check(pos: pl.DataFrame, cftc: pl.DataFrame) -> dict:
    """Weekly change in replicated position vs weekly change in leveraged-fund net contracts, on report dates."""
    weekly = (cftc.drop_nulls("lev_chg").join_asof(pos.select("ts", "position").sort("ts"), on="ts")
              .with_columns(pl.col("position").diff().alias("pos_chg")).drop_nulls("pos_chg"))
    corr = float(np.corrcoef(weekly["pos_chg"], weekly["lev_chg"])[0, 1]) if len(weekly) > 10 else None
    return {"weeks": len(weekly), "corr": corr}


def study(tenor: str = "10y", start: str = "2000-01-01") -> dict:
    """Everything the CTA tab shows."""
    pos = positions(load_yield(tenor, start))
    live = pos.drop_nulls("position")
    cftc = load_cftc(tenor, start)
    return dict(tenor=tenor, positions=live, triggers=triggers(live), flow=flow_curve(live),
                predictive=predictive(live), cftc=cftc, cftc_check=cftc_check(live, cftc))
