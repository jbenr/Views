"""Does 10s20s30s have any conditional relationship with long-end rate vol?

Before designing a dislocation trade we have to establish that there is
something to residualize. A dislocation signal is a regression residual; if
the regression explains nothing, the residual is just the target's own move
wearing a costume, and the vol conditioning is decoration.

Four questions, in order:

  1. coverage   -- how much overlapping sample actually exists?
  2. the fly    -- does 2*20y-10y-30y mean-revert, and around what?
  3. the link   -- do fly and vol co-move in levels? in changes? how stably?
  4. yardstick  -- how does that compare to 10s30s~10Y, a link we know is real?

Nothing here is a trade. It decides whether a trade is worth specifying.

    python -m dig.fly_vs_vol
"""

from __future__ import annotations

import numpy as np
import polars as pl

import utils
from stats import half_life, hurst_exponent, roll_lr, roll_lr_diff
from utils.helpers import query_db
from utils.market_data import coverage_report, load_wide

# ---- config -----------------------------------------------------------------

START = "2020-01-01"

TICKERS = {
    "10y": "USGG10YR Index",  # % -> scaled to bps at load
    "20y": "USGG20YR Index",
    "30y": "USGG30YR Index",
    "10s30s": "USYC1030 Index",  # already bps; yardstick only
}
BPS_COLS = ["10y", "20y", "30y"]

VOL_EXPIRY = "1Mo"  # MOVE-like tenor of the option
VOL_TENOR = 30      # long-end concentration of the fly's risk
LOOKBACK = 126

# ---- helpers ----------------------------------------------------------------


def load_vol(expiry: str = VOL_EXPIRY, tenor: int = VOL_TENOR) -> pl.DataFrame:
    """One ATM normal-vol series from md.swaption_vol, as (ts, vol) in bps/yr."""
    sql = """
        SELECT ts, vol::float AS vol
        FROM md.swaption_vol
        WHERE expiry = %s AND tenor = %s AND strike = 0
        ORDER BY ts
    """
    return pl.from_pandas(query_db(sql, params=[expiry, tenor])).with_columns(
        pl.col("ts").cast(pl.Date)
    )


def load_data(start: str = START) -> pl.DataFrame:
    """Rates panel joined to the vol series, with the fly built from generics."""
    rates = load_wide(TICKERS, start=start, bps_cols=BPS_COLS)
    return (
        rates.join(load_vol(), on="ts", how="left")
        .with_columns(
            (2 * pl.col("20y") - pl.col("10y") - pl.col("30y")).alias("fly")
        )
        .sort("ts")
    )


def describe_fly(data: pl.DataFrame) -> pl.DataFrame:
    """Level statistics and mean-reversion character of the raw fly."""
    fly = data["fly"].drop_nulls()
    return pl.DataFrame(
        {
            "n_obs": [len(fly)],
            "mean": [fly.mean()],
            "std": [fly.std()],
            "min": [fly.min()],
            "max": [fly.max()],
            "half_life": [half_life(fly)],
            "hurst": [hurst_exponent(fly)],
            "daily_chg_std": [fly.diff().std()],
        }
    )


def link_report(data: pl.DataFrame, y: str, x: str) -> pl.DataFrame:
    """Full-sample levels and changes association between two columns."""
    frame = data.select("ts", y, x).drop_nulls()
    dy, dx = frame[y].diff(), frame[x].diff()
    chg = pl.DataFrame({"dy": dy, "dx": dx}).drop_nulls()
    lvl_r = np.corrcoef(frame[y].to_numpy(), frame[x].to_numpy())[0, 1]
    chg_r = np.corrcoef(chg["dy"].to_numpy(), chg["dx"].to_numpy())[0, 1]
    return pl.DataFrame(
        {
            "pair": [f"{y} ~ {x}"],
            "n_obs": [len(frame)],
            "levels_corr": [lvl_r],
            "levels_r2": [lvl_r**2],
            "changes_corr": [chg_r],
            "changes_r2": [chg_r**2],
        }
    )


def rolling_link(data: pl.DataFrame, y: str, x: str, lookback: int = LOOKBACK) -> pl.DataFrame:
    """Distribution of the rolling changes-regression beta and R^2."""
    frame = data.select("ts", y, x).drop_nulls()
    reg = roll_lr_diff(frame[x], frame[y], lookback=lookback)
    r2, beta = reg["r2"].drop_nulls(), reg["beta"].drop_nulls()
    return pl.DataFrame(
        {
            "pair": [f"d{y} ~ d{x}"],
            "n_windows": [len(r2)],
            "r2_median": [r2.median()],
            "r2_p90": [r2.quantile(0.90)],
            "r2_max": [r2.max()],
            "beta_median": [beta.median()],
            "beta_std": [beta.std()],
            "beta_sign_flips": [
                int((beta.sign().diff().drop_nulls() != 0).sum())
            ],
        }
    )


def main() -> dict:
    """Establish whether the fly/vol link is strong enough to residualize."""
    # the panel: rates generics + one ATM vol series, fly built from generics
    data = load_data()
    overlap = data.drop_nulls(subset=["fly", "vol"])

    # question 1: how much joint sample is there really
    cover = coverage_report(data, ["10y", "20y", "30y", "fly", "vol", "10s30s"])

    # question 2: what kind of object is the fly
    fly_stats = describe_fly(overlap)

    # question 3: is there a contemporaneous link at all, levels and changes
    links = pl.concat([
        link_report(overlap, "fly", "vol"),
        link_report(overlap, "10s30s", "10y"),  # yardstick
    ])
    rolls = pl.concat([
        rolling_link(overlap, "fly", "vol"),
        rolling_link(overlap, "10s30s", "10y"),  # yardstick
    ])

    print(f"sample {overlap['ts'][0]} -> {overlap['ts'][-1]}  rows={len(overlap)}")
    print("\ncoverage:")
    utils.pdf(cover)
    print("\nthe fly (2*20y - 10y - 30y), on the overlapping sample:")
    utils.pdf(fly_stats)
    print("\ncontemporaneous link, full sample (fly/vol vs a known-real yardstick):")
    utils.pdf(links)
    print(f"\nrolling {LOOKBACK}d changes regression:")
    utils.pdf(rolls)
    return {
        "data": data, "overlap": overlap, "coverage": cover,
        "fly_stats": fly_stats, "links": links, "rolls": rolls,
    }


if __name__ == "__main__":
    state = main()
