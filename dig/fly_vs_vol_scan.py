"""Is the 10s20s30s / rate-vol null robust, or an artifact of one spec?

dig.fly_vs_vol found essentially no link between the fly and 1Mo x 30y ATM
vol. That was a single arbitrary choice of option expiry, swap tenor, and
1-day differencing. Before accepting a null we vary all three, and we also
test the other way vol could matter: not as a co-mover to residualize, but
as a regime that changes how the fly behaves.

  1. surface scan  -- changes R^2 across the whole expiry x tenor vol grid
  2. horizon scan  -- 1d / 5d / 21d changes, in case the link is slow
  3. regime test   -- does the fly mean-revert differently by vol tercile?

    python -m dig.fly_vs_vol_scan
"""

from __future__ import annotations

import numpy as np
import polars as pl

import utils
from stats import half_life, roll_lr_diff
from utils.helpers import query_db

from .fly_vs_vol import load_data

EXPIRIES = ("1Mo", "3Mo", "6Mo", "1Y")
TENORS = (2, 5, 10, 20, 30)
HORIZONS = (1, 5, 21)
LOOKBACK = 126

# ---- helpers ----------------------------------------------------------------


def load_vol_grid(expiries=EXPIRIES, tenors=TENORS) -> pl.DataFrame:
    """Wide (ts, vol_<expiry>_<tenor>) frame of ATM normal vols."""
    sql = """
        SELECT ts, expiry, tenor, vol::float AS vol
        FROM md.swaption_vol
        WHERE expiry = ANY(%s) AND tenor = ANY(%s) AND strike = 0
        ORDER BY ts
    """
    long = pl.from_pandas(query_db(sql, params=[list(expiries), list(tenors)]))
    return (
        long.with_columns(
            pl.col("ts").cast(pl.Date),
            ("vol_" + pl.col("expiry") + "_" + pl.col("tenor").cast(pl.Utf8)).alias("k"),
        )
        .pivot(index="ts", on="k", values="vol")
        .sort("ts")
    )


def changes_r2(y: pl.Series, x: pl.Series, horizon: int) -> tuple[int, float, float]:
    """n, correlation and R^2 of horizon-differenced y on x."""
    frame = pl.DataFrame({"y": y, "x": x}).drop_nulls()
    dy = frame["y"].diff(horizon)
    dx = frame["x"].diff(horizon)
    chg = pl.DataFrame({"dy": dy, "dx": dx}).drop_nulls()
    if len(chg) < 30:
        return len(chg), float("nan"), float("nan")
    r = np.corrcoef(chg["dy"].to_numpy(), chg["dx"].to_numpy())[0, 1]
    return len(chg), float(r), float(r**2)


def surface_scan(data: pl.DataFrame, vol_cols: list[str]) -> pl.DataFrame:
    """Changes R^2 of the fly on every vol point, at every horizon."""
    rows = []
    for col in vol_cols:
        frame = data.select("fly", col).drop_nulls()
        for h in HORIZONS:
            n, r, r2 = changes_r2(frame["fly"], frame[col], h)
            rows.append({"vol_point": col, "horizon_d": h, "n_obs": n,
                         "corr": r, "r2": r2})
    return pl.DataFrame(rows).sort("r2", descending=True)


def rolling_stability(data: pl.DataFrame, vol_col: str) -> pl.DataFrame:
    """Is the daily beta on the best vol point even stable in sign?"""
    frame = data.select("fly", vol_col).drop_nulls()
    reg = roll_lr_diff(frame[vol_col], frame["fly"], lookback=LOOKBACK)
    beta, r2 = reg["beta"].drop_nulls(), reg["r2"].drop_nulls()
    return pl.DataFrame({
        "vol_point": [vol_col],
        "n_windows": [len(beta)],
        "beta_median": [beta.median()],
        "beta_std": [beta.std()],
        "pct_beta_positive": [float((beta > 0).mean())],
        "r2_median": [r2.median()],
    })


def regime_test(data: pl.DataFrame, vol_col: str) -> pl.DataFrame:
    """Does the fly's level, volatility and mean reversion differ by vol tercile?"""
    frame = data.select("ts", "fly", vol_col).drop_nulls()
    q33, q66 = frame[vol_col].quantile(1 / 3), frame[vol_col].quantile(2 / 3)
    frame = frame.with_columns(
        pl.when(pl.col(vol_col) <= q33).then(pl.lit("low"))
        .when(pl.col(vol_col) <= q66).then(pl.lit("mid"))
        .otherwise(pl.lit("high")).alias("regime")
    )
    rows = []
    for regime in ("low", "mid", "high"):
        sub = frame.filter(pl.col("regime") == regime)
        fly = sub["fly"]
        rows.append({
            "regime": regime,
            "n_obs": len(sub),
            f"{vol_col}_mean": sub[vol_col].mean(),
            "fly_mean": fly.mean(),
            "fly_std": fly.std(),
            "fly_daily_chg_std": fly.diff().std(),
            "fly_half_life": half_life(fly),
        })
    return pl.DataFrame(rows)


def main() -> dict:
    # panel: rates + fly from dig.fly_vs_vol, joined to the full ATM vol grid
    data = load_data().drop("vol").join(load_vol_grid(), on="ts", how="inner")
    vol_cols = [c for c in data.columns if c.startswith("vol_")]

    # 1+2: expiry x tenor x horizon scan of the contemporaneous changes link
    surface = surface_scan(data, vol_cols)

    # is even the strongest cell a stable relationship, or a lucky window?
    best = surface.row(0, named=True)["vol_point"]
    stability = pl.concat([rolling_stability(data, best),
                           rolling_stability(data, "vol_1Mo_30")])

    # 3: vol as a regime variable rather than a co-mover
    regimes = regime_test(data, "vol_1Mo_30")

    print(f"sample {data['ts'][0]} -> {data['ts'][-1]}  rows={len(data)}  "
          f"vol points={len(vol_cols)}")
    print("\ntop 10 fly~vol cells by changes R^2 (of "
          f"{len(surface)} expiry x tenor x horizon cells):")
    utils.pdf(surface.head(10))
    print("\nworst 3, for scale:")
    utils.pdf(surface.tail(3))
    print(f"\nrolling {LOOKBACK}d beta stability on the best cell and on 1Mo x 30y:")
    utils.pdf(stability)
    print("\nfly behaviour by 1Mo x 30y vol tercile:")
    utils.pdf(regimes)
    return {"data": data, "surface": surface, "stability": stability,
            "regimes": regimes, "best": best}


if __name__ == "__main__":
    state = main()
