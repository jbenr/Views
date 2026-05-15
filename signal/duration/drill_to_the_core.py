"""Drill to the core — recursive regression diagnostics for rates research.

Pattern: pick (x, y). Rolling-regress y on x. Inspect the residual. Peel a
known factor z out of it. Repeat. The story of the analysis lives in main();
the helpers above are reusable building blocks.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import psycopg
from IPython.display import display
from statsmodels.tsa.stattools import acf, adfuller

_HERE = Path(__file__).resolve().parent
for _p in [_HERE, *_HERE.parents]:
    if (_p / "stats").exists():
        ROOT = _p
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stats import roll_lr, half_life
from utils.viz import Viz

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)


# ─── config ────────────────────────────────────────────────────────────────

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
START  = "2000-01-01"

TICKERS = {
    "2y":   "USGG2YR Index",
    "5y":   "USGG5YR Index",
    "10y":  "USGG10YR Index",
    "30y":  "USGG30YR Index",
    "be5":  "USGGBE05 Index",
    "be10": "USGGBE10 Index",
    "spx":  "SPX Index",
    "oil":  "CO1 Comdty",
    "dxy":  "DXY Curncy",
    "move": "MOVE Index",
}

BPS_COLS = ("2y", "5y", "10y", "30y", "be5", "be10")


# ─── data ──────────────────────────────────────────────────────────────────

def _query_to_pl(sql: str) -> pl.DataFrame:
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)})


def load_basket(tickers: dict[str, str] = TICKERS, start: str = START) -> pd.DataFrame:
    """Pull `tickers` from md.index_eod, pivot wide. Yields scaled to bps."""
    tlist = ", ".join(f"'{t}'" for t in tickers.values())
    raw = _query_to_pl(f"""
        SELECT ts, ticker, px_last::float AS px
        FROM md.index_eod
        WHERE ticker IN ({tlist}) AND ts >= '{start}'
        ORDER BY ts
    """)
    wide = (raw.to_pandas()
               .pivot(index="ts", columns="ticker", values="px")
               .sort_index()
               .rename(columns={v: k for k, v in tickers.items()}))
    for c in BPS_COLS:
        if c in wide:
            wide[c] = wide[c] * 100.0
    return wide.dropna(how="all")


# ─── residual diagnostics ──────────────────────────────────────────────────

def residual_fingerprint(resid: pd.Series) -> dict:
    """Compact stats describing what a residual looks like."""
    r = resid.dropna()
    if len(r) < 60:
        return {"n": len(r)}
    ac = acf(r, nlags=5, fft=True)
    try:
        adf_p = float(adfuller(r, autolag="AIC")[1])
    except Exception:
        adf_p = np.nan
    return {
        "n":         len(r),
        "std":       float(r.std()),
        "kurt":      float(r.kurt()),
        "skew":      float(r.skew()),
        "acf_lag1":  float(ac[1]),
        "acf_lag5":  float(ac[5]),
        "half_life": float(half_life(r)),
        "adf_p":     adf_p,
    }


def _print_fingerprint(label: str, fp: dict) -> None:
    print(f"residual fingerprint  ({label}):")
    for k, val in fp.items():
        if isinstance(val, float):
            print(f"  {k:<10s}  {val:+.3f}")
        else:
            print(f"  {k:<10s}  {val}")


def drill(
    x: pd.Series,
    y: pd.Series,
    *,
    lookback: int = 60,
    x_name: str = "x",
    y_name: str = "y",
    viz: Viz | None = None,
) -> pd.DataFrame:
    """Rolling OLS of y on x. Plots inputs, rolling β/R², and the residual.

    Returns pandas DataFrame indexed by date with columns
    [x, y, alpha, beta, yhat, resid, r2].
    """
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    res = roll_lr(pair["x"], pair["y"], lookback=lookback).to_pandas()
    res.index = pair.index

    if viz is not None:
        viz.line(pair.rename(columns={"x": x_name, "y": y_name}),
                 title=f"{y_name}  vs  {x_name}", yaxis_title="level")
        viz.line(res[["beta", "r2"]], left=["r2"],
                 title=f"rolling β and R²  (lookback={lookback})",
                 yaxis_title="β", yaxis_right_title="R²")
        viz.line(res[["resid"]].rename(columns={"resid": f"resid({y_name} | {x_name})"}),
                 title=f"residual:  {y_name} − ({lookback}d β·{x_name} + α)",
                 yaxis_title="resid",
                 hlines=[(0, None, "solid")])

    _print_fingerprint(f"{y_name} | {x_name}", residual_fingerprint(res["resid"]))
    return res


def corr_scan(resid: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    """Corr of resid against each candidate column (level and change). Sorted by |corr_level|."""
    rows = []
    for col in candidates.columns:
        joint = pd.concat([resid.rename("r"), candidates[col].rename("z")], axis=1).dropna()
        if len(joint) < 100:
            continue
        rows.append({
            "candidate":   col,
            "corr_level":  joint["r"].corr(joint["z"]),
            "corr_change": joint["r"].diff().corr(joint["z"].diff()),
            "n":           len(joint),
        })
    return (pd.DataFrame(rows)
              .assign(abs_corr=lambda d: d["corr_level"].abs())
              .sort_values("abs_corr", ascending=False)
              .drop(columns="abs_corr")
              .reset_index(drop=True))


def predict_scan(
    resid: pd.Series,
    candidates: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 20, 60),
) -> pd.DataFrame:
    """Spearman IC of resid_t vs fwd_change(candidate, h) for each h. Sorted by |ic|."""
    rows = []
    for col in candidates.columns:
        for h in horizons:
            fwd = candidates[col].diff(h).shift(-h)
            joint = pd.concat([resid.rename("r"), fwd.rename("f")], axis=1).dropna()
            if len(joint) < 100:
                continue
            rows.append({
                "target":  col,
                "horizon": h,
                "ic":      joint["r"].corr(joint["f"], method="spearman"),
                "n":       len(joint),
            })
    return (pd.DataFrame(rows)
              .assign(abs_ic=lambda d: d["ic"].abs())
              .sort_values("abs_ic", ascending=False)
              .drop(columns="abs_ic")
              .reset_index(drop=True))


def peel(
    resid: pd.Series,
    z: pd.Series,
    *,
    lookback: int = 60,
    resid_name: str = "resid",
    z_name: str = "z",
    viz: Viz | None = None,
) -> pd.DataFrame:
    """Strip the part of `resid` explained by `z`. Returns drill output for the new layer."""
    return drill(z, resid, lookback=lookback, x_name=z_name, y_name=resid_name, viz=viz)


# ─── main ──────────────────────────────────────────────────────────────────

def main() -> dict:
    viz = Viz(backend='plotly')

    # 1. data — wide pandas frame of all TICKERS, yields in bps
    df = load_basket()
    print(f"basket: {len(df)} obs · {df.index.min().date()} → {df.index.max().date()}")
    print(f"        cols: {list(df.columns)}")

    # 2. eyeball — UST yield curve
    yld_cols = [c for c in ("2y", "5y", "10y", "30y") if c in df]
    viz.line(df[yld_cols], title="UST yields", yaxis_title="yield, bps")

    # 3. drill — how much of 10y is just 2y?
    out_10_2   = drill(df["2y"], df["10y"], lookback=60,
                        x_name="2y", y_name="10y", viz=viz)
    resid_10_2 = out_10_2["resid"]

    # 4. interrogate the residual — who moves with it, who does it predict?
    others = df.drop(columns=["10y", "2y"])
    corr   = corr_scan(resid_10_2, others)
    pred   = predict_scan(resid_10_2, others)
    print("\n--- residual vs other series (contemporaneous) ---")
    display(corr.round(3).head(10))
    print("\n--- residual predicts which fwd change? ---")
    display(pred.round(3).head(15))

    # 5. peel 30y out — strip the next known factor
    out_peeled   = peel(resid_10_2, df["30y"], lookback=60,
                        resid_name="resid(10y|2y)", z_name="30y", viz=viz)
    resid_peeled = out_peeled["resid"]

    # 6. re-scan the peeled residual
    deeper_others = df.drop(columns=["10y", "2y", "30y"])
    pred2 = predict_scan(resid_peeled, deeper_others)
    print("\n--- after peeling 30y: predict_scan on the new residual ---")
    display(pred2.round(3).head(10))

    return {
        "df":            df,
        "out_10_2":      out_10_2,
        "resid_10_2":    resid_10_2,
        "corr":          corr,
        "pred":          pred,
        "out_peeled":    out_peeled,
        "resid_peeled":  resid_peeled,
        "pred2":         pred2,
    }


if __name__ == "__main__":
    state = main()
