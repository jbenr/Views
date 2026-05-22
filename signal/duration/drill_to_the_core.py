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

from stats import roll_lr, half_life, ou_params
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
                 residual=True,
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


def _latest_ou_metrics(resid: pd.Series, lookback: int = 252) -> dict:
    """Latest OU stats on a trailing residual window."""
    r = resid.dropna()
    if len(r) < max(20, lookback // 4):
        return {
            "ou_mean_resid": np.nan,
            "ou_zscore": np.nan,
            "half_life_d": np.nan,
            "ou_window_n": len(r),
        }

    window = r.tail(lookback)
    params = ou_params(window)
    sigma = float(window.std())
    mu = float(params["mu"])
    current = float(window.iloc[-1])
    z = (current - mu) / sigma if sigma > 0 and np.isfinite(mu) else np.nan
    return {
        "ou_mean_resid": mu,
        "ou_zscore": z,
        "half_life_d": float(params["half_life"]),
        "ou_window_n": len(window),
    }


def rolling_ou_zscore(resid: pd.Series, lookback: int = 252) -> pd.Series:
    """Rolling OU z-score: current residual vs trailing OU equilibrium."""
    out = pd.Series(np.nan, index=resid.index, name="ou_zscore")
    r = resid.astype(float)
    min_obs = max(20, lookback // 4)

    for i in range(len(r)):
        window = r.iloc[max(0, i - lookback + 1):i + 1].dropna()
        if len(window) < min_obs:
            continue
        params = ou_params(window)
        mu = float(params["mu"])
        sigma = float(window.std())
        current = r.iloc[i]
        if sigma > 0 and np.isfinite(mu) and np.isfinite(current):
            out.iloc[i] = (current - mu) / sigma
    return out


def single_factor_10y_table(
    df: pd.DataFrame,
    *,
    target: str = "10y",
    lookback: int = 60,
    ou_lookback: int = 252,
    exclude: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Regress 10y against each candidate one at a time and summarize latest OU residual state."""
    rows = []
    models: dict[str, pd.DataFrame] = {}
    candidates = [c for c in df.columns if c != target and c not in exclude]

    for anchor in candidates:
        pair = df[[anchor, target]].dropna()
        if len(pair) < lookback:
            continue

        out = roll_lr(pair[anchor], pair[target], lookback=lookback).to_pandas()
        out.index = pair.index
        out["ou_zscore"] = rolling_ou_zscore(out["resid"], lookback=ou_lookback)
        models[anchor] = out

        valid = out.dropna(subset=["y", "yhat", "resid"])
        if valid.empty:
            continue

        last = valid.iloc[-1]
        ou = _latest_ou_metrics(out["resid"], lookback=ou_lookback)
        ou_mean = ou["ou_mean_resid"]
        rows.append({
            "anchor": anchor,
            "date": valid.index[-1],
            "actual_10y": float(last["y"]),
            "model_10y": float(last["yhat"]),
            "residual": float(last["resid"]),
            "ou_mean_resid": ou_mean,
            "ou_fair_10y": float(last["yhat"] + ou_mean) if np.isfinite(ou_mean) else np.nan,
            "ou_zscore": ou["ou_zscore"],
            "half_life_d": ou["half_life_d"],
            "beta": float(last["beta"]),
            "r2": float(last["r2"]),
            "ou_window_n": ou["ou_window_n"],
        })

    table = pd.DataFrame(rows)
    if not table.empty:
        table = (table
                 .assign(abs_ou_zscore=lambda d: d["ou_zscore"].abs())
                 .sort_values("abs_ou_zscore", ascending=False)
                 .drop(columns="abs_ou_zscore")
                 .reset_index(drop=True))
    return table, models


# ─── main ──────────────────────────────────────────────────────────────────

def main() -> dict:
    viz = Viz(backend='plotly')

    df = load_basket()
    print(f"basket: {len(df)} obs · {df.index.min().date()} → {df.index.max().date()}")
    print(f"        cols: {list(df.columns)}")

    # Single-factor 10y models: each candidate gets its own rolling regression.
    ten_single, ten_single_models = single_factor_10y_table(
        df,
        target="10y",
        lookback=60,
        ou_lookback=252,
    )
    print("\n--- 10y single-factor rolling regression residual OU table ---")
    display_cols = [
        "anchor", "date", "actual_10y", "model_10y", "residual",
        "ou_mean_resid", "ou_fair_10y", "ou_zscore", "half_life_d",
        "beta", "r2",
    ]
    if ten_single.empty:
        print("No single-factor models had enough overlapping data.")
    else:
        display(ten_single[display_cols].round(3))
        viz.table(
            ten_single[display_cols].round(3),
            title="10y single-factor rolling regression residual OU table",
        )

    # App charts: table, then 10y/anchor, residual, and OU z-score per anchor.
    for anchor in ten_single["anchor"].tolist() if not ten_single.empty else []:
        model = ten_single_models[anchor]
        pair = pd.concat([
            model["y"].rename("10y"),
            model["x"].rename(anchor),
        ], axis=1).dropna()
        same_units = anchor in BPS_COLS

        viz.line(
            pair,
            left=[] if same_units else [anchor],
            title=f"10y vs {anchor}  (60d rolling model input)",
            yaxis_title="level, bps" if same_units else "10y, bps",
            yaxis_right_title=None if same_units else anchor,
        )

        viz.line(
            model[["resid"]].rename(columns={"resid": f"resid(10y | {anchor})"}),
            title=f"residual: 10y - model({anchor})",
            yaxis_title="resid, bps",
            residual=True,
            hlines=[(0, None, "solid")],
        )

        viz.line(
            model[["ou_zscore"]].rename(columns={"ou_zscore": f"OU z resid(10y | {anchor})"}),
            title=f"OU z-score: residual(10y | {anchor})",
            yaxis_title="z-score",
            residual=True,
            hlines=[(0, None, "solid"), (2, "+2", "dashed"), (-2, "-2", "dashed")],
        )

    return {
        "df":            df,
        "ten_single":    ten_single,
        "ten_single_models": ten_single_models,
    }


if __name__ == "__main__":
    state = main()
