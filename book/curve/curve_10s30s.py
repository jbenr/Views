"""Beta-weighted 10s30s curve.

Naive 10s30s = 30Y - 10Y carries hidden direction: 30Y DV01 > 10Y DV01, so
parallel rate moves steepen/flatten the naive spread with no real curve signal.
Fix: regress naive spread on 10Y level (rolling), trade the residual.

main() builds the beta-weighted residual, backtests IC/hit/Sharpe at
5d/20d/60d, and reports the current OU z-score.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from stats import half_life, ou_params, roll_half_life, roll_lr, roll_ou_zscore
from utils.helpers import query_db, timed
from utils.viz import Viz

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)

# ── config ────────────────────────────────────────────────────────────────────

START    = "2005-01-01"
LOOKBACK = 252

TICKERS = {
    "10y": "USGG10YR Index",
    "30y": "USGG30YR Index",
}
BPS_COLS = {"10y", "30y"}


# ── data ──────────────────────────────────────────────────────────────────────

def load_data(start: str = START) -> pd.DataFrame:
    tlist = ", ".join(f"'{t}'" for t in TICKERS.values())
    raw = query_db(f"""
        SELECT ts, ticker, px_last::float AS px
        FROM md.index_eod
        WHERE ticker IN ({tlist}) AND ts >= '{start}'
        ORDER BY ts
    """)
    df = (
        raw.pivot(index="ts", columns="ticker", values="px")
           .rename(columns={v: k for k, v in TICKERS.items()})
           .sort_index()
    )
    df.index = pd.to_datetime(df.index)
    for c in BPS_COLS:
        if c in df.columns:
            df[c] = df[c] * 100.0
    df["naive"] = df["30y"] - df["10y"]
    return df.dropna(subset=["10y", "30y"])


# ── model ─────────────────────────────────────────────────────────────────────

def build_model(df: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """Rolling OLS of naive 10s30s on 10Y level. Returns df with resid, beta, r2."""
    base = df.dropna(subset=["10y", "naive"])
    reg = roll_lr(base["10y"], base["naive"], lookback=lookback)
    out = base.copy()
    out["resid"] = reg["resid"].to_numpy()
    out["beta"]  = reg["beta"].to_numpy()
    out["r2"]    = reg["r2"].to_numpy()
    return out


# ── backtest ──────────────────────────────────────────────────────────────────

def backtest(resid: pd.Series, horizons: tuple[int, ...] = (5, 20, 60)) -> pd.DataFrame:
    """Fade the residual at multiple horizons. Returns IC / hit / Sharpe table."""
    clean = resid.dropna()
    hl = half_life(clean)
    rows = []
    for h in horizons:
        fwd = clean.diff(h).shift(-h)
        jh = pd.concat([clean.rename("s"), fwd.rename("f")], axis=1).dropna()
        jh = jh[jh["s"] != 0]
        if len(jh) < 30:
            rows.append({"h": h, "n": len(jh), "hl_days": round(hl, 0) if np.isfinite(hl) else np.nan})
            continue
        pnl = np.sign(jh["s"]) * (-jh["f"])
        rows.append({
            "h":       h,
            "n":       len(jh),
            "ic":      round(float(jh["s"].corr(-jh["f"], method="spearman")), 3),
            "hit":     round(float((np.sign(jh["s"]) == np.sign(-jh["f"])).mean()), 3),
            "sharpe":  round(float(pnl.mean() / pnl.std() * np.sqrt(252 / h)), 2) if pnl.std() > 0 else np.nan,
            "hl_days": round(hl, 0) if np.isfinite(hl) else np.nan,
        })
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    viz = Viz()

    with timed("loading data"):
        df = load_data()

    # build beta-weighted residual
    model = build_model(df)
    resid = model["resid"].dropna()

    # backtest: fade the residual
    bt = backtest(resid)
    viz.table(bt, title="10s30s beta-weighted — fade residual (IC / hit / Sharpe)")

    # naive vs residual
    viz.line(
        pd.DataFrame({
            "naive 10s30s (bps)": model["naive"],
            "beta_wtd resid":     model["resid"],
        }).dropna(),
        title="10s30s — naive spread vs beta-weighted residual",
    )

    # rolling beta (how much direction is in the naive spread)
    viz.line(
        model[["beta"]].dropna().rename(columns={"beta": "rolling beta (10Y on 10s30s)"}),
        title="10s30s — rolling beta of 10Y on naive spread",
    )

    # OU z-score
    zscore = roll_ou_zscore(resid, lookback=LOOKBACK)
    viz.line(
        pd.DataFrame({"ou_zscore": zscore.to_numpy()}, index=resid.index),
        title="10s30s beta-weighted — OU z-score",
    )

    # rolling half-life
    hl_series = roll_half_life(resid, lookback=LOOKBACK)
    viz.line(
        pd.DataFrame({"half_life_days": hl_series.to_numpy()}, index=resid.index),
        title="10s30s beta-weighted — rolling half-life",
    )

    # current signal state
    params = ou_params(resid.iloc[-LOOKBACK:])
    current_z = float(zscore.drop_nulls()[-1])
    print("\n── current signal ──")
    print(f"  residual now : {resid.iloc[-1]:.1f} bps")
    print(f"  OU mu        : {params['mu']:.1f} bps")
    print(f"  half-life    : {params['half_life']:.0f} days")
    print(f"  z-score now  : {current_z:.2f}")

    return {"df": df, "model": model, "bt": bt, "zscore": zscore, "params": params}


if __name__ == "__main__":
    state = main()
