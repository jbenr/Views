"""Pairs RV - trade two instruments simultaneously against their rolling model.

The key idea: the rolling OLS model of y on x gives you the fair-value
relationship. The residual is the spread dislocation. The beta is the hedge
ratio. When you fade the residual you are trading both legs - buy the cheap
one, sell the rich one, in beta-weighted size.

Example: 10Y nominal yield vs 5y5y forward inflation swap (ifs).
  - positive residual = 10Y is high vs forward inflation - 10Y is cheap
  - trade: buy 10Y, sell beta units of 5y5y_ifs
  - P&L ~ sign(entry_resid) * -(resid_exit - resid_entry)

The model and the trade are the same object.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

_HERE = Path(__file__).resolve().parent
for _p in [_HERE, *_HERE.parents]:
    if (_p / "stats").exists():
        ROOT = _p
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stats import half_life, roll_lr
from utils.helpers import timed
from utils.rates import linear_5y5y_forward
from utils.viz import Viz

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)

# ---- config -----------------------------------------------------------------

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets?connect_timeout=10")
START  = "2005-01-01"   # ZCIS data starts ~2005

TICKERS = {
    "2y":     "USGG2YR Index",
    "5y":     "USGG5YR Index",
    "10y":    "USGG10YR Index",
    "30y":    "USGG30YR Index",
    "be5":    "USGGBE05 Index",
    "be10":   "USGGBE10 Index",
    "zc5":    "USSWIT5 Curncy",
    "zc10":   "USSWIT10 Curncy",
    "sofr5":  "USOSFR5 Curncy",
    "sofr10": "USOSFR10 Curncy",
}
BPS_COLS = {"2y", "5y", "10y", "30y", "be5", "be10", "zc5", "zc10", "sofr5", "sofr10"}

# (x_col, y_col, label) - y is the target, x is what we're trading it against
PAIRS = [
    ("5y5y_ifs",  "10y", "10y vs 5y5y fwd inflation"),
    ("5y5y_sfr",  "10y", "10y vs 5y5y SOFR fwd"),
    ("be10",      "10y", "10y vs 10y breakeven"),
    ("be5",       "5y",  "5y vs 5y breakeven"),
    ("5y",        "10y", "10y vs 5y (curve slope)"),
    ("30y",       "10y", "10y vs 30y (long end)"),
]

# ---- data -------------------------------------------------------------------

def load_data(start: str = START) -> pd.DataFrame:
    tlist = ", ".join(f"'{t}'" for t in TICKERS.values())
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT ts, ticker, px_last::float AS px
                FROM md.index_eod
                WHERE ticker IN ({tlist}) AND ts >= '{start}'
                ORDER BY ts
            """)
            rows = cur.fetchall()

    df = (
        pd.DataFrame(rows, columns=["ts", "ticker", "px"])
        .pivot(index="ts", columns="ticker", values="px")
        .rename(columns={v: k for k, v in TICKERS.items()})
        .sort_index()
    )
    df.index = pd.to_datetime(df.index)

    for c in BPS_COLS:
        if c in df.columns:
            df[c] = df[c] * 100.0

    if "zc5" in df.columns and "zc10" in df.columns:
        df["5y5y_ifs"] = linear_5y5y_forward(df["zc5"], df["zc10"])
    if "sofr5" in df.columns and "sofr10" in df.columns:
        df["5y5y_sfr"] = linear_5y5y_forward(df["sofr5"], df["sofr10"])

    return df.dropna(how="all")


# ---- pair analytics ---------------------------------------------------------

def build_pair(df: pd.DataFrame, x_col: str, y_col: str, lookback: int = 252) -> pd.DataFrame:
    """Rolling OLS of y on x. Returns a df with residual, beta, r2."""
    data = df[[x_col, y_col]].dropna()
    reg  = roll_lr(data[x_col], data[y_col], lookback=lookback)
    out  = data.copy()
    out["resid"] = reg["resid"].to_numpy()
    out["beta"]  = reg["beta"].to_numpy()
    out["r2"]    = reg["r2"].to_numpy()
    return out


def backtest_spread(pair_df: pd.DataFrame, horizons: tuple[int, ...] = (5, 20, 60)) -> pd.DataFrame:
    """Backtest fading the spread residual at multiple horizons.

    P&L = sign(resid_t) * -(resid_{t+H} - resid_t)
    Positive when the spread reverts toward zero.
    """
    resid = pair_df["resid"].dropna()
    rows  = []
    for h in horizons:
        fwd_change = resid.diff(h).shift(-h)   # resid[t+h] - resid[t]
        jh = pd.concat([resid.rename("s"), fwd_change.rename("f")], axis=1).dropna()
        jh = jh[jh["s"] != 0]
        if len(jh) < 30:
            rows.append({"h": h, "n": len(jh)})
            continue
        pnl = np.sign(jh["s"]) * (-jh["f"])
        rows.append({
            "h":      h,
            "n":      len(jh),
            "ic":     round(float(jh["s"].corr(-jh["f"], method="spearman")), 3),
            "hit":    round(float((np.sign(jh["s"]) == np.sign(-jh["f"])).mean()), 3),
            "sharpe": round(float(pnl.mean() / pnl.std() * np.sqrt(252 / h)), 2) if pnl.std() > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ---- summary across all pairs -----------------------------------------------

def pairs_summary(df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    rows = []
    for x_col, y_col, label in PAIRS:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        pair = build_pair(df, x_col, y_col, lookback=lookback)
        bt   = backtest_spread(pair).set_index("h")
        resid_clean = pair["resid"].dropna()
        hl = half_life(resid_clean) if len(resid_clean) >= 60 else np.nan
        current_resid  = float(resid_clean.iloc[-1]) if len(resid_clean) else np.nan
        current_beta   = float(pair["beta"].dropna().iloc[-1]) if len(pair["beta"].dropna()) else np.nan
        row = {
            "pair":            label,
            "resid_now (bps)": round(current_resid, 1),
            "beta_now":        round(current_beta, 3),
            "half_life_d":     round(hl, 0) if np.isfinite(hl) else np.nan,
        }
        for h in (5, 20, 60):
            if h in bt.index:
                row[f"ic_{h}d"]     = bt.loc[h, "ic"]     if "ic"     in bt.columns else np.nan
                row[f"hit_{h}d"]    = bt.loc[h, "hit"]    if "hit"    in bt.columns else np.nan
                row[f"sharpe_{h}d"] = bt.loc[h, "sharpe"] if "sharpe" in bt.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---- main -------------------------------------------------------------------

def main():
    viz = Viz()

    with timed("loading data"):
        df = load_data()

    # summary table - all pairs at a glance
    summary = pairs_summary(df)
    viz.table(summary, title="spread RV - pairs backtest summary (fade the residual)")

    # deep dive: 10Y vs 5y5y fwd inflation
    if "5y5y_ifs" in df.columns and "10y" in df.columns:
        pair = build_pair(df, "5y5y_ifs", "10y")

        viz.line(
            pair[["resid"]].rename(columns={"resid": "spread (bps)"}),
            title="10Y vs 5y5y forward inflation - spread residual",
        )
        viz.line(
            pair[["beta"]].rename(columns={"beta": "hedge ratio (beta)"}),
            title="10Y vs 5y5y forward inflation - rolling hedge ratio",
        )

        # backtest detail for this pair
        bt = backtest_spread(pair)
        viz.table(bt, title="10Y vs 5y5y fwd inflation - backtest detail by horizon")

    return {"df": df, "summary": summary}


if __name__ == "__main__":
    state = main()
