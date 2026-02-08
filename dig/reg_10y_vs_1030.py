"""
10yr yield vs 10y30y curve — rolling regression + OU residual analysis

pull from timescaledb, run roll_lr, pipe residuals through OU,
visualize with residual shading.
"""

import os, sys
sys.path.append(os.path.expanduser("~/werk/Views"))

import pandas as pd
import psycopg

from utils.viz import Viz
from stats.ols import roll_lr
from stats.ou import half_life, ou_params, ou_zscore, ou_summary

# ── config ──────────────────────────────────────────────────────────────
DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
LOOKBACK = 100

X_TICKER = "USYC1030 Index"   # 10y30y curve
Y_TICKER = "USGG10YR Index"   # 10yr yield


# ── pull data ───────────────────────────────────────────────────────────
def pull(tickers: list[str]) -> pd.DataFrame:
    with psycopg.connect(DB_DSN) as conn:
        df = pd.read_sql(
            """
            SELECT ts, ticker, px_last
            FROM md.index_eod
            WHERE ticker = ANY(%s)
            ORDER BY ts
            """,
            conn,
            params=[tickers],
        )
    return df.pivot(index='ts', columns='ticker', values='px_last').dropna()


# ── run ─────────────────────────────────────────────────────────────────
raw = pull([X_TICKER, Y_TICKER])
raw.index = pd.to_datetime(raw.index)

x = raw[X_TICKER]
y = raw[Y_TICKER]

# rolling regression
reg = roll_lr(x, y, lookback=LOOKBACK)

# OU analysis on the residual
resid = reg['resid'].dropna()
print("── OU summary ──")
print(ou_summary(resid, lookback=LOOKBACK).to_string(index=False))
print(f"\nfull-sample half-life: {half_life(resid):.1f} days")

# z-score of residual
reg['ou_z'] = ou_zscore(resid, lookback=LOOKBACK)


# ── viz ─────────────────────────────────────────────────────────────────
v = Viz()

# 1) raw series
v.line(raw, title='10yr yield vs 10y30y curve')

# 2) rolling beta
v.line(reg[['beta']].dropna(), title=f'rolling {LOOKBACK}d beta: 10yr ~ 10y30y')

# 3) residual with shading
v.line(reg[['resid']].dropna(), title=f'rolling {LOOKBACK}d regression residual', residual=True)

# 4) OU z-score of residual
v.line(reg[['ou_z']].dropna(), title=f'OU z-score of residual ({LOOKBACK}d)', residual=True)

# 5) r-squared
v.line(reg[['r2']].dropna(), title=f'rolling {LOOKBACK}d R²')
