"""
Multi-spread curve book — cross-sectional mean reversion.

Trades multiple UST curve spreads simultaneously:
  2s5s, 2s10s, 2s30s, 5s10s, 5s30s, 10s30s

For each spread:
  SIGNAL: OU z-score of regression residual (spread ~ short-leg yield)
  SIZING: |z|/z_entry × inverse_vol × confidence (R², beta stability)

Cross-sectional layer:
  Each day, rank all spreads by |z| × confidence.
  Allocate to top-K with risk-parity (inverse-vol) weighting.

This is where Sharpe > 1 lives — diversification + confidence-gating.
"""

import polars as pl
import psycopg2

from backtest import SpreadBook, BookConfig, print_summary, trade_log
from utils import pdf

# ── config ────────────────────────────────────────────────────────────
DB_DSN = "postgresql://benjils:snickers@raptor:5432/markets"
LOOKBACK = 50
Z_ENTRY = 2.0
START = "2010-01-01"
MAX_SPREADS = 3  # trade top-3 spreads at any time

# ── pull data ─────────────────────────────────────────────────────────
conn = psycopg2.connect(DB_DSN)
cur = conn.cursor()


def query_to_pl(sql: str) -> pl.DataFrame:
    cur.execute(sql)
    cols = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    if not rows:
        return pl.DataFrame(schema=cols)
    data = {col: [row[i] for row in rows] for i, col in enumerate(cols)}
    return pl.DataFrame(data)


# Generics: yields + curve spreads (all in bps)
generics = (
    query_to_pl(f"""
        SELECT ts, ticker, px_last::float
        FROM md.index_eod
        WHERE ticker IN (
            'USGG2YR Index',  'USGG5YR Index',
            'USGG10YR Index', 'USGG30YR Index',
            'USYC2Y5Y Index', 'USYC2Y10 Index', 'USYC2Y30 Index',
            'USYC5Y10 Index', 'USYC5Y30 Index', 'USYC1030 Index'
        )
        AND ts >= '{START}'
        ORDER BY ts
    """)
    .pivot(index="ts", on="ticker", values="px_last")
    .rename({
        "USGG2YR Index": "gen_2Y",   "USGG5YR Index": "gen_5Y",
        "USGG10YR Index": "gen_10Y", "USGG30YR Index": "gen_30Y",
        "USYC2Y5Y Index": "gen_2s5s",  "USYC2Y10 Index": "gen_2s10s",
        "USYC2Y30 Index": "gen_2s30s", "USYC5Y10 Index": "gen_5s10s",
        "USYC5Y30 Index": "gen_5s30s", "USYC1030 Index": "gen_1030",
    })
    .sort("ts")
    .with_columns([pl.col(c) * 100 for c in [
        "gen_2Y", "gen_5Y", "gen_10Y", "gen_30Y",
        "gen_2s5s", "gen_2s10s", "gen_2s30s",
        "gen_5s10s", "gen_5s30s", "gen_1030",
    ]])
)

# On-the-run yields for PnL (all in bps)
otr = (
    query_to_pl(f"""
        SELECT ts, tenor, yld_ytm_mid::float
        FROM md.headline
        WHERE asset_class = 'nominal'
          AND status = 'c'
          AND tenor IN ('2-Year', '5-Year', '10-Year', '30-Year')
          AND ts >= '{START}'
        ORDER BY ts
    """)
    .pivot(index="ts", on="tenor", values="yld_ytm_mid")
    .rename({
        "2-Year": "otr_2Y", "5-Year": "otr_5Y",
        "10-Year": "otr_10Y", "30-Year": "otr_30Y",
    })
    .sort("ts")
    .with_columns([pl.col(c) * 100 for c in [
        "otr_2Y", "otr_5Y", "otr_10Y", "otr_30Y",
    ]])
)

conn.close()

data = generics.join(otr, on="ts", how="inner").drop_nulls()
print(f"Data: {data.shape[0]} rows, {data['ts'].min()} to {data['ts'].max()}\n")

# ── build the spread book ─────────────────────────────────────────────

book = SpreadBook(BookConfig(
    max_spreads=MAX_SPREADS,
    allocation="risk_parity",
    total_risk_budget=1.0,
    transaction_cost_bps=0.1,
))

# Each spread: regress generic spread on short-leg generic yield
# DV01 approx: 2Y~2, 5Y~5, 10Y~10, 30Y~30 (per $1M notional per bp)
book.add_spread("2s5s",   gen_x="gen_2Y",  gen_y="gen_2s5s",
                otr_short="otr_2Y",  otr_long="otr_5Y",
                dv01_short=2.0, dv01_long=5.0)
book.add_spread("2s10s",  gen_x="gen_2Y",  gen_y="gen_2s10s",
                otr_short="otr_2Y",  otr_long="otr_10Y",
                dv01_short=2.0, dv01_long=10.0)
book.add_spread("2s30s",  gen_x="gen_2Y",  gen_y="gen_2s30s",
                otr_short="otr_2Y",  otr_long="otr_30Y",
                dv01_short=2.0, dv01_long=30.0)
book.add_spread("5s10s",  gen_x="gen_5Y",  gen_y="gen_5s10s",
                otr_short="otr_5Y",  otr_long="otr_10Y",
                dv01_short=5.0, dv01_long=10.0)
book.add_spread("5s30s",  gen_x="gen_5Y",  gen_y="gen_5s30s",
                otr_short="otr_5Y",  otr_long="otr_30Y",
                dv01_short=5.0, dv01_long=30.0)
book.add_spread("10s30s", gen_x="gen_10Y", gen_y="gen_1030",
                otr_short="otr_10Y", otr_long="otr_30Y",
                dv01_short=10.0, dv01_long=30.0)

# ── run ───────────────────────────────────────────────────────────────

print(f"Running {len(book.spreads)} spreads, top-{MAX_SPREADS} "
      f"risk-parity, lookback={LOOKBACK}, z_entry={Z_ENTRY}\n")

result = book.run(data, lookback=LOOKBACK, z_entry=Z_ENTRY)

# ── results ───────────────────────────────────────────────────────────

print_summary(result)

tlog = trade_log(result.closed_trades)
if not tlog.is_empty():
    tlog = tlog.with_columns(
        pl.col("entry_date").cast(pl.Date).cast(pl.Utf8),
        pl.col("exit_date").cast(pl.Date).cast(pl.Utf8),
        pl.col("size").round(2),
        pl.col("pnl_bps").round(2),
        pl.col("pnl_dollar").round(2),
        pl.col("entry_level").round(1),
        pl.col("exit_level").round(1),
        pl.col("entry_signal").round(2),
        pl.col("exit_signal").round(2),
    )

    # Per-spread breakdown
    by_spread = (
        tlog.group_by("trade_name")
        .agg(
            pl.len().alias("n_trades"),
            pl.col("pnl_bps").sum().round(1).alias("total_bps"),
            pl.col("pnl_bps").mean().round(2).alias("avg_bps"),
            (pl.col("pnl_bps") > 0).mean().round(3).alias("hit_rate"),
            pl.col("bars_held").mean().round(1).alias("avg_hold"),
        )
        .sort("total_bps", descending=True)
    )
    print("\n── Per-Spread Breakdown ──")
    pdf(by_spread.to_pandas())

    print(f"\n── Last 20 Trades ──")
    pdf(tlog.tail(20).to_pandas())

    winners = tlog.filter(pl.col("pnl_bps") > 0)
    losers = tlog.filter(pl.col("pnl_bps") <= 0)
    print(f"\n  Total realized: {tlog['pnl_bps'].sum():.1f} bps | "
          f"Trades: {len(tlog)} | "
          f"Winners: {len(winners)} | Losers: {len(losers)} | "
          f"Avg win: {winners['pnl_bps'].mean():.1f} bps | "
          f"Avg loss: {losers['pnl_bps'].mean():.1f} bps")
