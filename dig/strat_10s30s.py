"""
10s30s curve trade — mean reversion on regression residual.

SIGNAL:  Regress generic 10s30s spread on generic 10Y yield (from md.index_eod).
         Take the residual, fit OU process, trade when z-score is extreme.
TRADE:   Long/short the 10s30s spread using on-the-run 10Y & 30Y (from md.headline).
EXIT:    Hold for rolling half-life duration (no lookahead — uses only historical data).
"""

import polars as pl
import psycopg2

from stats import roll_lr, ou_zscore, ou_summary, half_life, roll_half_life
from backtest import (
    Engine,
    BacktestConfig,
    TradeDef,
    SignalPipeline,
    SignalConfig,
    DV01Map,
    print_summary,
    equity_curve_pd,
    trade_log,
)
from utils import pdf, fix_outliers

# ── config ────────────────────────────────────────────────────────────
DB_DSN = "postgresql://benjils:snickers@raptor:5432/markets"
LOOKBACK = 500          # rolling regression window
Z_ENTRY = 1.5           # enter when |z| > threshold
START = "2010-01-01"

# ── signal filters (gate entry — null out signal when conditions fail) ─
# Permissive defaults = no filtering. Tighten from filter_research.py findings.
MIN_R2 = 0.0            # minimum R² of rolling regression (0 = off)
MAX_BETA_CV = 2.0        # maximum beta coefficient of variation (2 = off)
MIN_HALF_LIFE = 0.0      # minimum rolling half-life in days (0 = off)

# ── pull data ─────────────────────────────────────────────────────────
conn = psycopg2.connect(DB_DSN)

# helper: run query, return polars DataFrame
cur = conn.cursor()


def query_to_pl(sql: str) -> pl.DataFrame:
    cur.execute(sql)
    cols = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    if not rows:
        return pl.DataFrame(schema=cols)
    # Column-oriented avoids row-by-row schema inference issues with mixed types
    data = {col: [row[i] for row in rows] for i, col in enumerate(cols)}
    return pl.DataFrame(data)


# 1) GENERICS for signal generation (smooth Bloomberg generic series)
#    Scale to bps (*100) so all downstream PnL is natively in bps.
generics = (
    query_to_pl(f"""
        SELECT ts, ticker, px_last::float
        FROM md.index_eod
        WHERE ticker IN ('USGG10YR Index', 'USYC1030 Index')
          AND ts >= '{START}'
        ORDER BY ts
    """)
    .pivot(index="ts", on="ticker", values="px_last")
    .rename({"USGG10YR Index": "gen_10Y", "USYC1030 Index": "gen_1030"})
    .sort("ts")
    .with_columns(pl.col("gen_10Y") * 100, pl.col("gen_1030") * 100)
)

# 2) ON-THE-RUN yields for PnL tracking (actual current bonds)
#    Same *100 scaling — everything in bps.
otr = (
    query_to_pl(f"""
        SELECT ts, tenor, yld_ytm_mid::float
        FROM md.headline
        WHERE asset_class = 'nominal'
          AND status = 'c'
          AND tenor IN ('10-Year', '30-Year')
          AND ts >= '{START}'
        ORDER BY ts
    """)
    .pivot(index="ts", on="tenor", values="yld_ytm_mid")
    .rename({"10-Year": "otr_10Y", "30-Year": "otr_30Y"})
    .sort("ts")
    .with_columns(
        fix_outliers(pl.col("otr_10Y"), hi=20).alias("otr_10Y"),
        fix_outliers(pl.col("otr_30Y"), hi=20).alias("otr_30Y"),
    )
    .with_columns(pl.col("otr_10Y") * 100, pl.col("otr_30Y") * 100)
)

conn.close()

# Join generics + on-the-run on date
data = generics.join(otr, on="ts", how="inner").drop_nulls()

print(f"Data: {data.shape[0]} rows, {data['ts'].min()} to {data['ts'].max()}\n")

# ── explore the residual first ────────────────────────────────────────
# Regress generic 10s30s on generic 10Y
reg = roll_lr(data["gen_10Y"], data["gen_1030"], lookback=LOOKBACK)

resid = reg["resid"]
print("── OU Summary (full sample) ──")
print(ou_summary(resid, lookback=LOOKBACK))

hl_full = half_life(resid)
rhl = roll_half_life(resid, lookback=LOOKBACK)
rhl_clean = rhl.drop_nulls()
print(f"Full-sample half-life: {hl_full:.1f} days (reference only — NOT used in backtest)")
print(f"Rolling half-life: median={rhl_clean.median():.1f}, "
      f"p25={rhl_clean.quantile(0.25):.1f}, p75={rhl_clean.quantile(0.75):.1f} days\n")

# ── define the strategy ───────────────────────────────────────────────

# The TRADE references on-the-run columns for PnL.
# "long" the spread = steepener (30Y sells off more / rallies less than 10Y)
# "short" the spread = flattener
trade_10s30s = TradeDef.spread("10s30s", "otr_10Y", "otr_30Y")


def signal_fn(d: pl.DataFrame) -> pl.DataFrame:
    """OU z-score + confidence-weighted, vol-scaled sizing.

    Returns: signal, time_stop, size
    - signal: z-score (masked null when not mean-reverting)
    - time_stop: rolling half-life at entry (no lookahead)
    - size: |z|/Z_ENTRY * vol_scale * confidence (continuous, not binary)
    """
    reg = roll_lr(d["gen_10Y"], d["gen_1030"], lookback=LOOKBACK)
    z = ou_zscore(reg["resid"], lookback=LOOKBACK)
    rhl = roll_half_life(reg["resid"], lookback=LOOKBACK)

    # ── confidence score ──────────────────────────────────────────────
    # R²: how well regression explains the spread (0–1, higher = better)
    r2 = reg["r2"].fill_null(0).clip(0, 1)

    # Beta stability: coefficient of variation over trailing window
    # Low CV = stable relationship = high confidence
    beta_cv = (
        reg["beta"].rolling_std(LOOKBACK)
        / reg["beta"].rolling_mean(LOOKBACK).abs()
    ).fill_null(2.0).clip(0, 2)
    beta_conf = 1 - beta_cv / 2  # CV=0 → 1.0, CV≥2 → 0.0

    conf = (r2 * beta_conf).clip(0.05, 1.0)

    # ── vol-scaled sizing ─────────────────────────────────────────────
    # Signal strength: proportional to |z| (z=Z_ENTRY → 1.0, z=2*Z → 2.0)
    raw_size = z.abs() / Z_ENTRY

    # Inverse-vol: smaller position when spread vol is elevated
    resid_vol = reg["resid"].diff().rolling_std(20)
    vol_target = resid_vol.rolling_mean(252)  # trailing 1yr avg vol
    vol_scale = (vol_target / resid_vol).fill_null(1.0).clip(0.2, 5.0)

    # Final size = strength × vol_adj × confidence
    size = (raw_size * vol_scale * conf).clip(0.1, 3.0)

    # ── signal gate ─────────────────────────────────────────────────────
    # Suppress signal when conditions aren't met (null → engine skips entry)
    gate = (
        rhl.is_not_null()                   # must be mean-reverting
        & (r2 >= MIN_R2)                    # regression must explain enough
        & (beta_cv <= MAX_BETA_CV)          # relationship must be stable
        & (rhl >= MIN_HALF_LIFE)            # reversion can't be too fast to trade
    )

    # ── assemble output ───────────────────────────────────────────────
    out = pl.DataFrame({
        "signal": z,
        "time_stop": rhl,
        "size": size,
        "confidence": conf,
        "vol": resid_vol,
    })

    return out.with_columns(
        pl.when(pl.Series(gate))
        .then(pl.col("signal"))
        .otherwise(None)
        .alias("signal")
    )


pipeline = SignalPipeline(
    name="10s30s_ou_resid",
    trade_def=trade_10s30s,
    compute_fn=signal_fn,
    config=SignalConfig(
        entry_long=-Z_ENTRY,       # z < -2 → spread cheap → steepener
        entry_short=Z_ENTRY,       # z > +2 → spread rich  → flattener
        exit_long=None,            # disabled — time stop only
        exit_short=None,
        time_stop_bars=None,       # no fixed stop — rolling half-life drives it
        max_positions=1,
    ),
)

# ── run backtest ──────────────────────────────────────────────────────

engine = Engine(
    BacktestConfig(
        transaction_cost_bps=0.1,
        dv01_map=DV01Map.from_tenors({"otr_10Y": 10.0, "otr_30Y": 30.0}),
    )
)
engine.add_signal(pipeline)
result = engine.run(data)

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
        pl.col("entry_level").round(2),
        pl.col("exit_level").round(2),
        pl.col("entry_signal").round(2),
        pl.col("exit_signal").round(2),
    )
    print("\n── Trade Log ──")
    pdf(tlog.tail(20).to_pandas())
    winners = tlog.filter(pl.col("pnl_bps") > 0)
    losers = tlog.filter(pl.col("pnl_bps") <= 0)
    print(f"\n  Total realized: {tlog['pnl_bps'].sum():.1f} bps | "
          f"Winners: {len(winners)} | Losers: {len(losers)} | "
          f"Avg win: {winners['pnl_bps'].mean():.1f} bps | "
          f"Avg loss: {losers['pnl_bps'].mean():.1f} bps")

# ── visualization (uncomment in notebook) ─────────────────────────────
# from utils.viz import Viz
# v = Viz()
# eq = equity_curve_pd(result)
# v.line(eq[["cumulative_pnl"]], title="10s30s OU Residual Strategy", yaxis_title="bps")
