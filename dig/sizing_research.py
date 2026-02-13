"""
Sizing research — what observable metrics at entry actually predict trade PnL?

Approach:
  1. Run baseline backtest (size=1) to get trade outcomes
  2. Build a time series of candidate metrics (R², beta_cv, vol, |z|, half-life)
  3. For each closed trade, look up metrics at entry date
  4. Correlate each metric with trade PnL → find what predicts
  5. Test sizing formulas: re-run backtest weighting by predictive metrics
  6. Compare Sharpe across sizing approaches
"""

import polars as pl
import numpy as np
import psycopg2

from stats import roll_lr, ou_zscore, roll_half_life
from backtest import (
    Engine, BacktestConfig, TradeDef, SignalPipeline, SignalConfig,
    DV01Map, print_summary, trade_log,
)
from utils import pdf

# ── config ────────────────────────────────────────────────────────────
DB_DSN = "postgresql://benjils:snickers@raptor:5432/markets"
LOOKBACK = 100
Z_ENTRY = 2.5
START = "2010-01-01"

# ── pull data (same as strat_10s30s.py) ───────────────────────────────
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
    .with_columns(pl.col("otr_10Y") * 100, pl.col("otr_30Y") * 100)
)

conn.close()
data = generics.join(otr, on="ts", how="inner").drop_nulls()
print(f"Data: {data.shape[0]} rows, {data['ts'].min()} to {data['ts'].max()}\n")

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Build time series of candidate metrics
# ══════════════════════════════════════════════════════════════════════

reg = roll_lr(data["gen_10Y"], data["gen_1030"], lookback=LOOKBACK)
z = ou_zscore(reg["resid"], lookback=LOOKBACK)
rhl = roll_half_life(reg["resid"], lookback=LOOKBACK)

# Candidate metrics (all observable at time t, no lookahead)
r2 = reg["r2"].fill_null(0).clip(0, 1)
beta = reg["beta"]
beta_cv = (
    beta.rolling_std(LOOKBACK) / beta.rolling_mean(LOOKBACK).abs()
).fill_null(2.0).clip(0, 2)
resid_vol = reg["resid"].diff().rolling_std(20)
abs_z = z.abs()
beta_sign_stable = (
    beta.rolling_mean(LOOKBACK).sign() == beta.sign()
).cast(pl.Float64)  # 1 if beta sign matches its trailing mean

# Assemble metrics time series (same length as data, aligned by index)
# Need to handle that reg output may be shorter than data
n_data = data.shape[0]
n_reg = len(reg)
pad = n_data - n_reg

metrics_ts = pl.DataFrame({
    "ts": data["ts"],
    "z": z.extend_constant(None, pad) if pad > 0 else z,
    "abs_z": abs_z.extend_constant(None, pad) if pad > 0 else abs_z,
    "r2": r2.extend_constant(None, pad) if pad > 0 else r2,
    "beta_cv": beta_cv.extend_constant(None, pad) if pad > 0 else beta_cv,
    "resid_vol": resid_vol.extend_constant(None, pad) if pad > 0 else resid_vol,
    "half_life": rhl,
    "beta_sign_stable": beta_sign_stable.extend_constant(None, pad) if pad > 0 else beta_sign_stable,
})

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Run baseline backtest (size=1, no fancy sizing)
# ══════════════════════════════════════════════════════════════════════

trade_10s30s = TradeDef.spread("10s30s", "otr_10Y", "otr_30Y")


def baseline_signal_fn(d: pl.DataFrame) -> pl.DataFrame:
    """Simple z-score + rolling half-life, size=1."""
    r = roll_lr(d["gen_10Y"], d["gen_1030"], lookback=LOOKBACK)
    zscore = ou_zscore(r["resid"], lookback=LOOKBACK)
    rhl_local = roll_half_life(r["resid"], lookback=LOOKBACK)
    out = pl.DataFrame({"signal": zscore, "time_stop": rhl_local})
    return out.with_columns(
        pl.when(pl.col("time_stop").is_not_null())
        .then(pl.col("signal"))
        .otherwise(None)
        .alias("signal")
    )


pipeline = SignalPipeline(
    name="10s30s",
    trade_def=trade_10s30s,
    compute_fn=baseline_signal_fn,
    config=SignalConfig(
        entry_long=-Z_ENTRY,
        entry_short=Z_ENTRY,
        exit_long=None,
        exit_short=None,
        time_stop_bars=None,
        max_positions=1,
    ),
)

engine = Engine(BacktestConfig(
    transaction_cost_bps=0.1,
    dv01_map=DV01Map.from_tenors({"otr_10Y": 10.0, "otr_30Y": 30.0}),
))
engine.add_signal(pipeline)
result = engine.run(data)

print("── BASELINE (size=1) ──")
print_summary(result)

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Enrich trades with metrics at entry
# ══════════════════════════════════════════════════════════════════════

tlog = trade_log(result.closed_trades)
if tlog.is_empty():
    print("No closed trades — nothing to analyze.")
    exit()

# Normalize entry_date to Date for joining
tlog = tlog.with_columns(pl.col("entry_date").cast(pl.Date).alias("ts"))

# Join trade log with metrics at entry date
enriched = tlog.join(metrics_ts, on="ts", how="left")

# Compute per-trade return (PnL / entry_level for direction-neutral comparison)
enriched = enriched.with_columns(
    (pl.col("pnl_bps") / pl.col("entry_level").abs().clip(1, None)).alias("pnl_pct"),
    (pl.col("pnl_bps") > 0).cast(pl.Int32).alias("winner"),
)

print(f"\n{'='*70}")
print(f"  SIZING RESEARCH — {len(enriched)} trades")
print(f"{'='*70}\n")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Correlate metrics with trade PnL
# ══════════════════════════════════════════════════════════════════════

print("── Correlation: metric at entry vs trade PnL ──\n")
print(f"  {'metric':<20} {'corr(pnl)':>10} {'corr(win)':>10} {'avg(win)':>10} {'avg(lose)':>10}")
print(f"  {'-'*60}")

metric_cols = ["abs_z", "r2", "beta_cv", "resid_vol", "half_life", "beta_sign_stable"]

for col in metric_cols:
    s = enriched.select(col, "pnl_bps", "winner").drop_nulls()
    if len(s) < 10:
        continue

    m = s[col].to_numpy().astype(float)
    pnl = s["pnl_bps"].to_numpy().astype(float)
    win = s["winner"].to_numpy().astype(float)

    corr_pnl = np.corrcoef(m, pnl)[0, 1] if np.std(m) > 0 else 0
    corr_win = np.corrcoef(m, win)[0, 1] if np.std(m) > 0 else 0

    # Average metric value for winners vs losers
    w_mask = win > 0.5
    avg_w = m[w_mask].mean() if w_mask.sum() > 0 else float("nan")
    avg_l = m[~w_mask].mean() if (~w_mask).sum() > 0 else float("nan")

    print(f"  {col:<20} {corr_pnl:>10.3f} {corr_win:>10.3f} {avg_w:>10.2f} {avg_l:>10.2f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Quintile analysis — split trades into buckets by each metric
# ══════════════════════════════════════════════════════════════════════

print(f"\n── Quintile Analysis: PnL by metric bucket ──\n")

n_years = (data["ts"].max() - data["ts"].min()).days / 365.25


def _bucket_sharpe(pnl_arr):
    """Trade-level annualized Sharpe proxy: mean/std * sqrt(trades_per_year)."""
    if len(pnl_arr) < 2 or np.std(pnl_arr) == 0:
        return 0.0
    tpy = len(pnl_arr) / n_years
    return float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(tpy))


for col in metric_cols:
    s = enriched.select(col, "pnl_bps").drop_nulls()
    if len(s) < 20:
        continue

    # Split into 3 buckets (low / mid / high) for readability
    vals = s[col].to_numpy().astype(float)
    pnl = s["pnl_bps"].to_numpy().astype(float)

    p33 = np.percentile(vals, 33)
    p66 = np.percentile(vals, 66)

    # Skip degenerate splits (e.g. beta_sign_stable is all 1.0)
    if p33 == p66:
        low = pnl[vals <= p33]
        print(f"  {col}:")
        print(f"    All  (={p33:.2f}): n={len(low):3d}, avg PnL={low.mean():+.2f}, "
              f"hit={100*(low>0).mean():.0f}%, Sharpe={_bucket_sharpe(low):+.2f}, total={low.sum():+.1f}")
        print()
        continue

    low = pnl[vals <= p33]
    mid = pnl[(vals > p33) & (vals <= p66)]
    high = pnl[vals > p66]

    print(f"  {col}:")
    print(f"    Low  (≤{p33:.2f}): n={len(low):3d}, avg PnL={low.mean():+.2f}, "
          f"hit={100*(low>0).mean():.0f}%, Sharpe={_bucket_sharpe(low):+.2f}, total={low.sum():+.1f}")
    print(f"    Mid  ({p33:.2f}–{p66:.2f}): n={len(mid):3d}, avg PnL={mid.mean():+.2f}, "
          f"hit={100*(mid>0).mean():.0f}%, Sharpe={_bucket_sharpe(mid):+.2f}, total={mid.sum():+.1f}")
    print(f"    High (>{p66:.2f}): n={len(high):3d}, avg PnL={high.mean():+.2f}, "
          f"hit={100*(high>0).mean():.0f}%, Sharpe={_bucket_sharpe(high):+.2f}, total={high.sum():+.1f}")
    print()

# ══════════════════════════════════════════════════════════════════════
# STEP 6: Test sizing formulas — re-run backtest with different sizes
# ══════════════════════════════════════════════════════════════════════

print(f"{'='*70}")
print(f"  SIZING FORMULA COMPARISON")
print(f"{'='*70}\n")


def run_with_sizing(name: str, size_fn) -> dict:
    """Run backtest with a custom sizing function, return metrics."""

    def sized_signal_fn(d: pl.DataFrame) -> pl.DataFrame:
        r = roll_lr(d["gen_10Y"], d["gen_1030"], lookback=LOOKBACK)
        zscore = ou_zscore(r["resid"], lookback=LOOKBACK)
        rhl_local = roll_half_life(r["resid"], lookback=LOOKBACK)
        size = size_fn(r, zscore, rhl_local)
        out = pl.DataFrame({"signal": zscore, "time_stop": rhl_local, "size": size})
        return out.with_columns(
            pl.when(pl.col("time_stop").is_not_null())
            .then(pl.col("signal"))
            .otherwise(None)
            .alias("signal")
        )

    p = SignalPipeline(
        name="10s30s",
        trade_def=trade_10s30s,
        compute_fn=sized_signal_fn,
        config=SignalConfig(
            entry_long=-Z_ENTRY, entry_short=Z_ENTRY,
            exit_long=None, exit_short=None,
            time_stop_bars=None, max_positions=1,
        ),
    )
    e = Engine(BacktestConfig(
        transaction_cost_bps=0.1,
        dv01_map=DV01Map.from_tenors({"otr_10Y": 10.0, "otr_30Y": 30.0}),
    ))
    e.add_signal(p)
    res = e.run(data)
    m = res.summary()
    m["name"] = name
    return m


# ── Define sizing formulas to test ───────────────────────────────────

def sz_flat(r, z, rhl):
    """Baseline: size = 1.0 always."""
    return pl.Series("size", [1.0] * len(z))


def sz_abs_z(r, z, rhl):
    """Size proportional to |z|."""
    return (z.abs() / Z_ENTRY).clip(0.5, 3.0)


def sz_inv_vol(r, z, rhl):
    """Size inversely proportional to residual vol."""
    vol = r["resid"].diff().rolling_std(20)
    target = vol.rolling_mean(252)
    return (target / vol).fill_null(1.0).clip(0.2, 5.0)


def sz_r2(r, z, rhl):
    """Size proportional to R²."""
    return r["r2"].fill_null(0).clip(0.1, 1.0) * 2  # scale up so max=2


def sz_z_x_invvol(r, z, rhl):
    """|z| × inverse vol."""
    raw = z.abs() / Z_ENTRY
    vol = r["resid"].diff().rolling_std(20)
    target = vol.rolling_mean(252)
    vs = (target / vol).fill_null(1.0).clip(0.2, 5.0)
    return (raw * vs).clip(0.2, 3.0)


def sz_z_x_r2(r, z, rhl):
    """|z| × R²."""
    raw = z.abs() / Z_ENTRY
    r2_val = r["r2"].fill_null(0).clip(0.1, 1.0)
    return (raw * r2_val).clip(0.2, 3.0)


def sz_kitchen_sink(r, z, rhl):
    """|z| × R² × inverse vol × beta stability."""
    raw = z.abs() / Z_ENTRY
    r2_val = r["r2"].fill_null(0).clip(0, 1)
    vol = r["resid"].diff().rolling_std(20)
    target = vol.rolling_mean(252)
    vs = (target / vol).fill_null(1.0).clip(0.2, 5.0)
    bcv = (r["beta"].rolling_std(LOOKBACK) / r["beta"].rolling_mean(LOOKBACK).abs()).fill_null(2.0).clip(0, 2)
    bconf = 1 - bcv / 2
    return (raw * r2_val * vs * bconf).clip(0.1, 3.0)


# ── Run all sizing variants ──────────────────────────────────────────

formulas = [
    ("1. flat (size=1)",        sz_flat),
    ("2. |z| only",             sz_abs_z),
    ("3. inv_vol only",         sz_inv_vol),
    ("4. R² only",              sz_r2),
    ("5. |z| × inv_vol",       sz_z_x_invvol),
    ("6. |z| × R²",            sz_z_x_r2),
    ("7. kitchen sink",         sz_kitchen_sink),
]

results_rows = []
for name, fn in formulas:
    m = run_with_sizing(name, fn)
    results_rows.append(m)
    print(f"  {name:<25} Sharpe={m['sharpe']:+.2f}  PnL={m['total_pnl_bps']:+.1f}  "
          f"Hit={100*m['hit_rate']:.0f}%  DD={m['max_drawdown_bps']:.1f}  "
          f"PF={m['profit_factor']:.2f}  Trades={m['n_trades']}")

# Summary table
print(f"\n── Summary ──")
comparison = pl.DataFrame(results_rows).select(
    "name",
    pl.col("sharpe").round(3),
    pl.col("total_pnl_bps").round(1),
    pl.col("hit_rate").round(3),
    pl.col("max_drawdown_bps").round(1),
    pl.col("profit_factor").round(2),
    pl.col("n_trades"),
    pl.col("calmar").round(2),
)
pdf(comparison.to_pandas())
