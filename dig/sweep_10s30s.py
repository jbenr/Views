"""
Parameter sweep for 10s30s curve strategy — find optimal filter/signal params.

Answers: "WHEN should I trust the signal?" by sweeping over:
  - lookback      : rolling regression window
  - z_entry       : z-score threshold to enter
  - min_r2        : minimum R² to allow entry
  - max_beta_cv   : maximum beta CV to allow entry
  - min_half_life : minimum OU half-life to allow entry

Optimization: roll_lr / ou_zscore / roll_half_life only depend on lookback,
so we precompute those once per lookback value, then sweep thresholds cheaply.

Two-phase approach:
  1. Full-sample grid search → rank by Sharpe
  2. Walk-forward validation → check if top params survive out-of-sample
"""

import polars as pl
import psycopg2
import numpy as np
from itertools import product
from tqdm import tqdm
import time

from stats import roll_lr, ou_zscore, roll_half_life
from backtest import (
    Engine,
    BacktestConfig,
    TradeDef,
    SignalPipeline,
    SignalConfig,
    DV01Map,
    ParameterSweep,
    SweepConfig,
)
from utils import fix_outliers

# ── config ────────────────────────────────────────────────────────────
DB_DSN = "postgresql://benjils:snickers@raptor:5432/markets"
START = "2000-01-01"

# ── expanded parameter grid ───────────────────────────────────────────
LOOKBACKS = sorted(set(
    list(range(10, 101, 10))       # 10, 20, ..., 100
    + list(range(150, 301, 50))    # 150, 200, 250, 300
    + list(range(400, 1001, 100))  # 400, 500, ..., 1000
))
Z_ENTRIES       = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
MIN_R2S         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MAX_BETA_CVS    = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
MIN_HALF_LIFES  = [0, 2, 5, 10, 15, 20, 30, 50]

PARAM_GRID = {
    "lookback":      LOOKBACKS,
    "z_entry":       Z_ENTRIES,
    "min_r2":        MIN_R2S,
    "max_beta_cv":   MAX_BETA_CVS,
    "min_half_life": MIN_HALF_LIFES,
}

# ── pull data (once) ──────────────────────────────────────────────────
print("Pulling data...")
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
    .with_columns(
        fix_outliers(pl.col("otr_10Y"), hi=20).alias("otr_10Y"),
        fix_outliers(pl.col("otr_30Y"), hi=20).alias("otr_30Y"),
    )
    .with_columns(pl.col("otr_10Y") * 100, pl.col("otr_30Y") * 100)
)

conn.close()

data = generics.join(otr, on="ts", how="inner").drop_nulls()
print(f"Data: {data.shape[0]} rows, {data['ts'].min()} to {data['ts'].max()}\n")

# ── trade definition (constant across all combos) ────────────────────
trade_10s30s = TradeDef.spread("10s30s", "otr_10Y", "otr_30Y")

# ── Phase 1: Precompute + full-sample sweep ──────────────────────────
print("=" * 70)
print("PHASE 1: FULL-SAMPLE GRID SEARCH")
print("=" * 70)

threshold_combos = list(product(Z_ENTRIES, MIN_R2S, MAX_BETA_CVS, MIN_HALF_LIFES))
total_combos = len(LOOKBACKS) * len(threshold_combos)
print(f"  Lookback values:    {len(LOOKBACKS)}")
print(f"  Threshold combos:   {len(threshold_combos):,}")
print(f"  Total combinations: {total_combos:,}")

# Step 1: Precompute expensive intermediates per lookback
print("\nPrecomputing signal intermediates per lookback...")
precomputed = {}
for lb in tqdm(LOOKBACKS, desc="Lookbacks"):
    reg = roll_lr(data["gen_10Y"], data["gen_1030"], lookback=lb)
    z = ou_zscore(reg["resid"], lookback=lb)
    rhl = roll_half_life(reg["resid"], lookback=lb)

    r2 = reg["r2"].fill_null(0).clip(0, 1)
    beta_cv = (
        reg["beta"].rolling_std(lb)
        / reg["beta"].rolling_mean(lb).abs()
    ).fill_null(2.0).clip(0, 2)
    beta_conf = 1 - beta_cv / 2
    conf = (r2 * beta_conf).clip(0.05, 1.0)

    resid_vol = reg["resid"].diff().rolling_std(20)
    vol_target = resid_vol.rolling_mean(252)
    vol_scale = (vol_target / resid_vol).fill_null(1.0).clip(0.2, 5.0)

    precomputed[lb] = {
        "z": z, "rhl": rhl, "r2": r2, "beta_cv": beta_cv,
        "conf": conf, "vol_scale": vol_scale, "resid_vol": resid_vol,
    }

# Step 2: Sweep over all combos (cheap — just threshold ops + engine run)
print(f"\nSweeping {total_combos:,} combinations...")
records = []
engine = Engine(
    BacktestConfig(
        transaction_cost_bps=0.1,
        dv01_map=DV01Map.from_tenors({"otr_10Y": 10.0, "otr_30Y": 30.0}),
    )
)

t0 = time.time()
all_combos = list(product(LOOKBACKS, threshold_combos))

for lb, (z_ent, mr2, mbcv, mhl) in tqdm(all_combos, desc="Sweep", smoothing=0.01):
    pre = precomputed[lb]

    # Closure captures precomputed values by default-arg binding
    def signal_fn(
        d,
        _z=pre["z"], _rhl=pre["rhl"], _r2=pre["r2"],
        _bcv=pre["beta_cv"], _conf=pre["conf"],
        _vs=pre["vol_scale"], _rv=pre["resid_vol"],
        _z_ent=z_ent, _mr2=mr2, _mbcv=mbcv, _mhl=mhl,
    ):
        raw_size = _z.abs() / _z_ent
        size = (raw_size * _vs * _conf).clip(0.1, 3.0)

        gate = (
            _rhl.is_not_null()
            & (_r2 >= _mr2)
            & (_bcv <= _mbcv)
            & (_rhl >= _mhl)
        )

        out = pl.DataFrame({
            "signal": _z,
            "time_stop": _rhl,
            "size": size,
            "confidence": _conf,
            "vol": _rv,
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
            entry_long=-z_ent,
            entry_short=z_ent,
            exit_long=None,
            exit_short=None,
            time_stop_bars=None,
            max_positions=1,
        ),
    )

    engine.pipelines = [pipeline]
    try:
        result = engine.run(data)
        metrics = result.summary()
        records.append({
            "lookback": lb, "z_entry": z_ent, "min_r2": mr2,
            "max_beta_cv": mbcv, "min_half_life": mhl,
            **metrics,
        })
    except Exception as e:
        records.append({
            "lookback": lb, "z_entry": z_ent, "min_r2": mr2,
            "max_beta_cv": mbcv, "min_half_life": mhl,
            "error": str(e),
        })

elapsed = time.time() - t0
print(f"\nCompleted in {elapsed / 60:.1f} min "
      f"({elapsed / total_combos * 1000:.1f} ms/combo)")

results = pl.DataFrame(records)
results = results.filter(pl.col("sharpe").is_not_null() & ~pl.col("sharpe").is_nan())
results = results.sort("sharpe", descending=True)

# ── Display top 30 by Sharpe ─────────────────────────────────────────
display_cols = list(PARAM_GRID.keys()) + [
    "sharpe", "total_pnl_bps", "n_trades", "hit_rate",
    "profit_factor", "max_drawdown_bps", "calmar", "avg_holding_days",
]
display_cols = [c for c in display_cols if c in results.columns]

print("\n── Top 30 Parameter Combos by Sharpe ──")
print(results.head(30).select(display_cols)
      .to_pandas().to_string(index=False, float_format="%.3f"))

# ── Sensitivity analysis: which params matter most? ───────────────────
print("\n" + "=" * 70)
print("SENSITIVITY ANALYSIS: Average Sharpe by parameter value")
print("=" * 70)

valid = results.filter(pl.col("n_trades") > 5)

for param in PARAM_GRID.keys():
    agg = (
        valid.group_by(param)
        .agg(
            pl.col("sharpe").mean().alias("avg_sharpe"),
            pl.col("sharpe").median().alias("med_sharpe"),
            pl.col("sharpe").std().alias("std_sharpe"),
            pl.col("sharpe").max().alias("max_sharpe"),
            pl.col("n_trades").mean().alias("avg_trades"),
            pl.len().alias("count"),
        )
        .sort(param)
    )
    print(f"\n  {param}:")
    for row in agg.iter_rows(named=True):
        val = row[param]
        print(f"    {val:>8} -> avg={row['avg_sharpe']:+.3f}  "
              f"med={row['med_sharpe']:+.3f}  "
              f"max={row['max_sharpe']:+.3f}  "
              f"std={row['std_sharpe']:.3f}  "
              f"trades={row['avg_trades']:.0f}  "
              f"(n={row['count']})")

# ── Interaction heatmaps: pairwise parameter effects ─────────────────
print("\n" + "=" * 70)
print("INTERACTION ANALYSIS: Pairwise avg Sharpe")
print("=" * 70)

param_names = list(PARAM_GRID.keys())
for i in range(len(param_names)):
    for j in range(i + 1, len(param_names)):
        p1, p2 = param_names[i], param_names[j]
        pivot = (
            valid.group_by([p1, p2])
            .agg(pl.col("sharpe").mean().alias("avg_sharpe"))
            .sort([p1, p2])
            .pivot(index=p1, on=p2, values="avg_sharpe")
            .sort(p1)
        )
        print(f"\n  {p1} x {p2}:")
        print(f"  {pivot.to_pandas().to_string(index=False, float_format='%+.2f')}")


# ── Phase 2: Walk-forward validation ─────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 2: WALK-FORWARD VALIDATION")
print("=" * 70)


def build_and_run(params: dict, df: pl.DataFrame):
    """Full build_fn for walk-forward (must recompute on subset data)."""
    lookback = int(params["lookback"])
    z_entry = float(params["z_entry"])
    min_r2 = float(params["min_r2"])
    max_beta_cv = float(params["max_beta_cv"])
    min_half_life = float(params["min_half_life"])

    def signal_fn(d: pl.DataFrame) -> pl.DataFrame:
        reg = roll_lr(d["gen_10Y"], d["gen_1030"], lookback=lookback)
        z = ou_zscore(reg["resid"], lookback=lookback)
        rhl = roll_half_life(reg["resid"], lookback=lookback)

        r2 = reg["r2"].fill_null(0).clip(0, 1)
        beta_cv = (
            reg["beta"].rolling_std(lookback)
            / reg["beta"].rolling_mean(lookback).abs()
        ).fill_null(2.0).clip(0, 2)
        beta_conf = 1 - beta_cv / 2
        conf = (r2 * beta_conf).clip(0.05, 1.0)

        raw_size = z.abs() / z_entry
        resid_vol = reg["resid"].diff().rolling_std(20)
        vol_target = resid_vol.rolling_mean(252)
        vol_scale = (vol_target / resid_vol).fill_null(1.0).clip(0.2, 5.0)
        size = (raw_size * vol_scale * conf).clip(0.1, 3.0)

        gate = (
            rhl.is_not_null()
            & (r2 >= min_r2)
            & (beta_cv <= max_beta_cv)
            & (rhl >= min_half_life)
        )

        out = pl.DataFrame({
            "signal": z, "time_stop": rhl, "size": size,
            "confidence": conf, "vol": resid_vol,
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
            entry_long=-z_entry, entry_short=z_entry,
            exit_long=None, exit_short=None,
            time_stop_bars=None, max_positions=1,
        ),
    )

    eng = Engine(
        BacktestConfig(
            transaction_cost_bps=0.1,
            dv01_map=DV01Map.from_tenors({"otr_10Y": 10.0, "otr_30Y": 30.0}),
        )
    )
    eng.add_signal(pipeline)
    return eng.run(df)


# Narrow grid for walk-forward: top 3 values per param from full-sample
def top_values(param, n=3):
    agg = (
        valid.group_by(param)
        .agg(pl.col("sharpe").mean().alias("avg_sharpe"))
        .sort("avg_sharpe", descending=True)
    )
    return agg[param].head(n).to_list()


wf_grid = {p: top_values(p, n=3) for p in PARAM_GRID.keys()}
n_wf = 1
for v in wf_grid.values():
    n_wf *= len(v)
print(f"\nWalk-forward grid (narrowed from sensitivity): {n_wf} combos")
for p, vals in wf_grid.items():
    print(f"  {p}: {vals}")

wf_sweep = ParameterSweep(
    build_fn=build_and_run,
    data=data,
    config=SweepConfig(
        params=wf_grid,
        optimize_metric="sharpe",
        train_years=3,
        test_years=1,
        step_years=1,
    ),
)

wf_results = wf_sweep.walk_forward()

if not wf_results.is_empty():
    print("\n── Walk-Forward Results (Out-of-Sample) ──")

    wf_display = [c for c in wf_results.columns
                  if c.startswith(("window", "train", "test", "best_",
                                   "oos_sharpe", "oos_total", "oos_n_trades",
                                   "oos_hit", "is_train"))]
    print(wf_results.select(wf_display)
          .to_pandas().to_string(index=False, float_format="%.3f"))

    oos_sharpes = wf_results.filter(
        pl.col("oos_sharpe").is_not_null()
    )["oos_sharpe"]
    is_sharpes = wf_results.filter(
        pl.col("is_train_sharpe").is_not_null()
    )["is_train_sharpe"]

    print(f"\n── Walk-Forward Summary ──")
    print(f"  Windows:         {wf_results.height}")
    print(f"  IS avg Sharpe:   {is_sharpes.mean():.3f}")
    print(f"  OOS avg Sharpe:  {oos_sharpes.mean():.3f}")
    print(f"  OOS med Sharpe:  {oos_sharpes.median():.3f}")
    print(f"  OOS % positive:  {(oos_sharpes > 0).sum() / len(oos_sharpes) * 100:.0f}%")
    if is_sharpes.mean() > 0:
        print(f"  Sharpe decay:    "
              f"{(1 - oos_sharpes.mean() / is_sharpes.mean()) * 100:.0f}%")

    print("\n── Most Selected Parameters (across windows) ──")
    for p in PARAM_GRID.keys():
        best_col = f"best_{p}"
        if best_col in wf_results.columns:
            counts = (
                wf_results.group_by(best_col)
                .agg(pl.len().alias("times_chosen"))
                .sort("times_chosen", descending=True)
            )
            print(f"  {p}: {counts.to_pandas().to_string(index=False)}")
else:
    print("  No walk-forward windows completed (need more data history).")


# ── Final recommendation ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

best = results.row(0, named=True)
print(f"\nBest full-sample params (Sharpe = {best['sharpe']:.3f}):")
for p in PARAM_GRID.keys():
    print(f"  {p:>15} = {best[p]}")
print(f"  {'trades':>15} = {best['n_trades']}")
print(f"  {'hit_rate':>15} = {best['hit_rate']:.1%}")
print(f"  {'total_pnl':>15} = {best['total_pnl_bps']:.0f} bps")
print(f"  {'max_dd':>15} = {best['max_drawdown_bps']:.0f} bps")
print(f"  {'calmar':>15} = {best['calmar']:.2f}")

if not wf_results.is_empty() and len(oos_sharpes) > 0:
    print(f"\n  OOS avg Sharpe:  {oos_sharpes.mean():.3f}")
    if oos_sharpes.mean() > 0.3:
        print("  -> Signal appears robust out-of-sample.")
    elif oos_sharpes.mean() > 0:
        print("  -> Marginal OOS performance — use tight filters.")
    else:
        print("  -> Signal does not survive OOS — likely overfit.")

print(f"\nTotal elapsed: {(time.time() - t0) / 60:.1f} min")

# ── Save results ─────────────────────────────────────────────────────
results.write_parquet("sweep_10s30s_results.parquet")
print(f"\nSaved full-sample results → dig/sweep_10s30s_results.parquet ({results.height} rows)")

if not wf_results.is_empty():
    wf_results.write_parquet("sweep_10s30s_wf_results.parquet")
    print(f"Saved walk-forward results → dig/sweep_10s30s_wf_results.parquet ({wf_results.height} rows)")
