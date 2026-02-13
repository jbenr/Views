"""Intratrade research — study what happens during the life of a trade.

Core dataset: trade_paths — panel of (trade_id × day × all metrics).
For each closed trade, extract daily PnL and metric evolution from
entry through exit + 30 days beyond ("what if held longer?").

Questions this framework helps answer:
  - When do winners/losers diverge in PnL? When do winners peak?
  - How do metrics (R², beta_cv, vol, z) evolve during winning vs losing trades?
  - What fixed holding period maximizes Sharpe vs dynamic half-life?
  - Does a stop-loss or trailing stop improve Sharpe? At what level?
  - When z extends further after entry — add or cut?
  - If underwater at day N, what fraction of trades recover?
  - Do entry conditions (vol regime, R² level) change optimal management?
"""

import polars as pl
import numpy as np
import psycopg2

from stats import roll_lr, ou_zscore, roll_half_life
from backtest import (
    Engine, BacktestConfig, TradeDef, SignalPipeline, SignalConfig,
    DV01Map, print_summary, trade_log,
)
from utils import pdf, fix_outliers

# ── config ────────────────────────────────────────────────────────────
DB_DSN = "postgresql://benjils:snickers@raptor:5432/markets"
LOOKBACK = 100
Z_ENTRY = 2.0
MAX_HL = 60  # reject trades where half-life > this (too slow to mean-revert)
START = "2000-01-01"
MAX_EXTEND = 30  # days beyond actual exit for "what if held longer"

# ── pull data (same as strat_10s30s) ──────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Baseline backtest (size=1, no stops)
# ══════════════════════════════════════════════════════════════════════

trade_10s30s = TradeDef.spread("10s30s", "otr_10Y", "otr_30Y")


def baseline_signal_fn(d: pl.DataFrame) -> pl.DataFrame:
    r = roll_lr(d["gen_10Y"], d["gen_1030"], lookback=LOOKBACK)
    zscore = ou_zscore(r["resid"], lookback=LOOKBACK)
    rhl = roll_half_life(r["resid"], lookback=LOOKBACK)
    out = pl.DataFrame({"signal": zscore, "time_stop": rhl})
    return out.with_columns(
        pl.when(
            pl.col("time_stop").is_not_null()
            & (pl.col("time_stop") <= MAX_HL)
        )
        .then(pl.col("signal"))
        .otherwise(None)
        .alias("signal")
    )


pipeline = SignalPipeline(
    name="10s30s",
    trade_def=trade_10s30s,
    compute_fn=baseline_signal_fn,
    config=SignalConfig(
        entry_long=-Z_ENTRY, entry_short=Z_ENTRY,
        exit_long=None, exit_short=None,
        time_stop_bars=None, max_positions=1,
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

if not result.closed_trades:
    print("No closed trades — nothing to analyze.")
    exit()

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Build metrics time series + trade paths panel
# ══════════════════════════════════════════════════════════════════════

reg = roll_lr(data["gen_10Y"], data["gen_1030"], lookback=LOOKBACK)
z_ts = ou_zscore(reg["resid"], lookback=LOOKBACK)
rhl_ts = roll_half_life(reg["resid"], lookback=LOOKBACK)
r2_ts = reg["r2"].fill_null(0).clip(0, 1)
beta_ts = reg["beta"]
beta_cv_ts = (
    beta_ts.rolling_std(LOOKBACK) / beta_ts.rolling_mean(LOOKBACK).abs()
).fill_null(2.0).clip(0, 2)
resid_vol_ts = reg["resid"].diff().rolling_std(20)
vol_ratio_ts = (resid_vol_ts / resid_vol_ts.rolling_mean(252)).fill_null(1.0)

# Pad if reg shorter than data (roll_lr may drop leading nulls)
n_data = data.shape[0]
n_reg = len(reg)
pad = n_data - n_reg


def _pad(s):
    return s.extend_constant(None, pad) if pad > 0 else s


# Numpy arrays aligned to data (for fast path construction)
composite = trade_10s30s.composite_series(data).to_numpy().astype(float)
z_arr = _pad(z_ts).to_numpy().astype(float)
r2_arr = _pad(r2_ts).to_numpy().astype(float)
bcv_arr = _pad(beta_cv_ts).to_numpy().astype(float)
beta_arr = _pad(beta_ts).to_numpy().astype(float)
rv_arr = _pad(resid_vol_ts).to_numpy().astype(float)
vr_arr = _pad(vol_ratio_ts).to_numpy().astype(float)
hl_arr = rhl_ts.to_numpy().astype(float)

# Metrics time series (for regime conditioning join)
metrics_ts = pl.DataFrame({
    "ts": data["ts"],
    "z": _pad(z_ts),
    "r2": _pad(r2_ts),
    "beta_cv": _pad(beta_cv_ts),
    "resid_vol": _pad(resid_vol_ts),
    "vol_ratio": _pad(vol_ratio_ts),
    "half_life": rhl_ts,
})

# ── Build trade paths ────────────────────────────────────────────────
ts_list = data["ts"].to_list()
ts_to_idx = {t: i for i, t in enumerate(ts_list)}

path_rows = []
for tid, trade in enumerate(result.closed_trades):
    edate = trade.entry_date
    if hasattr(edate, "date"):
        edate = edate.date()
    entry_idx = ts_to_idx.get(edate)
    if entry_idx is None:
        continue

    max_day = min(trade.bars_held + MAX_EXTEND, len(data) - 1 - entry_idx)
    e_comp = composite[entry_idx]

    # Entry-time metric values (for computing deltas)
    e_z = z_arr[entry_idx]
    e_r2 = r2_arr[entry_idx]
    e_bcv = bcv_arr[entry_idx]
    e_beta = beta_arr[entry_idx]
    e_rv = rv_arr[entry_idx]
    e_vr = vr_arr[entry_idx]

    for day in range(max_day + 1):
        idx = entry_idx + day
        path_rows.append({
            "trade_id": tid,
            "day": day,
            "is_winner": bool(trade.pnl_bps > 0),
            "exit_day": trade.bars_held,
            "beyond_exit": day > trade.bars_held,
            "direction": trade.direction,
            # PnL (normalized, size=1)
            "pnl_norm": trade.direction * (composite[idx] - e_comp),
            # Current metrics
            "z": z_arr[idx],
            "r2": r2_arr[idx],
            "beta_cv": bcv_arr[idx],
            "beta": beta_arr[idx],
            "resid_vol": rv_arr[idx],
            "vol_ratio": vr_arr[idx],
            "half_life": hl_arr[idx],
            # Changes from entry
            "z_chg": z_arr[idx] - e_z,
            "r2_chg": r2_arr[idx] - e_r2,
            "bcv_chg": bcv_arr[idx] - e_bcv,
            "beta_chg": beta_arr[idx] - e_beta,
            "vol_chg": rv_arr[idx] - e_rv,
            "vr_chg": vr_arr[idx] - e_vr,
        })

paths = pl.DataFrame(path_rows)
actual = paths.filter(~pl.col("beyond_exit"))

n_trades = len(result.closed_trades)
n_winners = sum(1 for t in result.closed_trades if t.pnl_bps > 0)
n_losers = n_trades - n_winners
n_years = (data["ts"].max() - data["ts"].min()).days / 365.25

# Pre-compute per-trade PnL arrays (actual holding period only)
trade_paths_np = {}
for tid in range(n_trades):
    tp = actual.filter(pl.col("trade_id") == tid).sort("day")
    trade_paths_np[tid] = tp["pnl_norm"].to_numpy()

# Pre-compute extended PnL lookup: {trade_id: {day: pnl}}
ext_pnl = {}
for tid in range(n_trades):
    tp = paths.filter(pl.col("trade_id") == tid).sort("day")
    ext_pnl[tid] = dict(zip(tp["day"].to_list(), tp["pnl_norm"].to_list()))

print(f"\n{'='*70}")
print(f"  INTRATRADE RESEARCH — {n_trades} trades ({n_winners}W / {n_losers}L)")
print(f"  Paths: {actual.shape[0]} actual + "
      f"{paths.shape[0] - actual.shape[0]} extended days")
print(f"{'='*70}")

# ── flag outlier trades ──────────────────────────────────────────────
OUTLIER_THRESH = 50  # bps — flag any trade with |pnl_norm| > this on any day
outlier_tids = (
    paths.filter(pl.col("pnl_norm").abs() > OUTLIER_THRESH)["trade_id"]
    .unique()
    .to_list()
)
if outlier_tids:
    print(f"\n⚠  {len(outlier_tids)} trade(s) with |pnl| > {OUTLIER_THRESH} bps on some day:")
    print(f"  {'tid':>4}  {'dir':>4}  {'entry':>12}  {'exit':>12}  "
          f"{'bars':>5}  {'final_pnl':>10}  {'peak_day_pnl':>13}  {'peak_day':>9}")
    print(f"  {'-'*75}")
    for tid in sorted(outlier_tids):
        trade = result.closed_trades[tid]
        tp = paths.filter(pl.col("trade_id") == tid).sort("day")
        pnl_arr = tp["pnl_norm"].to_numpy()
        peak_idx = int(np.argmax(np.abs(pnl_arr)))
        peak_day = tp["day"].to_list()[peak_idx]
        peak_val = pnl_arr[peak_idx]
        edate = trade.entry_date
        xdate = trade.exit_date
        if hasattr(edate, "date"):
            edate = edate.date()
        if hasattr(xdate, "date"):
            xdate = xdate.date()
        print(f"  {tid:4d}  {trade.direction:+4d}  {str(edate):>12}  {str(xdate):>12}  "
              f"{trade.bars_held:5d}  {trade.pnl_bps:+10.2f}  {peak_val:+13.2f}  "
              f"day {peak_day:<5d}")


# ── helper ────────────────────────────────────────────────────────────


def _trade_sharpe(pnl_arr):
    """Annualized trade-level Sharpe proxy: mean/std * sqrt(trades/year)."""
    if len(pnl_arr) < 2 or np.std(pnl_arr) == 0:
        return 0.0
    return float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(len(pnl_arr) / n_years))


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: PnL Paths — winners vs losers
# ══════════════════════════════════════════════════════════════════════

print(f"\n── PnL Paths: Winners vs Losers ──\n")

key_days = [1, 2, 3, 5, 8, 12, 15, 20, 25, 30]
print(f"  {'Day':>4}  {'Winners':>10}  {'Losers':>10}  {'All':>10}  {'n':>4}")
print(f"  {'-'*44}")

for d in key_days:
    dd = paths.filter(pl.col("day") == d)
    if dd.is_empty():
        continue
    w = dd.filter(pl.col("is_winner"))["pnl_norm"].to_numpy()
    l = dd.filter(~pl.col("is_winner"))["pnl_norm"].to_numpy()
    a = dd["pnl_norm"].to_numpy()
    print(f"  {d:4d}  {np.nanmean(w) if len(w) else np.nan:+10.2f}  "
          f"{np.nanmean(l) if len(l) else np.nan:+10.2f}  "
          f"{np.nanmean(a):+10.2f}  {len(a):4d}")

# Peak / trough timing
for label, filt, desc_flag, word in [
    ("Winners", True, True, "peak"),
    ("Losers", False, False, "trough"),
]:
    sub = actual.filter(pl.col("is_winner") == filt).drop_nulls(subset=["pnl_norm"])
    sub = sub.filter(pl.col("pnl_norm").is_finite())
    if sub.is_empty():
        continue
    extremes = (
        sub.sort("pnl_norm", descending=desc_flag)
        .group_by("trade_id", maintain_order=True)
        .first()
    )
    print(f"\n  {label} {word}: avg day {extremes['day'].mean():.1f}, "
          f"avg PnL {extremes['pnl_norm'].mean():+.2f} bps")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Metric Evolution — winners vs losers
# ══════════════════════════════════════════════════════════════════════

print(f"\n\n── Metric Evolution: Winners vs Losers ──")

evo_metrics = [
    ("z_chg",   "Δz (signal reverting?)"),
    ("r2_chg",  "ΔR² (fit quality?)"),
    ("bcv_chg", "Δbeta_cv (stability?)"),
    ("vol_chg", "Δresid_vol (vol regime?)"),
    ("vr_chg",  "Δvol_ratio (vol normalizing?)"),
]
evo_days = [1, 3, 5, 8, 12]

for col, label in evo_metrics:
    print(f"\n  {label}:")
    print(f"    {'Day':>4}  {'Winners':>10}  {'Losers':>10}  {'Delta':>10}")
    for d in evo_days:
        dd = actual.filter(pl.col("day") == d)
        if dd.is_empty():
            continue
        w = dd.filter(pl.col("is_winner"))[col].drop_nulls().to_numpy()
        l = dd.filter(~pl.col("is_winner"))[col].drop_nulls().to_numpy()
        wm = np.nanmean(w) if len(w) else np.nan
        lm = np.nanmean(l) if len(l) else np.nan
        print(f"    {d:4d}  {wm:+10.4f}  {lm:+10.4f}  {wm - lm:+10.4f}")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Exit Timing Sweep
# ══════════════════════════════════════════════════════════════════════

print(f"\n\n── Exit Timing Sweep ──")
print(f"  (What if we used a fixed N-day hold instead of rolling half-life?)\n")

print(f"  {'Day':>4}  {'avg PnL':>8}  {'hit%':>6}  {'Sharpe':>8}  {'n':>4}")
print(f"  {'-'*36}")

best_exit = {"day": 0, "sharpe": -999}
for ed in range(1, MAX_EXTEND + 1):
    pnls = [ext_pnl[tid].get(ed) for tid in range(n_trades)
            if ed in ext_pnl.get(tid, {})]
    pnls = [p for p in pnls if p is not None]
    if len(pnls) < 5:
        continue
    pnls = np.array(pnls)
    sharpe = _trade_sharpe(pnls)
    if sharpe > best_exit["sharpe"]:
        best_exit = {"day": ed, "sharpe": sharpe, "pnl": pnls.mean()}
    if ed in key_days:
        print(f"  {ed:4d}  {pnls.mean():+8.2f}  {100*(pnls > 0).mean():5.0f}%  "
              f"{sharpe:+8.2f}  {len(pnls):4d}")

avg_hl = np.mean([t.bars_held for t in result.closed_trades])
print(f"\n  → Best fixed exit: Day {best_exit['day']} "
      f"(Sharpe {best_exit['sharpe']:+.2f}, avg PnL {best_exit.get('pnl', 0):+.2f})")
print(f"  → Current (half-life): avg {avg_hl:.1f} days "
      f"(Sharpe {result.summary()['sharpe']:+.2f})")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 4a: Stop-Loss Sweep
# ══════════════════════════════════════════════════════════════════════

print(f"\n\n── Stop-Loss Sweep ──")
print(f"  (Hard stop during actual hold — exit at first breach)\n")

stop_levels = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
print(f"  {'Stop':>6}  {'stopped':>8}  {'avg PnL':>8}  {'hit%':>6}  {'Sharpe':>8}")
print(f"  {'-'*42}")

# Baseline (no stop)
final_pnls = np.array([
    trade_paths_np[tid][-1] if len(trade_paths_np[tid]) > 0 else 0
    for tid in range(n_trades)
])
base_sharpe = _trade_sharpe(final_pnls)
print(f"  {'None':>6}  {'0%':>8}  {final_pnls.mean():+8.2f}  "
      f"{100*(final_pnls > 0).mean():5.0f}%  {base_sharpe:+8.2f}")

best_stop = {"level": None, "sharpe": base_sharpe}

for stop in stop_levels:
    pnls = []
    n_stopped = 0
    for tid in range(n_trades):
        path = trade_paths_np[tid]
        stopped = False
        for p in path:
            if p <= -stop:
                pnls.append(-stop)
                n_stopped += 1
                stopped = True
                break
        if not stopped:
            pnls.append(path[-1] if len(path) > 0 else 0)
    pnls = np.array(pnls)
    sharpe = _trade_sharpe(pnls)
    if sharpe > best_stop["sharpe"]:
        best_stop = {"level": stop, "sharpe": sharpe}
    print(f"  {stop:6.1f}  {100*n_stopped/n_trades:7.0f}%  {pnls.mean():+8.2f}  "
          f"{100*(pnls > 0).mean():5.0f}%  {sharpe:+8.2f}")

if best_stop["level"]:
    print(f"\n  → Best stop: {best_stop['level']:.1f} bps "
          f"(Sharpe {best_stop['sharpe']:+.2f})")
else:
    print(f"\n  → No stop improves Sharpe over baseline")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 4b: Trailing Stop Sweep
# ══════════════════════════════════════════════════════════════════════

print(f"\n\n── Trailing Stop Sweep ──")
print(f"  (Exit when PnL drops T bps from its peak during the trade)\n")

trail_levels = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0]
print(f"  {'Trail':>6}  {'trailed':>8}  {'avg PnL':>8}  {'hit%':>6}  {'Sharpe':>8}")
print(f"  {'-'*42}")

print(f"  {'None':>6}  {'0%':>8}  {final_pnls.mean():+8.2f}  "
      f"{100*(final_pnls > 0).mean():5.0f}%  {base_sharpe:+8.2f}")

best_trail = {"level": None, "sharpe": base_sharpe}

for trail in trail_levels:
    pnls = []
    n_trailed = 0
    for tid in range(n_trades):
        path = trade_paths_np[tid]
        peak = 0.0
        trailed = False
        for p in path:
            peak = max(peak, p)
            if peak - p > trail:
                pnls.append(p)
                n_trailed += 1
                trailed = True
                break
        if not trailed:
            pnls.append(path[-1] if len(path) > 0 else 0)
    pnls = np.array(pnls)
    sharpe = _trade_sharpe(pnls)
    if sharpe > best_trail["sharpe"]:
        best_trail = {"level": trail, "sharpe": sharpe}
    print(f"  {trail:6.1f}  {100*n_trailed/n_trades:7.0f}%  {pnls.mean():+8.2f}  "
          f"{100*(pnls > 0).mean():5.0f}%  {sharpe:+8.2f}")

if best_trail["level"]:
    print(f"\n  → Best trail: {best_trail['level']:.1f} bps "
          f"(Sharpe {best_trail['sharpe']:+.2f})")
else:
    print(f"\n  → No trailing stop improves Sharpe over baseline")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Signal Extension — add or cut when z extends?
# ══════════════════════════════════════════════════════════════════════

print(f"\n\n── Signal Extension Analysis ──")
print(f"  (After entry, is z reverting toward 0 or extending further?)")
print(f"  Reverting = trade working | Extending = signal getting stronger\n")

# direction=+1 (long, entered z<0): z_chg > 0 → reverting (good)
# direction=-1 (short, entered z>0): z_chg < 0 → reverting (good)
# In both cases: direction * z_chg > 0 = reverting

check_days = [1, 2, 3, 5, 8]
print(f"  {'Day':>4}  {'Group':>10}  {'n':>4}  {'avg final':>10}  "
      f"{'hit%':>6}  {'Sharpe':>8}")
print(f"  {'-'*50}")

for d in check_days:
    dd = actual.filter(pl.col("day") == d).drop_nulls(subset=["z_chg"])
    if dd.is_empty():
        continue
    dirs = dd["direction"].to_numpy().astype(float)
    z_chgs = dd["z_chg"].to_numpy().astype(float)
    tids = dd["trade_id"].to_list()
    reverting = dirs * z_chgs > 0

    for label, mask in [("Reverting", reverting), ("Extending", ~reverting)]:
        group_tids = [tids[i] for i in range(len(tids)) if mask[i]]
        if len(group_tids) < 3:
            continue
        finals = np.array([
            trade_paths_np[tid][-1] for tid in group_tids
            if tid in trade_paths_np and len(trade_paths_np[tid]) > 0
        ])
        if len(finals) < 3:
            continue
        sharpe = _trade_sharpe(finals)
        print(f"  {d:4d}  {label:>10}  {len(finals):4d}  "
              f"{finals.mean():+10.2f}  {100*(finals > 0).mean():5.0f}%  "
              f"{sharpe:+8.2f}")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 6: Drawdown Recovery
# ══════════════════════════════════════════════════════════════════════

print(f"\n\n── Drawdown Recovery ──")
print(f"  (If underwater at day N, what fraction eventually profit?)\n")

print(f"  {'Day':>4}  {'n_uw':>6}  {'recover%':>9}  {'avg final':>10}  "
      f"{'avg trough':>11}")
print(f"  {'-'*48}")

for d in [1, 2, 3, 5, 8]:
    dd = actual.filter(pl.col("day") == d)
    uw_tids = dd.filter(pl.col("pnl_norm") < 0)["trade_id"].to_list()
    if not uw_tids:
        continue
    finals, troughs = [], []
    for tid in uw_tids:
        path = trade_paths_np.get(tid, np.array([0]))
        finals.append(path[-1])
        troughs.append(path.min())
    finals, troughs = np.array(finals), np.array(troughs)
    print(f"  {d:4d}  {len(uw_tids):6d}  {100*(finals > 0).mean():8.0f}%  "
          f"{finals.mean():+10.2f}  {troughs.mean():+11.2f}")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 7: Regime Conditioning
# ══════════════════════════════════════════════════════════════════════

print(f"\n\n── Regime Conditioning ──")
print(f"  (How do entry conditions affect outcome & optimal exit?)\n")

tlog_rc = trade_log(result.closed_trades)
tlog_rc = tlog_rc.with_columns(pl.col("entry_date").cast(pl.Date).alias("ts"))
enriched_rc = tlog_rc.join(metrics_ts, on="ts", how="left")

for rm in ["r2", "beta_cv", "resid_vol", "vol_ratio", "half_life"]:
    col_vals = enriched_rc[rm].to_list()
    valid = [(tid, float(v)) for tid, v in enumerate(col_vals)
             if v is not None and not np.isnan(float(v))]
    if len(valid) < 10:
        continue

    med = float(np.median([v for _, v in valid]))

    for label, cond in [("Low ", lambda v: v <= med),
                        ("High", lambda v: v > med)]:
        ids = [tid for tid, v in valid if cond(v)]
        if len(ids) < 3:
            continue

        # Final PnL for this group
        group_pnls = np.array([
            trade_paths_np[tid][-1] for tid in ids
            if tid in trade_paths_np and len(trade_paths_np[tid]) > 0
        ])
        if len(group_pnls) < 3:
            continue

        # Optimal exit day for this regime
        best_day, best_sh = 1, -999.0
        for ed in range(1, MAX_EXTEND + 1):
            epnls = [ext_pnl[tid].get(ed) for tid in ids
                     if ed in ext_pnl.get(tid, {})]
            epnls = [p for p in epnls if p is not None]
            if len(epnls) < 3:
                continue
            sh = _trade_sharpe(np.array(epnls))
            if sh > best_sh:
                best_sh, best_day = sh, ed

        # Best stop for this regime
        best_stop_lvl, best_stop_sh = None, _trade_sharpe(group_pnls)
        for stop in [1.0, 2.0, 3.0, 5.0, 8.0]:
            spnls = []
            for tid in ids:
                path = trade_paths_np.get(tid, np.array([0]))
                stopped = False
                for p in path:
                    if p <= -stop:
                        spnls.append(-stop)
                        stopped = True
                        break
                if not stopped:
                    spnls.append(path[-1] if len(path) > 0 else 0)
            sh = _trade_sharpe(np.array(spnls))
            if sh > best_stop_sh:
                best_stop_sh, best_stop_lvl = sh, stop

        stop_str = f"stop={best_stop_lvl:.0f}" if best_stop_lvl else "no stop"
        print(f"  {rm:>10} {label} (cut={med:.2f}, n={len(ids):2d}): "
              f"PnL={group_pnls.mean():+.2f}, "
              f"hit={100*(group_pnls > 0).mean():.0f}%, "
              f"opt exit=Day {best_day} (Sharpe {best_sh:+.2f}), "
              f"{stop_str}")

    print()
