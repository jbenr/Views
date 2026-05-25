#!/usr/bin/env python
# coding: utf-8

# # 10Y Duration — Exploration Notebook
# 
# Before building a signal, characterize the 10Y yield as a time series. Establish what's stationary, what's predictable, and which of several candidate "support/resistance" features (if any) actually have edge.
# 
# **Sections:**
# - **A.** Just look at it — level, changes, rolling vol
# - **B.** Statistical properties — distribution, stationarity, autocorrelation, mean-reversion vs trending
# - **C.** Candidate S/R features — define cleanly, no lookahead, plot to sanity-check
# 
# Predictiveness testing comes after we've looked at A–C output together.

# ## Setup

# In[1]:


from __future__ import annotations

import os
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import polars as pl
import psycopg
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option("display.max_colwidth", None)   # show full strings
pd.set_option("display.width", 200)

ROOT = Path.cwd().resolve()
if not (ROOT / "stats").exists():
    for p in ROOT.parents:
        if (p / "stats").exists():
            ROOT = p
            break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stats import roll_lr_diff, ou_zscore, half_life, roll_half_life
from utils.viz import Viz

warnings.filterwarnings("ignore", category=RuntimeWarning)

v = Viz()  # default plotter for the notebook

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
START = "2000-01-01"
TICKER_10Y = "USGG10YR Index"  # the only series we need for now


# ## 1. Data Pull
# 
# Just the 10Y yield. We'll add more series only when something tells us we need them.

# In[3]:


def query_to_pl(sql: str) -> pl.DataFrame:
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame({col: [row[i] for row in rows] for i, col in enumerate(cols)})

raw = query_to_pl(f"""
    SELECT ts, px_last::float
    FROM md.index_eod
    WHERE ticker = '{TICKER_10Y}'
      AND ts >= '{START}'
    ORDER BY ts
""")

# ×100 to get bps
df = raw.to_pandas().set_index("ts").sort_index()
df.columns = ["y10"]
df["y10"] = df["y10"] * 100.0

print(f"10Y yield: {len(df)} obs  ·  {df.index.min().date()} → {df.index.max().date()}")
print(f"  Range: {df['y10'].min():.0f} → {df['y10'].max():.0f} bps  ·  Latest: {df['y10'].iloc[-1]:.0f}")
display(df.tail(5))


# ## A. Just look at it
# 
# Eyeball the level, changes, rolling vol. No models — just see the shape.

# In[4]:


y10 = df["y10"]
ch  = y10.diff().dropna()

# 10Y level
v.line(
    y10.to_frame("10Y"),
    title=f"10Y UST yield — {len(y10)} obs",
    yaxis_title="bps",
)

# Daily changes — bars colored by sign (green up, red down)
v.line(
    ch.to_frame("daily_change"),
    title="10Y daily change",
    yaxis_title="bps/day",
    bar=True,
    hlines=[(0, None, "solid")],
)

# Rolling vol of changes
vol_panel = pd.DataFrame({
    "vol_60d":  ch.rolling(60).std(),
    "vol_252d": ch.rolling(252).std(),
})
v.line(
    vol_panel,
    title="Rolling vol of daily changes",
    yaxis_title="bps/day",
)


# In[5]:


# Summary stats by year — does behavior shift across regimes?
yr = pd.DataFrame({
    "level_mean":  y10.groupby(y10.index.year).mean(),
    "level_min":   y10.groupby(y10.index.year).min(),
    "level_max":   y10.groupby(y10.index.year).max(),
    "level_range": y10.groupby(y10.index.year).max() - y10.groupby(y10.index.year).min(),
    "ch_std":      ch.groupby(ch.index.year).std(),
    "ch_skew":     ch.groupby(ch.index.year).skew(),
})
display(yr.round(1))


# ## B. Statistical properties
# 
# What kind of stochastic process are we dealing with?
# - **Distribution** of changes (Gaussian or fat-tailed?)
# - **Stationarity** of the level (almost certainly no) and of changes (likely yes)
# - **Autocorrelation** of changes (momentum / mean-reversion at what lags?)
# - **Vol clustering** (autocorr of squared changes — GARCH-y?)
# - **Variance ratio / Hurst** — random walk vs persistent vs anti-persistent?

# In[6]:


# Data hygiene: just the facts we actually need from yield changes.
#   - tails fatter than Normal? → naive z-scores will underweight tail risk
#   - lag-1 ACF of Δy meaningful? → if so, trade momentum, skip the FV model
#   - lag-1 ACF of Δy² meaningful? → vol clusters, use vol-targeted sizing
from statsmodels.tsa.stattools import acf

ch_clean = ch.dropna()
ac_lvl = acf(ch_clean, nlags=5, fft=True)
ac_sq  = acf(ch_clean ** 2, nlags=5, fft=True)

print(f"kurtosis(Δy10)        = {ch_clean.kurt():+.2f}   (Normal = 0, >3 → fat tails)")
print(f"skew(Δy10)            = {ch_clean.skew():+.2f}")
print(f"ACF(Δy10,   lag 1)    = {ac_lvl[1]:+.3f}   (≈0 → no naive daily momentum)")
print(f"ACF(Δy10,   lag 5)    = {ac_lvl[5]:+.3f}")
print(f"ACF(Δy10²,  lag 1)    = {ac_sq[1]:+.3f}   (>0 → vol clustering, vol-target later)")
print(f"ACF(Δy10²,  lag 5)    = {ac_sq[5]:+.3f}")


# In[7]:


# Stationarity — Augmented Dickey-Fuller
from statsmodels.tsa.stattools import adfuller

def adf(s, label):
    s = s.dropna()
    if len(s) < 100:
        print(f"{label:35s}  too short"); return
    stat, p, lags, n, crit, _ = adfuller(s, autolag="AIC")
    verdict = "STATIONARY" if p < 0.05 else "non-stationary"
    print(f"{label:35s}  stat={stat:+8.3f}  p={p:.4f}  n={n}  → {verdict}")

print("Augmented Dickey-Fuller (null = unit root / non-stationary):\n")
adf(y10,                                            "10Y level")
adf(y10.diff(),                                     "10Y daily change")
adf(y10 - y10.rolling(60).mean(),                   "10Y minus 60d MA")
adf(y10 - y10.rolling(252).mean(),                  "10Y minus 252d MA")
adf((y10 - y10.rolling(252).mean()) / y10.rolling(252).std(), "10Y rolling z-score (252d)")


# In[8]:


# Variance ratio + Hurst exponent
def variance_ratio(s, q):
    """VR(q) = Var(q-period change) / (q · Var(1-period change)).
    VR = 1 → random walk · VR < 1 → mean-reverting · VR > 1 → trending"""
    s = s.dropna()
    r1 = s.diff().dropna()
    rq = s.diff(q).dropna()
    return rq.var() / (q * r1.var())

def hurst(s, lags=range(2, 100)):
    """Power-law fit of std(diff_lag) ~ lag^H. H ≈ 0.5 random, <0.5 MR, >0.5 trend."""
    s = s.dropna().to_numpy()
    tau = [np.std(s[lag:] - s[:-lag]) for lag in lags]
    H, _ = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return H

print("Variance ratio (1.0 = random walk, < 1 = mean-reverting, > 1 = trending):")
for q in [5, 20, 60, 120, 252]:
    print(f"  VR({q:3d}d) = {variance_ratio(y10, q):.3f}")

print(f"\nHurst exponent on level: H = {hurst(y10):.3f}")
print(f"  (0.5 = random walk, < 0.5 = anti-persistent / MR, > 0.5 = persistent / trending)")


# ## C. Candidate S/R features
# 
# Define each cleanly, with no lookahead, in a way we can later test for predictiveness.
# 
# 1. **Rolling extrema** — distance to N-day high/low for N ∈ {20, 60, 252}
# 2. **Round numbers** — distance to nearest 25bp grid line
# 3. **Williams swing pivots** — confirmed K-bar pivot, only available K bars later
# 4. **Moving averages** — 50d/200d as dynamic S/R
# 
# Compute. Plot the last 2 years with all overlays. Sanity-check the math against the chart.

# In[9]:


sr = pd.DataFrame(index=df.index)
sr["y10"] = y10

# Rolling extrema — shift(1) so today's value never enters today's high/low (no lookahead)
for win in [20, 60, 252]:
    sr[f"high_{win}"]      = y10.shift(1).rolling(win).max()
    sr[f"low_{win}"]       = y10.shift(1).rolling(win).min()
    sr[f"dist_high_{win}"] = sr[f"high_{win}"] - y10   # +ve: yield is below the high
    sr[f"dist_low_{win}"]  = y10 - sr[f"low_{win}"]    # +ve: yield is above the low

# Round numbers (25bp grid) — signed distance to nearest
sr["nearest_25bp"] = (y10 / 25).round() * 25
sr["dist_25bp"]    = y10 - sr["nearest_25bp"]
sr["abs_25bp"]     = sr["dist_25bp"].abs()

# Moving averages — shift(1) again
for ma in [50, 200]:
    sr[f"ma_{ma}"]      = y10.shift(1).rolling(ma).mean()
    sr[f"dist_ma_{ma}"] = y10 - sr[f"ma_{ma}"]

print("Rolling extrema, round numbers, MAs computed.")
display(sr[["y10", "dist_high_60", "dist_low_60", "dist_25bp", "dist_ma_50", "dist_ma_200"]].tail(5).round(1))


# In[10]:


# Williams Fractal-style swing pivots
# A high at t requires y[t] >= y[t-K..t+K]. Confirmation only K bars later → shift output by K.
def swing_levels(y: pd.Series, K: int = 5):
    n = len(y); arr = y.to_numpy()
    is_high = pd.Series(False, index=y.index)
    is_low  = pd.Series(False, index=y.index)
    for i in range(K, n - K):
        w = arr[i-K:i+K+1]
        if arr[i] == w.max(): is_high.iloc[i] = True
        if arr[i] == w.min(): is_low.iloc[i]  = True
    return y.where(is_high).shift(K).ffill(), y.where(is_low).shift(K).ffill()

for K in [5, 10, 20]:
    sh, sl = swing_levels(y10, K=K)
    sr[f"swing_high_K{K}"] = sh
    sr[f"swing_low_K{K}"]  = sl
    sr[f"dist_sh_K{K}"]    = sh - y10
    sr[f"dist_sl_K{K}"]    = y10 - sl

print("Swing pivots computed for K = 5, 10, 20.")
display(sr[["y10", "swing_high_K5", "swing_low_K5", "dist_sh_K5", "dist_sl_K5"]].tail(5).round(1))


# In[11]:


# Split S/R levels and moving averages into separate charts.
# S/R chart: 10Y bold; rolling 60d extrema dashed; confirmed swing pivots dotted.
sr_cols = [
    "y10",
    "high_60", "low_60",
    "swing_high_K5", "swing_low_K5",
]
cur_60_max = sr["high_60"].iloc[-1]
cur_60_min = sr["low_60"].iloc[-1]

v.line(
    sr[sr_cols],
    title="10Y with candidate S/R levels",
    yaxis_title="bps",
    show_endpoint_marker=False,
    linestyles={
        "y10":            "bold",
        "high_60":        "dashed",
        "low_60":         "dashed",
        "swing_high_K5":  "dotted",
        "swing_low_K5":   "dotted",
    },
    hlines=[
        {"value": cur_60_max, "label": f"60d max now ({cur_60_max:.0f})", "style": "dashed", "color": "#C0392B"},
        {"value": cur_60_min, "label": f"60d min now ({cur_60_min:.0f})", "style": "dashed", "color": "#27AE60"},
    ],
)

ma_cols = ["y10", "ma_50", "ma_200"]
v.line(
    sr[ma_cols],
    title="10Y with moving averages",
    yaxis_title="bps",
    show_endpoint_marker=False,
    linestyles={
        "y10":   "bold",
        "ma_50": "thin",
        "ma_200": "thin",
    },
)


# ## D. Setup Battery
# 
# Systematic evaluation of trade hypotheses. Each `Setup` is `(trigger, direction)` —
# when the trigger fires, bet that yields move in the declared direction. Evaluated at
# multiple forward horizons against:
# - per-trade PnL, hit rate, profit factor, t-stat, annualized Sharpe
# - shuffled-trigger null distribution (significance vs random firing of same N triggers)
# 
# Then pairwise AND combinations among same-direction setups, with **lift-vs-components**
# reported so we can see whether a combo actually adds info or just selects a subset of
# one component's good trades.
# 
# **Starter library:** 8 setups across momentum, mean-reversion, and valuation families.

# In[12]:


# Pull additional series for valuation setups.
# The setup catalog dynamically uses whichever columns exist in `curve`.
from utils.rates import with_synthetic_real_rates

extra_tickers = {
    "30y": "USGG30YR Index",
    "2y":  "USGG2YR Index",

    # Synthetic real-rate inputs. These are optional: if the database has them,
    # `with_synthetic_real_rates` will create real_5y, real_10y, real_5y5y.
    "sofr_5y":  "USOSFR5 Curncy",
    "sofr_10y": "USOSFR10 Curncy",
    "zcis_5y":  "USSWIT5 Curncy",
    "zcis_10y": "USSWIT10 Curncy",
}
sql_extra = f"""
    SELECT ts, ticker, px_last::float
    FROM md.index_eod
    WHERE ticker = ANY(ARRAY{list(extra_tickers.values())!r}::text[])
      AND ts >= '{START}'
    ORDER BY ts
"""
extra_raw = query_to_pl(sql_extra)
extra_wide = (
    extra_raw.pivot(index="ts", on="ticker", values="px_last")
    .sort("ts")
    .rename({v: k for k, v in extra_tickers.items()})
)

curve = extra_wide.to_pandas().set_index("ts").sort_index()
curve = curve.reindex(df.index).ffill()

# Convert available rate columns from percent/decimal-ish quote units to bps,
# matching df["y10"]. Missing optional columns are ignored.
rate_cols = [c for c in extra_tickers if c in curve.columns]
curve[rate_cols] = curve[rate_cols] * 100.0

curve["10y"]    = df["y10"]
if {"30y", "10y"}.issubset(curve.columns):
    curve["10s30s"] = curve["30y"] - curve["10y"]
if {"2y", "10y"}.issubset(curve.columns):
    curve["2s10s"]  = curve["10y"] - curve["2y"]

curve = with_synthetic_real_rates(curve)

print(f"Curve frame: {len(curve)} rows  ?  {curve.index.min().date()} ? {curve.index.max().date()}")
print("Available curve columns:", list(curve.columns))
display(curve.tail(3))


# ### Setup library
# 
# Eight starter setups, balanced 4 long-yield / 4 short-yield, mix of families:

# In[13]:


# Add signal/duration to sys.path so we can import setups directly,
# bypassing the 'signal' package — Python's stdlib has a 'signal' module
# that gets loaded first by matplotlib etc., shadowing our local package.
import importlib
_setups_dir = str(ROOT / "signal" / "duration")
if _setups_dir not in sys.path:
    sys.path.insert(0, _setups_dir)
import setups as _setups_mod
importlib.reload(_setups_mod)  # pick up edits without kernel restart
from setups import (
    Setup, evaluate, evaluate_battery, combine_pairs, add_lift_vs_components,
    build_duration_setup_catalog, find_setup, describe_setup, setup_report,
    PositionRule, trade_log, episode_trade_log, position_trade_log,
    setup_signal_frame, plot_setup_diagnostics,
    evaluate_trade_log, evaluate_battery_from_trades, position_scoreboard,
    performance_stats, performance_stats_frame,
    default_exit_policies, compare_exit_policies, exit_policy_report,
    s_n_consecutive, s_at_extreme, s_bb_extreme, s_resid_extreme,
)

y10 = curve["10y"]

setups = [
    # Momentum / continuation
    s_n_consecutive(y10, n=3, direction=+1),  # 3 up days, expect more up
    s_n_consecutive(y10, n=3, direction=-1),  # 3 down days, expect more down

    # Mean reversion — at recent extreme
    s_at_extreme(y10, win=60, side="high", bps_tol=5.0),  # near 60d high → bet down
    s_at_extreme(y10, win=60, side="low",  bps_tol=5.0),  # near 60d low  → bet up

    # Mean reversion — Bollinger extreme (continuous score)
    s_bb_extreme(y10, win=60, side="high", z_thresh=2.0),
    s_bb_extreme(y10, win=60, side="low",  z_thresh=-2.0),

    # Valuation — 10y vs 10s30s residual (changes regression, OU z on cum residual)
    s_resid_extreme(curve["10y"], curve["10s30s"], lookback=120, side="high",
                    z_thresh=2.0, name_prefix="val_1030"),
    s_resid_extreme(curve["10y"], curve["10s30s"], lookback=120, side="low",
                    z_thresh=-2.0, name_prefix="val_1030"),
]

# Override the legacy 8-setup list with the expanded catalog.
setups = build_duration_setup_catalog(curve, y_col="10y", profile="wide")

print(f"{len(setups)} setups defined.\n")
for s in setups:
    n_trig = int(s.trigger.fillna(False).sum())
    print(f"  {s.name:30s}  family={s.family:<15} dir={s.direction:+d}  triggers={n_trig}")


# ### Singles leaderboard
# 
# Per-trade metrics at horizons 5/20/60d. `null_pct` is the percentile of the real
# per-trade Sharpe vs 500 shuffled-trigger null draws — values near 100 = beats almost
# every random firing; values near 50 = indistinguishable from noise.

# In[24]:


scoreboard_policies = [
    PositionRule(exit_on_signal_false=True, name="signal_false"),
    PositionRule(exit_on_signal_false=False, max_holding_period=5, name="fixed_5d"),
    PositionRule(exit_on_signal_false=False, max_holding_period=10, name="fixed_10d"),
    PositionRule(exit_on_signal_false=False, max_holding_period=20, name="fixed_20d"),
    PositionRule(exit_on_signal_false=False, max_holding_period=40, name="fixed_40d"),
    PositionRule(exit_on_signal_false=False, max_holding_period=60, name="fixed_60d"),
]
singles_df = position_scoreboard(setups, y10, policies=scoreboard_policies, min_trades=5, show_progress=True)

# Rank actual position trades, not trigger-days. sharpe_ann = sharpe_pertrade * sqrt(252/avg_hold)
# is the apples-to-apples cross-policy metric (a 2d trade can fire ~25x more often than a 50d one).
display_cols = [
    "name", "family", "direction", "exit_policy", "n_trades", "avg_hold",
    "mean_pnl_bps", "total_pnl_bps", "hit_rate", "profit_factor",
    "sharpe_pertrade", "sharpe_ann", "t_stat", "max_drawdown_bps",
]
print(f"\nSingles leaderboard ({len(singles_df)} setup-exit-policy rows, actual position trades):")
with pd.option_context("display.max_colwidth", None, "display.width", 240):
    display(
        singles_df[display_cols]
        .sort_values(["sharpe_ann", "mean_pnl_bps"], ascending=[False, False])
        .round(3)
        .reset_index(drop=True)
    )


# ### Pairwise combinations
# 
# AND-combine same-direction setups. `lift_vs_components` = combo Sharpe minus
# max(component Sharpes) — this is the column to scrutinize. A combo that doesn't
# beat both components by a meaningful margin is just selecting a subset.

# In[15]:


from setups import classify_type, trade_side

# Pairing every setup with every other setup gets noisy and expensive fast.
# Use the best single-name candidates as the pair universe.
pair_names = (
    singles_df.assign(rank_score=singles_df["sharpe_ann"] + 0.01 * singles_df["hit_rate"])
    .sort_values("rank_score", ascending=False)
    .drop_duplicates("name")
    .head(60)["name"]
    .tolist()
)
pair_base = [s for s in setups if s.name in pair_names]
pair_setups = combine_pairs(pair_base, min_triggers=20, show_progress=True)
print(f"Built {len(pair_setups)} pair combinations from {len(pair_base)} screened setups (min 20 triggers, same-direction).\n")

pairs_df = position_scoreboard(pair_setups, y10, policies=scoreboard_policies, min_trades=5, show_progress=True)
pairs_df = pairs_df.assign(
    type=pairs_df["family"].map(classify_type),
    side=pairs_df["direction"].map(trade_side),
)

pair_display_cols = [
    "name", "type", "side", "exit_policy", "n_trades", "avg_hold",
    "mean_pnl_bps", "total_pnl_bps", "hit_rate", "profit_factor",
    "sharpe_pertrade", "sharpe_ann", "t_stat", "max_drawdown_bps",
]

def show_pair_board(df, side, kind, n=20):
    sub = df[(df["side"] == side) & (df["type"] == kind)]
    print(f"\n── {side.upper():5s} {kind.upper():5s}  ({len(sub)} rows) ──")
    if sub.empty:
        print("  (nothing)")
        return
    display(
        sub[pair_display_cols]
        .sort_values(["sharpe_ann", "mean_pnl_bps"], ascending=[False, False])
        .round(3).reset_index(drop=True).head(n)
    )

with pd.option_context("display.max_colwidth", None, "display.width", 240):
    show_pair_board(pairs_df, "long",  "mr")
    show_pair_board(pairs_df, "long",  "mom")
    show_pair_board(pairs_df, "short", "mr")
    show_pair_board(pairs_df, "short", "mom")

# Mixed-style pairs (one leg mr, one leg mom) — often where orthogonal signals live.
mixed = pairs_df[pairs_df["type"] == "mixed"]
print(f"\n{len(mixed)} mixed-style pair rows (style disagreement between legs).")
if len(mixed):
    with pd.option_context("display.max_colwidth", None, "display.width", 240):
        show_pair_board(pairs_df, "long",  "mixed")
        show_pair_board(pairs_df, "short", "mixed")


# ### What to look for
# 
# - **Sharpe > 1.0 with `null_pct > 95`**: real signal. The setup beats nearly all random firings of the same trigger count.
# - **`lift_vs_components > 0.3`** on a combo: the AND is doing meaningful work, not just subset-selecting.
# - **`null_pct ≈ 50`**: noise. Don't get fooled by a high Sharpe if the null says random would do the same.
# - **Tiny `n_triggers`**: even high Sharpe is suspect. Min ~30 trades for any read; ~100 to be confident.
# 
# Next steps depending on what we see:
# - Expand the catalog (more lookbacks, more thresholds, more valuation anchors)
# - Triple combinations on the surviving pair leaders
# - Score-based composition (continuous z-aggregate instead of boolean AND)
# - In/out-of-sample split — currently this is full-sample

# In[16]:


all_setups = setups + pair_setups

name = "mr_ma_dist_low_20d_z2 & trend_ma_60_252_bull"
r = setup_report(name, setups + pair_setups, y10, plot=True, interactive_plot=True)

print(r["description"])
display(r["trades"].tail(20))
display(r["stats_frame"])


# In[17]:


exit_df = exit_policy_report(name, setups + pair_setups, y10, show_progress=True)
display(exit_df.round(3))


# In[18]:


rule = PositionRule(
    exit_on_signal_false=False,
    exit_score_operator="abs<=",
    exit_score_level=1.0,
    exit_on_any_component_score=True,
    max_holding_period=40,
    name="score_any_abs_le_1_or_40d",
)

r = setup_report(
    name,
    setups + pair_setups,
    y10,
    position_rule=rule,
    plot=True,
    interactive_plot=True
)
print(r["description"])
display(r["trades"])
display(r["stats_frame"])


# In[19]:


# ── OU exit policy deep-dive on one chosen setup ──────────────────────
import importlib
import exits_ou
importlib.reload(exits_ou)
from exits_ou import (
    components_from_pair, ou_time_stop_log, ou_level_target_log, compare_ou_exits, summarize_log
)

# Pick the setup to dig into. Change as desired.
TARGET_NAME = "val_2y_low_lb120 & val_5y5y_zc_low_lb252"

all_setups = setups + pair_setups
by_name = {s.name: s for s in all_setups}
if TARGET_NAME not in by_name:
    raise KeyError(f"{TARGET_NAME} not in catalog — pick another from the top-Sharpe table")
target = by_name[TARGET_NAME]
components = components_from_pair(target, all_setups)
print(f"Components: {[c.name for c in components]}")
for c in components:
    has_resid = "yes" if c.residual is not None else "no"
    print(f"  {c.name}: residual={has_resid}")

# 1) Baseline: existing default policies on this setup
baseline_rows = []
for rule in default_exit_policies(target):
    log = position_trade_log(target, y10, rule=rule, components=components)
    baseline_rows.append(summarize_log(log, target.name, rule.name))
baseline = pd.DataFrame(baseline_rows)

# 2) New OU policies
ou = compare_ou_exits(target, y10, all_setups,
                     ks=(0.5, 1.0, 1.5, 2.0, 3.0),
                     fractions=(0.25, 0.5, 0.75, 1.0),
                     hl_lookback=252, hard_cap=90)

combined = pd.concat([baseline, ou], ignore_index=True)
combined = combined.sort_values("sharpe_ann", ascending=False).reset_index(drop=True)
with pd.option_context("display.max_colwidth", None, "display.width", 220):
    display(combined)

# 3) Per-trade detail for the best OU variant
best_ou_row = ou.sort_values("sharpe_ann", ascending=False).iloc[0]
best_ou_policy = best_ou_row["exit_policy"]
print(f"\nBest OU variant on this setup: {best_ou_policy}  "
      f"(sharpe_ann={best_ou_row['sharpe_ann']:.2f}, n={int(best_ou_row['n_trades'])}, "
      f"avg_hold={best_ou_row['avg_hold']:.1f}d)")
print(f"Best overall on this setup:    {combined.iloc[0]['exit_policy']}  "
      f"(sharpe_ann={combined.iloc[0]['sharpe_ann']:.2f})")

if best_ou_policy.startswith("ou_time_stop_k"):
    k = float(best_ou_policy.split("_k")[1])
    detail = ou_time_stop_log(target, y10, components, k=k, hl_lookback=252, hard_cap=90)
elif best_ou_policy.startswith("ou_level_"):
    parts = best_ou_policy.split("_")
    req = parts[2]                                     # 'any' or 'all'
    f = int(parts[3].lstrip("f").rstrip("p")) / 100    # 'f50p' -> 0.5
    detail = ou_level_target_log(target, y10, components, fraction=f, require=req, hl_lookback=252, hard_cap=90)
with pd.option_context("display.max_colwidth", None, "display.width", 240):
    display(detail.head(15))


# ## E. Beta Model Discrepancy Study
# 
# Hypothesis: some market participants may estimate 10Y beta to an anchor using levels, while a tradable beta should usually be estimated on changes in levels. This section compares the two beta estimates and tests whether their discrepancy has forward edge for 10Y yield moves.

# In[20]:


def rolling_beta_levels(y, x, lookback=120):
    """Rolling beta from levels: y ~ alpha + beta * x."""
    cov = y.rolling(lookback).cov(x)
    var = x.rolling(lookback).var()
    return cov / var


def rolling_beta_changes(y, x, lookback=120):
    """Rolling beta from daily changes: ?y ~ alpha + beta * ?x."""
    dy = y.diff()
    dx = x.diff()
    cov = dy.rolling(lookback).cov(dx)
    var = dx.rolling(lookback).var()
    return cov / var


def beta_discrepancy_frame(curve, y_col="10y", anchor_col="2y", lookback=120, z_win=252):
    y = curve[y_col]
    x = curve[anchor_col]
    out = pd.DataFrame(index=curve.index)
    out["y"] = y
    out[f"beta_level_{anchor_col}_{lookback}d"] = rolling_beta_levels(y, x, lookback=lookback)
    out[f"beta_chg_{anchor_col}_{lookback}d"] = rolling_beta_changes(y, x, lookback=lookback)
    out["beta_gap"] = out[f"beta_level_{anchor_col}_{lookback}d"] - out[f"beta_chg_{anchor_col}_{lookback}d"]
    out["beta_gap_z"] = (out["beta_gap"] - out["beta_gap"].rolling(z_win).mean()) / out["beta_gap"].rolling(z_win).std()
    for h in [1, 5, 10, 20, 40, 60]:
        out[f"fwd_{h}d_chg_bps"] = y.shift(-h) - y
    return out


beta_anchor = "2y"
beta_lookback = 120
beta_z_win = 252

beta_gap = beta_discrepancy_frame(
    curve,
    y_col="10y",
    anchor_col=beta_anchor,
    lookback=beta_lookback,
    z_win=beta_z_win,
)

print(f"Beta discrepancy frame: {len(beta_gap)} rows ? anchor={beta_anchor} ? lookback={beta_lookback}d ? z_win={beta_z_win}d")
display(beta_gap.tail(5).round(3))


# In[21]:


beta_cols = [
    f"beta_level_{beta_anchor}_{beta_lookback}d",
    f"beta_chg_{beta_anchor}_{beta_lookback}d",
]
v.line(
    beta_gap[beta_cols],
    title=f"10Y beta to {beta_anchor}: levels vs changes",
    yaxis_title="beta",
    show_endpoint_marker=False,
    linestyles={
        beta_cols[0]: "thin",
        beta_cols[1]: "bold",
    },
)

v.line(
    beta_gap[["beta_gap_z"]],
    title=f"10Y beta model discrepancy z-score: levels beta minus changes beta",
    yaxis_title="z-score",
    show_endpoint_marker=False,
    hlines=[
        {"value": 2, "label": "+2", "style": "dashed", "color": "#C0392B"},
        {"value": -2, "label": "-2", "style": "dashed", "color": "#27AE60"},
        {"value": 0, "label": "0", "style": "dotted", "color": "#666"},
    ],
)


# In[22]:


def beta_gap_forward_study(beta_gap, z_col="beta_gap_z", thresholds=(1.0, 1.5, 2.0), horizons=(1, 5, 10, 20, 40, 60)):
    rows = []
    for thresh in thresholds:
        states = {
            f"gap_z_high_gt_{thresh:g}": beta_gap[z_col] >= thresh,
            f"gap_z_low_lt_-{thresh:g}": beta_gap[z_col] <= -thresh,
        }
        for state_name, mask in states.items():
            for h in horizons:
                fwd_col = f"fwd_{h}d_chg_bps"
                sample = beta_gap.loc[mask, fwd_col].dropna()
                if sample.empty:
                    continue
                rows.append({
                    "state": state_name,
                    "horizon": h,
                    "n_obs": int(sample.shape[0]),
                    "mean_fwd_chg_bps": sample.mean(),
                    "median_fwd_chg_bps": sample.median(),
                    "hit_yields_up": (sample > 0).mean(),
                    "hit_yields_down": (sample < 0).mean(),
                    "std_fwd_chg_bps": sample.std(ddof=1),
                    "t_stat": sample.mean() / (sample.std(ddof=1) / np.sqrt(sample.shape[0])) if sample.shape[0] > 1 and sample.std(ddof=1) > 0 else np.nan,
                })
    return pd.DataFrame(rows)

beta_gap_summary = beta_gap_forward_study(beta_gap)
display(
    beta_gap_summary
    .sort_values(["horizon", "state"])
    .round(3)
)


# In[23]:


# Optional: de-cluster discrepancy extremes into signal episodes instead of daily observations.
def beta_gap_episode_study(beta_gap, z_col="beta_gap_z", threshold=2.0, horizon=20, side="high"):
    if side == "high":
        mask = beta_gap[z_col] >= threshold
    elif side == "low":
        mask = beta_gap[z_col] <= -threshold
    else:
        raise ValueError("side must be 'high' or 'low'")
    entries = mask & ~mask.shift(1).fillna(False)
    fwd_col = f"fwd_{horizon}d_chg_bps"
    out = beta_gap.loc[entries, [z_col, fwd_col]].dropna().copy()
    out = out.rename(columns={z_col: "entry_gap_z", fwd_col: "fwd_chg_bps"})
    out["entry_date"] = out.index
    out["side"] = side
    out["horizon"] = horizon
    return out[["entry_date", "side", "horizon", "entry_gap_z", "fwd_chg_bps"]]

beta_gap_episodes = pd.concat([
    beta_gap_episode_study(beta_gap, threshold=2.0, horizon=20, side="high"),
    beta_gap_episode_study(beta_gap, threshold=2.0, horizon=20, side="low"),
], ignore_index=True)

display(beta_gap_episodes.round(3))
display(beta_gap_episodes.groupby("side")["fwd_chg_bps"].agg(["count", "mean", "median", "std"]).round(3))

