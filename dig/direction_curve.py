"""
Direction → Curve interaction.

Question: does 10Y level have predictive power for 10s30s slope,
and which factors (direction / curve / basis / vol) confound it?

Four diagnostics:
  1. Scatter 10Y vs 10s30s, colored by regime (vol, Fed cycle)
  2. Rolling 60d corr of Δ10Y vs Δ10s30s
  3. Event studies around known regime shifts
  4. Regression: Δ10s30s ~ Δ10Y + Δswap_spread + ΔMOVE + Δmortgage_OAS

Run from repo root:  python dig/direction_curve.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import psycopg2
import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils.viz import Viz
from stats.ols import roll_lr_diff

viz = Viz()  # applies PrismFP rcParams globally

DB_DSN = "postgresql://benjils:snickers@raptor:5432/markets"
START  = "2010-01-01"
OUT    = Path(__file__).parent / "out_direction_curve"
OUT.mkdir(exist_ok=True)


def query_pl(sql: str) -> pl.DataFrame:
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return pl.DataFrame(schema=cols)
    return pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)})


# ── data ─────────────────────────────────────────────────────────────
TICKERS = {
    "10y":     "USGG10YR Index",
    "30y":     "USGG30YR Index",
    "10s30s":  "USYC1030 Index",
    "move":    "MOVE Index",
    "ff":      "FEDL01 Index",
    "ss_10y":  "USSFCT10 Curncy",   # 10Y USD swap spread
    "ss_30y":  "USSFCT30 Curncy",   # 30Y USD swap spread
    "mtg_oas": "LUMSOAS Index",
}
inv = {v: k for k, v in TICKERS.items()}
ticker_list = "', '".join(TICKERS.values())

raw = query_pl(f"""
    SELECT ts, ticker, px_last::float AS px
    FROM md.index_eod
    WHERE ticker IN ('{ticker_list}')
      AND ts >= '{START}'
    ORDER BY ts
""")

available = set(raw["ticker"].unique().to_list())
missing = [v for v in TICKERS.values() if v not in available]
print(f"Available ({len(available)}): {sorted(available)}")
if missing:
    print(f"Missing  ({len(missing)}): {missing}\n")

data = raw.pivot(index="ts", on="ticker", values="px").sort("ts")
data = data.rename({c: inv[c] for c in data.columns if c in inv})

# Yields and fed funds are stored in % → convert to bps.
# USYC1030, MOVE, swap spreads (USSFCT), mtg_oas are already in bp.
for col in ["10y", "30y", "ff"]:
    if col in data.columns:
        data = data.with_columns((pl.col(col) * 100).alias(col))

last = data.sort("ts").row(-1, named=True)
print("Latest levels (sanity check, bps where applicable):")
for k in ["10y", "30y", "10s30s", "move", "ff", "ss_10y", "ss_30y", "mtg_oas"]:
    if k in last and last[k] is not None:
        print(f"  {k:<10} = {last[k]:.2f}")
print()

chg_cols = [c for c in ["10y", "30y", "10s30s", "move", "ss_10y", "ss_30y", "mtg_oas"] if c in data.columns]
data = data.with_columns([pl.col(c).diff().alias(f"d_{c}") for c in chg_cols])
data = data.drop_nulls(subset=["10y", "10s30s"])
print(f"Rows: {len(data)} | {data['ts'].min()} → {data['ts'].max()}\n")


# ── regimes ──────────────────────────────────────────────────────────
if "move" in data.columns:
    mv = data["move"].drop_nulls()
    q1, q2 = mv.quantile(1/3), mv.quantile(2/3)
    data = data.with_columns(
        pl.when(pl.col("move") < q1).then(pl.lit("low_vol"))
        .when(pl.col("move") < q2).then(pl.lit("mid_vol"))
        .otherwise(pl.lit("high_vol"))
        .alias("vol_regime")
    )

if "ff" in data.columns:
    data = data.with_columns((pl.col("ff") - pl.col("ff").shift(252)).alias("ff_yoy"))
    data = data.with_columns(
        pl.when(pl.col("ff_yoy") > 50).then(pl.lit("hiking"))
        .when(pl.col("ff_yoy") < -50).then(pl.lit("cutting"))
        .otherwise(pl.lit("holding"))
        .alias("fed_cycle")
    )


# ── diag 1: scatter by regime (PrismFP styling) ─────────────────────
def scatter_by(regime_col: str, fname: str, title: str):
    if regime_col not in data.columns:
        return
    sub = data.drop_nulls(subset=["10y", "10s30s", regime_col])
    regs = sorted(sub[regime_col].unique().to_list())

    fig, axes = plt.subplots(1, len(regs), figsize=(4.8 * len(regs), 4.4),
                              sharex=True, sharey=True)
    if len(regs) == 1:
        axes = [axes]
    for i, (ax, r) in enumerate(zip(axes, regs)):
        s = sub.filter(pl.col(regime_col) == r)
        x, y = s["10y"].to_numpy(), s["10s30s"].to_numpy()
        color = viz.COLORS[i % len(viz.COLORS)]
        ax.scatter(x, y, alpha=0.3, s=8, color=color, edgecolor='none')
        rho = np.corrcoef(x, y)[0, 1]
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color='#333', lw=1.2, linestyle='--')
        viz._style_ax(
            ax,
            title=f"{r}  ρ={rho:+.2f}  β={slope:+.2f}  N={len(s)}",
            xaxis_title="10Y (bps)",
            yaxis_title="10s30s (bps)",
        )
    fig.suptitle(title.upper(), fontsize=viz.TITLE_SIZE, color="#333", x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / fname, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  saved {fname}")


print("── diag 1: scatter 10Y vs 10s30s by regime ──")
scatter_by("vol_regime", "1_scatter_by_vol.png", "10Y vs 10s30s by MOVE regime")
scatter_by("fed_cycle",  "1_scatter_by_fed.png", "10Y vs 10s30s by Fed cycle (Δff_yoy)")


# ── diag 2: rolling 60d corr (PrismFP styling) ──────────────────────
print("\n── diag 2: rolling 60d corr Δ10Y vs Δ10s30s ──")
pdf = data.select(["ts", "d_10y", "d_10s30s"]).drop_nulls().to_pandas().set_index("ts")
rho = pdf["d_10y"].rolling(60).corr(pdf["d_10s30s"])

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(rho.index, rho.values, color=viz.COLORS[3], linewidth=1.2, label="60d Corr")
ax.axhline(0, color='#666', linestyle='--', linewidth=1, alpha=0.7)
viz._style_ax(
    ax,
    title="60d rolling corr: Δ10Y vs Δ10s30s   (negative = bear flattener / bull steepener)",
    yaxis_title="Correlation",
)
viz._format_dates(ax, rho.index.min(), rho.index.max())
viz._legend(ax)
fig.tight_layout()
fig.savefig(OUT / "2_rolling_corr.png", dpi=170, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  saved 2_rolling_corr.png  |  mean={rho.mean():+.2f}  min={rho.min():+.2f}  max={rho.max():+.2f}")


# ── diag 3: event studies — printed table + overlay plot ───────────
EVENTS = {
    "2019 inversion (Aug)":              "2019-08-14",
    "COVID shock (Mar 2020)":            "2020-03-09",
    "First Fed hike post-COVID":         "2022-03-16",
    "SVB collapse (Mar 2023)":           "2023-03-10",
    "Term premium repricing (Oct 2023)": "2023-10-19",
}

print("\n── diag 3: event studies — change over d0 to d0+20 ──")
header = f"{'event':<38} {'Δ10Y':>9} {'Δ10s30s':>10} {'ΔMOVE':>9}"
print(header)
print("-" * len(header))
for name, date in EVENTS.items():
    d0 = pd.Timestamp(date)
    win = data.filter((pl.col("ts") >= d0) & (pl.col("ts") <= d0 + pd.Timedelta(days=30)))
    if len(win) < 5:
        print(f"{name:<38}  (no data)")
        continue
    first, last = win.row(0, named=True), win.row(-1, named=True)
    d_10y    = last["10y"] - first["10y"]
    d_10s30s = last["10s30s"] - first["10s30s"]
    d_move   = (last["move"] - first["move"]) if "move" in win.columns else float("nan")
    print(f"{name:<38} {d_10y:>+9.1f} {d_10s30s:>+10.1f} {d_move:>+9.1f}")

# 10s30s overlay centered on each event (40d window)
fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, date) in enumerate(EVENTS.items()):
    d0 = pd.Timestamp(date)
    win = data.filter((pl.col("ts") >= d0 - pd.Timedelta(days=10))
                     & (pl.col("ts") <= d0 + pd.Timedelta(days=30))).to_pandas()
    if len(win) < 5:
        continue
    win["d_from_event"] = (win["ts"] - d0).dt.days
    base = win.loc[win["d_from_event"].abs().idxmin(), "10s30s"]
    ax.plot(win["d_from_event"], win["10s30s"] - base,
            color=viz.COLORS[i % len(viz.COLORS)], linewidth=1.5, label=name)
ax.axvline(0, color='#666', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(0, color='#666', linestyle='--', linewidth=1, alpha=0.7)
viz._style_ax(ax, title="10s30s reaction by event (Δ from event date)",
              xaxis_title="Days from event", yaxis_title="Δ10s30s (bps)")
from utils.viz import _SquareHandler
ax.legend(
    loc='upper left', bbox_to_anchor=(0, -0.14), ncol=3,
    fontsize=9, frameon=False, handlelength=2.0, handleheight=1,
    handler_map={plt.Line2D: _SquareHandler()},
)
fig.subplots_adjust(bottom=0.20)
fig.savefig(OUT / "3_event_overlay.png", dpi=170, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  saved 3_event_overlay.png")

# Rolling 60d β: Δ10s30s = α + β · Δ10y  (the slope, with event markers)
reg = roll_lr_diff(data["10y"], data["10s30s"], lookback=60)
beta_pdf = data.select("ts").slice(1).with_columns(
    reg["beta"].alias("beta"),
    reg["r2"].alias("r2"),
).drop_nulls().to_pandas().set_index("ts")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(beta_pdf.index, beta_pdf["beta"], color=viz.COLORS[3], linewidth=1.2, label="60d β (Δ10s30s / Δ10Y)")
ax.axhline(0, color='#666', linestyle='--', linewidth=1, alpha=0.7)

# Mark events with vertical lines + labels at the top
for name, date in EVENTS.items():
    d0 = pd.Timestamp(date)
    if d0 < beta_pdf.index.min() or d0 > beta_pdf.index.max():
        continue
    ax.axvline(d0, color='#333', linestyle=':', linewidth=1, alpha=0.6)
    ax.annotate(
        name.split(" (")[0], xy=(d0, 1.0), xycoords=('data', 'axes fraction'),
        xytext=(2, -2), textcoords='offset points',
        fontsize=7, color='#333', ha='left', va='top', rotation=90,
    )

viz._style_ax(
    ax,
    title="Rolling 60d β: Δ10s30s ~ Δ10Y   (curve response per 1 bp of direction)",
    yaxis_title="β (bp / bp)",
)
viz._format_dates(ax, beta_pdf.index.min(), beta_pdf.index.max())
viz._legend(ax)
fig.tight_layout()
fig.savefig(OUT / "3b_rolling_beta.png", dpi=170, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  saved 3b_rolling_beta.png  |  β mean={beta_pdf['beta'].mean():+.3f}  min={beta_pdf['beta'].min():+.3f}  max={beta_pdf['beta'].max():+.3f}")

# Event-driven overlay of rolling β  — normalized to event date so the SHAPE
# of the response is comparable across events with different baseline β levels.
# Window extended to +90d since rolling β needs ~one lookback to fully reflect
# the post-event regime.
fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, date) in enumerate(EVENTS.items()):
    d0 = pd.Timestamp(date)
    mask = (beta_pdf.index >= d0 - pd.Timedelta(days=15)) & (beta_pdf.index <= d0 + pd.Timedelta(days=90))
    win = beta_pdf.loc[mask].copy()
    if len(win) < 5:
        continue
    win["d_from_event"] = (win.index - d0).days
    base = win.loc[win["d_from_event"].abs().idxmin(), "beta"]
    ax.plot(win["d_from_event"], win["beta"] - base,
            color=viz.COLORS[i % len(viz.COLORS)], linewidth=1.5, label=name)
ax.axvline(0, color='#666', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(0, color='#666', linestyle='--', linewidth=1, alpha=0.7)
viz._style_ax(ax, title="Rolling 60d β around events  (Δβ from event date)",
              xaxis_title="Days from event", yaxis_title="Δβ (bp / bp)")
ax.legend(
    loc='upper left', bbox_to_anchor=(0, -0.14), ncol=3,
    fontsize=9, frameon=False, handlelength=2.0, handleheight=1,
    handler_map={plt.Line2D: _SquareHandler()},
)
fig.subplots_adjust(bottom=0.20)
fig.savefig(OUT / "3c_event_beta_overlay.png", dpi=170, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  saved 3c_event_beta_overlay.png")

# Distribution of β — one β per non-overlapping 21-day window (each independent)
from scipy.stats import gaussian_kde, norm, skew, kurtosis
WINDOW = 21
ld = data.select(["d_10y", "d_10s30s"]).drop_nulls().to_pandas()

betas = []
for start in range(0, len(ld) - WINDOW + 1, WINDOW):
    chunk = ld.iloc[start:start + WINDOW]
    x, y = chunk["d_10y"].values, chunk["d_10s30s"].values
    if np.var(x) < 1e-10:
        continue
    betas.append(np.polyfit(x, y, 1)[0])
beta_vals = pd.Series(betas)

fig, ax = plt.subplots(figsize=(11, 5))
color = viz.COLORS[3]

ax.hist(beta_vals, bins=40, density=True, color=color, alpha=0.35,
        edgecolor=color, linewidth=0.5)

kde = gaussian_kde(beta_vals)
x_grid = np.linspace(beta_vals.min(), beta_vals.max(), 300)
ax.plot(x_grid, kde(x_grid), color=color, linewidth=2, label="KDE")

mu, sigma = beta_vals.mean(), beta_vals.std()
ax.plot(x_grid, norm.pdf(x_grid, mu, sigma),
        color='#666', linewidth=1.5, linestyle='--', label="Normal")

ax.axvline(0, color='#333', linewidth=1, linestyle=':', alpha=0.7)
current = float(beta_vals.iloc[-1])
ax.axvline(current, color=color, linewidth=2, linestyle='-',
           label=f"Most recent: {current:+.3f}")

stats_text = (
    f"μ = {mu:+.3f}\n"
    f"σ = {sigma:.3f}\n"
    f"skew = {skew(beta_vals):+.2f}\n"
    f"kurt = {kurtosis(beta_vals):+.2f}\n"
    f"n = {len(beta_vals)}\n"
    f"%β>0 = {(beta_vals > 0).mean() * 100:.1f}%"
)
ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=8,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCC", alpha=0.9))

viz._style_ax(ax, title=f"Distribution of β  (Δ10s30s / Δ10Y, {WINDOW}-day non-overlapping windows)",
              xaxis_title="β (bp / bp)", yaxis_title="Density")
ax.legend(loc='upper left', fontsize=8, frameon=False, handlelength=1.5)
fig.tight_layout()
fig.savefig(OUT / "3d_beta_distribution.png", dpi=170, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  saved 3d_beta_distribution.png  |  n={len(beta_vals)}  μ={mu:+.3f}  σ={sigma:.3f}  %β>0={(beta_vals > 0).mean()*100:.1f}%")


# ── diag 4: regression Δ10s30s ~ Δ10Y + controls ───────────────────
print("\n── diag 4: regression Δ10s30s ~ Δ10Y + controls ──")
regressors = ["d_10y"]
for c in ["d_ss_30y", "d_move", "d_mtg_oas"]:
    if c in data.columns:
        regressors.append(c)

reg_df = data.select(["d_10s30s"] + regressors).drop_nulls().to_pandas()
y = reg_df["d_10s30s"]
X = sm.add_constant(reg_df[regressors])
m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

print(f"\nFull sample (N={len(reg_df)}, R²={m.rsquared:.3f}, Newey-West se):")
print(f"  {'regressor':<12} {'β':>9} {'t':>7}")
for c in regressors:
    print(f"  {c:<12} {m.params[c]:>+9.3f} {m.tvalues[c]:>+7.2f}")

if "vol_regime" in data.columns:
    print("\nBy vol regime:")
    sub_all = data.select(["d_10s30s", "vol_regime"] + regressors).drop_nulls().to_pandas()
    for reg in ["low_vol", "mid_vol", "high_vol"]:
        s = sub_all[sub_all["vol_regime"] == reg]
        if len(s) < 100:
            continue
        yy = s["d_10s30s"]
        XX = sm.add_constant(s[regressors])
        mm = sm.OLS(yy, XX).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        coefs = "  ".join(f"{c}: β={mm.params[c]:+.3f}(t={mm.tvalues[c]:+.1f})" for c in regressors)
        print(f"  {reg:<10} N={len(s):>5}  R²={mm.rsquared:.3f}   {coefs}")

print(f"\nDone.  Plots → {OUT}")
