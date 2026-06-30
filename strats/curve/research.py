"""
Beta-weighted 10s30s curve signal research.

Justifies three design choices in signal.py:
  (1) Changes-based OLS (roll_lr_diff) not levels — 10Y and 30Y are I(1),
      levels regression is spurious. Cumulated changes residual gives a
      stationary spread with directional contamination stripped out.
  (2) Hedge-ratio lookback — too short = noisy beta, too long = stale in
      regime shifts. Diag 2 picks the sweet spot via σ of rolling β.
  (3) z-score entry threshold — diag 4 calibrates to 40-day forward hit
      rates. Half-life from diag 3 drives the time stop.

Output: printed diagnostics + four charts in strats/curve/out/

Run:  mamba run -n 2s10s python strats/curve/research.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.helpers import query_db
from utils.viz import Viz
from stats.ols import roll_lr_diff
from stats.ou import ou_params, roll_ou_zscore, roll_half_life

viz = Viz()
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

START     = "2010-01-01"
CHOSEN_LB = 63      # hedge-ratio lookback — justified in diag 2
ENTRY_Z   = 2.0     # entry z-score — justified in diag 4
FWD_BARS  = 40      # forward window for hit-rate scan

TICKERS = {
    "10y":    "USGG10YR Index",
    "30y":    "USGG30YR Index",
    "move":   "MOVE Index",
    "ss_30y": "USSFCT30 Curncy",   # 30Y swap spread: non-fundamental steepener source
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_data() -> pl.DataFrame:
    inv = {v: k for k, v in TICKERS.items()}
    tickers_sql = "', '".join(TICKERS.values())
    pdf = query_db(f"""
        SELECT ts, ticker, px_last::float AS px
        FROM md.index_eod
        WHERE ticker IN ('{tickers_sql}')
          AND ts >= '{START}'
        ORDER BY ts
    """)
    wide = (
        pl.from_pandas(pdf)
        .with_columns(pl.col("ts").cast(pl.Date))
        .pivot(index="ts", on="ticker", values="px")
        .sort("ts")
    )
    wide = wide.rename({c: inv[c] for c in wide.columns if c in inv})
    for col in ["10y", "30y"]:
        if col in wide.columns:
            wide = wide.with_columns((pl.col(col) * 100).alias(col))
    return wide


def beta_weight(df: pl.DataFrame, lookback: int) -> pl.DataFrame:
    """Changes-based OLS: regress Δ30Y on Δ10Y. Returns df[1:] with beta, resid_cum, r2."""
    reg = roll_lr_diff(df["10y"], df["30y"], lookback=lookback)
    return df.slice(1).with_columns([
        reg["beta"].alias("beta"),
        reg["resid_cum"].alias("resid_cum"),
        reg["r2"].alias("r2"),
    ])


def threshold_scan(df: pl.DataFrame, z_col: str = "zscore") -> pd.DataFrame:
    """For each entry z, compute n entries and fwd FWD_BARS hit rate."""
    sub = df.select(["ts", z_col, "resid_cum"]).drop_nulls().to_pandas().reset_index(drop=True)
    fwd_chg = sub["resid_cum"].shift(-FWD_BARS) - sub["resid_cum"]
    rows = []
    for t in [1.0, 1.5, 2.0, 2.5, 3.0]:
        lm = sub[z_col] < -t
        sm = sub[z_col] >  t
        n_l, n_s = int(lm.sum()), int(sm.sum())
        hl = (fwd_chg[lm] > 0).mean() if n_l > 0 else np.nan
        hs = (fwd_chg[sm] < 0).mean() if n_s > 0 else np.nan
        rows.append({
            "z":         t,
            "n_long":    n_l,
            "n_short":   n_s,
            "hit_long":  round(hl, 3) if not np.isnan(hl) else np.nan,
            "hit_short": round(hs, 3) if not np.isnan(hs) else np.nan,
            "avg_hit":   round(np.nanmean([hl, hs]), 3),
        })
    return pd.DataFrame(rows)


# ── plots ────────────────────────────────────────────────────────────────────

def plot_naive_vs_bw(df: pl.DataFrame, bw: pl.DataFrame, c_naive: float, c_bw: float):
    naive_pd = df.with_columns((pl.col("30y") - pl.col("10y")).alias("naive")).to_pandas().set_index("ts")
    bw_pd    = bw.drop_nulls(subset=["resid_cum"]).to_pandas().set_index("ts")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax0 = axes[0]
    ax0.plot(naive_pd.index, naive_pd["naive"], color=viz.COLORS[0], lw=1.2)
    ax0r = ax0.twinx()
    ax0r.plot(naive_pd.index, naive_pd["10y"], color=viz.COLORS[2], lw=0.8, alpha=0.35)
    ax0r.set_ylabel("10Y (bps)", fontsize=8, color=viz.COLORS[2])
    viz._style_ax(ax0, title=f"Naive 10s30s (30Y − 10Y)  vs  10Y  (ρ = {c_naive:+.2f}) — direction in the spread",
                  yaxis_title="bps")

    ax1 = axes[1]
    ax1.plot(bw_pd.index, bw_pd["resid_cum"], color=viz.COLORS[1], lw=1.2)
    ax1r = ax1.twinx()
    ax1r.plot(bw_pd.index, bw_pd["10y"], color=viz.COLORS[2], lw=0.8, alpha=0.35)
    ax1r.set_ylabel("10Y (bps)", fontsize=8, color=viz.COLORS[2])
    viz._style_ax(ax1, title=f"Beta-weighted residual ({CHOSEN_LB}d)  vs  10Y  (ρ = {c_bw:+.2f}) — direction stripped",
                  yaxis_title="bps")
    viz._format_dates(ax1, pd.Timestamp(str(bw_pd.index.min())), pd.Timestamp(str(bw_pd.index.max())))

    fig.tight_layout()
    fig.savefig(OUT / "1_naive_vs_betawt.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved 1_naive_vs_betawt.png  |  naive ρ={c_naive:+.2f}  bw ρ={c_bw:+.2f}")


def plot_rolling_beta(df: pl.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 5))
    ts_vals = df.slice(1)["ts"].to_pandas()
    for i, lb in enumerate([21, 63, 126]):
        reg  = roll_lr_diff(df["10y"], df["30y"], lookback=lb)
        beta = pd.Series(reg["beta"].to_numpy(), index=ts_vals)
        ax.plot(beta.index, beta, color=viz.COLORS[i % len(viz.COLORS)],
                lw=1.1, label=f"{lb}d", alpha=0.85)
    ax.axhline(0, color="#666", ls="--", lw=0.8, alpha=0.5)
    viz._style_ax(ax, title="Rolling hedge ratio β (Δ30Y / Δ10Y) — 21 / 63 / 126-day lookbacks",
                  yaxis_title="β (bp / bp)")
    viz._format_dates(ax, pd.Timestamp(str(df["ts"].min())), pd.Timestamp(str(df["ts"].max())))
    viz._legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "2_rolling_beta.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved 2_rolling_beta.png")


def plot_rolling_hl(bw: pl.DataFrame):
    hl    = roll_half_life(bw["resid_cum"], lookback=252)
    ts    = bw["ts"].to_pandas()
    hl_np = hl.to_numpy().astype(float)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(ts, hl_np, color=viz.COLORS[3 % len(viz.COLORS)], lw=1.1)
    ax.axhline(0, color="#666", ls="--", lw=0.8, alpha=0.5)
    viz._style_ax(ax, title="Rolling half-life of beta-weighted residual (252d window)",
                  yaxis_title="half-life (days)")
    viz._format_dates(ax, pd.Timestamp(str(bw["ts"].min())), pd.Timestamp(str(bw["ts"].max())))
    fig.tight_layout()
    fig.savefig(OUT / "3_rolling_hl.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    valid = hl_np[~np.isnan(hl_np) & (hl_np > 0) & (hl_np < 500)]
    if len(valid):
        print(f"  saved 3_rolling_hl.png  |  "
              f"median={np.median(valid):.0f}d  p25={np.percentile(valid, 25):.0f}d  p75={np.percentile(valid, 75):.0f}d")
    else:
        print("  saved 3_rolling_hl.png")


def plot_zscore_bands(bw: pl.DataFrame):
    sub = bw.drop_nulls(subset=["zscore"]).to_pandas().set_index("ts")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(sub.index, sub["zscore"], color=viz.COLORS[0], lw=0.9)
    for band, color, ls in [(ENTRY_Z, viz.COLORS[2], "--"), (1.5, "#aaa", ":")]:
        ax.axhline( band, color=color, ls=ls, lw=1.0, alpha=0.8, label=f"±{band}")
        ax.axhline(-band, color=color, ls=ls, lw=1.0, alpha=0.8)
    ax.axhline(0, color="#666", ls=":", lw=0.7, alpha=0.4)
    viz._style_ax(ax, title=f"OU z-score of beta-weighted residual ({CHOSEN_LB}d lookback)",
                  yaxis_title="z-score")
    viz._format_dates(ax, pd.Timestamp(str(sub.index.min())), pd.Timestamp(str(sub.index.max())))
    viz._legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "4_zscore_bands.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved 4_zscore_bands.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> dict:
    # load and sanity check
    df = load_data()
    last = df.sort("ts").row(-1, named=True)
    print(f"Rows: {len(df)} | {df['ts'].min()} → {df['ts'].max()}")
    print(f"Latest: 10Y={last['10y']:.1f}bps  30Y={last['30y']:.1f}bps  "
          f"naive 10s30s={(last['30y'] - last['10y']):.1f}bps\n")

    # diag 1: naive 10s30s carries direction — bear flatteners shrink the spread
    # mechanically even when the curve shape hasn't changed
    df2 = df.with_columns((pl.col("30y") - pl.col("10y")).alias("naive"))
    c_naive = df2.drop_nulls(subset=["10y", "naive"]).select(
        pl.corr("10y", "naive")
    ).item()
    print("── diag 1: direction contamination ──")
    print(f"  naive 10s30s vs 10Y  ρ = {c_naive:+.3f}")
    print(f"  (high ρ means the spread is partly a proxy for rate level, not pure curve)\n")

    # diag 2: hedge ratio stability — shorter lookback adapts faster but is noisier;
    # longer is stable but lags regime shifts. σ of rolling β quantifies the tradeoff
    print("── diag 2: hedge ratio stability (Δ30Y / Δ10Y) ──")
    for lb in [21, 63, 126]:
        reg = roll_lr_diff(df["10y"], df["30y"], lookback=lb)
        b = reg["beta"].drop_nulls()
        print(f"  {lb:>3}d  mean={b.mean():+.3f}  σ={b.std():.3f}  "
              f"range=[{b.min():+.3f}, {b.max():+.3f}]")
    print(f"  → using CHOSEN_LB={CHOSEN_LB}d\n")

    # compute beta-weighted spread at chosen lookback
    bw = beta_weight(df, CHOSEN_LB)

    # direction check: resid_cum should have much lower ρ with 10Y than naive
    c_bw = bw.drop_nulls(subset=["10y", "resid_cum"]).select(
        pl.corr("10y", "resid_cum")
    ).item()
    print(f"  post-hedge: resid_cum vs 10Y  ρ = {c_bw:+.3f}  (vs naive {c_naive:+.3f})\n")

    # diag 3: OU params — is the residual stationary? half_life sets the time stop
    # if it hasn't mean-reverted in 2× half_life, the thesis is wrong and we exit
    print(f"── diag 3: OU params on residual ({CHOSEN_LB}d lookback) ──")
    params = ou_params(bw["resid_cum"].drop_nulls())
    hl = params["half_life"]
    time_stop = int(hl * 2) if not np.isnan(hl) else 40
    print(f"  theta     = {params['theta']:.4f}  (mean-reversion speed)")
    print(f"  mu        = {params['mu']:.2f} bps")
    print(f"  sigma     = {params['sigma']:.2f} bps/day")
    print(f"  half_life = {hl:.1f} days  → time_stop = {time_stop} bars\n")

    # add z-score
    bw = bw.with_columns(
        roll_ou_zscore(bw["resid_cum"], lookback=CHOSEN_LB).alias("zscore")
    )

    # diag 4: threshold scan — tighter entry = fewer but cleaner trades
    # pick the threshold where avg_hit meaningfully exceeds 50% with enough n
    print(f"── diag 4: entry threshold scan (fwd {FWD_BARS} bars) ──")
    scan = threshold_scan(bw)
    print(scan.to_string(index=False))
    chosen = scan[scan["z"] == ENTRY_Z].iloc[0]
    print(f"\n  → z=±{ENTRY_Z}: avg_hit={chosen['avg_hit']:.1%}  "
          f"n_entries≈{int(chosen['n_long'] + chosen['n_short'])}\n")

    # plots
    print("── plots ──")
    plot_naive_vs_bw(df, bw, c_naive, c_bw)
    plot_rolling_beta(df)
    plot_rolling_hl(bw)
    plot_zscore_bands(bw)
    print(f"\nDone.  Charts → {OUT}")

    return {
        "df":        df,
        "bw":        bw,
        "ou_params": params,
        "time_stop": time_stop,
        "scan":      scan,
    }


if __name__ == "__main__":
    main()
