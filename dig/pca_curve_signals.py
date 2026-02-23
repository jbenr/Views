"""PCA Curve Signal Scanner

Start with PCA on the full SOFR OIS curve, then scan tradeable
structures (curves, flies, spreads) for mean-reverting relationships
against each principal component.

The pipeline:
  1. Pull the SOFR OIS curve (8 tenors: 2y-30y + 4y).
  2. Rolling PCA on daily changes -> PC1 (level), PC2 (slope), PC3 (curvature).
  3. Cumulative PC scores become tradeable "factors".
  4. For each PC factor, scan a universe of curves/flies/spreads:
     - Roll regression: structure ~ PC factor
     - Test residual for mean reversion (OU z-score, half-life)
     - Score each structure by R2, beta stability, and mean-reversion strength
  5. Surface the best candidates per PC.

Usage:
    python dig/pca_curve_signals.py [--curve ust|sofr] [--start 2010-01-01]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from stats import roll_lr, ou_zscore, fit_pca, roll_pca, residual_from_pca, explain
from stats.ou import half_life, ou_params
from utils import tickers as tkr

# ── config ──────────────────────────────────────────────────────────────
DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
START = "2000-01-01"
PCA_LOOKBACK = 504          # 2y rolling window for PCA
SCAN_LOOKBACKS = [20, 30, 40, 50, 100, 120, 252, 504]
N_COMPONENTS = 3

# curves to PCA
CURVE_SETS = {
    "sofr": {
        "tickers": {
            "2y":  "USOSFR2 Curncy",
            "3y":  "USOSFR3 Curncy",
            "4y":  "USOSFR4 Curncy",
            "5y":  "USOSFR5 Curncy",
            "7y":  "USOSFR7 Curncy",
            "10y": "USOSFR10 Curncy",
            "20y": "USOSFR20 Curncy",
            "30y": "USOSFR30 Curncy",
        },
    },
    "ust": {
        "tickers": {
            "2y":  "USGG2YR Index",
            "3y":  "USGG3YR Index",
            "5y":  "USGG5YR Index",
            "7y":  "USGG7YR Index",
            "10y": "USGG10YR Index",
            "20y": "USGG20YR Index",
            "30y": "USGG30YR Index",
        },
    },
}

# structures to scan against PC factors
SCAN_UNIVERSE = {
    # curves
    **{f"{a}s{b}s": t for a, b, t in [
        (2, 5,   "USYC2Y5Y Index"),
        (2, 10,  "USYC2Y10Y Index"),
        (2, 30,  "USYC2Y30Y Index"),
        (5, 10,  "USYC5Y10Y Index"),
        (5, 30,  "USYC5Y30Y Index"),
        (10, 30, "USYC10Y30Y Index"),
        (3, 7,   "USYC3Y7Y Index"),
        (7, 10,  "USYC7Y10Y Index"),
        (7, 20,  "USYC7Y20Y Index"),
        (10, 20, "USYC10Y20Y Index"),
        (20, 30, "USYC20Y30Y Index"),
    ]},
    # butterflies
    **{f"bf_{a}s{b}s{c}s": t for a, b, c, t in [
        (2, 5, 10,   "BF020510 Index"),
        (2, 5, 30,   "BF020530 Index"),
        (2, 10, 30,  "BF021030 Index"),
        (3, 5, 7,    "BF030507 Index"),
        (3, 5, 10,   "BF030510 Index"),
        (3, 7, 10,   "BF030710 Index"),
        (5, 7, 10,   "BF050710 Index"),
        (5, 10, 30,  "BF051030 Index"),
        (5, 10, 20,  "BF051020 Index"),
        (7, 10, 20,  "BF071020 Index"),
        (7, 10, 30,  "BF071030 Index"),
        (10, 20, 30, "BF102030 Index"),
    ]},
    # swap spreads
    **{f"ss_{x}y": t for x, t in [
        (2,  "USSFCT02 Curncy"),
        (5,  "USSFCT05 Curncy"),
        (10, "USSFCT10 Curncy"),
        (30, "USSFCT30 Curncy"),
    ]},
    # outright yields for duration signal
    **{f"ust_{x}y": t for x, t in [
        (2,  "USGG2YR Index"),
        (5,  "USGG5YR Index"),
        (10, "USGG10YR Index"),
        (30, "USGG30YR Index"),
    ]},
}


# ── data ────────────────────────────────────────────────────────────────

def pull_px(conn, tickers: list[str], start: str) -> pd.DataFrame:
    q = """
    SELECT ts, ticker, px_last
    FROM md.index_eod
    WHERE ticker = ANY(%s)
      AND ts >= %s
    ORDER BY ts
    """
    with conn.cursor() as cur:
        cur.execute(q, (tickers, start))
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.pivot(index="ts", columns="ticker", values="px_last").sort_index()


# ── PCA ─────────────────────────────────────────────────────────────────

def run_pca(curve_df: pd.DataFrame, lookback: int, n_components: int):
    """Full-sample + rolling PCA on the curve.

    Returns
    -------
    full : dict from fit_pca (loadings, scores, etc.)
    rolling : dict from roll_pca (explained_variance, pc1_loadings)
    cum_scores : pd.DataFrame — cumulative PC scores (level-like series)
    """
    clean = curve_df.dropna()

    # full-sample
    full = fit_pca(clean, n_components=n_components, use_changes=True)

    # rolling
    rolling = roll_pca(clean, lookback=lookback, n_components=n_components, use_changes=True)

    # fix sign convention: flip so PC1 positive = yields up, etc.
    # convention: if mean loading is negative, flip that component
    loadings_np = full["loadings"].drop("column").to_numpy().copy()
    scores_np = full["scores"].to_numpy().copy()
    for i in range(n_components):
        if loadings_np[:, i].mean() < 0:
            loadings_np[:, i] *= -1
            scores_np[:, i] *= -1
    # write back
    import polars as pl
    pc_names = [f"PC{i+1}" for i in range(n_components)]
    col_series = full["loadings"]["column"]
    full["loadings"] = pl.DataFrame(
        {pc: loadings_np[:, i] for i, pc in enumerate(pc_names)}
    ).with_columns(col_series)
    full["scores"] = pl.DataFrame(
        {pc: scores_np[:, i] for i, pc in enumerate(pc_names)}
    )

    # cumulative scores -> tradeable factor series
    scores = full["scores"].to_pandas()
    scores.columns = pc_names
    scores.index = clean.index[1:]  # use_changes drops first row
    cum_scores = scores.cumsum()

    return full, rolling, cum_scores


# ── structure scanner ───────────────────────────────────────────────────

def adf_half_life(series: pd.Series) -> float:
    """Half-life from AR(1) on the series (Dickey-Fuller style)."""
    return half_life(series)


def scan_structure(
    pc_factor: pd.Series,
    structure: pd.Series,
    lookback: int,
) -> dict | None:
    """Regress a tradeable structure against a PC factor.

    Returns regression stats + mean-reversion diagnostics on the residual.
    """
    combo = pd.concat([pc_factor.rename("pc"), structure.rename("struct")], axis=1).dropna()
    if len(combo) < lookback + 20:
        return None

    reg = roll_lr(combo["pc"], combo["struct"], lookback=lookback).to_pandas()
    reg.index = combo.index

    # OUT-OF-SAMPLE residual: use YESTERDAY's alpha/beta on TODAY's observation.
    # roll_lr computes in-sample resid where time t is in its own regression window.
    # Shifting params by 1 removes the lookahead: we only use what was known at prior close.
    reg["resid"] = combo["struct"] - reg["alpha"].shift(1) - reg["beta"].shift(1) * combo["pc"]
    reg["yhat"] = reg["alpha"].shift(1) + reg["beta"].shift(1) * combo["pc"]

    # OU z-score on the out-of-sample residual
    z = ou_zscore(reg["resid"].dropna(), lookback=lookback).to_pandas()
    z_idx = reg["resid"].dropna().index
    z_series = pd.Series(np.nan, index=reg.index)
    z_series.loc[z_idx[-len(z):]] = z.values
    reg["ou_z"] = z_series

    # current stats
    tail = reg.dropna(subset=["beta", "r2", "resid", "ou_z"])
    if tail.empty:
        return None
    last = tail.iloc[-1]

    # rolling corr
    corr = combo["struct"].rolling(lookback).corr(combo["pc"])
    r2_corr = (corr ** 2).clip(0, 1)
    reg["r2_corr"] = r2_corr  # store rolling series for downstream filtering

    # beta stability
    beta_abs_mean = reg["beta"].abs().rolling(lookback).mean()
    beta_abs_mean = beta_abs_mean.replace(0, np.nan)
    beta_cv = reg["beta"].rolling(lookback).std() / beta_abs_mean
    reg["beta_cv"] = beta_cv  # store rolling series for downstream filtering

    # mean-reversion strength on OOS residual (full trailing window)
    resid_tail = reg["resid"].dropna().tail(lookback * 2)
    hl = adf_half_life(resid_tail)
    ou_p = ou_params(resid_tail)

    return {
        "beta": float(last["beta"]),
        "r2": float(last["r2"]),
        "r2_corr": float(r2_corr.iloc[-1]) if not r2_corr.empty else np.nan,
        "beta_cv": float(beta_cv.iloc[-1]) if not beta_cv.empty else np.nan,
        "resid": float(last["resid"]),
        "ou_z": float(last["ou_z"]),
        "half_life": hl,
        "ou_theta": ou_p["theta"],
        "ou_mu": ou_p["mu"],
        "reg": reg,  # keep full regression for downstream use
    }


def scan_all(
    cum_scores: pd.DataFrame,
    scan_px: pd.DataFrame,
    scan_names: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    """Scan every structure against every PC, across lookbacks.

    Returns a summary DataFrame sorted by signal quality + dict of all regs.
    """
    rows = []
    all_regs = {}

    for pc_col in cum_scores.columns:
        pc_series = cum_scores[pc_col]

        for struct_name, struct_ticker in scan_names.items():
            if struct_ticker not in scan_px.columns:
                continue
            struct_series = scan_px[struct_ticker]

            for lb in SCAN_LOOKBACKS:
                result = scan_structure(pc_series, struct_series, lb)
                if result is None:
                    continue

                reg = result.pop("reg")
                all_regs[(pc_col, struct_name, lb)] = reg

                # quality score: high R2, low beta CV, short half-life
                hl = result["half_life"]
                hl_score = max(0, 1.0 - hl / (lb * 2)) if np.isfinite(hl) and hl > 0 else 0.0
                cv = result["beta_cv"]
                cv_score = max(0, 1.0 - cv / 2.0) if np.isfinite(cv) else 0.0
                r2 = result["r2_corr"]
                quality = (r2 * 0.4 + hl_score * 0.4 + cv_score * 0.2) if np.isfinite(r2) else 0.0

                rows.append({
                    "pc": pc_col,
                    "structure": struct_name,
                    "lookback": lb,
                    **result,
                    "quality": quality,
                })

    if not rows:
        return pd.DataFrame(), all_regs

    df = pd.DataFrame(rows).sort_values("quality", ascending=False).reset_index(drop=True)
    return df, all_regs


# ── output ──────────────────────────────────────────────────────────────

def print_summary(scan_df: pd.DataFrame, top_n: int = 10):
    """Print the top candidates per PC."""
    display_cols = [
        "pc", "structure", "lookback", "quality",
        "r2_corr", "beta", "beta_cv", "half_life",
        "ou_z", "resid",
    ]
    cols = [c for c in display_cols if c in scan_df.columns]

    for pc in scan_df["pc"].unique():
        subset = scan_df[scan_df["pc"] == pc].head(top_n)
        print(f"\n{'='*70}")
        print(f"  TOP {top_n} structures for {pc}")
        print(f"{'='*70}")
        print(subset[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


def print_current_signals(
    scan_df: pd.DataFrame,
    z_threshold: float = 1.0,
    all_regs: dict | None = None,
):
    """Print structures with active z-score signals, sorted by typical Sharpe."""
    from dig.pca_dash import backtest_signal

    active = scan_df[scan_df["ou_z"].abs() >= z_threshold].copy()
    if active.empty:
        print(f"\nNo structures with |z| >= {z_threshold}")
        return

    # compute typical Sharpe for each active signal
    sharpes = []
    for _, row in active.iterrows():
        if all_regs is not None:
            key = (row["pc"], row["structure"], int(row["lookback"]))
            reg = all_regs.get(key)
            if reg is not None:
                try:
                    bt = backtest_signal(reg, entry_z=1.0, exit_z=0.0, lookback=int(row["lookback"]))
                    pnl = bt["pnl"].dropna()
                    ann = pnl.mean() * 252
                    vol = pnl.std() * np.sqrt(252)
                    sharpes.append(ann / vol if vol > 0 else np.nan)
                except Exception:
                    sharpes.append(np.nan)
            else:
                sharpes.append(np.nan)
        else:
            sharpes.append(np.nan)
    active["sharpe"] = sharpes
    active = active.sort_values("sharpe", ascending=False)

    print(f"\n{'='*70}")
    print(f"  ACTIVE SIGNALS (|z| >= {z_threshold}) — sorted by Sharpe")
    print(f"{'='*70}")
    cols = ["pc", "structure", "lookback", "sharpe", "ou_z", "resid", "r2_corr", "half_life", "quality"]
    cols = [c for c in cols if c in active.columns]
    print(active[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


# ── visualization ───────────────────────────────────────────────────────

def plot_all(result: dict):
    """Generate all diagnostic plots from a main() result dict."""
    import matplotlib.pyplot as plt
    from utils.viz import Viz

    v = Viz()
    full_pca = result["full_pca"]
    rolling_pca = result["rolling_pca"]
    cum_scores = result["cum_scores"]
    scan_df = result["scan_df"]
    all_regs = result["all_regs"]
    curve_df = result["curve_df"]

    # ── 1. PCA loadings: clustered bar chart ────────────────────────────
    _plot_loadings_bar(full_pca, v)

    # ── 2. Explained variance scree plot ────────────────────────────────
    v.pca_variance(full_pca["explained_variance"])

    # ── 3. Cumulative PC scores (the "factors") ────────────────────────
    v.line(cum_scores, title="Cumulative PC Scores (tradeable factors)")
    for pc_col in cum_scores.columns:
        v.line(cum_scores[[pc_col]], title=f"{pc_col} Factor", residual=(pc_col != "PC1"))

    # ── 4. Rolling PCA stability ────────────────────────────────────────
    _plot_rolling_pca(rolling_pca, curve_df, v)

    # ── 5. Top structure per PC: regression diagnostics ─────────────────
    _plot_top_structures(scan_df, all_regs, v)

    # ── 6. Active signals: z-score time series ──────────────────────────
    _plot_active_signals(scan_df, all_regs, v, z_threshold=1.5)

    plt.close("all")


def _plot_loadings_bar(full_pca: dict, v):
    """Clustered bar chart: PC1/PC2/PC3 loading per tenor."""
    import matplotlib.pyplot as plt

    loadings_pl = full_pca["loadings"].to_pandas()
    tenors = loadings_pl["column"].tolist()
    pc_cols = [c for c in loadings_pl.columns if c.startswith("PC")]
    ev = full_pca["explained_variance"]

    x = np.arange(len(tenors))
    width = 0.25
    n_pcs = min(len(pc_cols), 3)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(n_pcs):
        vals = loadings_pl[pc_cols[i]].values
        offset = (i - (n_pcs - 1) / 2) * width
        label = f"{pc_cols[i]} ({ev[i]*100:.1f}%)"
        ax.bar(x + offset, vals, width, color=v.colors[i], label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tenors, fontsize=10)
    ax.axhline(0, color="#666", linewidth=0.8, linestyle="--")
    ax.set_ylabel("LOADING", fontsize=10, color="#333")
    ax.set_xlabel("TENOR", fontsize=10, color="#333")
    ax.set_title("PCA LOADINGS BY TENOR", fontsize=11, color="#333", loc="left", pad=10)
    ax.legend(fontsize=9, framealpha=0.85, edgecolor="none")
    ax.set_facecolor("#F5F5F5")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def _plot_rolling_pca(rolling_pca: dict, curve_df: pd.DataFrame, v):
    """Rolling explained variance and PC1 loadings over time."""
    ev = rolling_pca["explained_variance"].to_pandas()
    ev.columns = [f"PC{i+1}" for i in range(ev.shape[1])]
    clean_idx = curve_df.dropna().index
    ev.index = clean_idx[-len(ev):]
    v.line(ev.dropna(), title=f"Rolling Explained Variance ({PCA_LOOKBACK}d)")

    pc1_load = rolling_pca["pc1_loadings"].to_pandas()
    tenor_names = list(curve_df.columns)
    pc1_load.columns = [f"{t}_load" for t in tenor_names]
    pc1_load.index = clean_idx[-len(pc1_load):]
    v.line(pc1_load.dropna(), title=f"Rolling PC1 Loadings ({PCA_LOOKBACK}d)")


def _plot_top_structures(scan_df: pd.DataFrame, all_regs: dict, v):
    """For the best structure per PC, plot residual, z-score, beta."""
    for pc in ["PC1", "PC2", "PC3"]:
        subset = scan_df[scan_df["pc"] == pc]
        if subset.empty:
            continue
        top = subset.iloc[0]
        key = (top["pc"], top["structure"], int(top["lookback"]))
        reg = all_regs.get(key)
        if reg is None:
            continue

        name = f"{top['structure']} ({int(top['lookback'])}d)"

        v.line(
            reg[["resid"]].dropna(),
            title=f"{pc}: Residual vs {name}",
            residual=True,
        )
        v.line(
            reg[["ou_z"]].dropna(),
            title=f"{pc}: OU Z-Score vs {name}",
            residual=True,
        )
        v.line(
            reg[["beta"]].dropna(),
            title=f"{pc}: Rolling Beta vs {name}",
        )


def _plot_active_signals(scan_df: pd.DataFrame, all_regs: dict, v, z_threshold: float = 1.5):
    """Plot z-score time series for top active signals."""
    from dig.pca_dash import backtest_signal

    active = scan_df[scan_df["ou_z"].abs() >= z_threshold].copy()
    if active.empty:
        return

    # sort by Sharpe (not |z|) — more extreme z doesn't mean better trade
    sharpes = []
    for _, row in active.iterrows():
        key = (row["pc"], row["structure"], int(row["lookback"]))
        reg = all_regs.get(key)
        if reg is not None:
            try:
                bt = backtest_signal(reg, entry_z=1.0, exit_z=0.0, lookback=int(row["lookback"]))
                pnl = bt["pnl"].dropna()
                ann = pnl.mean() * 252
                vol = pnl.std() * np.sqrt(252)
                sharpes.append(ann / vol if vol > 0 else np.nan)
            except Exception:
                sharpes.append(np.nan)
        else:
            sharpes.append(np.nan)
    active["sharpe"] = sharpes
    active = active.sort_values("sharpe", ascending=False)
    seen = set()
    to_plot = []
    for _, row in active.iterrows():
        if row["structure"] not in seen and len(to_plot) < 5:
            to_plot.append(row)
            seen.add(row["structure"])

    z_panel = pd.DataFrame()
    for row in to_plot:
        key = (row["pc"], row["structure"], int(row["lookback"]))
        reg = all_regs.get(key)
        if reg is not None:
            col_name = f"{row['structure']}|{row['pc']}|{int(row['lookback'])}d"
            z_panel[col_name] = reg["ou_z"]

    if not z_panel.empty:
        v.line(z_panel.dropna(how="all"), title="Active Signal Z-Scores", residual=True)


# ── main ────────────────────────────────────────────────────────────────

def main(curve_set: str = "sofr", do_plot: bool = True) -> dict:
    """Run the full pipeline. Returns everything for interactive use.

    Returns
    -------
    dict with keys: curve_df, full_pca, rolling_pca, cum_scores,
                    scan_df, all_regs, scan_px
    """
    cfg = CURVE_SETS[curve_set]
    curve_tickers = cfg["tickers"]

    # all tickers we need
    scan_tickers = list(SCAN_UNIVERSE.values())
    all_tickers = sorted(set(list(curve_tickers.values()) + scan_tickers))

    print(f"Curve: {curve_set} ({len(curve_tickers)} tenors)")
    print(f"Scan universe: {len(SCAN_UNIVERSE)} structures")

    # pull data
    with psycopg.connect(DB_DSN) as conn:
        raw = pull_px(conn, all_tickers, START)

    if raw.empty:
        raise RuntimeError("No data returned.")

    # build curve df with friendly names
    curve_df = pd.DataFrame()
    for name, ticker in curve_tickers.items():
        if ticker in raw.columns:
            curve_df[name] = raw[ticker]
    curve_df = curve_df * 100.0  # to bps

    print(f"Data: {raw.index.min().date()} to {raw.index.max().date()}")
    missing = [t for t in curve_tickers.values() if t not in raw.columns]
    if missing:
        print(f"WARNING: missing curve tickers: {missing}")

    # PCA
    print(f"\nRunning PCA (lookback={PCA_LOOKBACK}d, {N_COMPONENTS} components)...")
    full_pca, rolling_pca, cum_scores = run_pca(curve_df, PCA_LOOKBACK, N_COMPONENTS)

    # explain
    ev = full_pca["explained_variance"]
    for i in range(N_COMPONENTS):
        print(f"  PC{i+1}: {ev[i]*100:.1f}% of variance")

    loadings = full_pca["loadings"].to_pandas()
    print("\nLoadings:")
    print(loadings.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # scan
    print(f"\nScanning {len(SCAN_UNIVERSE)} structures x {len(SCAN_LOOKBACKS)} lookbacks x {N_COMPONENTS} PCs...")
    scan_df, all_regs = scan_all(cum_scores, raw, SCAN_UNIVERSE)

    if scan_df.empty:
        print("No valid scan results.")
        return {}

    print_summary(scan_df, top_n=8)
    print_current_signals(scan_df, z_threshold=1.0, all_regs=all_regs)

    out = {
        "curve_df": curve_df,
        "full_pca": full_pca,
        "rolling_pca": rolling_pca,
        "cum_scores": cum_scores,
        "scan_df": scan_df,
        "all_regs": all_regs,
        "scan_px": raw,
    }

    if do_plot:
        plot_all(out)

    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", choices=["ust", "sofr"], default="sofr")
    parser.add_argument("--start", default=START)
    parser.add_argument("--no-plot", action="store_true", help="skip plots")
    args = parser.parse_args()
    START = args.start
    main(args.curve, do_plot=not args.no_plot)
