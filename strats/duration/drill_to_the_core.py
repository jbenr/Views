"""Drill to the core — recursive regression diagnostics for rates research.

Pattern: pick (x, y). Rolling-regress y on x. Inspect the residual. Peel a
known factor z out of it. Repeat. The story of the analysis lives in main();
the helpers above are reusable building blocks.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import psycopg
from IPython.display import display
from statsmodels.tsa.stattools import acf, adfuller

_HERE = Path(__file__).resolve().parent
for _p in [_HERE, *_HERE.parents]:
    if (_p / "stats").exists():
        ROOT = _p
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stats import roll_lr, half_life, ou_params, roll_ou_zscore
from utils.rates import linear_5y5y_forward
from utils.viz import Viz
from strats.duration.signal_context import (
    build_signal_features,
    conditional_ic_table,
    oos_edge_summary,
    oos_edge_test_fast,
    filtered_sharpe_summary,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)


# ─── config ────────────────────────────────────────────────────────────────

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
START  = "2000-01-01"

TICKERS = {
    "2y":     "USGG2YR Index",
    "5y":     "USGG5YR Index",
    "10y":    "USGG10YR Index",
    "30y":    "USGG30YR Index",
    "be5":    "USGGBE05 Index",
    "be10":   "USGGBE10 Index",
    "spx":    "SPX Index",
    "oil":    "CO1 Comdty",
    "dxy":    "DXY Curncy",
    "move":   "MOVE Index",
    "mtg_cc": "MTGEFNCL Index",
    # components for 5y5y forwards
    "zc5":    "USSWIT5 Curncy",
    "zc10":   "USSWIT10 Curncy",
    "sofr5":  "USOSFR5 Curncy",
    "sofr10": "USOSFR10 Curncy",
}

BPS_COLS = ("2y", "5y", "10y", "30y", "be5", "be10", "mtg_cc", "zc5", "zc10", "sofr5", "sofr10")


# ─── data ──────────────────────────────────────────────────────────────────

def _query_to_pl(sql: str) -> pl.DataFrame:
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)})


def load_basket(tickers: dict[str, str] = TICKERS, start: str = START) -> pd.DataFrame:
    """Pull `tickers` from md.index_eod, pivot wide. Yields scaled to bps."""
    tlist = ", ".join(f"'{t}'" for t in tickers.values())
    raw = _query_to_pl(f"""
        SELECT ts, ticker, px_last::float AS px
        FROM md.index_eod
        WHERE ticker IN ({tlist}) AND ts >= '{start}'
        ORDER BY ts
    """)
    wide = (raw.to_pandas()
               .pivot(index="ts", columns="ticker", values="px")
               .sort_index()
               .rename(columns={v: k for k, v in tickers.items()}))
    for c in BPS_COLS:
        if c in wide:
            wide[c] = wide[c] * 100.0
    # derived: mortgage basis = current coupon yield - 10y (both already in bps)
    if "mtg_cc" in wide.columns and "10y" in wide.columns:
        wide["mtg_basis"] = wide["mtg_cc"] - wide["10y"]
    # derived: 5y5y forwards via linear approx (2×10y − 5y)
    if "zc5" in wide.columns and "zc10" in wide.columns:
        wide["5y5y_ifs"] = linear_5y5y_forward(wide["zc5"], wide["zc10"])
    if "sofr5" in wide.columns and "sofr10" in wide.columns:
        wide["5y5y_sfr"] = linear_5y5y_forward(wide["sofr5"], wide["sofr10"])
    return wide.dropna(how="all")


# ─── residual diagnostics ──────────────────────────────────────────────────

def residual_fingerprint(resid: pd.Series) -> dict:
    """Compact stats describing what a residual looks like."""
    r = resid.dropna()
    if len(r) < 60:
        return {"n": len(r)}
    ac = acf(r, nlags=5, fft=True)
    try:
        adf_p = float(adfuller(r, autolag="AIC")[1])
    except Exception:
        adf_p = np.nan
    return {
        "n":         len(r),
        "std":       float(r.std()),
        "kurt":      float(r.kurt()),
        "skew":      float(r.skew()),
        "acf_lag1":  float(ac[1]),
        "acf_lag5":  float(ac[5]),
        "half_life": float(half_life(r)),
        "adf_p":     adf_p,
    }


def _print_fingerprint(label: str, fp: dict) -> None:
    print(f"residual fingerprint  ({label}):")
    for k, val in fp.items():
        if isinstance(val, float):
            print(f"  {k:<10s}  {val:+.3f}")
        else:
            print(f"  {k:<10s}  {val}")


def drill(
    x: pd.Series,
    y: pd.Series,
    *,
    lookback: int = 60,
    x_name: str = "x",
    y_name: str = "y",
    viz: Viz | None = None,
) -> pd.DataFrame:
    """Rolling OLS of y on x. Plots inputs, rolling β/R², and the residual.

    Returns pandas DataFrame indexed by date with columns
    [x, y, alpha, beta, yhat, resid, r2].
    """
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    res = roll_lr(pair["x"], pair["y"], lookback=lookback).to_pandas()
    res.index = pair.index

    if viz is not None:
        viz.line(pair.rename(columns={"x": x_name, "y": y_name}),
                 title=f"{y_name}  vs  {x_name}", yaxis_title="level")
        viz.line(res[["beta", "r2"]], left=["r2"],
                 title=f"rolling β and R²  (lookback={lookback})",
                 yaxis_title="β", yaxis_right_title="R²")
        viz.line(res[["resid"]].rename(columns={"resid": f"resid({y_name} | {x_name})"}),
                 title=f"residual:  {y_name} − ({lookback}d β·{x_name} + α)",
                 yaxis_title="resid",
                 residual=True,
                 hlines=[(0, None, "solid")])

    _print_fingerprint(f"{y_name} | {x_name}", residual_fingerprint(res["resid"]))
    return res


def corr_scan(resid: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    """Corr of resid against each candidate column (level and change). Sorted by |corr_level|."""
    rows = []
    for col in candidates.columns:
        joint = pd.concat([resid.rename("r"), candidates[col].rename("z")], axis=1).dropna()
        if len(joint) < 100:
            continue
        rows.append({
            "candidate":   col,
            "corr_level":  joint["r"].corr(joint["z"]),
            "corr_change": joint["r"].diff().corr(joint["z"].diff()),
            "n":           len(joint),
        })
    return (pd.DataFrame(rows)
              .assign(abs_corr=lambda d: d["corr_level"].abs())
              .sort_values("abs_corr", ascending=False)
              .drop(columns="abs_corr")
              .reset_index(drop=True))


def predict_scan(
    resid: pd.Series,
    candidates: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 20, 60),
) -> pd.DataFrame:
    """Spearman IC of resid_t vs fwd_change(candidate, h) for each h. Sorted by |ic|."""
    rows = []
    for col in candidates.columns:
        for h in horizons:
            fwd = candidates[col].diff(h).shift(-h)
            joint = pd.concat([resid.rename("r"), fwd.rename("f")], axis=1).dropna()
            if len(joint) < 100:
                continue
            rows.append({
                "target":  col,
                "horizon": h,
                "ic":      joint["r"].corr(joint["f"], method="spearman"),
                "n":       len(joint),
            })
    return (pd.DataFrame(rows)
              .assign(abs_ic=lambda d: d["ic"].abs())
              .sort_values("abs_ic", ascending=False)
              .drop(columns="abs_ic")
              .reset_index(drop=True))


def peel(
    resid: pd.Series,
    z: pd.Series,
    *,
    lookback: int = 60,
    resid_name: str = "resid",
    z_name: str = "z",
    viz: Viz | None = None,
) -> pd.DataFrame:
    """Strip the part of `resid` explained by `z`. Returns drill output for the new layer."""
    return drill(z, resid, lookback=lookback, x_name=z_name, y_name=resid_name, viz=viz)


def _latest_ou_metrics(resid: pd.Series, lookback: int = 252) -> dict:
    """Latest OU stats on a trailing residual window."""
    r = resid.dropna()
    if len(r) < max(20, lookback // 4):
        return {
            "ou_mean_resid": np.nan,
            "ou_zscore": np.nan,
            "half_life_d": np.nan,
            "ou_window_n": len(r),
        }

    window = r.tail(lookback)
    params = ou_params(window)
    sigma = float(window.std())
    mu = float(params["mu"])
    current = float(window.iloc[-1])
    z = (current - mu) / sigma if sigma > 0 and np.isfinite(mu) else np.nan
    return {
        "ou_mean_resid": mu,
        "ou_zscore": z,
        "half_life_d": float(params["half_life"]),
        "ou_window_n": len(window),
    }


def single_factor_10y_table(
    df: pd.DataFrame,
    *,
    target: str = "10y",
    lookback: int = 60,
    ou_lookback: int = 252,
    exclude: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Regress 10y against each candidate one at a time and summarize latest OU residual state."""
    rows = []
    models: dict[str, pd.DataFrame] = {}
    candidates = [c for c in df.columns if c != target and c not in exclude]

    for anchor in candidates:
        pair = df[[anchor, target]].dropna()
        if len(pair) < lookback:
            continue

        out = roll_lr(pair[anchor], pair[target], lookback=lookback).to_pandas()
        out.index = pair.index
        out["ou_zscore"] = roll_ou_zscore(out["resid"], lookback=ou_lookback).to_numpy()
        models[anchor] = out

        valid = out.dropna(subset=["y", "yhat", "resid"])
        if valid.empty:
            continue

        last = valid.iloc[-1]
        ou = _latest_ou_metrics(out["resid"], lookback=ou_lookback)
        ou_mean = ou["ou_mean_resid"]
        rows.append({
            "anchor": anchor,
            "date": valid.index[-1],
            "actual_10y": float(last["y"]),
            "model_10y": float(last["yhat"]),
            "residual": float(last["resid"]),
            "ou_mean_resid": ou_mean,
            "ou_fair_10y": float(last["yhat"] + ou_mean) if np.isfinite(ou_mean) else np.nan,
            "ou_zscore": ou["ou_zscore"],
            "half_life_d": ou["half_life_d"],
            "beta": float(last["beta"]),
            "r2": float(last["r2"]),
            "ou_window_n": ou["ou_window_n"],
        })

    table = pd.DataFrame(rows)
    if not table.empty:
        table = (table
                 .assign(abs_ou_zscore=lambda d: d["ou_zscore"].abs())
                 .sort_values("abs_ou_zscore", ascending=False)
                 .drop(columns="abs_ou_zscore")
                 .reset_index(drop=True))
    return table, models


# ─── backtest ──────────────────────────────────────────────────────────────

def backtest_ou_signals(
    models: dict[str, pd.DataFrame],
    *,
    horizons: tuple[int, ...] = (5, 20, 60),
) -> pd.DataFrame:
    """Spearman IC, hit rate, and 1d Sharpe for each model's OU z-score signal.

    Signal convention: ou_zscore > 0 → 10y is rich → expect yield to fall.
    Correct prediction: sign(ou_zscore_t) == sign(-Δy_{t+H}).
    """
    rows = []
    for anchor, model in models.items():
        row: dict = {"anchor": anchor}
        signal = model["ou_zscore"]
        y = model["y"]

        fwd_1d = y.diff(1).shift(-1)
        j1 = pd.concat([signal.rename("s"), fwd_1d.rename("f")], axis=1).dropna()
        j1 = j1[j1["s"] != 0]
        if len(j1) >= 30:
            pnl = np.sign(j1["s"]) * (-j1["f"])
            row["sharpe_1d"] = float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else np.nan
        else:
            row["sharpe_1d"] = np.nan

        for h in horizons:
            fwd = y.diff(h).shift(-h)
            jh = pd.concat([signal.rename("s"), fwd.rename("f")], axis=1).dropna()
            jh = jh[jh["s"] != 0]
            if len(jh) < 30:
                row[f"ic_{h}d"] = np.nan
                row[f"hit_{h}d"] = np.nan
                row[f"sharpe_{h}d"] = np.nan
                continue
            pnl = np.sign(jh["s"]) * (-jh["f"])
            row[f"ic_{h}d"] = float(jh["s"].corr(-jh["f"], method="spearman"))
            row[f"hit_{h}d"] = float((np.sign(jh["s"]) == np.sign(-jh["f"])).mean())
            row[f"sharpe_{h}d"] = float(pnl.mean() / pnl.std() * np.sqrt(252.0 / h)) if pnl.std() > 0 else np.nan

        rows.append(row)

    col_order = ["anchor", "ic_5d", "ic_20d", "ic_60d", "hit_5d", "hit_20d", "hit_60d",
                 "sharpe_1d", "sharpe_5d", "sharpe_20d", "sharpe_60d"]
    bt = pd.DataFrame(rows)
    return bt[[c for c in col_order if c in bt.columns]]


# ─── trade simulation + deep-dive charts ─────────────────────────────────

def simulate_trades(
    model: pd.DataFrame,
    filter_mask: pd.Series,
    *,
    entry_z: float = 1.0,
) -> pd.DataFrame:
    """Simulate discrete OU trades with half-life exits.

    Entry  : |ou_zscore| >= entry_z AND filter active AND no open trade.
    Exit   : residual crosses trailing ou_mean in the right direction,
             OR max_hold = round(half_life) bars elapses.
    PnL    : direction × (entry_10y − exit_10y), bps.
    """
    ou_z_v  = model["ou_zscore"].values
    resid_v = model["resid"].values
    y_v     = model["y"].values
    dates   = model.index
    n       = len(dates)
    mask_v  = filter_mask.reindex(dates, fill_value=False).values

    trades = []
    i = 0
    while i < n:
        z_i = ou_z_v[i]
        if (np.isfinite(z_i) and abs(z_i) >= entry_z
                and mask_v[i] and np.isfinite(resid_v[i])):

            direction   = 1 if z_i > 0 else -1
            entry_y     = y_v[i]
            entry_r     = resid_v[i]

            win = resid_v[max(0, i - 252): i + 1]
            win = win[np.isfinite(win)]
            ou_mean = float(np.mean(win)) if len(win) >= 3 else 0.0
            hl      = float(half_life(pd.Series(win))) if len(win) >= 10 else 20.0
            if not np.isfinite(hl) or hl <= 0:
                hl = 20.0
            max_hold = max(1, round(hl))

            exit_i      = min(i + max_hold, n - 1)
            exit_reason = "time"
            for j in range(i + 1, min(i + max_hold + 1, n)):
                r_j = resid_v[j]
                if not np.isfinite(r_j):
                    continue
                if (direction == 1 and r_j <= ou_mean) or (direction == -1 and r_j >= ou_mean):
                    exit_i      = j
                    exit_reason = "mean"
                    break

            pnl = direction * (entry_y - y_v[exit_i])
            trades.append({
                "entry":       dates[i],
                "exit":        dates[exit_i],
                "dir":         "RCV" if direction == 1 else "PAY",
                "entry_10y":   round(entry_y, 1),
                "exit_10y":    round(y_v[exit_i], 1),
                "pnl":         round(pnl, 1),
                "entry_z":     round(z_i, 2),
                "entry_resid": round(entry_r, 1),
                "exit_resid":  round(resid_v[exit_i], 1),
                "ou_mean":     round(ou_mean, 1),
                "hl_d":        round(hl, 1),
                "hold_d":      exit_i - i,
                "exit_by":     exit_reason,
            })
            i = exit_i + 1
        else:
            i += 1

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def _best_signal_drill(
    best_row: pd.Series,
    models: dict[str, pd.DataFrame],
    df: pd.DataFrame,
    viz,
    *,
    train_window: int = 504,
    entry_z: float = 1.0,
) -> pd.DataFrame:
    """Three deep-dive charts + trade table for the best filtered signal."""
    anchor   = best_row["anchor"]
    h        = int(best_row["horizon"])
    feat_col = best_row["best_feature"]

    model  = models[anchor]
    feats  = build_signal_features(model)
    result = oos_edge_test_fast(feats, df["10y"], feat_col, horizon=h, train_window=train_window)

    mask       = result["filter_mask"]
    ou_z       = model["ou_zscore"]
    y_10       = model["y"]
    feat_ser   = feats[feat_col].reindex(model.index)
    trades     = simulate_trades(model, mask, entry_z=entry_z)
    sfx        = f"10y | {anchor}  ·  filter: {feat_col}  ·  h={h}d"

    # ── Chart 1: signal conditions ──────────────────────────────────────────
    # OU z-score (right) + filter feature (left) + shading where both fire.
    def _render_conditions(fig, ax, start, end):
        sub_z = ou_z.loc[start:end]
        sub_f = feat_ser.loc[start:end]
        sub_m = mask.reindex(sub_z.index, fill_value=False)

        yz = max(float(sub_z.abs().max(skipna=True)) * 1.2, entry_z * 2.5)
        rcv_on = (sub_m & (sub_z >= entry_z)).values.astype(bool)
        pay_on = (sub_m & (sub_z <= -entry_z)).values.astype(bool)

        ax.fill_between(sub_z.index, -yz, yz, where=rcv_on,
                        color='#27AE60', alpha=0.13, zorder=1, label='_nolegend_')
        ax.fill_between(sub_z.index, -yz, yz, where=pay_on,
                        color='#C0392B', alpha=0.13, zorder=1, label='_nolegend_')

        ax.plot(sub_z.index, sub_z.values, color='#2980B9', linewidth=1.5,
                label='OU z-score', zorder=3)
        ax.axhline( entry_z, color='#27AE60', linestyle='--', linewidth=0.9, alpha=0.8)
        ax.axhline(-entry_z, color='#C0392B', linestyle='--', linewidth=0.9, alpha=0.8)
        ax.axhline(0, color='#666', linestyle='-', linewidth=0.8)
        ax.set_ylim(-yz, yz)

        # entry dots on z-score line
        if not trades.empty:
            sub_t = trades[(trades["entry"] >= start) & (trades["entry"] <= end)]
            for _, t in sub_t.iterrows():
                d = t["entry"]
                if d in sub_z.index:
                    zv = sub_z.loc[d]
                    c  = '#27AE60' if t["dir"] == "RCV" else '#C0392B'
                    mk = 'v'       if t["dir"] == "RCV" else '^'
                    ax.scatter([d], [zv], marker=mk, color=c, s=90, zorder=6,
                               edgecolors='white', linewidths=0.5)

        # filter feature on left axis
        ax2 = ax.twinx()
        ax2.plot(sub_f.index, sub_f.values, color='#F39C12', linewidth=1.2,
                 linestyle='--', alpha=0.85, label=feat_col, zorder=2)
        ax2.axhline(0, color='#F39C12', linestyle=':', linewidth=0.6, alpha=0.5)
        ax2.grid(False)
        ax2.yaxis.tick_left()
        ax2.yaxis.set_label_position('left')
        ax2.set_ylabel(feat_col.upper(), fontsize=8, color='#F39C12')
        for sp in ('top', 'right', 'bottom'):
            ax2.spines[sp].set_visible(False)
        ax2.tick_params(axis='y', labelsize=8, colors='#F39C12')

        viz._style_ax(ax, yaxis_title='z-score')
        viz._format_dates(ax, start, end)
        viz._legend(ax)

    viz._make_time_nav(
        pd.DataFrame(index=ou_z.dropna().index),
        _render_conditions,
        title=f"signal conditions  ·  {sfx}",
    )

    # ── Chart 2: 10y yield with entry / exit markers ────────────────────────
    def _render_yield(fig, ax, start, end):
        sub_y = y_10.loc[start:end].dropna()
        ax.plot(sub_y.index, sub_y.values, color='#2C3E50', linewidth=1.5,
                label='10y', zorder=3)

        if not trades.empty:
            sub_t = trades[(trades["entry"] >= start) & (trades["entry"] <= end)]

            def _y_at(ds):
                return [float(y_10.loc[d]) if d in y_10.index else np.nan for d in ds]

            rcv = sub_t[sub_t["dir"] == "RCV"]
            pay = sub_t[sub_t["dir"] == "PAY"]
            if not rcv.empty:
                ax.scatter(pd.DatetimeIndex(rcv["entry"]), _y_at(rcv["entry"]),
                           marker='v', color='#27AE60', s=180, zorder=6,
                           label='receive ▼', edgecolors='white', linewidths=0.8)
            if not pay.empty:
                ax.scatter(pd.DatetimeIndex(pay["entry"]), _y_at(pay["entry"]),
                           marker='^', color='#C0392B', s=180, zorder=6,
                           label='pay ▲', edgecolors='white', linewidths=0.8)

            # exit circles
            ax.scatter(pd.DatetimeIndex(sub_t["exit"]), _y_at(sub_t["exit"]),
                       marker='o', color='#7F8C8D', s=60, zorder=5,
                       label='exit', edgecolors='white', linewidths=0.8)

            # entry→exit dotted connector per trade
            for _, t in sub_t.iterrows():
                if t["entry"] in y_10.index and t["exit"] in y_10.index:
                    c = '#27AE60' if t["dir"] == "RCV" else '#C0392B'
                    ax.plot([t["entry"], t["exit"]],
                            [y_10.loc[t["entry"]], y_10.loc[t["exit"]]],
                            color=c, linewidth=1.0, linestyle=':', alpha=0.5, zorder=4)

        viz._style_ax(ax, yaxis_title='yield, bps')
        viz._format_dates(ax, start, end)
        viz._legend(ax)

    viz._make_time_nav(
        pd.DataFrame(index=y_10.dropna().index),
        _render_yield,
        title=f"10y yield  ·  entries & exits  ·  {sfx}",
    )

    # ── Chart 3: cumulative PnL + per-trade bars ────────────────────────────
    def _render_pnl(fig, ax, start, end):
        if trades.empty:
            return
        all_t = trades[trades["exit"] <= end].copy()
        if all_t.empty:
            return

        cum = all_t.set_index("exit")["pnl"].cumsum()
        ax.step(cum.index, cum.values, where='post', color='#2980B9',
                linewidth=2.0, label='cum PnL', zorder=3)
        ax.axhline(0, color='#666', linestyle='-', linewidth=0.8)
        ax.fill_between(cum.index, 0, cum.values,
                        where=(cum.values >= 0), color='#27AE60', alpha=0.12, step='post')
        ax.fill_between(cum.index, 0, cum.values,
                        where=(cum.values < 0),  color='#C0392B', alpha=0.12, step='post')

        vis_t = all_t[(all_t["exit"] >= start) & (all_t["exit"] <= end)]
        if not vis_t.empty:
            ax2 = ax.twinx()
            bc  = ['#27AE60' if v >= 0 else '#C0392B' for v in vis_t["pnl"]]
            ax2.bar(pd.DatetimeIndex(vis_t["exit"]), vis_t["pnl"].values,
                    color=bc, width=3, alpha=0.4, zorder=2, label='trade PnL')
            ax2.axhline(0, color='#666', linestyle='-', linewidth=0.5, alpha=0.5)
            ax2.grid(False)
            ax2.yaxis.tick_left()
            ax2.yaxis.set_label_position('left')
            ax2.set_ylabel('TRADE PNL, BPS', fontsize=8)
            for sp in ('top', 'right', 'bottom'):
                ax2.spines[sp].set_visible(False)
            ax2.tick_params(axis='y', labelsize=8)

        viz._style_ax(ax, yaxis_title='cum pnl, bps')
        viz._format_dates(ax, start, end)
        viz._legend(ax)

    viz._make_time_nav(
        pd.DataFrame(index=y_10.dropna().index),
        _render_pnl,
        title=f"PnL  ·  {sfx}",
    )

    # trade table
    if not trades.empty:
        display(trades)
        viz.table(trades, title=f"trade log  ·  {sfx}")

    return trades


# ─── main ──────────────────────────────────────────────────────────────────

def main() -> dict:
    viz = Viz(backend='plotly')

    df = load_basket()
    print(f"basket: {len(df)} obs · {df.index.min().date()} → {df.index.max().date()}")
    print(f"        cols: {list(df.columns)}")

    # Single-factor 10y models: each candidate gets its own rolling regression.
    ten_single, ten_single_models = single_factor_10y_table(
        df,
        target="10y",
        lookback=60,
        ou_lookback=252,
    )
    print("\n--- 10y single-factor rolling regression residual OU table ---")
    display_cols = [
        "anchor", "date", "actual_10y", "model_10y", "residual",
        "ou_mean_resid", "ou_fair_10y", "ou_zscore", "half_life_d",
        "beta", "r2",
    ]
    if ten_single.empty:
        print("No single-factor models had enough overlapping data.")
    else:
        display(ten_single[display_cols].round(3))
        viz.table(
            ten_single[display_cols].round(3),
            title="10y single-factor rolling regression residual OU table",
        )

    bt = backtest_ou_signals(ten_single_models)
    if not ten_single.empty and not bt.empty:
        anchor_order = ten_single["anchor"].tolist()
        bt = bt.set_index("anchor").reindex(anchor_order).reset_index()
        display(bt.round(3))
        viz.table(
            bt.round(3),
            title="10y single-factor OU signal backtest  (IC + hit rate + sharpe)",
        )

    # Filtered sharpe: best OOS feature filter per anchor across all horizons.
    print("\n--- filtered sharpe summary (best feature×horizon per anchor) ---")
    filt = filtered_sharpe_summary(ten_single_models, df["10y"])
    if not filt.empty:
        display(filt.round(3))
        viz.table(filt.round(3), title="best OOS-filtered sharpe per anchor")

    # Deep-dive charts for the best filtered signal.
    if not filt.empty:
        _best_signal_drill(filt.iloc[0], ten_single_models, df, viz)

    return {
        "df":                df,
        "ten_single":        ten_single,
        "ten_single_models": ten_single_models,
        "bt":                bt,
        "filt":              filt,
    }


if __name__ == "__main__":
    state = main()
