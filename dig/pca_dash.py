"""PCA Curve Signal Explorer — Dash App

Interactive dashboard to explore PCA curve signals:
  - Select a PC component and structure
  - See regression diagnostics (residual, z-score, beta, R2)
  - Backtest the z-score signal with entry/exit flags
  - Trade log table
  - Strategy metrics

Usage:
    python dig/pca_dash.py [--curve sofr|ust] [--port 8050]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dig.pca_curve_signals import main as run_scanner

# ── colors ──────────────────────────────────────────────────────────────
C = {
    "bg": "#1a1a2e",
    "card": "#16213e",
    "accent": "#e94560",
    "accent2": "#0f3460",
    "text": "#eee",
    "muted": "#888",
    "green": "#27ae60",
    "red": "#e74c3c",
    "orange": "#f39c12",
    "blue": "#3498db",
    "grid": "#2a2a4a",
}

PC_COLORS = {"PC1": "#3498db", "PC2": "#e74c3c", "PC3": "#27ae60"}

INPUT_STYLE = {
    "backgroundColor": "#1a1a2e", "color": "#eee",
    "border": "1px solid #444", "borderRadius": "4px",
    "padding": "6px 8px", "fontSize": "12px",
}

HEATMAP_COLORSCALE = [
    [0.0, "#c0392b"],
    [0.25, "#78281f"],
    [0.5, C["card"]],
    [0.75, "#0e4429"],
    [1.0, "#27ae60"],
]

# ── backtest logic ──────────────────────────────────────────────────────

def backtest_signal(
    reg: pd.DataFrame,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
    hl_exit_pct: float = 0.0,
    lookback: int = 252,
    min_r2: float = 0.0,
    max_beta_cv: float = 99.0,
):
    """Backtest a mean-reversion signal on the residual.

    Entry filters (all backward-looking, checked at prior close):
      - |ou_z| >= entry_z
      - rolling R² >= min_r2
      - rolling beta CV <= max_beta_cv

    Exit conditions (whichever comes first):
      1. z crosses back through exit_z (z-based exit)
      2. held for hl_exit_pct * rolling_half_life days (time-based exit)
         set hl_exit_pct=0 to disable time-based exits
    """
    cols = ["y", "resid", "ou_z"]
    if "r2_corr" in reg.columns:
        cols.append("r2_corr")
    if "beta_cv" in reg.columns:
        cols.append("beta_cv")
    df = reg[cols].dropna(subset=["y", "resid", "ou_z"]).copy()
    df["d_y"] = df["y"].diff()

    use_hl_exit = hl_exit_pct > 0
    from stats.ou import roll_half_life
    # Compute rolling half-life on the FULL residual series (including warmup
    # data before ou_z is available) so that by the time the first trade can
    # fire, the half-life has already had enough lookback to produce values.
    full_resid = reg["resid"].dropna()
    hl_full = roll_half_life(full_resid, lookback=lookback).to_pandas()
    hl_full.index = full_resid.index[-len(hl_full):]
    df["roll_hl"] = hl_full.reindex(df.index)

    use_r2_filter = min_r2 > 0
    use_cv_filter = max_beta_cv < 99.0

    pos = np.zeros(len(df))
    hold_days = np.zeros(len(df), dtype=int)
    entry_hl = np.full(len(df), np.nan)

    for i in range(1, len(df)):
        z = df["ou_z"].iloc[i - 1]
        prev = pos[i - 1]
        held = hold_days[i - 1]
        hl_at_entry = entry_hl[i - 1]

        # quality gates (backward-looking: use prior bar's values)
        quality_ok = True
        if use_r2_filter and "r2_corr" in df.columns:
            r2_val = df["r2_corr"].iloc[i - 1]
            if np.isnan(r2_val) or r2_val < min_r2:
                quality_ok = False
        if quality_ok and use_cv_filter and "beta_cv" in df.columns:
            cv_val = df["beta_cv"].iloc[i - 1]
            if np.isnan(cv_val) or cv_val > max_beta_cv:
                quality_ok = False

        # time-based exit: held >= pct * half_life known at entry
        max_hold = int(hl_exit_pct * hl_at_entry) if (use_hl_exit and np.isfinite(hl_at_entry) and hl_at_entry > 0) else 0
        if max_hold > 0 and prev != 0 and held >= max_hold:
            pos[i] = 0.0
            hold_days[i] = 0
            entry_hl[i] = np.nan
        # z-based exit
        elif prev > 0 and z > -exit_z:
            pos[i] = 0.0
            hold_days[i] = 0
            entry_hl[i] = np.nan
        elif prev < 0 and z < exit_z:
            pos[i] = 0.0
            hold_days[i] = 0
            entry_hl[i] = np.nan
        # new entries — must pass quality gates
        elif z <= -entry_z and prev != 1.0 and quality_ok:
            pos[i] = 1.0
            hold_days[i] = 1
            entry_hl[i] = df["roll_hl"].iloc[i - 1]
        elif z >= entry_z and prev != -1.0 and quality_ok:
            pos[i] = -1.0
            hold_days[i] = 1
            entry_hl[i] = df["roll_hl"].iloc[i - 1]
        # hold existing
        elif prev != 0:
            pos[i] = prev
            hold_days[i] = held + 1
            entry_hl[i] = hl_at_entry
        else:
            pos[i] = 0.0
            hold_days[i] = 0
            entry_hl[i] = np.nan

    df["pos"] = pos
    df["hold_days"] = hold_days
    df["entry_hl"] = entry_hl
    df["pnl"] = df["pos"] * df["d_y"]
    df["cum_pnl"] = df["pnl"].cumsum()
    return df


def compute_metrics(bt: pd.DataFrame, trade_log: pd.DataFrame | None = None) -> dict:
    pnl = bt["pnl"].dropna()
    if len(pnl) < 10:
        return {}
    cum = pnl.cumsum()
    peak = cum.cummax()
    dd = cum - peak

    ann_pnl = pnl.mean() * 252
    ann_vol = pnl.std() * np.sqrt(252)
    sharpe = ann_pnl / ann_vol if ann_vol > 0 else np.nan
    max_dd = dd.min()
    calmar = ann_pnl / abs(max_dd) if max_dd != 0 else np.nan

    pos_s = bt["pos"]
    entries = ((pos_s != 0) & (pos_s.shift(1).fillna(0) == 0)).sum()

    metrics = {
        "Sharpe": f"{sharpe:.2f}",
        "Calmar": f"{calmar:.2f}",
        "Ann PnL": f"{ann_pnl:.1f}",
        "Ann Vol": f"{ann_vol:.1f}",
        "Max DD": f"{max_dd:.1f}",
    }

    # trade-level stats from the round-trip log
    if trade_log is not None and not trade_log.empty:
        closed = trade_log[trade_log["exit"] != "(open)"].copy()
        if not closed.empty:
            pnls = closed["pnl"].astype(float)
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            avg_win = wins.mean() if len(wins) > 0 else 0.0
            avg_loss = losses.mean() if len(losses) > 0 else 0.0
            gross_profit = wins.sum() if len(wins) > 0 else 0.0
            gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            win_rate = len(wins) / len(pnls) if len(pnls) > 0 else np.nan
            avg_dur = closed["duration_d"].astype(float).mean()

            metrics["Win Rate"] = f"{win_rate:.1%}"
            metrics["Profit Factor"] = f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "inf"
            metrics["# Trades"] = f"{entries}"
            metrics["Avg Win"] = f"{avg_win:+.2f}"
            metrics["Avg Loss"] = f"{avg_loss:+.2f}"
            metrics["Avg PnL"] = f"{pnls.mean():+.2f}"
            metrics["Best"] = f"{pnls.max():+.2f}"
            metrics["Worst"] = f"{pnls.min():+.2f}"
            metrics["Avg Dur"] = f"{avg_dur:.0f}d"
            metrics["% Invested"] = f"{(pos_s != 0).mean():.0%}"
    else:
        metrics["# Trades"] = f"{entries}"
        metrics["% Invested"] = f"{(pos_s != 0).mean():.0%}"

    return metrics


def build_trade_log(bt: pd.DataFrame) -> pd.DataFrame:
    """Build round-trip trade log with actual structure price entry/exit, PnL, duration."""
    trades = []
    entry_date = None
    entry_price = None
    entry_z = None
    entry_hl_val = np.nan
    direction = None
    trade_pnl = 0.0

    for i in range(len(bt)):
        row = bt.iloc[i]
        idx = bt.index[i]
        p = row["pos"]
        prev_p = bt["pos"].iloc[i - 1] if i > 0 else 0.0

        # new entry — signal fires at prior close, so entry price is y[i-1]
        if p != 0 and prev_p == 0:
            entry_date = idx
            entry_price = bt["y"].iloc[i - 1] if i > 0 else row["y"]
            entry_z = bt["ou_z"].iloc[i - 1] if i > 0 else row["ou_z"]
            entry_hl_val = row["entry_hl"] if "entry_hl" in bt.columns else np.nan
            direction = "LONG" if p > 0 else "SHORT"
            trade_pnl = 0.0

        # accumulate pnl while in trade
        if p != 0 and not np.isnan(row["pnl"]):
            trade_pnl += row["pnl"]

        # exit — last held bar is i-1, so exit price is y[i-1]
        if p == 0 and prev_p != 0 and entry_date is not None:
            exit_price = bt["y"].iloc[i - 1]
            exit_z = bt["ou_z"].iloc[i - 1]
            duration = (idx - entry_date).days
            trades.append({
                "entry": entry_date.strftime("%Y-%m-%d"),
                "exit": idx.strftime("%Y-%m-%d"),
                "dir": direction,
                "entry_px": round(entry_price, 2),
                "exit_px": round(exit_price, 2),
                "entry_z": round(entry_z, 2),
                "exit_z": round(exit_z, 2),
                "pnl": round(trade_pnl, 2),
                "duration_d": duration,
                "half_life": round(entry_hl_val, 1) if np.isfinite(entry_hl_val) else "—",
                "win": "Y" if trade_pnl > 0 else "N",
            })
            entry_date = None

    # handle open trade
    if entry_date is not None:
        last = bt.iloc[-1]
        duration = (bt.index[-1] - entry_date).days
        trades.append({
            "entry": entry_date.strftime("%Y-%m-%d"),
            "exit": "(open)",
            "dir": direction,
            "entry_px": round(entry_price, 2),
            "exit_px": round(last["y"], 2),
            "entry_z": round(entry_z, 2),
            "exit_z": round(last["ou_z"], 2),
            "pnl": round(trade_pnl, 2),
            "duration_d": duration,
            "half_life": round(entry_hl_val, 1) if np.isfinite(entry_hl_val) else "—",
            "win": "Y" if trade_pnl > 0 else "N",
        })

    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)


# ── chart builders ──────────────────────────────────────────────────────

def _layout(title: str = "", height: int = 400) -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor=C["card"],
        plot_bgcolor=C["card"],
        font=dict(color=C["text"], size=11),
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=50, r=50, t=40, b=30),
        height=height,
        xaxis=dict(gridcolor=C["grid"], showgrid=True),
        yaxis=dict(gridcolor=C["grid"], showgrid=True, side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )


def make_loadings_bar(full_pca: dict) -> go.Figure:
    loadings = full_pca["loadings"].to_pandas()
    tenors = loadings["column"].tolist()
    ev = full_pca["explained_variance"]
    pc_cols = [c for c in loadings.columns if c.startswith("PC")]

    fig = go.Figure()
    for i, pc in enumerate(pc_cols[:3]):
        fig.add_trace(go.Bar(
            x=tenors, y=loadings[pc].values,
            name=f"{pc} ({ev[i]*100:.1f}%)",
            marker_color=list(PC_COLORS.values())[i],
        ))
    fig.update_layout(**_layout("PCA Loadings by Tenor", height=350), barmode="group")
    return fig


def make_cum_scores(cum_scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in cum_scores.columns:
        fig.add_trace(go.Scatter(
            x=cum_scores.index, y=cum_scores[col],
            name=col, line=dict(color=PC_COLORS.get(col, C["blue"]), width=1.5),
        ))
    fig.update_layout(**_layout("Cumulative PC Scores (Factors)", height=350))
    return fig


def make_diagnostics(reg: pd.DataFrame, struct_name: str, pc: str, lb: int) -> go.Figure:
    """4-panel diagnostic: residual, z-score, beta, R2."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=["Residual", "OU Z-Score", "Rolling Beta", "Rolling R²"],
    )
    clean = reg.dropna()

    fig.add_trace(go.Scatter(
        x=clean.index, y=clean["resid"],
        line=dict(color=C["blue"], width=1), name="resid", showlegend=False,
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=C["muted"], row=1, col=1)

    fig.add_trace(go.Scatter(
        x=clean.index, y=clean["ou_z"],
        line=dict(color=C["accent"], width=1.2), name="z", showlegend=False,
    ), row=2, col=1)
    for lv in [-2, -1, 0, 1, 2]:
        fig.add_hline(y=lv, line_dash="dot", line_color=C["muted"],
                      line_width=0.5, row=2, col=1)

    fig.add_trace(go.Scatter(
        x=clean.index, y=clean["beta"],
        line=dict(color=C["orange"], width=1), name="beta", showlegend=False,
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=clean.index, y=clean["r2"],
        line=dict(color=C["green"], width=1), name="r2", showlegend=False,
    ), row=4, col=1)

    layout = _layout(f"{struct_name} ~ {pc} ({lb}d)", height=700)
    layout["yaxis"] = dict(gridcolor=C["grid"], side="right")
    layout["yaxis2"] = dict(gridcolor=C["grid"], side="right")
    layout["yaxis3"] = dict(gridcolor=C["grid"], side="right")
    layout["yaxis4"] = dict(gridcolor=C["grid"], side="right")
    fig.update_layout(**layout)
    return fig


def make_backtest_chart(bt: pd.DataFrame, reg: pd.DataFrame, struct_name: str) -> go.Figure:
    """Backtest chart: structure price + entries, residual, z-score, cum PnL."""
    clean = reg.dropna()
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=[f"{struct_name} (price + signals)", "Residual", "OU Z-Score + Entries", "Cumulative PnL"],
        row_heights=[0.3, 0.2, 0.25, 0.25],
    )

    fig.add_trace(go.Scatter(
        x=clean.index, y=clean["y"],
        line=dict(color=C["text"], width=1.2), name=struct_name, showlegend=False,
    ), row=1, col=1)

    # Detect position changes — these happen at bar i, but the signal fired
    # at bar i-1 (prior close). Shift markers back by one bar so they appear
    # where the signal was actually observed.
    def _shift_back(mask):
        """Return bt rows one bar before each True in mask (the signal bar)."""
        ipos = np.flatnonzero(mask.values) - 1
        ipos = ipos[ipos >= 0]
        return bt.iloc[ipos]

    sig_long = _shift_back((bt["pos"].diff() > 0) & (bt["pos"] > 0))
    sig_short = _shift_back((bt["pos"].diff() < 0) & (bt["pos"] < 0))
    sig_exit = _shift_back((bt["pos"].diff() != 0) & (bt["pos"] == 0))

    if not sig_long.empty:
        y_vals = clean["y"].reindex(sig_long.index)
        fig.add_trace(go.Scatter(
            x=sig_long.index, y=y_vals,
            mode="markers", marker=dict(symbol="triangle-up", size=9, color=C["green"]),
            name="Long", showlegend=True,
        ), row=1, col=1)
    if not sig_short.empty:
        y_vals = clean["y"].reindex(sig_short.index)
        fig.add_trace(go.Scatter(
            x=sig_short.index, y=y_vals,
            mode="markers", marker=dict(symbol="triangle-down", size=9, color=C["red"]),
            name="Short", showlegend=True,
        ), row=1, col=1)
    if not sig_exit.empty:
        y_vals = clean["y"].reindex(sig_exit.index)
        fig.add_trace(go.Scatter(
            x=sig_exit.index, y=y_vals,
            mode="markers", marker=dict(symbol="x", size=7, color=C["muted"]),
            name="Exit", showlegend=True,
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=clean.index, y=clean["resid"],
        line=dict(color=C["blue"], width=1), name="resid", showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=C["muted"], line_width=0.5, row=2, col=1)

    fig.add_trace(go.Scatter(
        x=bt.index, y=bt["ou_z"],
        line=dict(color=C["accent"], width=1.2), name="z", showlegend=False,
    ), row=3, col=1)
    for lv in [-2, -1, 0, 1, 2]:
        fig.add_hline(y=lv, line_dash="dot", line_color=C["muted"],
                      line_width=0.5, row=3, col=1)

    if not sig_long.empty:
        fig.add_trace(go.Scatter(
            x=sig_long.index, y=sig_long["ou_z"],
            mode="markers", marker=dict(symbol="triangle-up", size=7, color=C["green"]),
            name="Long", showlegend=False,
        ), row=3, col=1)
    if not sig_short.empty:
        fig.add_trace(go.Scatter(
            x=sig_short.index, y=sig_short["ou_z"],
            mode="markers", marker=dict(symbol="triangle-down", size=7, color=C["red"]),
            name="Short", showlegend=False,
        ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=bt.index, y=bt["cum_pnl"],
        line=dict(color=C["green"], width=1.5), fill="tozeroy",
        fillcolor="rgba(39,174,96,0.1)", name="PnL", showlegend=False,
    ), row=4, col=1)

    layout = _layout(f"Backtest: {struct_name}", height=750)
    layout["yaxis"] = dict(gridcolor=C["grid"], side="right")
    layout["yaxis2"] = dict(gridcolor=C["grid"], side="right")
    layout["yaxis3"] = dict(gridcolor=C["grid"], side="right")
    layout["yaxis4"] = dict(gridcolor=C["grid"], side="right")
    fig.update_layout(**layout)
    return fig


def make_monthly_heatmap(bt: pd.DataFrame) -> go.Figure:
    """Monthly PnL heatmap: rows=years, cols=months + yearly total."""
    pnl = bt["pnl"].dropna()
    if pnl.empty:
        fig = go.Figure()
        fig.update_layout(**_layout("Monthly PnL", height=350))
        return fig

    monthly = pnl.groupby([pnl.index.year, pnl.index.month]).sum()
    monthly.index.names = ["year", "month"]
    monthly = monthly.reset_index()
    pivot = monthly.pivot(index="year", columns="month", values="pnl").sort_index()

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    full_cols = list(range(1, 13))
    pivot = pivot.reindex(columns=full_cols)

    # add a NaN spacer column + yearly total
    monthly_vals = pivot[full_cols].to_numpy(dtype=float)
    yearly_totals = np.nansum(monthly_vals, axis=1, keepdims=True)
    spacer = np.full((len(pivot), 1), np.nan)

    # combine: months | spacer | year — single z matrix
    z = np.hstack([monthly_vals, spacer, yearly_totals])

    # normalize everything to a single scale (use the yearly range so months
    # still look vivid and yearly column isn't washed out)
    abs_max = max(np.nanmax(np.abs(monthly_vals)), 1.0)
    yearly_abs = max(np.nanmax(np.abs(yearly_totals)), 1.0)
    # scale yearly values into the monthly range so one colorbar works
    scale_factor = abs_max / yearly_abs if yearly_abs > 0 else 1.0
    z_scaled = z.copy()
    z_scaled[:, -1] = yearly_totals[:, 0] * scale_factor

    x_labels = month_names + ["", "Year"]
    y_labels = [str(y) for y in pivot.index]

    # text: actual values (not scaled)
    text = []
    for i in range(z.shape[0]):
        row = []
        for j in range(z.shape[1]):
            v = z[i, j]
            if j == len(month_names):  # spacer column
                row.append("")
            elif np.isnan(v):
                row.append("")
            elif j == len(month_names) + 1:  # yearly column — bold
                row.append(f"<b>{v:.1f}</b>")
            else:
                row.append(f"{v:.1f}")
        text.append(row)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z_scaled,
        x=x_labels,
        y=y_labels,
        colorscale=HEATMAP_COLORSCALE,
        zmid=0,
        zmin=-abs_max,
        zmax=abs_max,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=10, color=C["text"]),
        showscale=False,
        hoverongaps=False,
        xgap=2,
        ygap=2,
        hovertemplate="<b>%{y} %{x}</b>: %{text}<extra></extra>",
    ))
    layout = _layout("Monthly PnL (bps)", height=max(200, len(y_labels) * 32 + 80))
    layout["yaxis"] = dict(gridcolor=C["card"], autorange="reversed", showgrid=False)
    layout["xaxis"] = dict(
        gridcolor=C["card"], side="top", type="category",
        categoryorder="array", categoryarray=x_labels, showgrid=False,
    )
    fig.update_layout(**layout)
    return fig


# ── optimization ─────────────────────────────────────────────────────────

ENTRY_Z_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
HL_EXIT_GRID = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
R2_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def _adaptive_cv_grid(reg: pd.DataFrame, n_bins: int = 7) -> list[float]:
    """Build a beta_cv filter grid from the empirical distribution of the series.

    Returns [no_filter, p90, p75, p60, p50, p40, p25, p10] — descending so
    the heatmap y-axis goes from loose (top) to strict (bottom).
    """
    if "beta_cv" not in reg.columns:
        return [99.0]
    cv = reg["beta_cv"].dropna()
    if len(cv) < 20:
        return [99.0]
    pcts = [90, 75, 60, 50, 40, 25, 10]
    vals = np.percentile(cv, pcts)
    # round for display, deduplicate, descending order
    grid = sorted(set(round(float(v), 3) for v in vals), reverse=True)
    # prepend 99.0 as "no filter" baseline
    return [99.0] + grid


def optimize_grid(
    reg: pd.DataFrame,
    lookback: int,
    min_r2: float = 0.0,
    max_beta_cv: float = 99.0,
) -> pd.DataFrame:
    """Grid search over entry_z x hl_exit_pct, return Sharpe + stats per combo."""
    rows = []
    for ez in ENTRY_Z_GRID:
        for hlp in HL_EXIT_GRID:
            try:
                bt = backtest_signal(
                    reg, entry_z=ez, exit_z=0.0, hl_exit_pct=hlp,
                    lookback=lookback, min_r2=min_r2, max_beta_cv=max_beta_cv,
                )
                pnl = bt["pnl"].dropna()
                if len(pnl) < 60:
                    rows.append({"entry_z": ez, "hl_exit_%": hlp, "sharpe": np.nan,
                                 "trades": 0, "win%": np.nan, "ann_pnl": np.nan})
                    continue
                ann = pnl.mean() * 252
                vol = pnl.std() * np.sqrt(252)
                sharpe = ann / vol if vol > 0 else np.nan

                pos_s = bt["pos"]
                n_trades = int(((pos_s != 0) & (pos_s.shift(1).fillna(0) == 0)).sum())

                tlog = build_trade_log(bt)
                closed = tlog[tlog["exit"] != "(open)"] if not tlog.empty else tlog
                if not closed.empty:
                    pnls = closed["pnl"].astype(float)
                    wr = (pnls > 0).sum() / len(pnls) if len(pnls) > 0 else np.nan
                else:
                    wr = np.nan

                rows.append({"entry_z": ez, "hl_exit_%": hlp, "sharpe": sharpe,
                             "trades": n_trades, "win%": wr, "ann_pnl": ann})
            except Exception:
                rows.append({"entry_z": ez, "hl_exit_%": hlp, "sharpe": np.nan,
                             "trades": 0, "win%": np.nan, "ann_pnl": np.nan})
    return pd.DataFrame(rows)


def optimize_filters(
    reg: pd.DataFrame,
    lookback: int,
    entry_z: float = 1.0,
    hl_exit_pct: float = 0.0,
) -> pd.DataFrame:
    """Grid search over min_r2 x max_beta_cv (adaptive CV grid from data)."""
    cv_grid = _adaptive_cv_grid(reg)
    rows = []
    for r2 in R2_GRID:
        for cv in cv_grid:
            try:
                bt = backtest_signal(
                    reg, entry_z=entry_z, exit_z=0.0, hl_exit_pct=hl_exit_pct,
                    lookback=lookback, min_r2=r2, max_beta_cv=cv,
                )
                pnl = bt["pnl"].dropna()
                if len(pnl) < 60:
                    rows.append({"min_r2": r2, "max_cv": cv, "sharpe": np.nan,
                                 "trades": 0, "win%": np.nan, "ann_pnl": np.nan})
                    continue
                ann = pnl.mean() * 252
                vol = pnl.std() * np.sqrt(252)
                sharpe = ann / vol if vol > 0 else np.nan
                pos_s = bt["pos"]
                n_trades = int(((pos_s != 0) & (pos_s.shift(1).fillna(0) == 0)).sum())

                tlog = build_trade_log(bt)
                closed = tlog[tlog["exit"] != "(open)"] if not tlog.empty else tlog
                wr = np.nan
                if not closed.empty:
                    pnls = closed["pnl"].astype(float)
                    wr = (pnls > 0).sum() / len(pnls) if len(pnls) > 0 else np.nan

                rows.append({"min_r2": r2, "max_cv": cv, "sharpe": sharpe,
                             "trades": n_trades, "win%": wr, "ann_pnl": ann})
            except Exception:
                rows.append({"min_r2": r2, "max_cv": cv, "sharpe": np.nan,
                             "trades": 0, "win%": np.nan, "ann_pnl": np.nan})
    return pd.DataFrame(rows)


def _opt_heatmap(pivot: pd.DataFrame, title: str, x_title: str, y_title: str) -> go.Figure:
    """Shared Sharpe heatmap builder for optimization grids."""
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(x) for x in pivot.columns],
        y=[str(y) for y in pivot.index],
        colorscale=HEATMAP_COLORSCALE,
        zmid=0,
        text=np.round(pivot.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, color=C["text"]),
        colorbar=dict(
            title=dict(text="Sharpe", font=dict(size=10)),
            tickfont=dict(size=9), len=0.8,
            outlinecolor=C["grid"], outlinewidth=1,
        ),
        xgap=2, ygap=2,
        hoverongaps=False,
    ))
    layout = _layout(title, height=380)
    layout["xaxis"] = dict(title=x_title, side="bottom", gridcolor=C["card"], showgrid=False, type="category")
    layout["yaxis"] = dict(title=y_title, side="left", gridcolor=C["card"], showgrid=False, type="category")
    fig.update_layout(**layout)
    return fig


def make_opt_heatmap(opt_df: pd.DataFrame, struct_name: str) -> go.Figure:
    pivot = opt_df.pivot(index="hl_exit_%", columns="entry_z", values="sharpe")
    return _opt_heatmap(pivot, f"Entry Z x HL Exit: {struct_name}", "Entry |z|", "Exit @ % HL")


def make_filter_heatmap(filt_df: pd.DataFrame, struct_name: str) -> go.Figure:
    pivot = filt_df.pivot(index="max_cv", columns="min_r2", values="sharpe")
    pivot = pivot.sort_index(ascending=False)
    return _opt_heatmap(pivot, f"R² x Beta CV Filters: {struct_name}", "Min R²", "Max Beta CV")


# ── app ─────────────────────────────────────────────────────────────────

def build_app(result: dict) -> dash.Dash:
    scan_df = result["scan_df"].copy()
    all_regs = result["all_regs"]
    full_pca = result["full_pca"]
    cum_scores = result["cum_scores"]

    # precompute Sharpe for each combo (default entry_z=1.0, z-exit only)
    print("Precomputing Sharpe for signal table...")
    sharpes = []
    for _, row in scan_df.iterrows():
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
    scan_df["sharpe"] = sharpes
    scan_df = scan_df.sort_values("sharpe", ascending=False)

    pcs = sorted(scan_df["pc"].unique())
    structures = sorted(scan_df["structure"].unique())
    lookbacks = sorted(scan_df["lookback"].unique())

    app = dash.Dash(__name__)

    # ── metric card helper ──────────────────────────────────────────────
    def metric_card(label, value, tone="neutral"):
        v = str(value)
        signed_pos = v.startswith("+")
        signed_neg = v.startswith("-")

        tone_styles = {
            "positive": {
                "bg": "rgba(39,174,96,0.10)",
                "border": "rgba(39,174,96,0.45)",
                "label": C["green"],
                "default": "#d9f7e5",
            },
            "risk": {
                "bg": "rgba(231,76,60,0.10)",
                "border": "rgba(231,76,60,0.45)",
                "label": C["red"],
                "default": "#ffdede",
            },
            "activity": {
                "bg": "rgba(52,152,219,0.10)",
                "border": "rgba(52,152,219,0.45)",
                "label": C["blue"],
                "default": "#dfefff",
            },
            "neutral": {
                "bg": C["card"],
                "border": C["grid"],
                "label": C["muted"],
                "default": C["text"],
            },
        }
        style = tone_styles.get(tone, tone_styles["neutral"])

        if signed_neg:
            value_color = C["red"]
        elif signed_pos:
            value_color = C["green"]
        else:
            value_color = style["default"]

        return html.Div([
            html.Div(label, style={
                "fontSize": "9px", "color": style["label"], "textTransform": "uppercase",
                "letterSpacing": "0.5px", "marginBottom": "2px",
            }),
            html.Div(value, style={
                "fontSize": "16px", "fontWeight": "600", "color": value_color,
                "fontFamily": "monospace",
            }),
        ], style={
            "backgroundColor": style["bg"], "padding": "8px 12px",
            "borderRadius": "6px", "minWidth": "80px", "textAlign": "center",
            "border": f"1px solid {style['border']}",
        })

    # ── control input helper ────────────────────────────────────────────
    def ctrl_input(label, id, value, **kwargs):
        return html.Div([
            html.Div(label, style={
                "fontSize": "10px", "color": C["muted"], "marginBottom": "4px",
                "textTransform": "uppercase", "letterSpacing": "0.5px",
            }),
            dcc.Input(id=id, type="number", value=value,
                      style={**INPUT_STYLE, "width": "72px"}, **kwargs),
        ])

    # ── layout ──────────────────────────────────────────────────────────
    app.layout = html.Div(style={"backgroundColor": C["bg"], "minHeight": "100vh", "padding": "20px 30px",
                                  "fontFamily": "'Inter', Arial, sans-serif", "color": C["text"]}, children=[

        html.H2("PCA Curve Signal Explorer", style={"marginBottom": "2px", "fontSize": "22px"}),
        html.P("SOFR OIS curve PCA  /  scan structures  /  mean-reversion signals",
               style={"color": C["muted"], "marginBottom": "20px", "fontSize": "12px"}),

        # PCA overview row
        html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "25px"}, children=[
            html.Div(dcc.Graph(figure=make_loadings_bar(full_pca), config={"displayModeBar": False}),
                     style={"flex": "1"}),
            html.Div(dcc.Graph(figure=make_cum_scores(cum_scores), config={"displayModeBar": False}),
                     style={"flex": "1"}),
        ]),

        # ── controls: two rows ──────────────────────────────────────────
        html.Div(style={
            "backgroundColor": C["card"], "borderRadius": "8px",
            "padding": "14px 18px", "marginBottom": "18px",
            "border": f"1px solid {C['grid']}",
        }, children=[
            # row 1: selectors
            html.Div(style={"display": "flex", "gap": "14px", "alignItems": "flex-end",
                             "marginBottom": "12px"}, children=[
                html.Div([
                    html.Div("PC", style={"fontSize": "10px", "color": C["muted"],
                                           "marginBottom": "4px", "textTransform": "uppercase"}),
                    dcc.Dropdown(id="pc-select", options=[{"label": p, "value": p} for p in pcs],
                                 value="PC1", clearable=False,
                                 style={"width": "100px", "backgroundColor": C["bg"], "color": "#333"}),
                ]),
                html.Div([
                    html.Div("STRUCTURE", style={"fontSize": "10px", "color": C["muted"],
                                                   "marginBottom": "4px", "textTransform": "uppercase"}),
                    dcc.Dropdown(id="struct-select", options=[{"label": s, "value": s} for s in structures],
                                 value=structures[0], clearable=False,
                                 style={"width": "200px", "backgroundColor": C["bg"], "color": "#333"}),
                ]),
                html.Div([
                    html.Div("LOOKBACK", style={"fontSize": "10px", "color": C["muted"],
                                                  "marginBottom": "4px", "textTransform": "uppercase"}),
                    dcc.Dropdown(id="lb-select",
                                 options=[{"label": f"{lb}d", "value": lb} for lb in lookbacks],
                                 value=252, clearable=False,
                                 style={"width": "100px", "backgroundColor": C["bg"], "color": "#333"}),
                ]),
            ]),

            # row 2: strategy params + quality filters
            html.Div(style={"display": "flex", "gap": "28px", "alignItems": "flex-end"}, children=[
                # strategy params group
                html.Div(style={"display": "flex", "gap": "10px", "alignItems": "flex-end"}, children=[
                    html.Div(html.Span("STRATEGY", style={
                        "fontSize": "9px", "color": C["accent"], "fontWeight": "bold",
                        "textTransform": "uppercase", "letterSpacing": "1px",
                    }), style={"paddingBottom": "8px"}),
                    ctrl_input("Entry |z|", "entry-z", 1.0, min=0.25, max=3.0, step=0.25),
                    ctrl_input("HL Exit %", "hl-exit-pct", 0, min=0, max=5.0, step=0.25),
                    html.Div(html.Span("0 = z-exit only", style={
                        "fontSize": "9px", "color": C["muted"], "fontStyle": "italic",
                    }), style={"paddingBottom": "8px"}),
                ]),

                # divider
                html.Div(style={"borderLeft": f"1px solid {C['grid']}", "height": "40px"}),

                # quality filters group
                html.Div(style={"display": "flex", "gap": "10px", "alignItems": "flex-end"}, children=[
                    html.Div(html.Span("FILTERS", style={
                        "fontSize": "9px", "color": C["orange"], "fontWeight": "bold",
                        "textTransform": "uppercase", "letterSpacing": "1px",
                    }), style={"paddingBottom": "8px"}),
                    ctrl_input("Min R²", "min-r2", 0.0, min=0.0, max=1.0, step=0.05),
                    ctrl_input("Max Beta CV", "max-beta-cv", 99.0, min=0.0, max=99.0, step=0.1),
                ]),
            ]),
        ]),

        dcc.Loading(
            id="analysis-loading",
            type="dot",
            color=C["accent"],
            children=html.Div([
                # ── metrics strip ───────────────────────────────────────
                html.Div(id="metrics-row", style={"marginBottom": "12px"}),

                # scan stats bar
                html.Div(id="scan-stats", style={
                    "backgroundColor": C["card"], "padding": "8px 14px", "borderRadius": "6px",
                    "marginBottom": "18px", "fontSize": "11px", "fontFamily": "monospace",
                    "border": f"1px solid {C['grid']}",
                }),

                # ── charts ──────────────────────────────────────────────
                html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "20px"}, children=[
                    html.Div(dcc.Graph(id="diag-chart", config={"displayModeBar": False}), style={"flex": "1"}),
                    html.Div(dcc.Graph(id="bt-chart", config={"displayModeBar": False}), style={"flex": "1"}),
                ]),

                # ── monthly pnl heatmap ────────────────────────────────
                html.Div(dcc.Graph(id="monthly-heatmap", config={"displayModeBar": False}),
                         style={"marginBottom": "20px"}),

                # ── trade log ───────────────────────────────────────────
                html.H4("Trade Log (last 50)", style={"marginBottom": "8px", "fontSize": "14px"}),
                dash_table.DataTable(
                    id="trade-table",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "backgroundColor": C["card"], "color": C["text"],
                        "border": f"1px solid {C['grid']}", "fontSize": "11px",
                        "padding": "4px 8px", "textAlign": "center",
                    },
                    style_header={
                        "backgroundColor": C["accent2"], "fontWeight": "bold",
                        "border": f"1px solid {C['grid']}",
                    },
                    style_data_conditional=[
                        {"if": {"state": "active"}, "backgroundColor": C["accent2"], "color": C["text"], "border": f"1px solid {C['grid']}"},
                        {"if": {"filter_query": '{dir} = "LONG"'}, "color": C["green"]},
                        {"if": {"filter_query": '{dir} = "SHORT"'}, "color": C["red"]},
                        {"if": {"filter_query": '{win} = "N"'}, "backgroundColor": "rgba(231,76,60,0.1)"},
                    ],
                    css=[
                        {"selector": "td.dash-cell.focused", "rule": f"background-color: {C['accent2']} !important; color: {C['text']} !important;"},
                        {"selector": "tr:hover td", "rule": f"background-color: {C['accent2']} !important; color: {C['text']} !important;"},
                        {"selector": "td.cell--selected", "rule": f"background-color: {C['accent2']} !important; color: {C['text']} !important;"},
                        {"selector": "td.focused", "rule": f"background-color: {C['accent2']} !important; color: {C['text']} !important; outline: none !important;"},
                    ],
                    page_size=50,
                ),

                # ── optimization ────────────────────────────────────────
                html.H4("Parameter Optimization", style={"marginTop": "30px", "marginBottom": "10px", "fontSize": "14px"}),
                html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "20px"}, children=[
                    html.Div(dcc.Graph(id="opt-heatmap", config={"displayModeBar": False}), style={"flex": "1"}),
                    html.Div(dcc.Graph(id="filt-heatmap", config={"displayModeBar": False}), style={"flex": "1"}),
                ]),
            ]),
        ),

        # ── top signals ─────────────────────────────────────────────────
        html.H4("Top Signals by Sharpe (z=1.0, z-exit, no filters)", style={
            "marginTop": "30px", "marginBottom": "10px", "fontSize": "14px"}),
        dash_table.DataTable(
            id="top-signals-table",
            columns=[{"name": c, "id": c} for c in
                     ["pc", "structure", "lookback", "sharpe", "quality", "r2_corr", "beta",
                      "beta_cv", "half_life", "ou_z", "resid"]],
            data=scan_df.head(30).round(4).to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={
                "backgroundColor": C["card"], "color": C["text"],
                "border": f"1px solid {C['grid']}", "fontSize": "11px",
                "padding": "4px 8px", "textAlign": "center",
            },
            style_header={
                "backgroundColor": C["accent2"], "fontWeight": "bold",
                "border": f"1px solid {C['grid']}",
            },
            sort_action="native",
            filter_action="native",
            page_size=30,
        ),
    ])

    # ── callbacks ───────────────────────────────────────────────────────
    @app.callback(
        [Output("diag-chart", "figure"),
         Output("bt-chart", "figure"),
         Output("metrics-row", "children"),
         Output("scan-stats", "children"),
         Output("trade-table", "data"),
         Output("trade-table", "columns"),
         Output("opt-heatmap", "figure"),
         Output("filt-heatmap", "figure"),
         Output("monthly-heatmap", "figure")],
        [Input("pc-select", "value"),
         Input("struct-select", "value"),
         Input("lb-select", "value"),
         Input("entry-z", "value"),
         Input("hl-exit-pct", "value"),
         Input("min-r2", "value"),
         Input("max-beta-cv", "value")],
    )
    def update(pc, struct, lb, entry_z, hl_exit_pct, min_r2, max_beta_cv):
        entry_z = float(entry_z or 1.0)
        hl_exit_pct = float(hl_exit_pct or 0.0)
        min_r2 = float(min_r2 or 0.0)
        max_beta_cv = float(max_beta_cv or 99.0)
        key = (pc, struct, int(lb))
        reg = all_regs.get(key)

        empty_fig = go.Figure().update_layout(**_layout("No data"))
        if reg is None:
            return (empty_fig, empty_fig, [], "No regression data for this combination.",
                    [], [], empty_fig, empty_fig, empty_fig)

        match = scan_df[(scan_df["pc"] == pc) & (scan_df["structure"] == struct) & (scan_df["lookback"] == lb)]

        diag_fig = make_diagnostics(reg, struct, pc, lb)

        bt = backtest_signal(
            reg, entry_z=entry_z, exit_z=0.0,
            hl_exit_pct=hl_exit_pct, lookback=int(lb),
            min_r2=min_r2, max_beta_cv=max_beta_cv,
        )
        bt_fig = make_backtest_chart(bt, reg, struct)

        # trade log
        all_trades = build_trade_log(bt)
        metrics = compute_metrics(bt, all_trades)

        def metrics_panel(title: str, cards_list, accent: str):
            return html.Div([
                html.Div(title, style={
                    "fontSize": "10px", "fontWeight": "700", "letterSpacing": "0.8px",
                    "textTransform": "uppercase", "color": accent, "marginBottom": "6px",
                }),
                html.Div(cards_list, style={"display": "flex", "gap": "6px", "flexWrap": "wrap"}),
            ], style={
                "backgroundColor": C["card"], "border": f"1px solid {C['grid']}",
                "borderRadius": "8px", "padding": "8px 10px",
            })

        # ── headline metrics (top row) ──
        headline_keys = [
            ("Sharpe", "neutral"), ("Ann PnL", "positive"), ("Win Rate", "positive"), ("# Trades", "activity"),
        ]
        # ── returns section ──
        returns_keys = [
            ("Ann PnL", "positive"), ("Ann Vol", "risk"), ("Max DD", "risk"), ("Calmar", "positive"),
        ]
        # ── trade quality section ──
        trade_keys = [
            ("Avg Win", "positive"), ("Avg Loss", "risk"), ("Avg PnL", "neutral"),
            ("Profit Factor", "positive"), ("Best", "positive"), ("Worst", "risk"),
        ]
        # ── activity section ──
        activity_keys = [
            ("# Trades", "activity"), ("% Invested", "activity"), ("Avg Dur", "activity"),
        ]

        def make_cards(key_tone_list):
            return [metric_card(k, metrics[k], tone=t) for k, t in key_tone_list if k in metrics]

        # Build headline row with larger Sharpe card
        headline_cards = []
        if "Sharpe" in metrics:
            headline_cards.append(html.Div([
                html.Div("SHARPE", style={
                    "fontSize": "9px", "color": C["muted"], "textTransform": "uppercase",
                    "letterSpacing": "0.5px", "marginBottom": "2px",
                }),
                html.Div(metrics["Sharpe"], style={
                    "fontSize": "22px", "fontWeight": "700", "color": C["text"],
                    "fontFamily": "monospace",
                }),
            ], style={
                "backgroundColor": C["card"], "padding": "10px 18px",
                "borderRadius": "6px", "minWidth": "90px", "textAlign": "center",
                "border": f"1px solid {C['accent']}",
            }))
        for k, t in headline_keys[1:]:
            if k in metrics:
                headline_cards.append(metric_card(k, metrics[k], tone=t))

        metric_cards = []
        # Headline row
        if headline_cards:
            metric_cards.append(html.Div(headline_cards, style={
                "display": "flex", "gap": "8px", "alignItems": "center",
                "marginBottom": "6px",
            }))
        # Detail row: three panels side by side
        detail_panels = []
        rc = make_cards(returns_keys)
        if rc:
            detail_panels.append(metrics_panel("Returns", rc, C["accent"]))
        tc = make_cards(trade_keys)
        if tc:
            detail_panels.append(metrics_panel("Trade Quality", tc, C["green"]))
        ac = make_cards(activity_keys)
        if ac:
            detail_panels.append(metrics_panel("Activity", ac, C["blue"]))
        if detail_panels:
            metric_cards.append(html.Div(detail_panels, style={
                "display": "flex", "gap": "8px", "flexWrap": "wrap",
            }))
        trades = all_trades.tail(50)

        # scan stats
        if not match.empty:
            row = match.iloc[0]
            curr_hl = bt["roll_hl"].dropna().iloc[-1] if "roll_hl" in bt.columns and not bt["roll_hl"].dropna().empty else np.nan
            exit_desc = "z-cross" if hl_exit_pct <= 0 else f"z-cross | {hl_exit_pct:.0%} x HL"
            filt = ""
            if min_r2 > 0:
                filt += f"  R2>={min_r2:.2f}"
            if max_beta_cv < 99.0:
                filt += f"  CV<={max_beta_cv:.1f}"
            stats_text = (
                f"R2={row['r2_corr']:.3f}  "
                f"Beta={row['beta']:.4f}  "
                f"CV={row['beta_cv']:.3f}  "
                f"HL={curr_hl:.0f}d  "
                f"z={row['ou_z']:+.2f}  "
                f"resid={row['resid']:+.2f}  "
                f"exit: {exit_desc}{filt}"
            )
        else:
            stats_text = "No scan data."

        trade_cols = [{"name": c, "id": c} for c in trades.columns] if not trades.empty else []
        trade_data = trades.to_dict("records") if not trades.empty else []

        # optimization grids
        opt_df = optimize_grid(reg, lookback=int(lb), min_r2=min_r2, max_beta_cv=max_beta_cv)
        opt_heatmap = make_opt_heatmap(opt_df, struct)

        filt_df = optimize_filters(reg, lookback=int(lb), entry_z=entry_z, hl_exit_pct=hl_exit_pct)
        filt_heatmap = make_filter_heatmap(filt_df, struct)

        # monthly heatmap
        monthly_fig = make_monthly_heatmap(bt)

        return (diag_fig, bt_fig, metric_cards, stats_text, trade_data, trade_cols,
                opt_heatmap, filt_heatmap, monthly_fig)

    return app


# ── main ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", choices=["ust", "sofr"], default="sofr")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--start", default="2021-01-01")
    args = parser.parse_args()

    print("Running PCA scanner...")
    from dig.pca_curve_signals import START as _s
    import dig.pca_curve_signals as scanner
    scanner.START = args.start
    result = scanner.main(args.curve, do_plot=False)

    if not result:
        print("Scanner returned no results.")
        return

    print(f"\nStarting Dash app on http://localhost:{args.port}")
    app = build_app(result)
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
