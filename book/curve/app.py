#!/usr/bin/env python3
"""
Beta-weighted 10s30s curve explorer.
Run:  mamba run -n 2s10s python book/curve/app.py
      Open http://localhost:8051
"""

from __future__ import annotations
import sys
from functools import lru_cache
from pathlib import Path

_here = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _here]

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from dash import Input, Output, ctx, dcc, html

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.helpers import query_db
from utils.research_app import (
    BG, PANEL, BORDER, TEXT, DIM, GRID, ORANGE, C0, C1, C2,
    styled_fig, stat_block, slider_with_input, graph, make_app, run,
)
from backtest import (
    BooleanSignalPipeline, BacktestConfig, Engine, SignalConfig, TradeDef,
    half_drift_residual,
)
from stats.ols import roll_lr_diff
from stats.ou import ou_params, roll_half_life, roll_ou_zscore

# ── config ────────────────────────────────────────────────────────────────────

TICKERS  = {"10y": "USGG10YR Index", "30y": "USGG30YR Index"}
START    = "2010-01-01"
FWD_BARS = 40
BETA_LB_VALS = [10, 15, 21, 30, 42, 63, 90, 126]
Z_LB_VALS = [21, 42, 63, 90, 126]
Z_VALS   = [1.0, 1.5, 2.0, 2.5, 3.0]
RESID_VALS = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30]
EXIT_TARGET_VALS = [0.25, 0.5, 0.75, 1.0]
HL_MULT_VALS = [0.5, 1.0, 1.5, 2.0, 3.0]
SIGNAL_MODES = [
    {"label": "OU z-score", "value": "z"},
    {"label": "Residual bps", "value": "resid"},
]
EXIT_POLICIES = [
    {"label": "Zero-cross", "value": "zero"},
    {"label": "Residual target", "value": "target"},
    {"label": "OU half-life stop", "value": "hl"},
    {"label": "Target or half-life", "value": "target_or_hl"},
    {"label": "Zero-cross or half-life", "value": "zero_or_hl"},
]

METRICS = [
    {"label": "Avg hit rate",   "value": "avg_hit"},
    {"label": "Long hit rate",  "value": "hit_long"},
    {"label": "Short hit rate", "value": "hit_short"},
    {"label": "Realized Sharpe", "value": "sharpe"},
    {"label": "Cum PnL",        "value": "pnl"},
]
METRIC_LABEL = {m["value"]: m["label"] for m in METRICS}

TRADE_COLUMNS = [
    {"name": "#", "id": "trade_id", "type": "numeric"},
    {"name": "Side", "id": "side"},
    {"name": "Entry", "id": "entry_date"},
    {"name": "Exit", "id": "exit_date"},
    {"name": "Entry sig", "id": "entry_signal", "type": "numeric", "format": {"specifier": "+.2f"}},
    {"name": "Exit sig", "id": "exit_signal", "type": "numeric", "format": {"specifier": "+.2f"}},
    {"name": "Entry lvl", "id": "entry_level", "type": "numeric", "format": {"specifier": "+.2f"}},
    {"name": "Exit lvl", "id": "exit_level", "type": "numeric", "format": {"specifier": "+.2f"}},
    {"name": "Entry resid", "id": "entry_resid", "type": "numeric", "format": {"specifier": "+.2f"}},
    {"name": "Exit resid", "id": "exit_resid", "type": "numeric", "format": {"specifier": "+.2f"}},
    {"name": "Beta", "id": "entry_beta", "type": "numeric", "format": {"specifier": ".3f"}},
    {"name": "OU HL", "id": "entry_ou_hl", "type": "numeric", "format": {"specifier": ".1f"}},
    {"name": "Exp resid", "id": "entry_expected_resid_pnl", "type": "numeric", "format": {"specifier": "+.2f"}},
    {"name": "Bars", "id": "bars_held", "type": "numeric"},
    {"name": "Exit reason", "id": "exit_reason"},
    {"name": "PnL", "id": "pnl", "type": "numeric", "format": {"specifier": "+.2f"}},
    {"name": "Cum PnL", "id": "cum_pnl", "type": "numeric", "format": {"specifier": "+.2f"}},
    {"name": "Run Sharpe", "id": "running_sharpe", "type": "numeric", "format": {"specifier": "+.2f"}},
]
_TRADE_LEFT = {"side", "entry_date", "exit_date", "exit_reason"}
_TRADE_SIGNED = {"entry_signal", "exit_signal", "entry_level", "exit_level",
                 "entry_resid", "exit_resid", "entry_expected_resid_pnl",
                 "pnl", "cum_pnl", "running_sharpe"}


def _render_stats_section(label: str, blocks: list) -> html.Div:
    """One labeled row of stat_block()s within the summary bar."""
    return html.Div([
        html.Span(label, style={
            "color": DIM, "fontSize": 9, "fontWeight": "bold",
            "textTransform": "uppercase", "letterSpacing": "0.05em",
            "display": "block", "marginBottom": 6,
        }),
        html.Div(blocks, style={"display": "flex", "gap": 40, "flexWrap": "wrap"}),
    ])


def _fmt_trade_cell(col_id: str, value) -> str:
    if value is None or pd.isna(value):
        return "-"
    if col_id in {"trade_id", "bars_held"}:
        return str(int(value))
    if col_id in _TRADE_SIGNED:
        return f"{float(value):+.2f}"
    if col_id == "entry_beta":
        return f"{float(value):.3f}"
    if col_id == "entry_ou_hl":
        return f"{float(value):.1f}"
    return str(value)


def _render_trade_table(trades: pd.DataFrame):
    if trades.empty:
        return html.Div("No closed trades for this parameter set.", style={
            "color": DIM, "fontSize": 11, "padding": "10px 12px",
        })

    header_style = {
        "position": "sticky", "top": 0, "zIndex": 1,
        "backgroundColor": PANEL, "color": TEXT,
        "fontWeight": "bold", "fontSize": 10,
        "padding": "6px 7px", "borderBottom": f"1px solid {BORDER}",
        "borderRight": f"1px solid {GRID}", "whiteSpace": "nowrap",
        "textAlign": "right",
    }
    cell_base = {
        "fontSize": 10, "padding": "5px 7px",
        "borderBottom": f"1px solid {GRID}",
        "borderRight": f"1px solid {GRID}",
        "whiteSpace": "nowrap", "textAlign": "right",
    }
    rows = []
    for _, row in trades.iterrows():
        side = row.get("side")
        bg = "rgba(39,174,96,0.04)" if side == "long" else "rgba(231,76,60,0.04)"
        cells = []
        for col in TRADE_COLUMNS:
            col_id = col["id"]
            style = dict(cell_base)
            if col_id in _TRADE_LEFT:
                style["textAlign"] = "left"
            if col_id == "pnl":
                pnl = row.get("pnl")
                if pd.notna(pnl):
                    style["color"] = C1 if float(pnl) > 0 else C0 if float(pnl) < 0 else TEXT
            cells.append(html.Td(_fmt_trade_cell(col_id, row.get(col_id)), style=style))
        rows.append(html.Tr(cells, style={"backgroundColor": bg}))

    return html.Table([
        html.Thead(html.Tr([
            html.Th(col["name"], style={
                **header_style,
                "textAlign": "left" if col["id"] in _TRADE_LEFT else "right",
            })
            for col in TRADE_COLUMNS
        ])),
        html.Tbody(rows),
    ], style={
        "borderCollapse": "collapse",
        "width": "100%",
        "minWidth": 1180,
        "fontFamily": "Arial, Helvetica, sans-serif",
    })


# ── data ──────────────────────────────────────────────────────────────────────

def _load() -> pl.DataFrame:
    inv = {v: k for k, v in TICKERS.items()}
    tickers_sql = "', '".join(TICKERS.values())
    pdf = query_db(f"""
        SELECT ts, ticker, px_last::float AS px
        FROM md.index_eod
        WHERE ticker IN ('{tickers_sql}') AND ts >= '{START}'
        ORDER BY ts
    """)
    wide = (
        pl.from_pandas(pdf)
        .with_columns(pl.col("ts").cast(pl.Date))
        .pivot(index="ts", on="ticker", values="px")
        .sort("ts")
    )
    wide = wide.rename({c: inv[c] for c in wide.columns if c in inv})
    for col in ("10y", "30y"):
        if col in wide.columns:
            wide = wide.with_columns((pl.col(col) * 100).alias(col))
    return wide


print("Loading data...", end="", flush=True)
DATA = _load()
print(f" {len(DATA)} rows  ({DATA['ts'].min()} → {DATA['ts'].max()})")


# ── compute ───────────────────────────────────────────────────────────────────

def _compute(beta_lb: int, zscore_lb: int) -> pd.DataFrame:
    reg       = roll_lr_diff(DATA["10y"], DATA["30y"], lookback=beta_lb)
    null1     = pl.Series("_", [None], dtype=pl.Float64)
    resid_roll = pl.concat([
        null1,
        reg["resid"].rolling_sum(beta_lb, min_samples=beta_lb),
    ]).alias("resid_roll")
    beta      = pl.concat([null1, reg["beta"]]).alias("beta")
    z         = roll_ou_zscore(resid_roll, lookback=zscore_lb).alias("zscore")
    hl        = roll_half_life(resid_roll, lookback=zscore_lb).alias("ou_half_life")
    ou_mean   = resid_roll.rolling_mean(zscore_lb, min_samples=max(20, zscore_lb // 4)).alias("ou_mean")
    df = DATA.with_columns([beta, resid_roll, z, hl,
                             ou_mean, (pl.col("30y") - pl.col("10y")).alias("naive")])
    df = df.with_columns(
        (pl.col("30y") - pl.col("beta") * pl.col("10y")).alias("beta_wtd")
    )
    pdf = df.to_pandas()
    pdf["ts"] = pd.to_datetime(pdf["ts"])
    return pdf.set_index("ts")


def _hit_rates(pdf: pd.DataFrame) -> pd.DataFrame:
    sub     = pdf[["zscore", "resid_roll"]].dropna()
    fwd_chg = sub["resid_roll"].shift(-FWD_BARS) - sub["resid_roll"]
    rows = []
    for t in Z_VALS:
        lm, sm   = sub["zscore"] < -t, sub["zscore"] > t
        n_l, n_s = int(lm.sum()), int(sm.sum())
        hl = float((fwd_chg[lm] > 0).mean()) if n_l > 0 else np.nan
        hs = float((fwd_chg[sm] < 0).mean()) if n_s > 0 else np.nan
        rows.append({"z": t, "n_long": n_l, "n_short": n_s,
                     "hit_long": hl, "hit_short": hs})
    return pd.DataFrame(rows)


def _trade_pnl(
    pdf: pd.DataFrame,
    entry_z: float,
    exit_policy: str = "zero",
    target_frac: float = 0.5,
    hl_mult: float = 1.0,
) -> float:
    """Final walk-forward PnL in naive 10s30s bps for a given entry threshold."""
    _, _, _, cum_pnl, _, _, equity_curve = _simulate(
        pdf, entry_z, exit_policy=exit_policy, target_frac=target_frac, hl_mult=hl_mult
    )
    if len(equity_curve) == 0:
        return np.nan
    return float(equity_curve.iloc[-1])


def _sharpe(
    pdf: pd.DataFrame,
    entry_z: float,
    exit_policy: str = "zero",
    target_frac: float = 0.5,
    hl_mult: float = 1.0,
) -> float:
    """Annualized daily Sharpe from actual naive-spread mark-to-market PnL."""
    _, _, _, _, _, daily_pnl, _ = _simulate(
        pdf, entry_z, exit_policy=exit_policy, target_frac=target_frac, hl_mult=hl_mult
    )
    return _daily_sharpe(daily_pnl)


def _daily_sharpe(daily_pnl: pd.Series) -> float:
    """Annualized Sharpe from daily mark-to-market PnL."""
    pnl = daily_pnl.dropna()
    if len(pnl) < 20 or pnl.std(ddof=1) == 0:
        return np.nan
    return float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252))


def _predictability_stats(pdf: pd.DataFrame, horizon: int = FWD_BARS) -> dict:
    """Forecast quality of the fade-residual signal vs forward naive 10s30s change.

    Distinct from realized backtest performance (Sharpe, PnL): these measure
    whether the signal has predictive content at all, independent of the
    current entry/exit rule.
    """
    empty = {"ic": np.nan, "r2": np.nan, "rank_ic": np.nan, "hit": np.nan}
    sub = pdf[["resid_roll", "naive"]].dropna()
    if len(sub) < horizon + 20:
        return empty
    fwd_naive = sub["naive"].shift(-horizon) - sub["naive"]
    pred = -sub["resid_roll"]
    aligned = pd.concat([pred.rename("pred"), fwd_naive.rename("fwd")], axis=1).dropna()
    aligned = aligned[aligned["pred"] != 0]
    if len(aligned) < 20 or aligned["pred"].std(ddof=1) == 0 or aligned["fwd"].std(ddof=1) == 0:
        return empty
    ic = float(aligned["pred"].corr(aligned["fwd"]))
    rank_ic = float(aligned["pred"].corr(aligned["fwd"], method="spearman"))
    hit = float((np.sign(aligned["pred"]) == np.sign(aligned["fwd"])).mean())
    return {"ic": ic, "r2": ic * ic, "rank_ic": rank_ic, "hit": hit}


def _uses_zero_exit(exit_policy: str) -> bool:
    return exit_policy in {"zero", "zero_or_hl"}


def _uses_hl_stop(exit_policy: str) -> bool:
    return exit_policy in {"hl", "target_or_hl", "zero_or_hl"}


def _uses_resid_target(exit_policy: str) -> bool:
    return exit_policy in {"target", "target_or_hl"}


def _build_z_signal_frame(
    pdf: pd.DataFrame,
    entry_z: float,
    exit_policy: str = "zero",
    hl_mult: float = 1.0,
) -> pd.DataFrame:
    """Strategy layer: convert OU z-score into boolean entry/exit arrays."""
    cols = ["zscore", "resid_roll", "naive", "beta", "ou_half_life", "ou_mean"]
    sub = pdf[cols].dropna(subset=["zscore", "resid_roll", "naive"])
    zero_exit = _uses_zero_exit(exit_policy)
    time_stop = (sub["ou_half_life"] * hl_mult).round().clip(lower=1) if _uses_hl_stop(exit_policy) else np.nan
    return pd.DataFrame({
        "signal_value": sub["zscore"],
        "level": sub["naive"],
        "resid": sub["resid_roll"],
        "beta": sub["beta"],
        "ou_half_life": sub["ou_half_life"],
        "ou_mean": sub["ou_mean"],
        "time_stop": time_stop,
        "enter_long": sub["zscore"] < -entry_z,
        "enter_short": sub["zscore"] > entry_z,
        "exit_long": (sub["zscore"] >= 0) if zero_exit else False,
        "exit_short": (sub["zscore"] <= 0) if zero_exit else False,
    }, index=sub.index)


def _build_resid_signal_frame(
    pdf: pd.DataFrame,
    entry_bps: float,
    exit_bps: float = 0.0,
    exit_policy: str = "zero",
    hl_mult: float = 1.0,
) -> pd.DataFrame:
    """Strategy layer: pure residual-bps entry/exit arrays."""
    cols = ["zscore", "resid_roll", "naive", "beta", "ou_half_life", "ou_mean"]
    sub = pdf[cols].dropna(subset=["resid_roll", "naive"])
    resid = sub["resid_roll"]
    zero_exit = _uses_zero_exit(exit_policy)
    time_stop = (sub["ou_half_life"] * hl_mult).round().clip(lower=1) if _uses_hl_stop(exit_policy) else np.nan
    return pd.DataFrame({
        "signal_value": resid,
        "level": sub["naive"],
        "resid": resid,
        "beta": sub["beta"],
        "ou_half_life": sub["ou_half_life"],
        "ou_mean": sub["ou_mean"],
        "time_stop": time_stop,
        "enter_long": resid < -entry_bps,
        "enter_short": resid > entry_bps,
        "exit_long": (resid >= -exit_bps) if zero_exit else False,
        "exit_short": (resid <= exit_bps) if zero_exit else False,
    }, index=sub.index)


def _run_signal_engine(
    signals: pd.DataFrame,
    max_hold_bars: int = FWD_BARS,
    exit_policy: str = "zero",
    target_frac: float = 0.5,
):
    """Adapter over the generic backtest Engine for boolean signal frames."""
    if signals.empty:
        empty = pd.Series(dtype=float)
        return [], [], [], empty, pd.DataFrame(), empty, empty

    engine_frame = signals.rename(columns={"signal_value": "signal"}).copy()
    engine_frame["ts"] = pd.to_datetime(engine_frame.index).date
    cols = ["ts", "level", "signal", "enter_long", "enter_short", "exit_long",
            "exit_short", "resid", "beta", "ou_half_life", "ou_mean", "time_stop"]
    data = pl.from_pandas(engine_frame[cols])
    exit_fn = half_drift_residual(target_frac) if _uses_resid_target(exit_policy) else None

    pipeline = BooleanSignalPipeline(
        name="curve_signal",
        trade_def=TradeDef.outright("naive_10s30s", "level"),
        compute_fn=lambda df: df.select([
            "signal", "enter_long", "enter_short", "exit_long", "exit_short",
            "resid", "beta", "ou_half_life", "ou_mean", "time_stop",
        ]),
        config=SignalConfig(time_stop_bars=max_hold_bars, exit_fn=exit_fn),
    )
    result = Engine(BacktestConfig(max_total_positions=1)).add_signal(pipeline).run(data)

    closed = result.closed_trades
    open_trades = result.open_trades
    long_entries = [pd.Timestamp(t.entry_date) for t in closed + open_trades if t.direction == 1]
    short_entries = [pd.Timestamp(t.entry_date) for t in closed + open_trades if t.direction == -1]
    exit_dates = [pd.Timestamp(t.exit_date) for t in closed]

    pnls = [float(t.pnl_bps) for t in closed]
    cum_pnl = pd.Series(dtype=float)
    if pnls:
        cum_pnl = pd.Series(np.cumsum(pnls), index=pd.DatetimeIndex(exit_dates))

    rows = []
    for idx, trade in enumerate(closed, start=1):
        cum = float(np.sum(pnls[:idx]))
        held = [t.bars_held for t in closed[:idx]]
        avg_bars = np.mean(held) if held else np.nan
        running_sharpe = np.nan
        if idx > 1 and np.std(pnls[:idx], ddof=1) > 0 and avg_bars > 0:
            running_sharpe = float(np.mean(pnls[:idx]) / np.std(pnls[:idx], ddof=1) * np.sqrt(252 / avg_bars))
        entry_resid = trade.entry_extras.get("resid")
        entry_mu = trade.entry_extras.get("ou_mean", 0.0)
        expected_resid_pnl = np.nan
        if entry_resid is not None and pd.notna(entry_resid):
            target_resid = entry_mu + (entry_resid - entry_mu) * (1.0 - target_frac)
            expected_resid_pnl = abs(float(entry_resid) - float(target_resid))
        rows.append({
            "trade_id": idx,
            "side": "long" if trade.direction == 1 else "short",
            "entry_date": pd.Timestamp(trade.entry_date).strftime("%Y-%m-%d"),
            "exit_date": pd.Timestamp(trade.exit_date).strftime("%Y-%m-%d"),
            "entry_signal": round(float(trade.entry_signal), 3),
            "exit_signal": round(float(trade.exit_signal), 3),
            "entry_level": round(float(trade.entry_level), 3),
            "exit_level": round(float(trade.exit_level), 3),
            "entry_resid": _round_or_none(trade.entry_extras.get("resid"), 3),
            "exit_resid": _round_or_none(trade.exit_extras.get("resid"), 3),
            "entry_beta": _round_or_none(trade.entry_extras.get("beta"), 4),
            "entry_ou_hl": _round_or_none(trade.entry_extras.get("ou_half_life"), 2),
            "entry_expected_resid_pnl": _round_or_none(expected_resid_pnl, 3),
            "bars_held": int(trade.bars_held),
            "exit_reason": trade.exit_reason,
            "pnl": round(float(trade.pnl_bps), 3),
            "cum_pnl": round(cum, 3),
            "running_sharpe": _round_or_none(running_sharpe, 3),
        })

    daily_pdf = result.daily_pnl.to_pandas()
    equity_pdf = result.equity_curve.to_pandas()
    if daily_pdf.empty:
        daily_pnl = pd.Series(dtype=float)
    else:
        daily_pnl = pd.Series(daily_pdf["pnl_bps"].to_numpy(), index=pd.to_datetime(daily_pdf["ts"]))
    if equity_pdf.empty:
        equity_curve = pd.Series(dtype=float)
    else:
        equity_curve = pd.Series(equity_pdf["cumulative_pnl"].to_numpy(), index=pd.to_datetime(equity_pdf["ts"]))

    return long_entries, short_entries, exit_dates, cum_pnl, pd.DataFrame(rows), daily_pnl, equity_curve


def _simulate(
    pdf: pd.DataFrame,
    entry_z: float,
    exit_policy: str = "zero",
    target_frac: float = 0.5,
    hl_mult: float = 1.0,
):
    """Compatibility wrapper for the current OU-z strategy."""
    signals = _build_z_signal_frame(pdf, entry_z, exit_policy=exit_policy, hl_mult=hl_mult)
    return _run_signal_engine(signals, exit_policy=exit_policy, target_frac=target_frac)


def _simulate_resid(
    pdf: pd.DataFrame,
    entry_bps: float,
    exit_policy: str = "zero",
    target_frac: float = 0.5,
    hl_mult: float = 1.0,
):
    """Pure residual-bps strategy wrapper."""
    signals = _build_resid_signal_frame(
        pdf, entry_bps, exit_policy=exit_policy, hl_mult=hl_mult
    )
    return _run_signal_engine(signals, exit_policy=exit_policy, target_frac=target_frac)


def _round_or_none(value, ndigits: int):
    try:
        if pd.isna(value):
            return None
        return round(float(value), ndigits)
    except Exception:
        return None

def _marker_xy(pdf: pd.DataFrame, dates, col: str = "10y"):
    """Align marker dates to pdf column, drop nulls."""
    if not dates:
        return [], []
    vals = pdf[col].reindex(pd.DatetimeIndex(dates)).dropna()
    return list(vals.index), list(vals.values)


@lru_cache(maxsize=64)
def _grid(
    metric: str,
    entry_z: float,
    exit_policy: str = "zero",
    target_frac: float = 0.5,
    hl_mult: float = 1.0,
) -> tuple:
    """β-lb (y) × z-lb (x) at fixed entry_z. Cached."""
    grid = np.full((len(BETA_LB_VALS), len(Z_LB_VALS)), np.nan)
    for i, beta_lb in enumerate(BETA_LB_VALS):
        for j, zscore_lb in enumerate(Z_LB_VALS):
            try:
                pdf  = _compute(beta_lb, zscore_lb)
                scan = _hit_rates(pdf)
                crow = scan.iloc[(scan["z"] - entry_z).abs().argsort()].iloc[0]
                if metric == "hit_long":
                    grid[i, j] = crow["hit_long"]
                elif metric == "hit_short":
                    grid[i, j] = crow["hit_short"]
                elif metric == "avg_hit":
                    grid[i, j] = (crow["hit_long"] + crow["hit_short"]) / 2
                elif metric == "sharpe":
                    grid[i, j] = _sharpe(pdf, entry_z, exit_policy, target_frac, hl_mult)
                elif metric == "pnl":
                    grid[i, j] = _trade_pnl(pdf, entry_z, exit_policy, target_frac, hl_mult)
            except Exception:
                pass
    return tuple(tuple(float(v) if not np.isnan(v) else None for v in row)
                 for row in grid)


@lru_cache(maxsize=64)
def _grid2(
    metric: str,
    zscore_lb_snap: int,
    exit_policy: str = "zero",
    target_frac: float = 0.5,
    hl_mult: float = 1.0,
) -> tuple:
    """β-lb (y) × entry threshold (x) at fixed zscore_lb. Cached."""
    grid = np.full((len(BETA_LB_VALS), len(Z_VALS)), np.nan)
    for i, beta_lb in enumerate(BETA_LB_VALS):
        for j, ez in enumerate(Z_VALS):
            try:
                pdf  = _compute(beta_lb, zscore_lb_snap)
                scan = _hit_rates(pdf)
                crow = scan.iloc[(scan["z"] - ez).abs().argsort()].iloc[0]
                if metric == "hit_long":
                    grid[i, j] = crow["hit_long"]
                elif metric == "hit_short":
                    grid[i, j] = crow["hit_short"]
                elif metric == "avg_hit":
                    grid[i, j] = (crow["hit_long"] + crow["hit_short"]) / 2
                elif metric == "sharpe":
                    grid[i, j] = _sharpe(pdf, ez, exit_policy, target_frac, hl_mult)
                elif metric == "pnl":
                    grid[i, j] = _trade_pnl(pdf, ez, exit_policy, target_frac, hl_mult)
            except Exception:
                pass
    return tuple(tuple(float(v) if not np.isnan(v) else None for v in row)
                 for row in grid)


# ── value parsers ─────────────────────────────────────────────────────────────


def _resid_metric(
    pdf: pd.DataFrame,
    metric: str,
    entry_bps: float,
    exit_policy: str = "zero",
    target_frac: float = 0.5,
    hl_mult: float = 1.0,
) -> float:
    """Heatmap metric for the residual-bps strategy."""
    _, _, _, _, trades, daily_pnl, equity_curve = _simulate_resid(
        pdf, entry_bps, exit_policy=exit_policy, target_frac=target_frac, hl_mult=hl_mult
    )
    if metric == "sharpe":
        return _daily_sharpe(daily_pnl)
    if metric == "pnl":
        return float(equity_curve.iloc[-1]) if len(equity_curve) else np.nan
    if trades.empty:
        return np.nan
    if metric == "hit_long":
        sub = trades[trades["side"] == "long"]
        return float((sub["pnl"] > 0).mean()) if len(sub) else np.nan
    if metric == "hit_short":
        sub = trades[trades["side"] == "short"]
        return float((sub["pnl"] > 0).mean()) if len(sub) else np.nan
    if metric == "avg_hit":
        vals = []
        for side in ("long", "short"):
            sub = trades[trades["side"] == side]
            if len(sub):
                vals.append(float((sub["pnl"] > 0).mean()))
        return float(np.mean(vals)) if vals else np.nan
    return np.nan


@lru_cache(maxsize=64)
def _grid_resid_zlb(
    metric: str,
    entry_bps: float,
    exit_policy: str,
    target_frac: float,
    hl_mult: float,
) -> tuple:
    """Beta-lb (y) x OU z-lb (x) for residual strategy. z-lb is diagnostic only."""
    grid = np.full((len(BETA_LB_VALS), len(Z_LB_VALS)), np.nan)
    for i, beta_lb in enumerate(BETA_LB_VALS):
        for j, zscore_lb in enumerate(Z_LB_VALS):
            try:
                pdf = _compute(beta_lb, zscore_lb)
                grid[i, j] = _resid_metric(pdf, metric, entry_bps, exit_policy, target_frac, hl_mult)
            except Exception:
                pass
    return tuple(tuple(float(v) if not np.isnan(v) else None for v in row)
                 for row in grid)


@lru_cache(maxsize=64)
def _grid_resid_entry(
    metric: str,
    zscore_lb_snap: int,
    exit_policy: str,
    target_frac: float,
    hl_mult: float,
) -> tuple:
    """Beta-lb (y) x residual entry threshold (x) at fixed OU z-lb."""
    grid = np.full((len(BETA_LB_VALS), len(RESID_VALS)), np.nan)
    for i, beta_lb in enumerate(BETA_LB_VALS):
        for j, entry_bps in enumerate(RESID_VALS):
            try:
                pdf = _compute(beta_lb, zscore_lb_snap)
                grid[i, j] = _resid_metric(pdf, metric, entry_bps, exit_policy, target_frac, hl_mult)
            except Exception:
                pass
    return tuple(tuple(float(v) if not np.isnan(v) else None for v in row)
                 for row in grid)
def _parse_lb(typed, slider, default=63) -> int:
    try:
        return max(5, int(float(typed or slider or default)))
    except Exception:
        return int(slider or default)

def _parse_z(typed, slider, default=2.0) -> float:
    try:
        return max(0.1, round(float(typed or slider or default), 2))
    except Exception:
        return float(slider or default)

def _crow(scan, entry_z):
    return scan.iloc[(scan["z"] - entry_z).abs().argsort()].iloc[0]


# ── heatmap figure helper ─────────────────────────────────────────────────────

def _heatmap_fig(
    title: str,
    grid: tuple,
    x_labels,
    y_labels,
    x_title: str,
    y_title: str,
    metric: str,
    cur_x=None,
    cur_y=None,
) -> go.Figure:
    z = [list(row) for row in grid]
    values = [v for row in z for v in row if v is not None]

    if metric in {"avg_hit", "hit_long", "hit_short"}:
        fmt = lambda v: f"{v:.0%}"
        zmin, zmax, tickformat = 0.42, 0.82, ".0%"
    elif metric == "pnl":
        fmt = lambda v: f"{v:+.1f}"
        lim = max([abs(v) for v in values], default=1.0)
        zmin, zmax, tickformat = -max(lim, 1.0), max(lim, 1.0), ".1f"
    else:
        fmt = lambda v: f"{v:.2f}"
        zmin, zmax, tickformat = -0.3, 1.8, ".2f"

    text = [[fmt(v) if v is not None else "—" for v in row] for row in z]

    fig = styled_fig(title, height=360)
    fig.add_trace(go.Heatmap(
        x=[str(v) for v in x_labels],
        y=[str(v) for v in y_labels],
        z=z,
        colorscale="RdYlGn",
        zmin=zmin,
        zmax=zmax,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=10, color="black"),
        hovertemplate=f"{y_title}=%{{y}}  {x_title}=%{{x}}<br>%{{text}}<extra></extra>",
        showscale=True,
        colorbar=dict(thickness=10, len=0.85, tickfont=dict(size=8),
                      tickformat=tickformat),
    ))
    # current-selection ring
    cx = str(cur_x) if cur_x is not None and str(cur_x) in [str(v) for v in x_labels] else None
    cy = str(cur_y) if cur_y is not None and str(cur_y) in [str(v) for v in y_labels] else None
    if cx and cy:
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers",
            marker=dict(symbol="circle-open", size=22, color="white",
                        line=dict(color="white", width=3)),
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(
        xaxis=dict(title=x_title, showgrid=False, tickfont=dict(size=9), linecolor=BORDER),
        yaxis=dict(title=y_title, showgrid=False, tickfont=dict(size=9), linecolor=BORDER),
        margin=dict(l=66, r=74, t=44, b=54),
    )
    return fig


# ── layout ────────────────────────────────────────────────────────────────────

_ROW = {"padding": "16px 20px 0",
        "display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": 16}

app = make_app(
    title    = "10s30s",
    subtitle = "beta-weighted curve explorer",
    data_info= f"{len(DATA):,} rows · {DATA['ts'].min()} → {DATA['ts'].max()}",
    sliders  = [
        # row 1: entry side — signal construction + entry threshold
        [
            html.Div([
                html.Span("signal mode", style={
                    "color": DIM, "fontSize": 10, "fontWeight": "bold",
                    "letterSpacing": "0.05em", "textTransform": "uppercase",
                    "display": "block", "marginBottom": 6,
                }),
                dcc.Dropdown(
                    id="signal-mode",
                    options=SIGNAL_MODES,
                    value="resid",
                    clearable=False,
                    searchable=False,
                    style={"fontSize": 12, "fontFamily": "Arial, Helvetica, sans-serif"},
                ),
            ]),
            slider_with_input("beta-lb",   BETA_LB_VALS, 63, "beta lookback (days)"),
            slider_with_input("zscore-lb", Z_LB_VALS,    63, "z-score lookback (days)"),
            # entry threshold: both sliders share this one grid cell; only the
            # slider matching the active signal mode is shown (see callback below)
            html.Div([
                html.Div(
                    id="entry-z-wrap",
                    children=[slider_with_input("entry-z", Z_VALS, 2.0, "entry threshold  ±z")],
                    style={"display": "none"},
                ),
                html.Div(
                    id="entry-resid-wrap",
                    children=[slider_with_input("entry-resid", RESID_VALS, 10, "entry threshold  ±resid bps")],
                    style={"display": "block"},
                ),
            ]),
        ],
        # row 2: exit params
        [
            html.Div([
                html.Span("exit policy", style={
                    "color": DIM, "fontSize": 10, "fontWeight": "bold",
                    "letterSpacing": "0.05em", "textTransform": "uppercase",
                    "display": "block", "marginBottom": 6,
                }),
                dcc.Dropdown(
                    id="exit-policy",
                    options=EXIT_POLICIES,
                    value="target_or_hl",
                    clearable=False,
                    searchable=False,
                    style={"fontSize": 12, "fontFamily": "Arial, Helvetica, sans-serif"},
                ),
            ]),
            slider_with_input("exit-target-frac", EXIT_TARGET_VALS, 0.5, "target fraction"),
            slider_with_input("exit-hl-mult",     HL_MULT_VALS,     1.0, "OU half-life multiple"),
        ],
    ],
    body=html.Div([
        # stats bar
        html.Div(id="stats-row", style={
            "padding": "10px 28px 12px", "background": PANEL,
            "borderBottom": f"1px solid {BORDER}",
            "display": "flex", "flexDirection": "column", "gap": 10,
        }),
        # row 1: naive vs beta-weighted spread + signal spread
        html.Div(style=_ROW, children=[graph("chart-yields"), graph("chart-spread")]),
        # row 2: resid + zscore
        html.Div(style=_ROW, children=[graph("chart-resid"),  graph("chart-zscore")]),
        # row 3: pnl + hit rates
        html.Div(style=_ROW, children=[graph("chart-pnl"),    graph("chart-hits")]),
        # heatmap header
        html.Div(style={
            "padding": "20px 20px 6px",
            "display": "flex", "alignItems": "center", "gap": 12,
        }, children=[
            html.Span("heatmap metric", style={
                "color": DIM, "fontSize": 10, "fontWeight": "bold",
                "textTransform": "uppercase", "letterSpacing": "0.05em",
            }),
            dcc.Dropdown(
                id="metric-select",
                options=METRICS, value="avg_hit",
                clearable=False, searchable=False,
                style={"width": 180, "fontSize": 12,
                       "fontFamily": "Arial, Helvetica, sans-serif"},
            ),
        ]),
        # row 4: heatmaps
        html.Div(style=_ROW, children=[graph("chart-heatmap1"), graph("chart-heatmap2")]),
        html.Div(style={"padding": "20px 20px 0"}, children=[
            html.Div("trade log", style={
                "color": DIM, "fontSize": 10, "fontWeight": "bold",
                "textTransform": "uppercase", "letterSpacing": "0.05em",
                "marginBottom": 6,
            }),
            html.Div(
                id="trade-log",
                children=_render_trade_table(pd.DataFrame()),
                style={
                    "overflowX": "auto",
                    "overflowY": "auto",
                    "border": f"1px solid {BORDER}",
                    "maxHeight": "318px",
                },
            ),
        ]),
    ]),
    debug=True,
)


# ── signal-mode visibility ──────────────────────────────────────────────────

@app.callback(
    Output("entry-z-wrap",     "style"),
    Output("entry-resid-wrap", "style"),
    Input("signal-mode", "value"),
)
def _toggle_entry_threshold(signal_mode):
    if signal_mode == "resid":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


# ── sync callbacks (slider ↔ typed input) ─────────────────────────────────────

def _register_sync(id_: str, vals: list):
    @app.callback(
        Output(id_, "value"),
        Output(f"{id_}-typed", "value"),
        Input(id_, "value"),
        Input(f"{id_}-typed", "value"),
        prevent_initial_call=True,
    )
    def _sync(slider_val, typed_val):
        if ctx.triggered_id == f"{id_}-typed":
            try:
                v    = float(typed_val)
                snap = min(vals, key=lambda x: abs(x - v))
                return snap, typed_val
            except Exception:
                return slider_val, str(slider_val)
        sv = slider_val if slider_val is not None else vals[0]
        return sv, str(int(sv) if sv == int(sv) else sv)


_register_sync("beta-lb",   BETA_LB_VALS)
_register_sync("zscore-lb", Z_LB_VALS)
_register_sync("entry-z",   Z_VALS)
_register_sync("entry-resid", RESID_VALS)
_register_sync("exit-target-frac", EXIT_TARGET_VALS)
_register_sync("exit-hl-mult", HL_MULT_VALS)


# ── main callback: time-series charts + stats ─────────────────────────────────

@app.callback(
    Output("chart-yields", "figure"),
    Output("chart-spread", "figure"),
    Output("chart-resid",  "figure"),
    Output("chart-zscore", "figure"),
    Output("chart-pnl",    "figure"),
    Output("chart-hits",   "figure"),
    Output("stats-row",    "children"),
    Output("trade-log",    "children"),
    Input("beta-lb-typed",   "value"),
    Input("zscore-lb-typed", "value"),
    Input("entry-z-typed",   "value"),
    Input("entry-resid-typed", "value"),
    Input("signal-mode", "value"),
    Input("exit-policy", "value"),
    Input("exit-target-frac-typed", "value"),
    Input("exit-hl-mult-typed", "value"),
    Input("beta-lb",   "value"),
    Input("zscore-lb", "value"),
    Input("entry-z",   "value"),
    Input("entry-resid", "value"),
    Input("exit-target-frac", "value"),
    Input("exit-hl-mult", "value"),
)
def update_charts(
    beta_t, zscore_t, ez_t, er_t, signal_mode, exit_policy, target_t, hl_t,
    beta_s, zscore_s, ez_s, er_s, target_s, hl_s,
):
    beta_lb   = _parse_lb(beta_t,   beta_s,   63)
    zscore_lb = _parse_lb(zscore_t, zscore_s, 63)
    entry_z   = _parse_z(ez_t,     ez_s,     2.0)
    entry_resid = _parse_z(er_t, er_s, 10.0)
    target_frac = min(1.0, max(0.01, _parse_z(target_t, target_s, 0.5)))
    hl_mult = max(0.1, _parse_z(hl_t, hl_s, 1.0))
    exit_policy = exit_policy or "target_or_hl"

    pdf  = _compute(beta_lb, zscore_lb)
    scan = _hit_rates(pdf)
    crow = _crow(scan, entry_z)

    if signal_mode == "resid":
        long_entries, short_entries, exit_dates, cum_pnl, trades, daily_pnl, equity_curve = _simulate_resid(
            pdf, entry_resid, exit_policy=exit_policy, target_frac=target_frac, hl_mult=hl_mult
        )
        signal_label = f"resid +/-{entry_resid:g}bp"
    else:
        long_entries, short_entries, exit_dates, cum_pnl, trades, daily_pnl, equity_curve = _simulate(
            pdf, entry_z, exit_policy=exit_policy, target_frac=target_frac, hl_mult=hl_mult
        )
        signal_label = f"OU z +/-{entry_z:g}"
    exit_label = f"{exit_policy.replace('_', ' ')} | target={target_frac:.0%} | HLx={hl_mult:g}"
    le_x, le_y = _marker_xy(pdf, long_entries, "naive")
    se_x, se_y = _marker_xy(pdf, short_entries, "naive")
    ex_x, ex_y = _marker_xy(pdf, exit_dates,   "naive")

    # ── chart 1: naive vs beta-weighted 10s30s ───────────────────────────────
    fig1 = styled_fig(f"Naive vs beta-weighted 10s30s  (beta lookback={beta_lb}d)",
                      "bps", height=320)
    fig1.add_trace(go.Scatter(
        x=pdf.index, y=pdf["naive"], name="naive 10s30s",
        line=dict(color=ORANGE, width=1.2),
    ))
    fig1.add_trace(go.Scatter(
        x=pdf.index, y=pdf["beta_wtd"], name="beta-weighted 10s30s",
        line=dict(color=C2, width=1.2),
    ))

    # ── chart 2: naive 10s30s spread + signal markers ────────────────────────
    fig2 = styled_fig("10s30s naive spread  (30Y − 10Y)", "bps", height=320)
    fig2.add_trace(go.Scatter(
        x=pdf.index, y=pdf["naive"], name="spread",
        line=dict(color=C2, width=1.3),
    ))
    if le_x:
        fig2.add_trace(go.Scatter(
            x=le_x, y=le_y, mode="markers", name="long entry",
            marker=dict(symbol="triangle-up", size=9, color=C1,
                        line=dict(color="white", width=0.5)),
        ))
    if se_x:
        fig2.add_trace(go.Scatter(
            x=se_x, y=se_y, mode="markers", name="short entry",
            marker=dict(symbol="triangle-down", size=9, color=C0,
                        line=dict(color="white", width=0.5)),
        ))
    if ex_x:
        fig2.add_trace(go.Scatter(
            x=ex_x, y=ex_y, mode="markers", name="exit",
            marker=dict(symbol="x", size=6, color=DIM,
                        line=dict(color=DIM, width=1.5)),
        ))

    # chart 3: rolling residual
    fig3 = styled_fig(f"Rolling residual  (beta lookback={beta_lb}d)", "bps", height=280)
    fig3.add_hline(y=0, line=dict(color=BORDER, dash="dot", width=0.8))
    if signal_mode == "resid":
        fig3.add_hline(y= entry_resid, line=dict(color=C0, dash="dash", width=1.1))
        fig3.add_hline(y=-entry_resid, line=dict(color=C1, dash="dash", width=1.1),
                       annotation_text=f"±{entry_resid:g}bp", annotation_position="right")
    fig3.add_trace(go.Scatter(
        x=pdf.index, y=pdf["resid_roll"], name="rolling residual", showlegend=False,
        line=dict(color=C2, width=1.2),
    ))

    # ── chart 4: OU z-score ───────────────────────────────────────────────────
    z_clean = pdf["zscore"].dropna()
    fig4 = styled_fig(f"OU z-score — z-lb={zscore_lb}d  active signal={signal_label}", "z", height=280)
    if signal_mode == "z" and len(z_clean) > 0:
        pad = 0.5
        fig4.add_hrect(y0= entry_z, y1=z_clean.max() + pad,
                       fillcolor=C0, opacity=0.06, line_width=0)
        fig4.add_hrect(y0=z_clean.min() - pad, y1=-entry_z,
                       fillcolor=C1, opacity=0.06, line_width=0)
        fig4.add_hline(y= entry_z, line=dict(color=C0, dash="dash", width=1.2))
        fig4.add_hline(y=-entry_z, line=dict(color=C1, dash="dash", width=1.2),
                       annotation_text=f"±{entry_z}", annotation_position="right")
    fig4.add_hline(y=0, line=dict(color=BORDER, dash="dot", width=0.8))
    fig4.add_trace(go.Scatter(
        x=pdf.index, y=pdf["zscore"], showlegend=False,
        line=dict(color=C2, width=1.2),
    ))

    # ── chart 5: cumulative PnL ───────────────────────────────────────────────
    n_trades = len(long_entries) + len(short_entries)
    fig5 = styled_fig(f"Cumulative PnL  ({n_trades} trades, {signal_label}, {exit_label}, daily MTM)", "bps", height=240)
    fig5.add_hline(y=0, line=dict(color=BORDER, width=1))
    if len(equity_curve) > 0:
        final = float(equity_curve.iloc[-1])
        color = C1 if final >= 0 else C0
        fig5.add_trace(go.Scatter(
            x=equity_curve.index, y=equity_curve.values,
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba(39,174,96,0.10)" if final >= 0 else "rgba(231,76,60,0.10)",
            showlegend=False,
        ))

    # ── chart 6: hit rates ────────────────────────────────────────────────────
    labels    = [f"±{v}" for v in Z_VALS]
    highlight = [abs(v - entry_z) < 0.01 for v in Z_VALS]
    fig6 = styled_fig(f"Forward {FWD_BARS}-day hit rate by entry threshold",
                      "hit rate", height=280)
    fig6.add_trace(go.Bar(
        x=labels, y=scan["hit_long"], name="long",
        marker_color=[C1 if not h else "#1A8A4A" for h in highlight],
        text=[f"n={int(r.n_long)}" for _, r in scan.iterrows()],
        textposition="outside", textfont=dict(size=9, color=DIM),
    ))
    fig6.add_trace(go.Bar(
        x=labels, y=scan["hit_short"], name="short",
        marker_color=[C0 if not h else "#C0392B" for h in highlight],
        text=[f"n={int(r.n_short)}" for _, r in scan.iterrows()],
        textposition="outside", textfont=dict(size=9, color=DIM),
    ))
    fig6.add_hline(y=0.5, line=dict(color=DIM, dash="dash", width=1))
    fig6.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.15], tickformat=".0%",
                   showgrid=True, gridcolor=GRID, linecolor=BORDER,
                   tickfont=dict(size=9)),
    )

    # ── stats ─────────────────────────────────────────────────────────────────
    sub     = pdf[["naive", "10y", "resid_roll"]].dropna()
    c_naive = float(sub["naive"].corr(sub["10y"]))
    c_bw    = float(sub["resid_roll"].corr(sub["10y"]))
    p       = ou_params(pdf["resid_roll"].dropna())
    hl      = p["half_life"]
    hl_val  = crow["hit_long"]
    hs_val  = crow["hit_short"]
    sh      = _daily_sharpe(daily_pnl)
    final   = float(equity_curve.iloc[-1]) if len(equity_curve) > 0 else np.nan
    pred    = _predictability_stats(pdf)

    predictability_stats = [
        stat_block("direction ρ naive",       f"{c_naive:+.3f}"),
        stat_block("direction ρ β-weighted",  f"{c_bw:+.3f}"),
        stat_block("OU half-life",            f"{hl:.0f}d" if not np.isnan(hl) else "—"),
        stat_block("OU σ / day",              f"{p['sigma']:.2f} bps"),
        stat_block(f"resid→naive {FWD_BARS}d IC",
                   f"{pred['ic']:+.3f}" if not np.isnan(pred["ic"]) else "—",
                   alert=not np.isnan(pred["ic"]) and pred["ic"] > 0.10),
        stat_block(f"resid→naive {FWD_BARS}d R2",
                   f"{pred['r2']:.1%}" if not np.isnan(pred["r2"]) else "—"),
        stat_block(f"resid→naive {FWD_BARS}d rank IC",
                   f"{pred['rank_ic']:+.3f}" if not np.isnan(pred["rank_ic"]) else "—",
                   alert=not np.isnan(pred["rank_ic"]) and pred["rank_ic"] > 0.10),
        stat_block(f"resid→naive {FWD_BARS}d sign hit",
                   f"{pred['hit']:.1%}" if not np.isnan(pred["hit"]) else "—",
                   alert=not np.isnan(pred["hit"]) and pred["hit"] > 0.55),
    ]
    performance_stats = [
        stat_block(f"hit long  z={entry_z}",
                   f"{hl_val:.1%}" if not np.isnan(hl_val) else "—",
                   alert=not np.isnan(hl_val) and hl_val > 0.6),
        stat_block(f"hit short  z={entry_z}",
                   f"{hs_val:.1%}" if not np.isnan(hs_val) else "—",
                   alert=not np.isnan(hs_val) and hs_val < 0.45),
        stat_block("realized sharpe",
                   f"{sh:.2f}" if not np.isnan(sh) else "—",
                   alert=not np.isnan(sh) and sh > 0.8),
        stat_block("cum PnL",
                   f"{final:+.1f} bps" if not np.isnan(final) else "—",
                   alert=not np.isnan(final) and final > 0),
        stat_block("signal mode", signal_label),
        stat_block("exit policy", exit_label),
        stat_block("n long entries",  str(len(long_entries))),
        stat_block("n short entries", str(len(short_entries))),
    ]

    stats = [
        _render_stats_section("signal predictability", predictability_stats),
        _render_stats_section("strategy performance", performance_stats),
    ]

    if not trades.empty:
        trades = trades.sort_values(["exit_date", "trade_id"], ascending=[False, False])

    return fig1, fig2, fig3, fig4, fig5, fig6, stats, _render_trade_table(trades)


# ── heatmap callback ──────────────────────────────────────────────────────────

@app.callback(
    Output("chart-heatmap1", "figure"),
    Output("chart-heatmap2", "figure"),
    Input("metric-select",   "value"),
    Input("entry-z-typed",   "value"),
    Input("entry-z",         "value"),
    Input("entry-resid-typed", "value"),
    Input("entry-resid",       "value"),
    Input("signal-mode",       "value"),
    Input("exit-policy",       "value"),
    Input("exit-target-frac-typed", "value"),
    Input("exit-target-frac",       "value"),
    Input("exit-hl-mult-typed", "value"),
    Input("exit-hl-mult",       "value"),
    Input("beta-lb-typed",   "value"),
    Input("beta-lb",         "value"),
    Input("zscore-lb-typed", "value"),
    Input("zscore-lb",       "value"),
)
def update_heatmaps(
    metric, ez_t, ez_s, er_t, er_s, signal_mode,
    exit_policy, target_t, target_s, hl_t, hl_s,
    beta_t, beta_s, zscore_t, zscore_s,
):
    metric    = metric or "avg_hit"
    entry_z   = _parse_z(ez_t,    ez_s,    2.0)
    entry_resid = _parse_z(er_t, er_s, 10.0)
    target_frac = min(1.0, max(0.01, _parse_z(target_t, target_s, 0.5)))
    hl_mult = max(0.1, _parse_z(hl_t, hl_s, 1.0))
    exit_policy = exit_policy or "target_or_hl"
    beta_lb   = _parse_lb(beta_t,  beta_s,  63)
    zscore_lb = _parse_lb(zscore_t, zscore_s, 63)

    zs_snap = min(Z_LB_VALS, key=lambda x: abs(x - zscore_lb))
    if signal_mode == "resid":
        exit_short = f"{exit_policy.replace('_', ' ')} target={target_frac:.0%} HLx={hl_mult:g}"
        g1 = _grid_resid_zlb(metric, round(entry_resid, 2), exit_policy, round(target_frac, 4), round(hl_mult, 4))
        fig_h1 = _heatmap_fig(
            title    = f"beta-lb x OU z-lb - {METRIC_LABEL[metric]}  (resid +/-{entry_resid:g}bp; {exit_short})",
            grid     = g1,
            x_labels = Z_LB_VALS, y_labels = BETA_LB_VALS,
            x_title  = "OU z-score lookback (diagnostic)",
            y_title  = "beta lookback (days)",
            metric   = metric,
            cur_x    = zscore_lb if zscore_lb in Z_LB_VALS else None,
            cur_y    = beta_lb   if beta_lb   in BETA_LB_VALS else None,
        )

        g2 = _grid_resid_entry(metric, zs_snap, exit_policy, round(target_frac, 4), round(hl_mult, 4))
        fig_h2 = _heatmap_fig(
            title    = f"beta-lb x entry +/-resid bps - {METRIC_LABEL[metric]}  (z-lb={zs_snap}d; {exit_short})",
            grid     = g2,
            x_labels = RESID_VALS, y_labels = BETA_LB_VALS,
            x_title  = "entry threshold +/-resid bps",
            y_title  = "beta lookback (days)",
            metric   = metric,
            cur_x    = entry_resid if entry_resid in RESID_VALS else None,
            cur_y    = beta_lb if beta_lb in BETA_LB_VALS else None,
        )
    else:
        g1 = _grid(metric, round(entry_z, 2), exit_policy, round(target_frac, 4), round(hl_mult, 4))
        fig_h1 = _heatmap_fig(
            title    = f"beta-lb x z-lb - {METRIC_LABEL[metric]}  (entry +/-{entry_z})",
            grid     = g1,
            x_labels = Z_LB_VALS, y_labels = BETA_LB_VALS,
            x_title  = "z-score lookback (days)",
            y_title  = "beta lookback (days)",
            metric   = metric,
            cur_x    = zscore_lb if zscore_lb in Z_LB_VALS else None,
            cur_y    = beta_lb   if beta_lb   in BETA_LB_VALS else None,
        )

        g2 = _grid2(metric, zs_snap, exit_policy, round(target_frac, 4), round(hl_mult, 4))
        fig_h2 = _heatmap_fig(
            title    = f"beta-lb x entry +/-z - {METRIC_LABEL[metric]}  (z-lb={zs_snap}d)",
            grid     = g2,
            x_labels = Z_VALS, y_labels = BETA_LB_VALS,
            x_title  = "entry threshold +/-z",
            y_title  = "beta lookback (days)",
            metric   = metric,
            cur_x    = entry_z if entry_z in Z_VALS else None,
            cur_y    = beta_lb if beta_lb in BETA_LB_VALS else None,
        )

    return fig_h1, fig_h2

# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()
    if args.no_debug:
        app._ra_debug = False
    run(app, port=args.port, host=args.host)
