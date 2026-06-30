#!/usr/bin/env python3
"""
Beta-weighted 10s30s curve explorer.
Run:  mamba run -n 2s10s python strats/curve/app.py
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
from stats.ols import roll_lr_diff
from stats.ou import ou_params, roll_ou_zscore

# ── config ────────────────────────────────────────────────────────────────────

TICKERS  = {"10y": "USGG10YR Index", "30y": "USGG30YR Index"}
START    = "2010-01-01"
FWD_BARS = 40
LB_VALS  = [21, 42, 63, 90, 126]
Z_VALS   = [1.0, 1.5, 2.0, 2.5, 3.0]

METRICS = [
    {"label": "Avg hit rate",   "value": "avg_hit"},
    {"label": "Long hit rate",  "value": "hit_long"},
    {"label": "Short hit rate", "value": "hit_short"},
    {"label": "Approx Sharpe",  "value": "sharpe"},
]
METRIC_LABEL = {m["value"]: m["label"] for m in METRICS}


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
    resid_cum = pl.concat([null1, reg["resid_cum"]]).alias("resid_cum")
    beta      = pl.concat([null1, reg["beta"]]).alias("beta")
    z         = roll_ou_zscore(resid_cum, lookback=zscore_lb).alias("zscore")
    df = DATA.with_columns([beta, resid_cum, z,
                             (pl.col("30y") - pl.col("10y")).alias("naive")])
    pdf = df.to_pandas()
    pdf["ts"] = pd.to_datetime(pdf["ts"])
    return pdf.set_index("ts")


def _hit_rates(pdf: pd.DataFrame) -> pd.DataFrame:
    sub     = pdf[["zscore", "resid_cum"]].dropna()
    fwd_chg = sub["resid_cum"].shift(-FWD_BARS) - sub["resid_cum"]
    rows = []
    for t in Z_VALS:
        lm, sm   = sub["zscore"] < -t, sub["zscore"] > t
        n_l, n_s = int(lm.sum()), int(sm.sum())
        hl = float((fwd_chg[lm] > 0).mean()) if n_l > 0 else np.nan
        hs = float((fwd_chg[sm] < 0).mean()) if n_s > 0 else np.nan
        rows.append({"z": t, "n_long": n_l, "n_short": n_s,
                     "hit_long": hl, "hit_short": hs})
    return pd.DataFrame(rows)


def _sharpe(pdf: pd.DataFrame, entry_z: float) -> float:
    sub = pdf[["zscore", "resid_cum"]].dropna()
    fwd = sub["resid_cum"].shift(-FWD_BARS) - sub["resid_cum"]
    pnl = pd.concat([
        fwd[sub["zscore"] < -entry_z],
        -fwd[sub["zscore"] > entry_z],
    ]).dropna()
    if len(pnl) < 5 or pnl.std() == 0:
        return np.nan
    return float(pnl.mean() / pnl.std() * np.sqrt(252 / FWD_BARS))


def _simulate(pdf: pd.DataFrame, entry_z: float):
    """Walk-forward simulation. Returns (long_entries, short_entries, exit_dates, cum_pnl)."""
    sub   = pdf[["zscore", "resid_cum"]].dropna()
    z_arr = sub["zscore"].values
    r_arr = sub["resid_cum"].values
    dates = sub.index
    n     = len(z_arr)

    long_entries:  list = []
    short_entries: list = []
    exit_dates:    list = []
    pnls:          list = []

    state = e_idx = e_val = None

    for i in range(n):
        zi, ri = z_arr[i], r_arr[i]
        if np.isnan(zi) or np.isnan(ri):
            continue

        if state == "long":
            if zi >= 0 or (i - e_idx) >= FWD_BARS:
                exit_dates.append(dates[i])
                pnls.append(ri - e_val)
                state = None
        elif state == "short":
            if zi <= 0 or (i - e_idx) >= FWD_BARS:
                exit_dates.append(dates[i])
                pnls.append(-(ri - e_val))
                state = None

        if state is None:
            if zi < -entry_z:
                long_entries.append(dates[i])
                state, e_idx, e_val = "long", i, ri
            elif zi > entry_z:
                short_entries.append(dates[i])
                state, e_idx, e_val = "short", i, ri

    cum_pnl = pd.Series(dtype=float)
    if pnls:
        cum_pnl = pd.Series(np.cumsum(pnls), index=pd.DatetimeIndex(exit_dates))
    return long_entries, short_entries, exit_dates, cum_pnl


def _marker_xy(pdf: pd.DataFrame, dates, col: str = "10y"):
    """Align marker dates to pdf column, drop nulls."""
    if not dates:
        return [], []
    vals = pdf[col].reindex(pd.DatetimeIndex(dates)).dropna()
    return list(vals.index), list(vals.values)


@lru_cache(maxsize=64)
def _grid(metric: str, entry_z: float) -> tuple:
    """β-lb (y) × z-lb (x) at fixed entry_z. Cached."""
    grid = np.full((len(LB_VALS), len(LB_VALS)), np.nan)
    for i, beta_lb in enumerate(LB_VALS):
        for j, zscore_lb in enumerate(LB_VALS):
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
                    grid[i, j] = _sharpe(pdf, entry_z)
            except Exception:
                pass
    return tuple(tuple(float(v) if not np.isnan(v) else None for v in row)
                 for row in grid)


@lru_cache(maxsize=64)
def _grid2(metric: str, zscore_lb_snap: int) -> tuple:
    """β-lb (y) × entry threshold (x) at fixed zscore_lb. Cached."""
    grid = np.full((len(LB_VALS), len(Z_VALS)), np.nan)
    for i, beta_lb in enumerate(LB_VALS):
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
                    grid[i, j] = _sharpe(pdf, ez)
            except Exception:
                pass
    return tuple(tuple(float(v) if not np.isnan(v) else None for v in row)
                 for row in grid)


# ── value parsers ─────────────────────────────────────────────────────────────

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
    is_pct = metric != "sharpe"
    fmt    = (lambda v: f"{v:.0%}") if is_pct else (lambda v: f"{v:.2f}")
    z      = [list(row) for row in grid]
    text   = [[fmt(v) if v is not None else "—" for v in row] for row in z]

    fig = styled_fig(title, height=270)
    fig.add_trace(go.Heatmap(
        x=[str(v) for v in x_labels],
        y=[str(v) for v in y_labels],
        z=z,
        colorscale="RdYlGn",
        zmin=0.42 if is_pct else -0.3,
        zmax=0.82 if is_pct else 1.8,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=10, color="black"),
        hovertemplate=f"{y_title}=%{{y}}  {x_title}=%{{x}}<br>%{{text}}<extra></extra>",
        showscale=True,
        colorbar=dict(thickness=10, len=0.85, tickfont=dict(size=8),
                      tickformat=".0%" if is_pct else ".2f"),
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
        margin=dict(l=60, r=70, t=36, b=46),
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
        slider_with_input("beta-lb",   LB_VALS, 63,  "beta lookback (days)"),
        slider_with_input("zscore-lb", LB_VALS, 63,  "z-score lookback (days)"),
        slider_with_input("entry-z",   Z_VALS,  2.0, "entry threshold  ±z"),
    ],
    body=html.Div([
        # stats bar
        html.Div(id="stats-row", style={
            "padding": "10px 28px 10px", "background": PANEL,
            "borderBottom": f"1px solid {BORDER}",
            "display": "flex", "gap": 40, "flexWrap": "wrap",
        }),
        # row 1: yields + spread
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
    ]),
    debug=True,
)


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


_register_sync("beta-lb",   LB_VALS)
_register_sync("zscore-lb", LB_VALS)
_register_sync("entry-z",   Z_VALS)


# ── main callback: time-series charts + stats ─────────────────────────────────

@app.callback(
    Output("chart-yields", "figure"),
    Output("chart-spread", "figure"),
    Output("chart-resid",  "figure"),
    Output("chart-zscore", "figure"),
    Output("chart-pnl",    "figure"),
    Output("chart-hits",   "figure"),
    Output("stats-row",    "children"),
    Input("beta-lb-typed",   "value"),
    Input("zscore-lb-typed", "value"),
    Input("entry-z-typed",   "value"),
    Input("beta-lb",   "value"),
    Input("zscore-lb", "value"),
    Input("entry-z",   "value"),
)
def update_charts(beta_t, zscore_t, ez_t, beta_s, zscore_s, ez_s):
    beta_lb   = _parse_lb(beta_t,   beta_s,   63)
    zscore_lb = _parse_lb(zscore_t, zscore_s, 63)
    entry_z   = _parse_z(ez_t,     ez_s,     2.0)

    pdf  = _compute(beta_lb, zscore_lb)
    scan = _hit_rates(pdf)
    crow = _crow(scan, entry_z)

    long_entries, short_entries, exit_dates, cum_pnl = _simulate(pdf, entry_z)

    le_x, le_y = _marker_xy(pdf, long_entries, "naive")
    se_x, se_y = _marker_xy(pdf, short_entries, "naive")
    ex_x, ex_y = _marker_xy(pdf, exit_dates,   "naive")

    # ── chart 1: 10Y + 30Y yields ────────────────────────────────────────────
    fig1 = styled_fig(f"10Y & 30Y yields — β-lb={beta_lb}d  z-lb={zscore_lb}d",
                      "bps (×100)", height=320)
    fig1.add_trace(go.Scatter(
        x=pdf.index, y=pdf["10y"], name="10Y",
        line=dict(color=C2, width=1.3),
    ))
    fig1.add_trace(go.Scatter(
        x=pdf.index, y=pdf["30y"], name="30Y",
        line=dict(color=ORANGE, width=1.3),
    ))

    # ── chart 2: naive 10s30s spread + signal markers ────────────────────────
    fig2 = styled_fig("10s30s naive spread  (30Y − 10Y)", "bps", height=320)
    fig2.add_hline(y=float(pdf["naive"].mean()),
                   line=dict(color=DIM, dash="dot", width=1),
                   annotation_text="mean", annotation_position="right")
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

    # ── chart 3: cumulated residual ───────────────────────────────────────────
    fig3 = styled_fig(f"Direction-stripped spread  (β={beta_lb}d, 30Y − β·10Y cumulated)", "bps", height=280)
    fig3.add_hline(y=0, line=dict(color=BORDER, dash="dot", width=0.8))
    fig3.add_trace(go.Scatter(
        x=pdf.index, y=pdf["resid_cum"], showlegend=False,
        line=dict(color=C2, width=1.2),
    ))

    # ── chart 4: OU z-score ───────────────────────────────────────────────────
    z_clean = pdf["zscore"].dropna()
    fig4 = styled_fig(f"OU z-score — z-lb={zscore_lb}d  entry=±{entry_z}", "z", height=280)
    if len(z_clean) > 0:
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
    fig5 = styled_fig(f"Cumulative PnL  ({n_trades} trades)", "bps", height=240)
    fig5.add_hline(y=0, line=dict(color=BORDER, width=1))
    if len(cum_pnl) > 0:
        plot_pnl = pd.concat([
            pd.Series([0.0], index=[pdf.index[0]]),
            cum_pnl,
        ]).sort_index()
        final = float(cum_pnl.iloc[-1])
        color = C1 if final >= 0 else C0
        fig5.add_trace(go.Scatter(
            x=plot_pnl.index, y=plot_pnl.values,
            mode="lines", line_shape="hv",
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
    sub     = pdf[["naive", "10y", "resid_cum"]].dropna()
    c_naive = float(sub["naive"].corr(sub["10y"]))
    c_bw    = float(sub["resid_cum"].corr(sub["10y"]))
    p       = ou_params(pdf["resid_cum"].dropna())
    hl      = p["half_life"]
    hl_val  = crow["hit_long"]
    hs_val  = crow["hit_short"]
    sh      = _sharpe(pdf, entry_z)
    final   = float(cum_pnl.iloc[-1]) if len(cum_pnl) > 0 else np.nan

    stats = [
        stat_block("direction ρ naive",       f"{c_naive:+.3f}"),
        stat_block("direction ρ β-weighted",  f"{c_bw:+.3f}"),
        stat_block("OU half-life",            f"{hl:.0f}d" if not np.isnan(hl) else "—"),
        stat_block("OU σ / day",              f"{p['sigma']:.2f} bps"),
        stat_block(f"hit long  z={entry_z}",
                   f"{hl_val:.1%}" if not np.isnan(hl_val) else "—",
                   alert=not np.isnan(hl_val) and hl_val > 0.6),
        stat_block(f"hit short  z={entry_z}",
                   f"{hs_val:.1%}" if not np.isnan(hs_val) else "—",
                   alert=not np.isnan(hs_val) and hs_val < 0.45),
        stat_block("approx sharpe",
                   f"{sh:.2f}" if not np.isnan(sh) else "—",
                   alert=not np.isnan(sh) and sh > 0.8),
        stat_block("cum PnL",
                   f"{final:+.1f} bps" if not np.isnan(final) else "—",
                   alert=not np.isnan(final) and final > 0),
        stat_block("n long entries",  str(len(long_entries))),
        stat_block("n short entries", str(len(short_entries))),
    ]

    return fig1, fig2, fig3, fig4, fig5, fig6, stats


# ── heatmap callback ──────────────────────────────────────────────────────────

@app.callback(
    Output("chart-heatmap1", "figure"),
    Output("chart-heatmap2", "figure"),
    Input("metric-select",   "value"),
    Input("entry-z-typed",   "value"),
    Input("entry-z",         "value"),
    Input("beta-lb-typed",   "value"),
    Input("beta-lb",         "value"),
    Input("zscore-lb-typed", "value"),
    Input("zscore-lb",       "value"),
)
def update_heatmaps(metric, ez_t, ez_s, beta_t, beta_s, zscore_t, zscore_s):
    metric    = metric or "avg_hit"
    entry_z   = _parse_z(ez_t,    ez_s,    2.0)
    beta_lb   = _parse_lb(beta_t,  beta_s,  63)
    zscore_lb = _parse_lb(zscore_t, zscore_s, 63)

    # Heatmap 1: β-lb × z-lb  (at fixed entry_z)
    g1 = _grid(metric, round(entry_z, 2))
    fig_h1 = _heatmap_fig(
        title    = f"β-lb × z-lb  — {METRIC_LABEL[metric]}  (entry ±{entry_z})",
        grid     = g1,
        x_labels = LB_VALS, y_labels = LB_VALS,
        x_title  = "z-score lookback (days)",
        y_title  = "β lookback (days)",
        metric   = metric,
        cur_x    = zscore_lb if zscore_lb in LB_VALS else None,
        cur_y    = beta_lb   if beta_lb   in LB_VALS else None,
    )

    # Heatmap 2: β-lb × entry_z  (at fixed z-lb snapped to nearest LB_VALS)
    zs_snap = min(LB_VALS, key=lambda x: abs(x - zscore_lb))
    g2 = _grid2(metric, zs_snap)
    fig_h2 = _heatmap_fig(
        title    = f"β-lb × entry ±z  — {METRIC_LABEL[metric]}  (z-lb={zs_snap}d)",
        grid     = g2,
        x_labels = Z_VALS, y_labels = LB_VALS,
        x_title  = "entry threshold ±z",
        y_title  = "β lookback (days)",
        metric   = metric,
        cur_x    = entry_z if entry_z in Z_VALS else None,
        cur_y    = beta_lb if beta_lb in LB_VALS else None,
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
