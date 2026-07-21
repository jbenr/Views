#!/usr/bin/env python3
"""10s30s signal comparison - how similar are the funnel's winning setups?

Reads the trade log saved by `python -m book.curve.tens_10s30s --sweep`
(TRADES_FILE) plus the market data, and compares the sweep winners side by
side: stacked long/short position ribbons under the 10s30s level, pairwise
position correlation, per-setup stats, and an equal-weight combined book.
The question: do the signals diversify each other (low overlap, higher
combined sharpe) or are they the same trade in different clothes?

PnL here is gross of costs, marked daily as position[t-1] x d(level); the
engine's official numbers live in
data/tens_10s30s/tens_10s30s_sweep_results.parquet.

Run:  mamba run -n 2s10s python book/curve/app.py
      Open http://localhost:8051
"""

from __future__ import annotations

import argparse

import numpy as np
import plotly.graph_objects as go
import polars as pl
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots

from book.curve.tens_10s30s import TARGET, TRADES_FILE, load_data, model_frame
from utils.research_app import (
    BG, BORDER, C0, C1, C2, DIM, GRID, ORANGE, PANEL, TEXT,
    graph, make_app, run, stat_block, styled_fig,
)

# -- data (loaded once at startup) --------------------------------------------

if not TRADES_FILE.exists():
    raise SystemExit(f"{TRADES_FILE} not found - run --sweep first")

TRADES = pl.read_parquet(TRADES_FILE)
DATA = model_frame(load_data())
DATES = DATA["ts"].to_list()
LEVEL = DATA[TARGET].to_numpy().astype(float)
SETUPS = TRADES["setup"].unique(maintain_order=True).to_list()

_DATE_IX = {d: i for i, d in enumerate(DATES)}

LONG_FILL = "rgba(39, 174, 96, 0.75)"   # C1
SHORT_FILL = "rgba(231, 76, 60, 0.75)"  # C0
INDIV_LINE = "#B8B8B8"


def _positions() -> dict[str, np.ndarray]:
    """Daily position (+1/-1/0) per setup, rebuilt from its trade log."""
    out = {}
    for setup in SETUPS:
        pos = np.zeros(len(DATES))
        for t in TRADES.filter(pl.col("setup") == setup).iter_rows(named=True):
            i0 = _DATE_IX.get(t["entry_date"])
            i1 = _DATE_IX.get(t["exit_date"])
            if i0 is None or i1 is None:
                continue
            pos[i0:i1] = 1.0 if t["direction"] == "long" else -1.0
        out[setup] = pos
    return out


POS = _positions()
# position held over (t-1, t] earns d(level) at t
PNL = {
    s: np.concatenate([[0.0], p[:-1] * np.diff(LEVEL)]) for s, p in POS.items()
}


def _sharpe(daily_pnl: np.ndarray) -> float:
    sd = float(daily_pnl.std())
    return float(daily_pnl.mean()) / sd * np.sqrt(252) if sd > 0 else 0.0


def _corr_matrix(selected: list[str]) -> np.ndarray:
    mat = np.column_stack([POS[s] for s in selected])
    if mat.shape[1] == 1:
        return np.ones((1, 1))
    return np.corrcoef(mat.T)


def _avg_pairwise_corr(corr: np.ndarray) -> float:
    n = corr.shape[0]
    if n < 2:
        return 1.0
    iu = np.triu_indices(n, k=1)
    return float(np.nanmean(corr[iu]))


# -- figures -------------------------------------------------------------------


def _style(fig: go.Figure, height: int) -> go.Figure:
    """styled_fig()'s layout applied to a make_subplots figure."""
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Arial, Helvetica, sans-serif", size=10, color=TEXT),
        margin=dict(l=54, r=16, t=30, b=36),
        height=height, showlegend=False, hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER,
                     tickfont=dict(size=9), zeroline=False)
    return fig


def _fig_ribbons(selected: list[str]) -> go.Figure:
    """10s30s level on top; one long/short position ribbon per setup below,
    sharing the x-axis so overlap is visible at a glance."""
    n = len(selected)
    fig = make_subplots(
        rows=n + 1, cols=1, shared_xaxes=True,
        row_heights=[0.30] + [0.70 / n] * n, vertical_spacing=0.008,
    )
    fig.add_trace(
        go.Scatter(
            x=DATES, y=LEVEL, name="10s30s",
            line=dict(color=C2, width=1.2),
            hovertemplate="%{y:.1f} bps<extra>10s30s</extra>",
        ),
        row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, linecolor=BORDER, zeroline=False,
        tickfont=dict(size=9), title=dict(text="bps", font=dict(size=9)),
        row=1, col=1,
    )
    for i, s in enumerate(selected, start=2):
        pos = POS[s]
        for clipped, fill in [
            (np.clip(pos, 0.0, None), LONG_FILL),
            (np.clip(pos, None, 0.0), SHORT_FILL),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=DATES, y=clipped, mode="lines",
                    line=dict(width=0, shape="hv"),
                    fill="tozeroy", fillcolor=fill, hoverinfo="skip",
                ),
                row=i, col=1,
            )
        fig.update_yaxes(
            range=[-1.15, 1.15], showticklabels=False, showgrid=False,
            zeroline=True, zerolinecolor=GRID, linecolor=BORDER,
            row=i, col=1,
        )
        fig.add_annotation(
            text=s, xref="paper", x=0.003, yref=f"y{i} domain", y=0.5,
            xanchor="left", showarrow=False,
            font=dict(size=9, color=DIM), bgcolor="rgba(255,255,255,0.6)",
        )
    fig.update_layout(
        title=dict(text="positions - long (green) / short (red) per setup",
                   font=dict(size=11, color=TEXT), x=0, xanchor="left"),
    )
    return _style(fig, height=200 + 42 * n)


def _fig_overlap(selected: list[str]) -> go.Figure:
    corr = _corr_matrix(selected)
    text = [[f"{v:+.2f}" for v in row] for row in corr]
    fig = styled_fig("position overlap - correlation of daily positions",
                     height=420)
    fig.add_trace(
        go.Heatmap(
            z=corr, x=selected, y=selected, zmin=-1, zmax=1,
            colorscale=[[0.0, C0], [0.5, "#F7F7F7"], [1.0, C2]],
            text=text, texttemplate="%{text}",
            textfont=dict(size=9, color=TEXT),
            hovertemplate="%{y}<br>%{x}<br>corr %{z:+.2f}<extra></extra>",
            showscale=False, xgap=2, ygap=2,
        )
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(tickangle=40, tickfont=dict(size=8), showgrid=False)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=8),
                     showgrid=False)
    return fig


def _fig_equity(selected: list[str], combo_pnl: np.ndarray) -> go.Figure:
    fig = styled_fig(
        "cumulative pnl - setups (gray) vs equal-weight combo (gross, bps)",
        "bps", height=420,
    )
    for k, s in enumerate(selected):
        fig.add_trace(
            go.Scatter(
                x=DATES, y=PNL[s].cumsum(),
                name="individual setups", legendgroup="indiv",
                showlegend=(k == 0),
                line=dict(color=INDIV_LINE, width=1),
                hovertemplate="%{y:.1f} bps<extra>" + s + "</extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=DATES, y=combo_pnl.cumsum(), name="equal-weight combo",
            line=dict(color=ORANGE, width=2.4),
            hovertemplate="%{y:.1f} bps<extra>combo</extra>",
        )
    )
    fig.update_layout(showlegend=True, hovermode="closest")
    return fig


# -- per-setup table -----------------------------------------------------------

_CELL = {
    "fontSize": 10, "padding": "5px 8px", "textAlign": "right",
    "borderBottom": f"1px solid {GRID}", "borderRight": f"1px solid {GRID}",
    "whiteSpace": "nowrap", "fontFamily": "monospace",
}
_HEAD = {**_CELL, "fontWeight": "bold", "backgroundColor": PANEL,
         "fontFamily": "Arial, Helvetica, sans-serif"}


def _setup_table(selected: list[str], combo_pnl: np.ndarray) -> html.Table:
    header = ["setup", "trades", "hit%", "pnl (bps)", "ex-best", "best trd%",
              "med trd", "sharpe", "avg hold", "days active%", "corr to combo"]
    rows = []
    for s in selected:
        tr = TRADES.filter(pl.col("setup") == s)
        pnl = PNL[s]
        active = float((POS[s] != 0).mean())
        c_sd = combo_pnl.std()
        corr = (
            float(np.corrcoef(pnl, combo_pnl)[0, 1])
            if pnl.std() > 0 and c_sd > 0 else 0.0
        )
        total = float(tr["pnl_bps"].sum())
        best = float(tr["pnl_bps"].max())
        best_share = f"{100 * best / total:.0f}" if total > 0 else "-"
        cells = [
            html.Td(s, style={**_CELL, "textAlign": "left",
                              "fontFamily": "Arial, Helvetica, sans-serif"}),
            html.Td(f"{len(tr)}", style=_CELL),
            html.Td(f"{(tr['pnl_bps'] > 0).mean() * 100:.0f}", style=_CELL),
            html.Td(f"{total:+.1f}", style=_CELL),
            html.Td(f"{total - best:+.1f}", style=_CELL),
            html.Td(best_share, style=_CELL),
            html.Td(f"{tr['pnl_bps'].median():+.2f}", style=_CELL),
            html.Td(f"{_sharpe(pnl):+.2f}", style=_CELL),
            html.Td(f"{tr['bars_held'].mean():.1f}", style=_CELL),
            html.Td(f"{active * 100:.0f}", style=_CELL),
            html.Td(f"{corr:+.2f}", style=_CELL),
        ]
        rows.append(html.Tr(cells))
    return html.Table(
        [html.Thead(html.Tr([html.Th(h, style={**_HEAD, "textAlign":
                                               "left" if h == "setup" else "right"})
                             for h in header]))]
        + [html.Tbody(rows)],
        style={"borderCollapse": "collapse", "width": "100%",
               "border": f"1px solid {BORDER}"},
    )


# -- app -----------------------------------------------------------------------

_span = f"{DATES[0]} - {DATES[-1]}  |  {len(SETUPS)} setups  |  {len(TRADES)} trades"

_checklist = html.Div([
    html.Span("setups in combo", style={
        "color": DIM, "fontSize": 10, "fontWeight": "bold",
        "letterSpacing": "0.05em", "textTransform": "uppercase",
        "display": "block", "marginBottom": 6,
    }),
    dcc.Checklist(
        id="setup-select",
        options=[{"label": s, "value": s} for s in SETUPS],
        value=SETUPS, inline=True,
        style={"fontSize": 11},
        inputStyle={"marginRight": 4},
        labelStyle={"marginRight": 16, "whiteSpace": "nowrap"},
    ),
])

app = make_app(
    title="10s30s",
    subtitle="signal comparison - sweep winners, overlap, combined book",
    data_info=_span,
    sliders=[_checklist],
    body=html.Div(
        style={"padding": "14px 24px", "display": "grid", "gap": 14},
        children=[
            html.Div(id="stats-bar",
                     style={"display": "flex", "gap": 40, "flexWrap": "wrap",
                            "padding": "10px 12px", "background": PANEL,
                            "border": f"1px solid {BORDER}"}),
            graph("fig-ribbons"),
            html.Div(
                style={"display": "grid",
                       "gridTemplateColumns": "1fr 1fr", "gap": 14},
                children=[graph("fig-overlap"), graph("fig-equity")],
            ),
            html.Div(id="setup-table"),
        ],
    ),
)


@app.callback(
    Output("stats-bar", "children"),
    Output("fig-ribbons", "figure"),
    Output("fig-overlap", "figure"),
    Output("fig-equity", "figure"),
    Output("setup-table", "children"),
    Input("setup-select", "value"),
)
def update(selected):
    selected = [s for s in SETUPS if s in (selected or [])]
    if not selected:
        empty = styled_fig("select at least one setup", height=200)
        return [stat_block("setups", "0")], empty, empty, empty, html.Div()

    pnl_mat = np.column_stack([PNL[s] for s in selected])
    combo_pnl = pnl_mat.mean(axis=1)
    corr = _corr_matrix(selected)
    pos_any = np.column_stack([POS[s] for s in selected]).any(axis=1)
    best_single = max(_sharpe(PNL[s]) for s in selected)
    combo_sharpe = _sharpe(combo_pnl)

    stats = [
        stat_block("combo sharpe", f"{combo_sharpe:+.2f}",
                   alert=combo_sharpe > best_single),
        stat_block("best single", f"{best_single:+.2f}"),
        stat_block("avg pairwise corr", f"{_avg_pairwise_corr(corr):+.2f}"),
        stat_block("combo pnl", f"{combo_pnl.sum():+.1f} bps"),
        stat_block("days active", f"{pos_any.mean() * 100:.0f}%"),
        stat_block("setups", f"{len(selected)}"),
    ]
    return (
        stats,
        _fig_ribbons(selected),
        _fig_overlap(selected),
        _fig_equity(selected, combo_pnl),
        _setup_table(selected, combo_pnl),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    run(app, port=args.port, host=args.host)
