"""
Shared layout and Plotly helpers for research Dash apps.

Each strategy app imports palette constants, `styled_fig`, `stat_block`,
`slider`, `graph`, and `make_app` from here instead of repeating boilerplate.

Usage:
    from utils.research_app import (
        BG, PANEL, BORDER, TEXT, DIM, GRID, ORANGE, C0, C1, C2,
        styled_fig, stat_block, slider, graph, make_app,
    )
"""

from __future__ import annotations
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, dcc, html

# ── palette ───────────────────────────────────────────────────────────────────
BG     = "#FFFFFF"
PANEL  = "#F5F5F5"
BORDER = "#DCDCDC"
TEXT   = "#333333"
DIM    = "#888888"
GRID   = "#EBEBEB"
ORANGE = "#D35400"  # matches Viz.COLORS[0]
C0     = "#E74C3C"  # red  (short / sell)
C1     = "#27AE60"  # green (long / buy)
C2     = "#2980B9"  # blue


# ── Plotly figure factory ─────────────────────────────────────────────────────

def styled_fig(title: str, yaxis_title: str = "", height: int = 340) -> go.Figure:
    """Return an empty Plotly figure with standard research-app styling."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color=TEXT),
                   x=0, xanchor="left", pad=dict(l=4)),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Arial, Helvetica, sans-serif", size=10, color=TEXT),
        margin=dict(l=54, r=16, t=36, b=36),
        height=height,
        legend=dict(orientation="h", x=0, y=1.08, xanchor="left",
                    font=dict(size=9), bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hovermode="x unified",
        xaxis=dict(showgrid=False, linecolor=BORDER, tickfont=dict(size=9), zeroline=False),
        yaxis=dict(
            showgrid=True, gridcolor=GRID, linecolor=BORDER,
            tickfont=dict(size=9),
            title=dict(text=yaxis_title, font=dict(size=9)),
            zeroline=False,
        ),
    )
    return fig


# ── layout components ─────────────────────────────────────────────────────────

def stat_block(label: str, value: str, alert: bool = False) -> html.Div:
    """Labeled metric block for the stats bar."""
    return html.Div([
        html.Span(label, style={
            "color": DIM, "fontSize": 10, "display": "block",
            "textTransform": "uppercase", "letterSpacing": "0.05em",
        }),
        html.Span(value, style={
            "fontWeight": "bold", "fontSize": 15, "fontFamily": "monospace",
            "color": ORANGE if alert else TEXT,
        }),
    ])


def slider(id_: str, values: list, default, label: str) -> html.Div:
    """Labeled slider control. values must be a sorted list (used as discrete steps)."""
    return html.Div([
        html.Span(label, style={
            "color": DIM, "fontSize": 10, "fontWeight": "bold",
            "letterSpacing": "0.05em", "textTransform": "uppercase",
            "display": "block", "marginBottom": 6,
        }),
        dcc.Slider(
            id=id_, min=values[0], max=values[-1], step=None,
            marks={v: {"label": str(v), "style": {"fontSize": 10, "color": DIM}}
                   for v in values},
            value=default, tooltip={"always_visible": False},
        ),
    ])


def slider_with_input(id_: str, values: list, default, label: str) -> html.Div:
    """Labeled slider + text input. Slider snaps to nearest mark; input accepts any value.
    Slider id: id_   Input id: f'{id_}-typed'
    Wire a sync callback so they stay in sync (see app.py for pattern).
    """
    _lbl_style = {
        "color": DIM, "fontSize": 10, "fontWeight": "bold",
        "letterSpacing": "0.05em", "textTransform": "uppercase",
        "display": "block", "marginBottom": 6,
    }
    _input_style = {
        "width": 58, "fontSize": 12, "border": f"1px solid {BORDER}",
        "borderRadius": 3, "padding": "4px 6px",
        "fontFamily": "monospace", "color": TEXT, "outline": "none",
        "MozAppearance": "textfield",
    }
    return html.Div([
        html.Span(label, style=_lbl_style),
        html.Div(style={"display": "flex", "alignItems": "center", "gap": 10}, children=[
            html.Div(style={"flex": 1}, children=[
                dcc.Slider(
                    id=id_, min=values[0], max=values[-1], step=None,
                    marks={v: {"label": str(v), "style": {"fontSize": 10, "color": DIM}}
                           for v in values},
                    value=default, tooltip={"always_visible": False},
                ),
            ]),
            dcc.Input(
                id=f"{id_}-typed", type="number", value=default,
                debounce=True,
                style=_input_style,
            ),
        ]),
    ])


def graph(id_: str) -> dcc.Graph:
    """dcc.Graph with standard config (no mode bar)."""
    return dcc.Graph(
        id=id_,
        config={"displayModeBar": False},
        style={"border": f"1px solid {BORDER}"},
    )


# ── app factory ───────────────────────────────────────────────────────────────

def make_app(
    title: str,
    subtitle: str,
    data_info: str,
    sliders: list,
    body: html.Div,
    debug: bool = True,
) -> Dash:
    """
    Build a Dash app with a standard header + slider row(s) + body area.

    title     : bold coloured title in the header (e.g. "10s30s")
    subtitle  : grey subtitle (e.g. "beta-weighted curve explorer")
    data_info : right-aligned data range string
    sliders   : list of slider() Divs, or a list of such lists to lay out
                as separate rows (e.g. group entry params on one row and
                exit params on another)
    body      : html.Div containing charts, stats, etc.
    debug     : True shows callback errors in browser (recommended during dev)
    """
    _assets = Path(__file__).parent / "assets"
    _assets.mkdir(exist_ok=True)

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        assets_folder=str(_assets),
    )
    app.title = title

    slider_rows = sliders if sliders and isinstance(sliders[0], list) else [sliders]
    slider_row_divs = [
        html.Div(
            style={
                "padding": "10px 28px",
                "display": "grid",
                "gridTemplateColumns": " ".join(["1fr"] * len(row)),
                "gap": "8px 48px",
            },
            children=row,
        )
        for row in slider_rows
    ]

    app.layout = html.Div(
        style={"background": BG, "minHeight": "100vh", "paddingBottom": 64,
               "fontFamily": "Arial, Helvetica, sans-serif", "color": TEXT},
        children=[
            # header bar
            html.Div(style={
                "background": PANEL, "borderBottom": f"2px solid {BORDER}",
                "padding": "10px 24px", "display": "flex",
                "alignItems": "center", "gap": 14,
            }, children=[
                html.Span(title, style={"fontSize": 18, "fontWeight": "bold",
                                         "color": ORANGE, "letterSpacing": 3}),
                html.Span(subtitle, style={"fontSize": 12, "color": DIM}),
                html.Span(data_info, style={"marginLeft": "auto", "fontSize": 11,
                                             "color": DIM, "fontStyle": "italic"}),
            ]),
            # slider rows
            html.Div(
                style={"borderBottom": f"1px solid {BORDER}", "paddingTop": 4},
                children=slider_row_divs,
            ),
            # body
            body,
        ],
    )

    # stash debug flag so run() callers can pick it up
    app._ra_debug = debug
    return app


def run(app: Dash, port: int = 8050, host: str = "127.0.0.1") -> None:
    """Start the app. Uses the debug flag set in make_app()."""
    debug = getattr(app, "_ra_debug", True)
    print(f"  {app.title}  →  http://{host}:{port}  (debug={debug})")
    app.run(host=host, port=port, debug=debug)
