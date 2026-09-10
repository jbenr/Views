"""Research app.

    python -m research.app            # http://localhost:8052
    python -m research.app --port N
"""

from __future__ import annotations

import argparse
import time
import traceback

import polars as pl
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from dashboard.charts import (
    WINDOW_PRESETS,
    coverage_chart,
    hedge_weights_chart,
    level_chart,
)

from research.panel import (
    BETA_LOOKBACK,
    CATALOG,
    START,
    SWAP_SPREADS,
    VOLS,
    YIELDS,
    build_panel,
    dependent_leg,
    diagnostics,
    resolve_target,
)
from backtest.lab import scan_backend
from research.dislocation import dislocation_scan
from research.dislocation_backtest import run_grid
from utils.market_data import last_updated, swaption_last_updated
from utils.research_app import (
    BORDER, C0, C1, DIM, ORANGE, PANEL, TEXT, make_app, run,
    shimmer_loader, stat_block,
)
from utils.viz import table_div

DEFAULT_TARGET = "10s20s30s"
DEFAULT_FEATURES: list[str] = []
BETA_LOOKBACKS = [63, 126, 189, 252, 504]
DEFAULT_CHART_WINDOW = "6M"
RESEARCH_CHART_MAX_WIDTH = "1120px"
DISLOCATION_BETA_LBS = [63, 126, 252]
DISLOCATION_RESIDUAL_LBS = [5, 10, 20, 40, 60, 100, 126]
DISLOCATION_NORM_LBS = [63, 126]
DISLOCATION_THRESHOLDS = [1.0, 1.5, 2.0, 2.5]
DISLOCATION_HORIZONS = [5, 10, 20, 40]
DISLOCATION_GATES = {
    "feature_level": "Feature · level",
    "feature_move20": "Feature · 20d change",
    "feature_vol20": "Feature · 20d volatility",
    "target_level": "Target · level",
    "target_move20": "Target · 20d change",
    "target_vol20": "Target · 20d volatility",
    "r2": "Model · R²",
    "beta": "Model · beta",
    "beta_cv": "Model · beta stability",
    "model_quality": "Model · quality",
    "beta_vol20": "Model · beta 20d volatility",
    "beta_mom10": "Model · beta 10d change",
    "r2_vol20": "Model · R² 20d volatility",
    "r2_mom10": "Model · R² 10d change",
    "resid_vol20": "Residual · 20d volatility",
    "resid_vol60": "Residual · 60d volatility",
    "resid_mom10": "Residual · 10d change",
}
DEFAULT_DISLOCATION_GATES = [
    "feature_level", "feature_move20", "target_vol20",
    "r2", "beta", "beta_cv", "model_quality", "beta_vol20",
    "beta_mom10", "r2_vol20", "r2_mom10", "resid_vol20", "resid_mom10",
]
DISLOCATION_FIT_BASES = {
    "changes": "Changes · accumulated residual",
    "levels": "Levels · regression residual",
}

FEATURE_GROUPS = {
    "Treasury yields": sorted(YIELDS, key=lambda a: int(a[:-1])),
    "Curves & flies": sorted(name for name in CATALOG if name not in YIELDS),
    "Swap spreads": ["swsp2", "swsp5", "swsp10", "swsp20", "swsp30"],
    "Real rates & inflation": [
        "real5y", "real10y", "real30y", "be5", "be10", "be30",
    ],
    "SOFR OIS": ["sofr2", "sofr10", "sofr30"],
    "Mortgage": ["mtg_cc"],
    "Macro": ["dxy", "gold", "spx", "oil"],
    "Rate vol": sorted(VOLS),
}

FEATURE_LABELS = {
    **{name: f"{name[:-1]}Y Treasury yield" for name in YIELDS},
    "swsp2": "2Y Treasury swap spread",
    "swsp5": "5Y Treasury swap spread",
    "swsp10": "10Y Treasury swap spread",
    "swsp20": "20Y Treasury swap spread",
    "swsp30": "30Y Treasury swap spread",
    "real5y": "5Y real yield",
    "real10y": "10Y real yield",
    "real30y": "30Y real yield",
    "be5": "5Y breakeven inflation",
    "be10": "10Y breakeven inflation",
    "be30": "30Y breakeven inflation",
    "sofr2": "2Y SOFR OIS",
    "sofr10": "10Y SOFR OIS",
    "sofr30": "30Y SOFR OIS",
    "mtg_cc": "Fannie 30Y current-coupon yield",
    "dxy": "US dollar index",
    "gold": "Gold",
    "spx": "S&P 500",
    "oil": "WTI crude oil",
    **{
        name: f"{expiry} x {tenor}Y ATM normal vol"
        for name, (expiry, tenor) in VOLS.items()
    },
}

TARGET_LABELS = {
    **{name: name for name in CATALOG if name not in SWAP_SPREADS},
    **{name: FEATURE_LABELS[name] for name in SWAP_SPREADS},
}

TAB_STYLE = {
    "padding": "10px 18px", "fontSize": 12, "fontWeight": "bold",
    "background": PANEL, "border": f"1px solid {BORDER}", "color": DIM,
}
SELECTED_TAB_STYLE = {
    **TAB_STYLE, "background": "#FFFFFF", "color": ORANGE,
    "borderTop": f"3px solid {ORANGE}",
}
LABEL = {
    "color": DIM, "fontSize": 10, "fontWeight": "bold",
    "letterSpacing": "0.05em", "textTransform": "uppercase",
    "display": "block", "marginBottom": 6,
}
INPUT = {
    "width": "100%", "fontSize": 12, "fontFamily": "monospace",
    "padding": "5px 8px", "border": f"1px solid {BORDER}",
    "borderRadius": 3, "color": TEXT, "outline": "none",
}
HEADING = {"fontSize": 14, "fontWeight": "bold", "color": TEXT}


def btn_style(primary: bool = False) -> dict:
    return {
        "padding": "6px 14px", "fontSize": 12, "cursor": "pointer",
        "border": f"1px solid {ORANGE if primary else BORDER}",
        "background": ORANGE if primary else "#FFFFFF",
        "color": "#FFFFFF" if primary else TEXT,
        "borderRadius": 3,
    }


# ---- layout pieces ----------------------------------------------------------


def field(label: str, control) -> html.Div:
    return html.Div([html.Span(label, style=LABEL), control],
                    style={"marginBottom": 12})


def note(text: str, tone: str = "dim") -> html.Div:
    colour = {"dim": DIM, "warn": ORANGE, "bad": C0, "good": C1}[tone]
    return html.Div(text, style={"fontSize": 11, "color": colour,
                                 "marginTop": 6, "lineHeight": 1.5})


def heading(text: str, right=None) -> html.Div:
    kids = [html.Div(text.upper(), style=HEADING)]
    if right is not None:
        kids.append(html.Div(right, style={"marginLeft": "auto"}))
    return html.Div(kids, style={"display": "flex", "alignItems": "center",
                                 "gap": 12, "marginBottom": 12})


def stub_tab(title: str, needs: list[str]) -> html.Div:
    return html.Div(style={"padding": "18px 24px", "maxWidth": 860}, children=[
        heading(title),
        note("Not built yet. Load a panel on the Setup tab first; this tab "
             "will read it. Outstanding before it can be trusted:"),
        html.Ul([html.Li(n, style={"fontSize": 12, "color": TEXT,
                                   "marginBottom": 7}) for n in needs],
                style={"marginTop": 10, "lineHeight": 1.6}),
    ])


def dislocation_tab() -> html.Div:
    """Discovery then exact trade-mechanics testing for one selected row."""
    values = [
        ("beta lookbacks", "dis-beta-lbs", DISLOCATION_BETA_LBS),
        ("residual windows", "dis-residual-lbs", DISLOCATION_RESIDUAL_LBS),
        ("normalization", "dis-norm-lbs", DISLOCATION_NORM_LBS),
        ("entry thresholds (z)", "dis-thresholds", DISLOCATION_THRESHOLDS),
        ("forward horizons", "dis-horizons", DISLOCATION_HORIZONS),
    ]
    controls = [field("feature (load it on Setup first)", dcc.Dropdown(
        id="dis-feature", value="vol_1Mo_30", clearable=False,
        options=[
            {"label": FEATURE_LABELS.get(name, name), "value": name}
            for name in FEATURE_LABELS
        ], style={"fontSize": 12},
    ))]
    controls += [field("regression basis", dcc.Checklist(
        id="dis-fit-on", value=["changes", "levels"],
        options=[{"label": label, "value": value}
                 for value, label in DISLOCATION_FIT_BASES.items()],
        labelStyle={"display": "block", "fontSize": 12, "marginBottom": 6,
                    "color": TEXT},
        inputStyle={"marginRight": 5},
    ))]
    controls += [field(label, dcc.Dropdown(
        id=id_, value=items, multi=True, clearable=False,
        options=[{"label": str(item), "value": item} for item in items],
        style={"fontSize": 12},
    )) for label, id_, items in values]
    controls += [
        field("gate conditions", dcc.Dropdown(
            id="dis-gates", value=DEFAULT_DISLOCATION_GATES, multi=True,
            clearable=True,
            options=[{"label": label, "value": name}
                     for name, label in DISLOCATION_GATES.items()],
            style={"fontSize": 12},
        )),
        html.Div(style={"display": "flex", "gap": 7, "marginTop": -7,
                        "marginBottom": 12}, children=[
            html.Button("All", id="dis-gates-all", n_clicks=0,
                        className="ref-btn", style=btn_style()),
            html.Button("None", id="dis-gates-none", n_clicks=0,
                        className="ref-btn", style=btn_style()),
            html.Span("None = ungated only", style={
                "fontSize": 10, "color": DIM, "alignSelf": "center",
            }),
        ]),
        field("gate percentile lookbacks", dcc.Dropdown(
            id="dis-gate-windows", value=[126, 252, 504], multi=True,
            clearable=False,
            options=[{"label": str(value), "value": value}
                     for value in [63, 126, 252, 504]],
            style={"fontSize": 12},
        )),
        note("Changes uses a windowed accumulated residual. Levels uses the raw "
             "level-regression residual, so residual window is shown as —. Ungated is "
             "always included. Each checked gate adds causal bucket variants.", "dim"),
        html.Button("Run discovery", id="dis-run", n_clicks=0,
                    className="ref-btn", style={**btn_style(primary=True), "width": "100%"}),
        html.Div(id="dis-run-info"),
    ]
    return html.Div(style={"padding": "18px 24px"}, children=[
        heading("dislocation discovery"),
        html.Div(style={"display": "grid", "gridTemplateColumns": "300px minmax(0, 1fr)",
                        "gap": 26, "alignItems": "start"}, children=[
            html.Div(controls),
            dcc.Loading(html.Div(id="dis-out"),
                        custom_spinner=shimmer_loader(image="guy.png", caption="searching"),
                        overlay_style={"visibility": "visible", "opacity": 0.35},
                        parent_className="research-panel-loader"),
        ]),
        html.Div(style={"borderTop": f"1px solid {BORDER}", "marginTop": 24,
                        "paddingTop": 18}, children=[
            heading("trade mechanics"),
            html.Div(style={"display": "grid", "gridTemplateColumns": "300px minmax(0, 1fr)",
                            "gap": 26, "alignItems": "start"}, children=[
                dislocation_backtest_controls(),
                dcc.Loading(html.Div(id="bt-out"),
                            custom_spinner=shimmer_loader(image="guy.png", caption="backtesting"),
                            overlay_style={"visibility": "visible", "opacity": 0.35},
                            parent_className="research-panel-loader"),
            ]),
        ]),
        dcc.Store(id="dis-board"),
        dcc.Store(id="dis-candidate"),
        dcc.Store(id="bt-grid"),
    ])


def dislocation_backtest_controls() -> html.Div:
    """Controls deliberately limited to execution choices, not model refitting."""
    return html.Div([
        html.Div(id="bt-candidate", children=note(
            "Choose Backtest on a row above to freeze its relationship and gate.", "dim"
        )),
        field("entry thresholds (z)", dcc.Dropdown(
            id="bt-entry-zs", value=[1.5, 2.0, 2.5, 3.0], multi=True,
            clearable=False,
            options=[{"label": str(v), "value": v}
                     for v in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]],
            style={"fontSize": 12},
        )),
        field("exit styles", dcc.Checklist(
            id="bt-exit-styles", value=["time", "band", "revert_frac"],
            options=[
                {"label": " time stop (business days)", "value": "time"},
                {"label": " residual band (z)", "value": "band"},
                {"label": " reversion fraction", "value": "revert_frac"},
            ],
            labelStyle={"display": "block", "fontSize": 12, "marginBottom": 6,
                        "color": TEXT}, inputStyle={"marginRight": 5},
        )),
        field("time stops (d)", dcc.Dropdown(
            id="bt-time-stops", value=[5, 10, 20, 40, 60], multi=True,
            clearable=False,
            options=[{"label": str(v), "value": v} for v in [3, 5, 10, 15, 20, 30, 40, 60, 80]],
            style={"fontSize": 12},
        )),
        field("exit bands (z)", dcc.Dropdown(
            id="bt-bands", value=[0.0, 0.5, 1.0], multi=True, clearable=False,
            options=[{"label": str(v), "value": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]],
            style={"fontSize": 12},
        )),
        field("reversion fractions", dcc.Dropdown(
            id="bt-revert-fracs", value=[0.25, 0.5, 0.75, 1.0], multi=True,
            clearable=False,
            options=[{"label": str(v), "value": v} for v in [0.25, 0.5, 0.75, 1.0]],
            style={"fontSize": 12},
        )),
        field("hard stop (bp)", dcc.Dropdown(
            id="bt-stop", value=25.0, clearable=False,
            options=[
                {"label": "none", "value": 0.0},
                *[{"label": str(v), "value": v} for v in [10.0, 15.0, 25.0, 40.0, 60.0]],
            ], style={"fontSize": 12},
        )),
        field("round-trip cost (bp)", dcc.Dropdown(
            id="bt-cost", value=0.1, clearable=False,
            options=[{"label": str(v), "value": v} for v in [0.0, 0.1, 0.25, 0.5, 1.0]],
            style={"fontSize": 12},
        )),
        html.Button("Run backtest grid", id="bt-run", n_clicks=0,
                    className="ref-btn", style={**btn_style(primary=True), "width": "100%"}),
        html.Div(id="bt-run-info"),
    ])


# ---- setup tab --------------------------------------------------------------


def controls() -> html.Div:
    return html.Div(children=[
        field("target", dcc.Dropdown(
            id="target", value=DEFAULT_TARGET, clearable=False,
            options=[{"label": TARGET_LABELS[n], "value": n} for n in sorted(CATALOG)]
                    + [{"label": "custom weights...", "value": "custom"}],
            style={"fontSize": 12})),
        field("custom weights", dcc.Input(
            id="custom", type="text", placeholder="20y:2, 10y:-1, 30y:-1",
            debounce=True, style=INPUT)),
        field("leg weighting", dcc.RadioItems(
            id="weighting", value="fixed",
            options=[
                {"label": " fixed", "value": "fixed"},
                {"label": " beta-weighted", "value": "beta"},
            ],
            inline=True,
            labelStyle={"marginRight": 16, "fontSize": 12, "color": TEXT},
            inputStyle={"marginRight": 4})),
        field("beta lookback (d)", dcc.Dropdown(
            id="beta-lb", value=BETA_LOOKBACK, clearable=False,
            options=[{"label": str(v), "value": v} for v in BETA_LOOKBACKS],
            style={"fontSize": 12})),
        field("beta dependent leg", dcc.Input(
            id="beta-dependent", type="text", placeholder="auto (20y for 10s20s30s)",
            debounce=True, style=INPUT)),
        field("features", dcc.Dropdown(
            id="features", value=DEFAULT_FEATURES, multi=True,
            options=[{"label": f"{group} · {FEATURE_LABELS.get(name, name)}", "value": name}
                     for group, names in FEATURE_GROUPS.items() for name in names],
            style={"fontSize": 12})),
        dcc.Checklist(
            id="invert-feature", value=[],
            options=[{"label": " Invert feature in chart", "value": "invert"}],
            style={"fontSize": 11, "color": DIM, "marginTop": -6, "marginBottom": 12},
            inputStyle={"marginRight": 4},
        ),
        field("start date", dcc.Dropdown(
            id="start", value=START, clearable=False,
            options=[
                {"label": "All available · 2000", "value": "2000-01-01"},
                {"label": "2010 onward", "value": "2010-01-01"},
                {"label": "2020 onward", "value": "2020-01-01"},
                {"label": "Swaption-vol clean history · Sep 2021", "value": "2021-09-20"},
            ], style={"fontSize": 12})),
        html.Div(style={"display": "flex", "gap": 8, "marginTop": 2}, children=[
            html.Button("Load", id="load", n_clicks=0,
                        className="ref-btn",
                        style={**btn_style(primary=True), "flex": 1}),
            html.Button("Ping DB", id="ping", n_clicks=0, className="ref-btn",
                        style=btn_style()),
        ]),
        html.Div(id="ping-out"),
    ])


def setup_tab() -> html.Div:
    return html.Div(style={"padding": "18px 24px"}, children=[
        heading("panel setup"),
        html.Div(style={"display": "grid",
                        "gridTemplateColumns": "300px minmax(0, 1fr)",
                        "gap": 26, "alignItems": "start"}, children=[
            controls(),
            dcc.Loading(
                html.Div(id="panel-out"),
                custom_spinner=shimmer_loader(image="guy.png", caption="loading"),
                overlay_style={"visibility": "visible", "opacity": 0.35},
                parent_className="research-panel-loader",
            ),
        ]),
        dcc.Store(id="research-level-data"),
        dcc.Store(id="research-level-window", data=DEFAULT_CHART_WINDOW),
    ])


def level_window_nav(current: str) -> html.Div:
    """The live dashboard's exact chart-window control, reused for research."""
    return html.Div(
        style={"display": "flex", "gap": 8, "marginBottom": 8,
               "alignItems": "center", "flexWrap": "wrap"},
        children=[
            html.Span("Chart window", style={
                "fontSize": 10, "color": DIM, "textTransform": "uppercase",
                "marginRight": 2,
            }),
            *[
                html.Button(
                    key, id=f"research-level-window-{key}", n_clicks=0,
                    className="ref-btn", style=btn_style(primary=(key == current)),
                )
                for key in WINDOW_PRESETS
            ],
        ],
    )


def level_view(
    data: pl.DataFrame, target: str, features: list[str], window: str,
    invert_features: bool,
) -> html.Div:
    """The target chart exactly as rendered in the live signal dashboard."""
    png = level_chart(data, target, features=features, invert_features=invert_features,
                      window_bars=WINDOW_PRESETS[window], fig_height=4.2)
    return html.Div([
        level_window_nav(window),
        html.Img(
            id="research-level-chart", src=f"data:image/png;base64,{png}",
            style={"width": "100%", "border": f"1px solid {BORDER}"},
        ),
    ], style={"marginBottom": 14, "maxWidth": RESEARCH_CHART_MAX_WIDTH})


def weights_view(panel, window: str) -> html.Div:
    """Rolling betas versus fixed ratios, on the main chart's window."""
    beta_cols = panel.beta_diagnostic_cols
    if not beta_cols:
        return html.Img(id="research-weight-chart", style={"display": "none"})
    dependent = dependent_leg(panel.target, panel.beta_dependent)
    scale = float(panel.target.legs[dependent])
    priors = {
        col: -float(panel.target.legs[col.removeprefix("w_")]) / scale
        for col in beta_cols
    }
    png = hedge_weights_chart(
        panel.data, beta_cols, priors, WINDOW_PRESETS[window], fig_height=3.5
    )
    return html.Div(html.Img(
        id="research-weight-chart", src=f"data:image/png;base64,{png}",
        style={"width": "100%", "border": f"1px solid {BORDER}"},
    ), style={"marginBottom": 14, "maxWidth": RESEARCH_CHART_MAX_WIDTH})


def coverage_view(coverage: pl.DataFrame) -> html.Div:
    png = coverage_chart(coverage)
    return html.Div(html.Img(
        src=f"data:image/png;base64,{png}",
        style={"width": "100%", "border": f"1px solid {BORDER}"},
    ), style={"marginBottom": 14, "maxWidth": RESEARCH_CHART_MAX_WIDTH})


# ---- the load callback's output ---------------------------------------------


def summary_bar(panel, trade, aligned_n: int, diag: dict) -> html.Div:
    legs = "  ".join(f"{w:+g}·{leg}" for leg, w in trade.legs.items())
    level = panel.data[trade.name].drop_nulls()

    def move_block(bars: int) -> html.Div:
        if len(level) <= bars:
            return stat_block(f"target move · {bars}d", "—")
        value = float(level[-1] - level[-1 - bars])
        color = C0 if value > 0 else C1 if value < 0 else TEXT
        return html.Div([
            html.Span(f"target move · {bars}d", style={
                "color": DIM, "fontSize": 10, "display": "block",
                "textTransform": "uppercase", "letterSpacing": "0.05em",
            }),
            html.Span(f"{value:+.1f} bp", style={
                "fontWeight": "bold", "fontSize": 15, "fontFamily": "monospace",
                "color": color,
            }),
        ])

    blocks = [
        stat_block("target", trade.name),
        stat_block("legs", legs if panel.weighting == "fixed" else "fitted"),
        stat_block("weighting", panel.weighting
                   + (f" · {panel.beta_lookback}d" if panel.weighting == "beta" else "")),
        stat_block("loaded bars", f"{len(panel.data):,}"),
        stat_block("usable", f"{aligned_n:,}", alert=aligned_n < 500),
        stat_block("range", f"{panel.data['ts'].min()} → {panel.data['ts'].max()}"),
        move_block(1),
        move_block(5),
        move_block(20),
    ]
    remark = diag["remark"]
    if not remark.is_empty():
        worst = remark.sort("remark_share_of_var", descending=True).row(0, named=True)
        blocks.append(stat_block(
            "hedge re-mark @60d", f"{worst['remark_share_of_var']:.0%}", alert=True))
    return html.Div(blocks, style={"display": "flex", "gap": 28,
                                   "flexWrap": "wrap", "marginBottom": 14,
                                   "paddingBottom": 12,
                                   "borderBottom": f"1px solid {BORDER}"})


def warnings_for(panel, diag: dict, aligned_n: int) -> list:
    out = []
    gaps = diag["gaps"].row(0, named=True) if not diag["gaps"].is_empty() else {}
    if gaps.get("gaps_over_5d"):
        aligned = panel.data.drop_nulls(subset=panel.columns).sort("ts")
        dates = aligned["ts"].to_list()
        pairs = list(zip(dates[:-1], dates[1:]))
        start, end = max(pairs, key=lambda pair: (pair[1] - pair[0]).days)
        out.append(note(
            f"Missing-data interval: {start} → {end} "
            f"({gaps['largest_gap_days']} calendar days). The next observation "
            f"must not be treated as a one-day move.",
            "warn"))
    stale = diag["stale"].filter(pl.col("longest_repeat_run") >= 5)
    if len(stale):
        worst = stale.sort("longest_repeat_run", descending=True).row(0, named=True)
        out.append(note(
            f"{worst['series']} repeats one value for up to "
            f"{worst['longest_repeat_run']} bars ({worst['pct_unchanged']:.1%} of "
            f"bars unchanged). A stale print reads as a real observation that "
            f"did not move, and drags any correlation toward zero.", "warn"))
    if aligned_n < 500:
        out.append(note(
            f"common sample is only {aligned_n} bars -- too thin for regime or "
            f"era splits, whatever a scorecard reports.", "warn"))
    return out


def panel_view(
    panel, trade, diag: dict, chart_window: str, invert_features: bool
) -> html.Div:
    aligned = panel.data.drop_nulls(subset=panel.columns)
    return html.Div([
        summary_bar(panel, trade, len(aligned), diag),
        *warnings_for(panel, diag, len(aligned)),
        level_view(
            panel.data, trade.name, list(panel.features), chart_window, invert_features
        ),
        weights_view(panel, chart_window),
        coverage_view(diag["coverage"]),
    ])


def board_row_picker(records: list[dict]) -> html.Div:
    """One line per discovery-board row, aligned to its index, with a Backtest
    action. table_div is the shared house table and has no room for a button
    cell, so the picker is a separate compact list rather than a table column.
    """
    if not records:
        return html.Div()
    lines = []
    for i, row in enumerate(records):
        gate_desc = (
            f"{row['gate']}:{row['gate_bucket']} w={row['gate_window']}"
            if row["gate"] != "(none)" else "ungated"
        )
        label = (
            f"#{i}  ic={row['ic']:.3f}  {row['fit_on']}  beta_lb={row['beta_lb']}  "
            f"resid_lb={row['residual_lb'] if row['residual_lb'] is not None else '—'}  "
            f"norm_lb={row['norm_lb']}  {gate_desc}"
        )
        lines.append(html.Div([
            html.Span(label, style={"fontSize": 11, "fontFamily": "monospace",
                                    "color": TEXT}),
            html.Button("Backtest", id={"type": "dis-pick", "index": i},
                        n_clicks=0, className="ref-btn",
                        style={**btn_style(), "padding": "2px 10px", "fontSize": 11,
                               "marginLeft": 12, "flexShrink": 0}),
        ], style={"display": "flex", "alignItems": "center", "gap": 8,
                  "padding": "3px 0", "borderBottom": f"1px solid {BORDER}"}))
    return html.Div(lines, style={"maxHeight": 260, "overflowY": "auto",
                                  "marginTop": 10})


def dislocation_view(
    results: pl.DataFrame, feature: str, run_info: dict
) -> tuple[html.Div, list[dict]]:
    """Compact discovery board. It deliberately does not call a cell a winner."""
    valid = results.filter(pl.col("n_obs") >= 30)
    gated = valid.filter(pl.col("gate") != "(none)")
    agreement = (
        gated.filter(pl.col("ic") > 0)
        .group_by("fit_on", "beta_lb", "residual_lb", "norm_lb", "entry_z", "horizon", "gate", "gate_bucket")
        .agg(pl.col("gate_window").n_unique().alias("positive_windows_30plus"))
    )
    board = (
        valid.join(
            agreement,
            on=["fit_on", "beta_lb", "residual_lb", "norm_lb", "entry_z", "horizon", "gate", "gate_bucket"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("gate") == "(none)").then(None)
            .otherwise(pl.col("positive_windows_30plus").fill_null(0))
            .alias("positive_windows_30plus")
        )
        .sort("ic", descending=True)
        .head(40)
    )
    # The board picker needs the real (nullable) residual_lb; the display
    # table wants the "—" placeholder. Keep records before that string cast.
    board_records = board.to_dicts()
    stats = [
        stat_block("feature", FEATURE_LABELS.get(feature, feature)),
        stat_block("cells tested", f"{len(results):,}"),
        stat_block("cells with ≥30 events", f"{len(valid):,}"),
        stat_block("gated cells", f"{len(gated):,}"),
        stat_block("backend", run_info["backend"]),
        stat_block("elapsed", f"{run_info['elapsed_s']:.2f}s"),
        stat_block("rate", f"{run_info['cells_per_second']:,.0f} cells/s"),
    ]
    cols = [
        "fit_on", "beta_lb", "residual_lb", "norm_lb", "entry_z", "horizon", "gate",
        "gate_bucket", "gate_window", "positive_windows_30plus", "ic", "hit_rate",
        "n_obs", "events_per_year",
    ]
    display = board.with_columns(pl.col("residual_lb").cast(pl.Utf8).fill_null("—"))
    view = html.Div([
        html.Div(stats, style={"display": "flex", "gap": 28, "flexWrap": "wrap",
                               "marginBottom": 14, "paddingBottom": 12,
                               "borderBottom": f"1px solid {BORDER}"}),
        note("Discovery only. 'Positive windows' means positive IC with ≥30 events; "
             "it is agreement, not proof. Exits, costs, time in market, and trade "
             "robustness are deliberately not scored here."),
        table_div(
            display.select([col for col in cols if col in display.columns]).to_pandas(),
            title="IC discovery board", max_rows=40, float_fmt=",.3f",
        ),
        note("Pick a row's Backtest button to freeze its relationship and gate "
             "for exact trade-mechanics testing below.", "dim"),
        board_row_picker(board_records),
    ])
    return view, board_records


def backtest_grid_view(grid: pl.DataFrame, selected: dict) -> html.Div:
    """Every requested entry/exit cell run through the exact Engine.

    No refitting happens here -- the relationship and gate are frozen from
    the discovery candidate; only trade mechanics vary across the grid.
    """
    best = selected["metrics"]
    stats = [
        stat_block("cells run", f"{len(grid):,}"),
        stat_block("best sharpe", f"{best['sharpe']:.2f}"),
        stat_block("best cell", f"z={best['entry_z']} · {best['exit_style']}={best['exit_param']}"),
        stat_block("total pnl (best)", f"{best['total_pnl_bps']:,.0f} bps"),
        stat_block("trades (best)", f"{best['n_trades']:,}"),
        stat_block("hit rate (best)", f"{best['hit_rate']:.1%}"),
        stat_block("time in market (best)", f"{best['time_in_market_pct']:.1%}"),
    ]
    cols = [
        "entry_z", "exit_style", "exit_param", "sharpe", "total_pnl_bps",
        "n_trades", "hit_rate", "avg_pnl_per_trade_bps", "median_pnl_bps",
        "time_in_market_pct", "max_drawdown_bps", "open_trades",
    ]
    return html.Div([
        html.Div(stats, style={"display": "flex", "gap": 28, "flexWrap": "wrap",
                               "marginBottom": 14, "paddingBottom": 12,
                               "borderBottom": f"1px solid {BORDER}"}),
        note("Exact Engine re-mark on the actual legs, transaction cost applied. "
             "One row per requested entry/exit cell."),
        table_div(
            grid.select([c for c in cols if c in grid.columns])
                .sort("sharpe", descending=True).to_pandas(),
            title="backtest grid", max_rows=60, float_fmt=",.2f",
        ),
    ])


# ---- app --------------------------------------------------------------------


def tabs() -> dcc.Tabs:
    return dcc.Tabs(id="bench-tabs", value="setup", children=[
        dcc.Tab(label="Setup", value="setup", style=TAB_STYLE,
                selected_style=SELECTED_TAB_STYLE, children=setup_tab()),
        dcc.Tab(label="Dislocation", value="dis", style=TAB_STYLE,
                selected_style=SELECTED_TAB_STYLE, children=dislocation_tab()),
        dcc.Tab(label="Relative Value", value="rv", style=TAB_STYLE,
                selected_style=SELECTED_TAB_STYLE, children=stub_tab(
                    "Relative Value", [
                        "BLOCKED: PairRVStudy.research scores the forward change in "
                        "rv_value, which is re-marked by a drifting hedge ratio. "
                        "That re-marking is 60.6% of scored variance and correlates "
                        "only 0.63 with holdable P&L (dig/audit_research.py).",
                        "Score a held position instead: d(left) - beta_entry * "
                        "d(right), with the entry beta frozen. The Setup tab's "
                        "re-marking table already computes exactly this split.",
                        "PCRelativeValueStudy fades against rv_value the same way "
                        "and needs the same fix.",
                    ])),
        dcc.Tab(label="Fair Value", value="fv", style=TAB_STYLE,
                selected_style=SELECTED_TAB_STYLE, children=stub_tab(
                    "Fair Value", [
                        "Audited clean for lookahead and alignment, and it fades "
                        "against the target level, so its scorecard is tradeable.",
                        "Do not gate on roll_lr's r2: it accumulates each bar's own "
                        "residual rather than refitting the window, and its max gap "
                        "to true in-window R2 is 0.114 on relationships whose R2 is "
                        "about 0.05.",
                        "Surface factor_condition_number prominently -- multi-factor "
                        "levels regressions on collinear rates go unstable quietly.",
                    ])),
    ])


def register_callbacks(app) -> None:
    @app.callback(
        Output("dis-gates", "value"),
        Input("dis-gates-all", "n_clicks"),
        Input("dis-gates-none", "n_clicks"),
        prevent_initial_call=True,
    )
    def _set_dislocation_gates(_all, _none):
        return list(DISLOCATION_GATES) if ctx.triggered_id == "dis-gates-all" else []

    @app.callback(
        Output("ping-out", "children"),
        Input("ping", "n_clicks"),
        prevent_initial_call=True,
    )
    def _ping(_n):
        try:
            idx, vol = last_updated(list(YIELDS.values())), swaption_last_updated()
        except Exception as exc:
            return note(f"database unreachable: {exc}", "bad")
        out = []
        for name, info in (("md.index_eod", idx), ("md.swaption_vol", vol)):
            out.append(note(f"{name}: no rows", "bad") if info is None else
                       note(f"{name}: {info['last_ts']} · written "
                            f"{info['last_written']}", "good"))
        return html.Div(out)

    @app.callback(
        Output("dis-out", "children"),
        Output("dis-run-info", "children"),
        Output("dis-board", "data"),
        Input("dis-run", "n_clicks"),
        State("research-level-data", "data"),
        State("dis-feature", "value"),
        State("dis-fit-on", "value"),
        State("dis-beta-lbs", "value"), State("dis-residual-lbs", "value"),
        State("dis-norm-lbs", "value"), State("dis-thresholds", "value"),
        State("dis-horizons", "value"), State("dis-gates", "value"),
        State("dis-gate-windows", "value"),
        prevent_initial_call=True,
        running=[
            (Output("dis-run", "children"), "Running discovery…", "Run discovery"),
            (Output("dis-run", "disabled"), True, False),
        ],
    )
    def _run_dislocation(
        _n, stored, feature, fit_on, beta_lbs, residual_lbs, norm_lbs, thresholds,
        horizons, gates, gate_windows,
    ):
        if not stored:
            return note("Load a target and this feature on Setup first.", "warn"), "", no_update
        if feature not in stored.get("features", []):
            return note(
                f"{FEATURE_LABELS.get(feature, feature)} is not in the loaded panel. "
                "Add it on Setup, then Load.", "warn"
            ), "", no_update
        try:
            target = stored["target"]
            rows = stored["rows"]
            frame = pl.DataFrame({
                "ts": pl.Series([row["ts"] for row in rows], dtype=pl.Utf8),
                target: pl.Series([row[target] for row in rows], dtype=pl.Float64),
                feature: pl.Series([row[feature] for row in rows], dtype=pl.Float64),
            }).with_columns(pl.col("ts").str.to_date())
            backend = scan_backend("auto")
            started = time.perf_counter()
            _, results = dislocation_scan(
                frame, target=target, feature=feature,
                fit_on=fit_on or ["changes"], device="auto",
                beta_lookbacks=beta_lbs or DISLOCATION_BETA_LBS,
                residual_lookbacks=residual_lbs or DISLOCATION_RESIDUAL_LBS,
                normalization_lookbacks=norm_lbs or DISLOCATION_NORM_LBS,
                thresholds=thresholds or DISLOCATION_THRESHOLDS,
                horizons=horizons or DISLOCATION_HORIZONS,
                gate_names=gates or [],
                gate_windows=gate_windows or [126, 252, 504],
            )
            elapsed_s = time.perf_counter() - started
        except Exception as exc:
            return html.Div([
                note(f"{type(exc).__name__}: {exc}", "bad"),
                html.Pre(traceback.format_exc(), style={"fontSize": 10, "color": DIM,
                                                        "whiteSpace": "pre-wrap"}),
            ]), "", no_update
        info = {
            "backend": backend,
            "elapsed_s": elapsed_s,
            "cells_per_second": len(results) / max(elapsed_s, 1e-9),
        }
        status = note(
            f"Completed · {backend} · {len(results):,} cells in {elapsed_s:.2f}s",
            "good",
        )
        view, board_records = dislocation_view(results, feature, info)
        board = {"target": target, "feature": feature, "rows": board_records}
        return view, status, board

    @app.callback(
        Output("dis-candidate", "data"),
        Output("bt-candidate", "children"),
        Input({"type": "dis-pick", "index": ALL}, "n_clicks"),
        State("dis-board", "data"),
        prevent_initial_call=True,
    )
    def _pick_dislocation_candidate(n_clicks, board):
        # Dash fires this the moment the row buttons first appear (all zero
        # clicks), before any real click. Only a genuine click should freeze
        # a candidate.
        if not board or not any(n_clicks or []):
            return no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update, no_update
        rows = board["rows"]
        idx = triggered["index"]
        if idx >= len(rows):
            return no_update, no_update
        row = rows[idx]
        candidate = {
            "target": board["target"], "feature": board["feature"],
            "fit_on": row["fit_on"], "beta_lb": int(row["beta_lb"]),
            "residual_lb": (
                None if row["residual_lb"] is None else int(row["residual_lb"])
            ),
            "norm_lb": int(row["norm_lb"]), "gate": row["gate"],
            "gate_bucket": row["gate_bucket"], "gate_window": row["gate_window"],
        }
        gate_desc = (
            f"{candidate['gate']}:{candidate['gate_bucket']} "
            f"(window={candidate['gate_window']})"
            if candidate["gate"] != "(none)" else "ungated"
        )
        label = (
            f"{candidate['target']} vs "
            f"{FEATURE_LABELS.get(candidate['feature'], candidate['feature'])} · "
            f"{candidate['fit_on']} · beta_lb={candidate['beta_lb']} · "
            f"resid_lb={candidate['residual_lb'] or '—'} · "
            f"norm_lb={candidate['norm_lb']} · {gate_desc}"
        )
        return candidate, note(f"Frozen: {label}", "good")

    @app.callback(
        Output("bt-out", "children"),
        Output("bt-run-info", "children"),
        Output("bt-grid", "data"),
        Input("bt-run", "n_clicks"),
        State("dis-candidate", "data"),
        State("research-level-data", "data"),
        State("target", "value"), State("custom", "value"),
        State("bt-entry-zs", "value"), State("bt-exit-styles", "value"),
        State("bt-time-stops", "value"), State("bt-bands", "value"),
        State("bt-revert-fracs", "value"), State("bt-stop", "value"),
        State("bt-cost", "value"),
        prevent_initial_call=True,
        running=[
            (Output("bt-run", "children"), "Running backtest…", "Run backtest grid"),
            (Output("bt-run", "disabled"), True, False),
        ],
    )
    def _run_backtest_grid(
        _n, candidate, stored, target, custom, entry_zs, exit_styles,
        time_stops, bands, revert_fracs, stop_bp, cost_bp,
    ):
        if not candidate:
            return note("Choose Backtest on a discovery row first.", "warn"), "", no_update
        if not stored:
            return note("Load a target on Setup first.", "warn"), "", no_update
        if candidate["target"] != stored["target"]:
            return note(
                "Loaded target has changed since discovery ran; rerun discovery "
                "for the current target.", "warn",
            ), "", no_update
        if candidate["feature"] not in stored.get("features", []):
            return note(
                f"{FEATURE_LABELS.get(candidate['feature'], candidate['feature'])} is "
                "not in the loaded panel. Add it on Setup, then Load.", "warn",
            ), "", no_update
        exit_params = {
            style: values
            for style, values in {
                "time": time_stops, "band": bands, "revert_frac": revert_fracs,
            }.items()
            if style in (exit_styles or [])
        }
        if not exit_params or not entry_zs:
            return note(
                "Choose at least one entry threshold and one exit style.", "warn",
            ), "", no_update
        try:
            selected_target = "custom" if custom and custom.strip() else target
            trade = resolve_target(selected_target, custom)
            rows = stored["rows"]
            needed = list(dict.fromkeys([trade.name, candidate["feature"], *trade.legs]))
            columns = {"ts": pl.Series([row["ts"] for row in rows], dtype=pl.Utf8)}
            columns.update({
                col: pl.Series([row[col] for row in rows], dtype=pl.Float64)
                for col in needed
            })
            data = pl.DataFrame(columns).with_columns(pl.col("ts").str.to_date())
            started = time.perf_counter()
            grid, selected = run_grid(
                data, trade, candidate,
                entry_zs=entry_zs, exit_params=exit_params,
                stop_loss_bps=(stop_bp or None), round_trip_cost_bps=cost_bp or 0.0,
            )
            elapsed_s = time.perf_counter() - started
        except Exception as exc:
            return html.Div([
                note(f"{type(exc).__name__}: {exc}", "bad"),
                html.Pre(traceback.format_exc(), style={"fontSize": 10, "color": DIM,
                                                        "whiteSpace": "pre-wrap"}),
            ]), "", no_update
        status = note(f"Completed · {len(grid):,} cells in {elapsed_s:.2f}s", "good")
        return backtest_grid_view(grid, selected), status, grid.to_dicts()

    @app.callback(
        Output("panel-out", "children"),
        Output("research-level-data", "data"),
        Output("target", "value"),
        Input("load", "n_clicks"),
        State("target", "value"), State("custom", "value"),
        State("weighting", "value"), State("beta-lb", "value"),
        State("beta-dependent", "value"),
        State("features", "value"), State("start", "value"),
        State("invert-feature", "value"),
        State("research-level-window", "data"),
    )
    def _load(
        _n, target, custom, weighting, beta_lb, beta_dependent, features, start,
        invert_feature, chart_window,
    ):
        try:
            # A populated basket is an explicit instruction, not decoration.
            # Reflect that in the dropdown after Load so the selected target
            # and the calculation can never disagree on screen.
            selected_target = "custom" if custom and custom.strip() else target
            trade = resolve_target(selected_target, custom)
            panel = build_panel(trade, features or [], start=start or START,
                                weighting=weighting or "fixed",
                                beta_lookback=int(beta_lb or BETA_LOOKBACK),
                                beta_dependent=beta_dependent or None)
            diag = diagnostics(panel)
        except Exception as exc:
            return html.Div([
                note(f"{type(exc).__name__}: {exc}", "bad"),
                html.Pre(traceback.format_exc(),
                         style={"fontSize": 10, "color": DIM,
                                "whiteSpace": "pre-wrap"}),
            ]), None, no_update
        chart_features = list(panel.features)
        weight_cols = panel.beta_diagnostic_cols
        # Legs ride along so a later exact-mechanics backtest (Dislocation tab)
        # can re-mark the actual tradeable instruments, not just the composite.
        store_cols = list(dict.fromkeys(
            [trade.name, *chart_features, *weight_cols, *trade.legs]
        ))
        chart_data = panel.data.select("ts", *store_cols).with_columns(
            pl.col("ts").cast(pl.Utf8)
        )
        if weight_cols:
            dependent = dependent_leg(trade, panel.beta_dependent)
            scale = float(trade.legs[dependent])
            weight_priors = {
                col: -float(trade.legs[col.removeprefix("w_")]) / scale
                for col in weight_cols
            }
        else:
            weight_priors = {}
        return panel_view(
            panel, trade, diag, chart_window or DEFAULT_CHART_WINDOW,
            "invert" in (invert_feature or []),
        ), {
            "target": trade.name,
            "features": chart_features,
            "weight_cols": weight_cols,
            "weight_priors": weight_priors,
            "invert_features": "invert" in (invert_feature or []),
            "rows": chart_data.to_dicts(),
        }, selected_target

    @app.callback(
        Output("research-level-chart", "src"),
        Output("research-weight-chart", "src"),
        Output("research-level-window", "data"),
        *[
            Output(f"research-level-window-{key}", "style")
            for key in WINDOW_PRESETS
        ],
        *[
            Input(f"research-level-window-{key}", "n_clicks")
            for key in WINDOW_PRESETS
        ],
        State("research-level-data", "data"),
        State("research-level-window", "data"),
        prevent_initial_call=True,
    )
    def _resize_level(*args):
        *clicks, chart_data, current = args
        # Adding a freshly loaded chart also adds its buttons. Dash may invoke
        # this callback at that point with every n_clicks still zero; choosing
        # the first input would silently reset the user's window to 1M.
        if not any(clicks):
            return no_update, no_update, current, *[no_update] * len(WINDOW_PRESETS)
        selected = ctx.triggered_id.removeprefix("research-level-window-")
        if not chart_data:
            return no_update, no_update, current, *[no_update] * len(WINDOW_PRESETS)
        target = chart_data["target"]
        features = chart_data.get("features", [])
        invert_features = bool(chart_data.get("invert_features"))
        rows = chart_data["rows"]
        # Dash serializes leading beta-warmup nulls as JSON null. Declare the
        # dtype rather than letting Polars infer a Null column from those rows.
        columns = {
            "ts": pl.Series([row["ts"] for row in rows], dtype=pl.Utf8),
            target: pl.Series([row[target] for row in rows], dtype=pl.Float64),
        }
        columns.update({
            feature: pl.Series([row[feature] for row in rows], dtype=pl.Float64)
            for feature in features
        })
        weight_cols = chart_data.get("weight_cols", [])
        columns.update({
            col: pl.Series([row[col] for row in rows], dtype=pl.Float64)
            for col in weight_cols
        })
        frame = pl.DataFrame(columns).with_columns(pl.col("ts").str.to_date())
        png = level_chart(
            frame, target, features=features, invert_features=invert_features,
            window_bars=WINDOW_PRESETS[selected], fig_height=4.2
        )
        weights_png = (
            f"data:image/png;base64,{hedge_weights_chart(
                frame, weight_cols, chart_data.get('weight_priors', {}),
                WINDOW_PRESETS[selected], fig_height=3.5
            )}"
            if weight_cols
            else no_update
        )
        return (
            f"data:image/png;base64,{png}", weights_png, selected,
            *[btn_style(primary=(key == selected)) for key in WINDOW_PRESETS],
        )


def build_app():
    app = make_app(title="Research", sliders=[], body=tabs)
    register_callbacks(app)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="research bench")
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    run(build_app(), port=args.port, host=args.host)


if __name__ == "__main__":
    main()
