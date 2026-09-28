"""Research app.

    python -m research.app            # http://localhost:8052
    python -m research.app --port N
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from uuid import uuid4

import polars as pl
import numpy as np
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
from research.dislocation import dislocation_scan
from backtest.lab import REGIME_GATE_BUCKETS
from research.dislocation_backtest import run_grid, signal_frame, _entry_gate
from backtest.validation import event_overlap_diagnostics
from research.artifacts import save_run
from research.saved_runs import grid_spec, input_hash, list_runs, compare_run, load_run, run_path
from research.saved_runs import target_definition, definition_match, target_label
from research import progress as work
from research.preferences import load_preferences, save_preferences
from backtest.engine import TradeDef
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
RAW_THRESHOLDS = [1.0, 2.0, 3.0, 5.0, 10.0, 15.0]
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
    "resid_phi": "Residual · OU persistence",
    "resid_half_life": "Residual · OU half-life",
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
    "width": "100%", "fontSize": 12, "fontFamily": "Arial, Helvetica, sans-serif",
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
    control.style = {"fontFamily": "Arial, Helvetica, sans-serif", "fontSize": 12,
                     **(getattr(control, "style", None) or {})}
    return html.Div([html.Span(label, style=LABEL), control],
                    id=f"{control.id}-field", className="research-field",
                    style={"marginBottom": 12, **({"display": "none"} if control.id in {"custom", "beta-lb"} else {})})


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


def loading_panel(output_id: str, task: str, caption: str) -> html.Div:
    return html.Div([
        dcc.Loading(html.Div(id=output_id),
                    # Progress owns the logo and log as one positioned unit.
                    custom_spinner=html.Div(),
                    overlay_style={"visibility": "visible", "opacity": 0.35},
                    parent_className="research-panel-loader"),
        html.Div(id=f"{task}-progress", role="status"),
    ])


def progress_view(state: dict, caption: str):
    if not state:
        return ""
    done, total = state["done"], state["total"]
    failed = state["message"].startswith("Failed:")
    finished = state["message"].startswith(("Completed ·", "Failed:"))
    fraction = min(1.0, max(0.0, done / total)) if total else None
    count = f"{done:,} / {total:,} · {fraction:.0%}" if total else "Working"
    elapsed = state["elapsed"]
    clock = f"{int(elapsed)//60}:{int(elapsed)%60:02d}"
    phase_elapsed = state.get("phase_elapsed", elapsed)
    eta_seconds = phase_elapsed / done * (total-done) if total and 2 <= done < total else None
    eta = ""
    if eta_seconds is not None:
        eta = (f" · ~{eta_seconds / 3600:.1f}h remaining" if eta_seconds >= 3600 else
               f" · ~{eta_seconds / 60:.0f}m remaining" if eta_seconds >= 60 else
               f" · ~{eta_seconds:.0f}s remaining")
    bar = html.Div(html.Div(className="research-work-bar-fill", style={
        "width": f"{fraction * 100:.1f}%" if fraction is not None else "28%",
    }), className="research-work-bar", role="progressbar", **{
        "aria-label": caption, "aria-valuemin": "0", "aria-valuemax": "100",
        **({"aria-valuenow": str(round(fraction * 100))} if fraction is not None else {}),
    })
    content = [
        html.Div(f"Stopped · {clock} elapsed" if failed else f"{count} · {clock} elapsed{eta}",
                 className="research-work-count"),
        *([] if failed else [bar]),
        html.Div([html.Div(line, className="research-work-line", title=line,
                          key=str(state.get("message_count", len(state['history'])) - len(state['history']) + i),
                          **{"data-line-id": str(state.get("message_count", len(state['history'])) - len(state['history']) + i)})
                  for i, line in enumerate(state["history"])],
                 className="research-work-log", tabIndex=0,
                 title="Scroll up for earlier messages (latest 200 retained)",
                 **{"aria-label": "Progress history, newest messages at the bottom",
                    "data-log-key": caption, "data-run-id": str(state.get('started', ''))}),
    ]
    if finished:
        return html.Details([html.Summary(state["message"]), *content], className="research-work-finished")
    return html.Div([shimmer_loader(image="guy.png", caption=caption), *content],
                    className="research-work-status", **{"aria-live": "polite"})


def stub_tab(title: str, needs: list[str]) -> html.Div:
    return html.Div(style={"padding": "18px 24px", "maxWidth": 860}, children=[
        heading(title),
        html.Div(id=f"{title.lower().replace(' ', '-')}-context"),
        note("Not built yet. Load a panel on the Setup tab first; this tab "
             "will read it. Outstanding before it can be trusted:"),
        html.Ul([html.Li(n, style={"fontSize": 12, "color": TEXT,
                                   "marginBottom": 7}) for n in needs],
                style={"marginTop": 10, "lineHeight": 1.6}),
    ])


def dislocation_tab() -> html.Div:
    """Discovery then exact trade-mechanics testing for one selected row."""
    values = [
        ("regression beta lookbacks", "dis-beta-lbs", DISLOCATION_BETA_LBS),
        ("residual windows", "dis-residual-lbs", DISLOCATION_RESIDUAL_LBS),
        ("normalization / OU lookbacks", "dis-norm-lbs", DISLOCATION_NORM_LBS),
        ("entry thresholds (z)", "dis-thresholds", DISLOCATION_THRESHOLDS),
        ("forward horizons", "dis-horizons", DISLOCATION_HORIZONS),
    ]
    controls = [html.Button("Select all sweep options", id="dis-select-all", n_clicks=0,
                           className="ref-btn", style={**btn_style(), "marginBottom": 8}),
        note("Selects every discovery and exit-grid option. Keeps target, feature, data split, costs and execution timing unchanged. Does not start a run; the full grid can be very large.", "dim"),
        html.Div(id="dis-grid-size"),
        field("feature (load it on Setup first)", dcc.Dropdown(
        id="dis-feature", value=None, clearable=False,
        options=[], style={"fontSize": 12},
    ))]
    controls += [
        html.Div(id="dis-context"),
        field("signals (select any combination)", dcc.Dropdown(id="dis-signal", value=["normalized"], multi=True, clearable=False,
            options=[{"label": "Residual / rolling standard deviation", "value": "normalized"},
                     {"label": "OU z-score", "value": "ou_z"},
                     {"label": "Raw residual (target units)", "value": "raw"}])),
        note("Both start from the same residual r. Scaled residual = r / rolling std(r). OU z = (r - fitted OU equilibrium) / rolling std(r).", "dim"),
        field("raw entry thresholds (target units; bp for rates)", dcc.Dropdown(
            id="dis-raw-thresholds", value=RAW_THRESHOLDS, multi=True, clearable=False,
            options=[{'label': str(v), 'value': v} for v in [.5, 1., 2., 3., 5., 10., 15., 20., 25., 40., 50.]])),
        note("Raw uses r directly, with no volatility scaling or centering. The normalization / OU window only affects OU gates and half-life exits for raw candidates; raw signals repeat across those windows.", "dim"),
        field("discovery period", dcc.Dropdown(id="dis-train", value=0.7, clearable=False,
            options=[{"label": "First 70%; evaluate later 30%", "value": 0.7},
                     {"label": "First 80%; evaluate later 20%", "value": 0.8},
                     {"label": "Full history (exploratory)", "value": 1.0}])),
        field("minimum discovery events", dcc.Input(id="dis-min-events", type="number", value=30, min=5, step=5, style=INPUT)),
    ]
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
        options=[{"label": str(item), "value": item} for item in sorted(set(items + (
            [20, 40, 60, 100, 130, 140, 190, 252, 360, 410, 504] if "lbs" in id_
            else [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0] if id_ == "dis-thresholds"
            else [5, 10, 20, 40, 60, 100])))],
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
                     for value in [126, 252, 504, 756, 1260, 1764]],
            style={"fontSize": 12},
        )),
        note("Percentile gates require 126 valid observations; gate lookbacks must be at least 126. "
             "Changes uses a windowed accumulated residual. Levels uses the raw "
             "level-regression residual, so residual window is shown as —. Ungated is "
             "always included. Each checked gate adds causal bucket variants.", "dim"),
        html.Button("Run discovery", id="dis-run", n_clicks=0,
                    className="ref-btn", style={**btn_style(primary=True), "width": "100%"}),
        html.Div(id="dis-run-info"),
        html.Details([
            html.Summary("Saved discovery runs"),
            dcc.Checklist(id="dis-saved-other", value=[], options=[{'label': ' Include other / unverified target definitions', 'value': 'show'}]),
            field("same target definition / feature", dcc.Dropdown(id="dis-saved-run", options=[],
                  placeholder="Load a panel to find saved runs", style={"fontSize": 12})),
            html.Div(id="dis-saved-summary"),
            html.Button("Open saved run", id="dis-saved-open", n_clicks=0,
                        className="ref-btn", style=btn_style()),
            note("Opens the original snapshot. Run discovery above always starts a fresh run.", "dim"),
        ], open=True, style={"marginTop": 16}),
        dcc.Store(id="dis-saved-query"),
    ]
    return html.Div(style={"padding": "18px 24px"}, children=[
        heading("dislocation discovery"),
        html.Div(style={"display": "grid", "gridTemplateColumns": "300px minmax(0, 1fr)",
                        "gap": 26, "alignItems": "start"}, children=[
            html.Div(controls),
            loading_panel("dis-out", "dis", "searching"),
        ]),
        html.Div(style={"borderTop": f"1px solid {BORDER}", "marginTop": 24,
                        "paddingTop": 18}, children=[
            heading("trade mechanics"),
            html.Div(style={"display": "grid", "gridTemplateColumns": "300px minmax(0, 1fr)",
                            "gap": 26, "alignItems": "start"}, children=[
                dislocation_backtest_controls(),
                loading_panel("bt-out", "bt", "backtesting"),
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
        field("entry thresholds (selected signal units)", dcc.Dropdown(
            id="bt-entry-zs", value=[0.5], multi=True,
            clearable=False,
            options=[{"label": str(v), "value": v}
                     for v in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 5., 10., 15., 20., 25., 40., 50.]],
            style={"fontSize": 12},
        )),
        field("exit styles", dcc.Checklist(
            id="bt-exit-styles", value=["time", "band", "revert_frac", "half_life_frac"],
            options=[
                {"label": " time stop (business days)", "value": "time"},
                {"label": " residual band (selected signal units)", "value": "band"},
                {"label": " reversion fraction", "value": "revert_frac"},
                {"label": " entry half-life multiple", "value": "half_life_frac"},
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
        field("exit bands (selected signal units)", dcc.Dropdown(
            id="bt-bands", value=[0.0, 0.25], multi=True, clearable=False,
            options=[{"label": str(v), "value": v} for v in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 2., 3., 5., 10., 15., 20.]],
            style={"fontSize": 12},
        )),
        field("reversion fractions", dcc.Dropdown(
            id="bt-revert-fracs", value=[0.25, 0.5, 0.75, 1.0], multi=True,
            clearable=False,
            options=[{"label": str(v), "value": v} for v in [0.25, 0.5, 0.75, 1.0]],
            style={"fontSize": 12},
        )),
        field("half-life multiples", dcc.Dropdown(id="bt-half-lives", value=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0], multi=True,
            options=[{"label": str(v), "value": v} for v in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]])),
        field("execution", dcc.Dropdown(id="bt-lag", value=1, clearable=False,
            options=[{"label": "Next observation (signal known before fill)", "value": 1},
                     {"label": "Same observation (optimistic diagnostic)", "value": 0}])),
        field("hard stops (bp)", dcc.Dropdown(
            id="bt-stop", value=[15.0], multi=True, clearable=False,
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
    preferences = load_preferences(CATALOG, {name for names in FEATURE_GROUPS.values() for name in names},
                                   DEFAULT_TARGET, DEFAULT_FEATURES)
    return html.Div(children=[
        field("target", dcc.Dropdown(
            id="target", value=preferences['target'], clearable=False,
            options=[{"label": TARGET_LABELS[n], "value": n} for n in sorted(CATALOG)]
                    + [{"label": "custom weights...", "value": "custom"}],
            style={"fontSize": 12})),
        field("custom weights", dcc.Input(
            id="custom", type="text", value=preferences['custom'], placeholder="20y:2, 10y:-1, 30y:-1",
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
        html.Details([html.Summary("Advanced beta settings"), field("beta dependent leg", dcc.Input(
            id="beta-dependent", type="text", placeholder="auto (20y for 10s20s30s)",
            debounce=True, style=INPUT))], id="beta-advanced", style={"display": "none"}),
        field("features", dcc.Dropdown(
            id="features", value=preferences['features'], multi=True,
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
            html.Button("Fill tabs", id="fill", n_clicks=0, className="ref-btn",
                        style=btn_style()),
        ]),
        html.Div(id="fill-out"),
    ])


def setup_tab() -> html.Div:
    return html.Div(style={"padding": "18px 24px"}, children=[
        heading("panel setup"),
        html.Div(style={"display": "grid",
                        "gridTemplateColumns": "300px minmax(0, 1fr)",
                        "gap": 26, "alignItems": "start"}, children=[
            controls(),
            loading_panel("panel-out", "load", "loading"),
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


def board_row_picker(records: list[dict], enabled: bool = True) -> html.Div:
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
        ic_label = f"{row['ic']:.3f}" if row['ic'] is not None else "n/a"
        label = (
            f"#{i + 1}  ic={ic_label}  {row.get('signal_kind', 'normalized')}  {row['fit_on']}  beta_lb={row['beta_lb']}  "
            f"resid_lb={row['residual_lb'] if row['residual_lb'] is not None else '—'}  "
            f"norm_lb={row['norm_lb']}  {gate_desc}"
        )
        lines.append(html.Div([
            html.Span(label, style={"fontSize": 11, "fontFamily": "monospace",
                                    "color": TEXT}),
            html.Button("Backtest", id={"type": "dis-pick", "index": i},
                        disabled=not enabled,
                        n_clicks=0, className="ref-btn",
                        style={**btn_style(), "padding": "2px 10px", "fontSize": 11,
                               "marginLeft": 12, "flexShrink": 0}),
        ], style={"display": "flex", "alignItems": "center", "gap": 8,
                  "padding": "3px 0", "borderBottom": f"1px solid {BORDER}"}))
    return html.Div(lines, style={"maxHeight": 260, "overflowY": "auto",
                                  "marginTop": 10})


def dislocation_view(
    results: pl.DataFrame, feature: str, run_info: dict, min_events: int = 30,
    data: pl.DataFrame | None = None,
) -> tuple[html.Div, list[dict]]:
    """Compact discovery board. It deliberately does not call a cell a winner."""
    if "signal_kind" not in results.columns:
        results = results.with_columns(pl.lit("normalized").alias("signal_kind"))
    discovery = results.filter(pl.col("sample") != "test") if "sample" in results.columns else results
    valid = discovery.filter(pl.col("n_obs") >= min_events)
    valid = valid.with_columns(pl.col("ic").median().over(
        ["signal_kind", "fit_on", "entry_z", "horizon", "gate", "gate_bucket", "gate_window"]
    ).alias("model_family_median_ic"))
    gated = valid.filter(pl.col("gate") != "(none)")
    agreement = (
        gated.filter(pl.col("ic") > 0)
        .group_by("signal_kind", "fit_on", "beta_lb", "residual_lb", "norm_lb", "entry_z", "horizon", "gate", "gate_bucket")
        .agg(pl.col("gate_window").n_unique().alias("positive_windows_30plus"))
    )
    board = (
        valid.join(
            agreement,
            on=["signal_kind", "fit_on", "beta_lb", "residual_lb", "norm_lb", "entry_z", "horizon", "gate", "gate_bucket"],
            how="left",
            nulls_equal=True,
        )
        .with_columns(
            pl.when(pl.col("gate") == "(none)").then(None)
            .otherwise(pl.col("positive_windows_30plus").fill_null(0))
            .alias("positive_windows_30plus")
        )
        .sort(["model_family_median_ic", "ic"], descending=True, nulls_last=True)
        .head(40)
    )
    if "sample" in results.columns and "test" in results["sample"].to_list():
        keys = ["signal_kind", "fit_on", "beta_lb", "residual_lb", "norm_lb", "entry_z", "horizon", "gate", "gate_bucket", "gate_window"]
        later = results.filter(pl.col("sample") == "test").select(
            *keys, pl.col("ic").alias("later_ic"),
            pl.col("hit_rate").alias("later_event_hit_rate"), pl.col("n_obs").alias("later_events"))
        board = board.join(later, on=keys, how="left", nulls_equal=True)
    # The board picker needs the real (nullable) residual_lb; the display
    # table wants the "—" placeholder. Keep records before that string cast.
    board_records = board.to_dicts()
    if data is not None:
        cache = {}
        cutoff = int(len(data) * run_info.get("train_fraction", 1.0))
        for row in board_records:
            key = tuple(row.get(k) for k in ("fit_on", "beta_lb", "residual_lb", "norm_lb", "signal_kind"))
            if key not in cache:
                cache[key] = signal_frame(data, target=run_info["target"], feature=feature,
                    fit_on=row["fit_on"], beta_lb=row["beta_lb"], residual_lb=row["residual_lb"],
                    norm_lb=row["norm_lb"], signal_kind=row.get("signal_kind", "normalized"))
            state = cache[key]
            z = state["signal"].to_numpy()
            prev = np.r_[np.nan, z[:-1]]
            e = row["entry_z"]
            crossed = ((z >= e) & ~(prev >= e)) | ((z <= -e) & ~(prev <= -e))
            events = np.flatnonzero(crossed & np.isfinite(z) & _entry_gate(state, row)
                                   & (np.arange(len(z)) + row["horizon"] < cutoff))
            evidence = event_overlap_diagnostics(events, row["horizon"])
            row.update({k: evidence[k] for k in ("n_non_overlapping", "overlap_fraction")})
        if board_records:
            board = pl.DataFrame(board_records, infer_schema_length=None)
    stats = [
        stat_block("result target", target_label(run_info)),
        stat_block("result source", run_info.get('result_source', 'Fresh discovery')),
        stat_block("result data period", run_info.get('data_period', 'Not recorded')),
        stat_block("feature", FEATURE_LABELS.get(feature, feature)),
        stat_block("cells tested", f"{len(results):,}"),
        stat_block(f"cells with ≥{min_events} events", f"{len(valid):,}"),
        stat_block("gated cells", f"{len(gated):,}"),
        stat_block("backend", run_info["backend"]),
        stat_block("elapsed", f"{run_info['elapsed_s']:.2f}s"),
        stat_block("rate", f"{run_info['cells_per_second']:,.0f} cells/s"),
    ]
    cols = [
        "#", "signal_kind", "signal_units", "fit_on", "beta_lb", "residual_lb", "norm_lb", "entry_z", "horizon", "gate",
        "gate_bucket", "gate_window", "positive_windows_30plus", "model_family_median_ic", "ic", "hit_rate",
        "n_obs", "events_per_year", "later_ic", "later_event_hit_rate", "later_events",
        "n_non_overlapping", "overlap_fraction",
    ]
    display = board.with_row_index("#", offset=1).with_columns(pl.col("residual_lb").cast(pl.Utf8).fill_null("—"))
    view = html.Div([
        html.Div(stats, style={"display": "flex", "gap": 28, "flexWrap": "wrap",
                               "marginBottom": 14, "paddingBottom": 12,
                               "borderBottom": f"1px solid {BORDER}"}),
        note("Ranked by median discovery IC across tested model lookbacks, then individual IC. Event hit rate is the fraction of "
             "threshold-crossing events with a favorable forward move, not winning trades. "
             "Forward windows can overlap. Later-period columns evaluate the same candidates; "
             "repeatedly choosing from them turns that period into research data."),
        table_div(
            display.select([col for col in cols if col in display.columns]).rename({"hit_rate": "event_hit_rate", "entry_z": "entry_threshold"}).to_pandas(),
            title="IC discovery board", max_rows=40, float_fmt=",.3f", sortable=True,
        ),
        note("Pick a row's Backtest button to freeze its relationship and gate "
             "for exact trade-mechanics testing below.", "dim"),
        board_row_picker(board_records, enabled=run_info.get('can_backtest', True)),
    ])
    return view, board_records


def backtest_grid_view(grid: pl.DataFrame, selected: dict) -> html.Div:
    """Every requested entry/exit cell run through the exact Engine.

    No refitting happens here -- the relationship and gate are frozen from
    the discovery candidate; only trade mechanics vary across the grid.
    """
    best = selected["metrics"]
    rank_metric = selected.get("rank_metric", "sharpe")
    stats = [
        stat_block("cells run", f"{len(grid):,}"),
        stat_block(f"selected {rank_metric}", f"{best[rank_metric]:.2f}"),
        stat_block("best cell", f"entry={best['entry_z']} ({best.get('signal_units', 'standard deviations')}) · {best['exit_style']}={best['exit_param']}"),
        stat_block("total pnl (best)", f"{best['total_pnl_bps']:,.0f} bps"),
        stat_block("trades (best)", f"{best['n_trades']:,}"),
        stat_block("winning closed trades (best)", f"{best['trade_win_rate']:.1%}"),
        stat_block("time in market (best)", f"{best['time_in_market_pct']:.1%}"),
    ]
    cols = [
        "config_id", "signal_kind", "signal_units", "entry_z", "exit_style", "exit_param", "stop_loss_bps", "sharpe", "total_pnl_bps",
        "n_trades", "trade_win_rate", "avg_pnl_per_trade_bps", "median_pnl_bps",
        "avg_holding_days", "time_in_market_pct", "max_drawdown_bps", "open_trades",
        "earlier_sharpe", "later_sharpe", "earlier_pnl_bps", "later_pnl_bps",
        "closed_pnl_bps",
    ]
    return html.Div([
        html.Div(stats, style={"display": "flex", "gap": 28, "flexWrap": "wrap",
                               "marginBottom": 14, "paddingBottom": 12,
                               "borderBottom": f"1px solid {BORDER}"}),
        note("Exact Engine re-mark on the actual legs, transaction cost applied. "
             "Every requested exit is tested. Stops are evaluated on observations, not "
             "intraday fills. Ranking uses the earlier period when a split is selected. "
             "Later daily P&L includes positions carried across the split. "
             "This is a chronological diagnostic, not a complete walk-forward validation."),
        table_div(
            grid.select([c for c in cols if c in grid.columns])
                .sort(rank_metric, descending=True).to_pandas(),
            title="backtest grid", max_rows=60, float_fmt=",.2f",
        ),
        field("inspect configuration", dcc.Dropdown(id="bt-inspect", value=str(best["config_id"]),
            options=[{"label": f"#{r['config_id']} · {r['exit_style']} {r['exit_param']} · entry {r['entry_z']} · stop {r['stop_loss_bps']}",
                      "value": str(r["config_id"])} for r in grid.iter_rows(named=True)])),
        html.Div(id="bt-detail"),
    ])


# ---- app --------------------------------------------------------------------


def tabs() -> html.Div:
    return html.Div([dcc.Store(id="research-session", data=uuid4().hex),
        dcc.Interval(id="research-progress-poll", interval=750),
        dcc.Tabs(id="bench-tabs", value="setup", children=[
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
    ])], className="research-bench")


def register_callbacks(app) -> None:
    sweep_controls = ["dis-signal", "dis-fit-on", "dis-beta-lbs", "dis-residual-lbs",
        "dis-norm-lbs", "dis-thresholds", "dis-horizons", "dis-gates", "dis-gate-windows", "dis-raw-thresholds",
        "bt-entry-zs", "bt-exit-styles", "bt-time-stops", "bt-bands", "bt-revert-fracs",
        "bt-half-lives", "bt-stop"]

    @app.callback(*[Output(control, "value", allow_duplicate=True) for control in sweep_controls],
        Input("dis-select-all", "n_clicks"),
        *[State(control, "options") for control in sweep_controls], prevent_initial_call=True)
    def _select_all_sweeps(_clicks, *options):
        return [[option['value'] for option in choices if not option.get('disabled', False)]
                for choices in options]

    @app.callback(Output("dis-grid-size", "children"),
        *[Input(control, "value") for control in sweep_controls[:10]], Input("dis-train", "value"))
    def _grid_size(signals, bases, beta, residual, norm, entries, horizons, gates, windows, raw_entries, train):
        signals = [signals] if isinstance(signals, str) else signals or []
        models = len(signals)*len(beta or [])*len(norm or [])*(
            (len(residual or []) if 'changes' in (bases or []) else 0)
            + (1 if 'levels' in (bases or []) else 0))
        threshold_count = sum(len(raw_entries or []) if s == 'raw' else len(entries or []) for s in signals)
        cells = (models // max(len(signals), 1))*threshold_count*len(horizons or [])*(
            1+len(gates or [])*len(windows or [])*len(REGIME_GATE_BUCKETS))*(2 if train < 1 else 1)
        warning = (" Very large grid: saved scores alone may require many GB of RAM; consider narrowing selections."
                   if cells > 10_000_000 else "")
        return note(f"{models:,} models · {cells:,} discovery cells (including evaluation period)."+warning,
                    'warn' if warning else 'dim')

    @app.callback(
        Output("dis-gates", "value"),
        Input("dis-gates-all", "n_clicks"),
        Input("dis-gates-none", "n_clicks"),
        prevent_initial_call=True,
    )
    def _set_dislocation_gates(_all, _none):
        return list(DISLOCATION_GATES) if ctx.triggered_id == "dis-gates-all" else []

    @app.callback(Output("dis-raw-thresholds-field", "style"), Input("dis-signal", "value"))
    def _raw_threshold_visibility(signals):
        return {"marginBottom": 12} if 'raw' in (signals or []) else {"display": "none"}

    @app.callback(Output("custom-field", "style"), Output("beta-lb-field", "style"),
                  Output("beta-advanced", "style"), Input("target", "value"), Input("weighting", "value"))
    def _weight_controls(target, weighting):
        show, hide = {"marginBottom": 12}, {"display": "none"}
        return show if target == "custom" else hide, show if weighting == "beta" else hide, show if weighting == "beta" else hide

    @app.callback(*[Output(f"{name}-field", "style") for name in
                   ("bt-time-stops", "bt-bands", "bt-revert-fracs", "bt-half-lives")], Input("bt-exit-styles", "value"))
    def _exit_controls(styles):
        return [{"marginBottom": 12} if style in (styles or []) else {"display": "none"}
                for style in ("time", "band", "revert_frac", "half_life_frac")]

    @app.callback(Output("dis-feature", "options"), Output("dis-feature", "value"),
                  Output("dis-context", "children"), Output("fill-out", "children"),
                  Output("relative-value-context", "children"), Output("fair-value-context", "children"),
                  Input("research-level-data", "data"), State("dis-feature", "value"))
    def _fill(stored, current):
        if not stored:
            return [], None, "", "", "", ""
        features = [f for f in stored.get("features", []) if f != stored["target"]]
        options = [{"label": FEATURE_LABELS.get(f, f), "value": f} for f in features]
        message = f"Current Setup: {target_label(stored)} · {len(stored['rows']):,} observations"
        context = note(f"{message} · features: {', '.join(features) or 'none selected'}")
        return options, current if current in features else (features[0] if features else None), context, note("Loaded setup carried into discovery. Choose a discovery row to fill trade mechanics.", "good"), context, context

    @app.callback(
        Output("dis-out", "children", allow_duplicate=True), Output("dis-board", "data", allow_duplicate=True),
        Output("dis-candidate", "data", allow_duplicate=True), Output("bt-out", "children", allow_duplicate=True),
        Output("bt-grid", "data", allow_duplicate=True), Output("bt-candidate", "children", allow_duplicate=True),
        Input("research-level-data", "data"), prevent_initial_call=True,
    )
    def _invalidate_panel(_stored):
        return "", None, None, "", None, note("Choose Backtest on a discovery row to fill trade mechanics.")

    @app.callback(
        Output("dis-candidate", "data", allow_duplicate=True),
        Output("bt-out", "children", allow_duplicate=True),
        Output("bt-grid", "data", allow_duplicate=True),
        Output("bt-candidate", "children", allow_duplicate=True),
        Input("dis-board", "data"), prevent_initial_call=True,
    )
    def _invalidate_candidate_for_board(_board):
        return None, "", None, note("Choose Backtest on a row from this discovery run.", "dim")

    @app.callback(*[Output(f"{name}-progress", "children") for name in ("load", "dis", "bt")],
                  Input("research-progress-poll", "n_intervals"), State("research-session", "data"))
    def _progress(_tick, session):
        return [progress_view(work.snapshot(session, name), caption)
                for name, caption in (("load", "loading"), ("dis", "searching"), ("bt", "backtesting"))]

    @app.callback(Output("bt-detail", "children"), Input("bt-inspect", "value"), State("bt-grid", "data"))
    def _inspect(config_id, stored):
        # save_run writes trades/equity/periods for every config_id to disk;
        # read just the one being inspected rather than shipping the whole
        # grid's detail to the browser upfront (see _run_backtest_grid).
        saved_dir = Path((stored or {}).get("run_path", "")) if stored else None
        if not saved_dir or not (saved_dir / "equity.parquet").is_file():
            return note("Run a backtest grid to inspect trades.")
        cid = str(config_id)
        equity = pl.read_parquet(saved_dir / "equity.parquet").filter(pl.col("config_id") == cid)
        if equity.is_empty():
            return note("No saved detail for this configuration.")
        periods_file = saved_dir / "periods.parquet"
        periods = (pl.read_parquet(periods_file).filter(pl.col("config_id") == cid).drop("config_id")
                   if periods_file.is_file() else pl.DataFrame())
        trades_file = saved_dir / "trades.parquet"
        trades = (pl.read_parquet(trades_file).filter(pl.col("config_id") == cid).drop("config_id")
                  if trades_file.is_file() else pl.DataFrame())
        return html.Div([
            dcc.Graph(figure={"data": [{"x": equity["ts"].to_list(),
                                       "y": equity["cumulative_pnl"].to_list(),
                                       "type": "scatter", "mode": "lines", "name": "Net P&L"}],
                              "layout": {"title": "Selected configuration · cumulative P&L", "yaxis": {"title": "bp"},
                                         "height": 320, "margin": {"l": 55, "r": 20, "t": 45, "b": 40}}}),
            table_div(periods.to_pandas(), title="P&L by year", max_rows=50) if len(periods) else None,
            table_div(trades.group_by("exit_reason").agg(pl.len().alias("trades"), pl.col("pnl_bps").mean().alias("mean_pnl_bps")).to_pandas(),
                      title="Exit reasons") if len(trades) else None,
            table_div(trades.to_pandas(), title="Closed trades · MAE/MFE measured on observations", max_rows=500)
            if len(trades) else note("No closed trades for this configuration."),
        ])

    @app.callback(
        Output("dis-saved-run", "options"), Output("dis-saved-run", "value"),
        Output("dis-saved-query", "data"),
        Input("research-level-data", "data"), Input("dis-feature", "value"),
        Input("dis-fit-on", "value"), Input("dis-beta-lbs", "value"),
        Input("dis-residual-lbs", "value"), Input("dis-norm-lbs", "value"),
        Input("dis-thresholds", "value"), Input("dis-horizons", "value"),
        Input("dis-gates", "value"), Input("dis-gate-windows", "value"),
        Input("dis-signal", "value"), Input("dis-train", "value"),
        Input("dis-board", "data"), State("dis-saved-run", "value"),
        Input("dis-raw-thresholds", "value"),
        Input("dis-saved-other", "value"),
    )
    def _find_saved(stored, feature, bases, beta, residual, norm, entries, horizons,
                    gates, windows, signal, train, _board, selected, raw_entries=None, show_other=None):
        if not stored or feature not in stored.get('features', []):
            return [], None, None
        request = grid_spec(bases or ['changes'], beta or DISLOCATION_BETA_LBS,
            residual or DISLOCATION_RESIDUAL_LBS, norm or DISLOCATION_NORM_LBS,
            entries or DISLOCATION_THRESHOLDS, horizons or DISLOCATION_HORIZONS,
            gates or [], windows or [126, 252, 504], signal, train, raw_entries or RAW_THRESHOLDS)
        fingerprint = input_hash(stored, feature)
        runs = list_runs(stored['target'], feature)
        definition = target_definition(stored)
        hidden = sum(definition_match(m, definition) != 'match' for m in runs)
        if not show_other:
            runs = [m for m in runs if definition_match(m, definition) == 'match']
        runs.sort(key=lambda m: (m.get('input_sha256') == fingerprint,
                                m.get('grid') == request), reverse=True)
        options = [dict(value=m['run_id'], label=f"{target_label(m)} | {definition_match(m, definition)} | {m.get('created_at', m['run_id'])} | "
                    f"{m['data_start']} to {m['data_end']} | "
                    + ('same inputs' if m.get('input_sha256') == fingerprint else 'historical / unverified')) for m in runs]
        value = selected if selected in {m['run_id'] for m in runs} else (runs[0]['run_id'] if runs else None)
        return options, value, dict(grid=request, fingerprint=fingerprint,
            target_definition=definition, hidden_definitions=hidden,
            target=stored['target'], feature=feature,
            current_start=min(r['ts'] for r in stored['rows']),
            current_end=max(r['ts'] for r in stored['rows']))

    @app.callback(Output("dis-saved-summary", "children"),
                  Input("dis-saved-run", "value"), Input("dis-saved-query", "data"))
    def _saved_summary(run_id, query):
        if not run_id or not query:
            hidden = (query or {}).get('hidden_definitions', 0)
            return note(f"No saved runs with a verified matching target definition. {hidden} other / unverified runs are hidden; use the checkbox to browse them explicitly.", "dim")
        meta = next((m for m in list_runs(query['target'], query['feature']) if m['run_id'] == run_id), None)
        if meta is None:
            return note("Saved run is no longer available.", "warn")
        explanation = note(f"Saved target: {target_label(meta)}. Current target: {target_label(query['target_definition'])}. "
                    f"Target-definition comparison: {definition_match(meta, query['target_definition'])}. "
                    f"Saved: {meta['data_start']} to {meta['data_end']} ({meta['data_rows']:,} observations). "
                    f"Loaded: {query['current_start']} to {query['current_end']}. "
                    + compare_run(meta, query['grid'], query['fingerprint']), "dim")
        settings = meta.get('grid', {})
        return html.Div([explanation, html.Details([
            html.Summary('Saved settings'),
            *[note(f'{key}: {value}', 'dim') for key, value in settings.items()],
            note(f"Leg weights: {meta.get('legs', 'not recorded')} · "
                 f"Gate minimum history: {meta.get('gate_min_history', 'not recorded')}", 'dim'),
        ], style={'marginBottom': 10})])

    @app.callback(
        Output("dis-out", "children", allow_duplicate=True),
        Output("dis-run-info", "children", allow_duplicate=True),
        Output("dis-board", "data", allow_duplicate=True),
        Input("dis-saved-open", "n_clicks"), State("dis-saved-run", "value"),
        State("dis-min-events", "value"), State("research-session", "data"),
        State("dis-saved-query", "data"),
        prevent_initial_call=True,
        running=[(Output("dis-saved-open", "disabled"), True, False)],
    )
    def _open_saved(_n, run_id, min_events, session, query=None):
        if not run_id:
            return no_update, note("Choose a saved run first.", "warn"), no_update
        try:
            work.start(session, 'dis', 'Opening saved snapshot and ranking saved scores')
            meta, frame, results = load_run(run_id)
            info = dict(meta, backend='Saved results (no scan)',
                        result_source=f'Archive {run_id}', data_period=f"{frame['ts'].min()} to {frame['ts'].max()}",
                        can_backtest=bool(meta.get('legs')) and meta.get('weighting') in {'fixed', 'beta'}
                            and (meta.get('weighting') != 'beta' or bool(meta.get('weight_columns'))),
                        elapsed_s=meta.get('elapsed_s', 0), cells_per_second=meta.get('cells_per_second', 0))
            view, records = dislocation_view(results, meta['feature'], info, int(min_events or 30), frame)
            fraction = float(meta.get('train_fraction', 1))
            split = str(frame['ts'][int(len(frame)*fraction)]) if fraction < 1 else None
            board = dict(target=meta['target'], feature=meta['feature'], rows=records,
                panel_id='archive:'+run_id, split_date=split, run_path=str(run_path(run_id)),
                archive_id=run_id, weight_columns=meta.get('weight_columns', {}),
                target_definition=target_definition(meta))
            text = (f"Saved result target: {target_label(meta)}. Opened run {run_id}: {frame['ts'].min()} to {frame['ts'].max()} · "
                    f"{len(results):,} saved cells. Original settings/data; not a new scan. "
                    + ("Exit tests use this saved snapshot and current engine code." if info['can_backtest']
                       else "View only: saved trade weights are incomplete; exit testing is disabled."))
            mismatch = query and definition_match(meta, query.get('target_definition', {})) != 'match'
            if mismatch:
                warning = f"Historical comparison only: this is NOT a verified match for Current Setup ({target_label(query['target_definition'])})."
                view = html.Div([note(warning, 'warn'), view])
                text = warning + ' ' + text
            work.update(session, 'dis', 'Completed · '+text, 1, 1)
            return view, note(text, 'warn' if mismatch else 'good'), board
        except Exception as exc:
            work.update(session, 'dis', f'Failed: {exc}')
            return no_update, note(f'Cannot open saved run: {exc}', 'bad'), no_update

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
        State("dis-signal", "value"), State("dis-train", "value"),
        State("dis-min-events", "value"), State("research-session", "data"),
        State("dis-raw-thresholds", "value"),
        prevent_initial_call=True,
        running=[
            (Output("dis-run", "children"), "Running discovery…", "Run discovery"),
            (Output("dis-run", "disabled"), True, False),
        ],
    )
    def _run_dislocation(
        _n, stored, feature, fit_on, beta_lbs, residual_lbs, norm_lbs, thresholds,
        horizons, gates, gate_windows, signal_kind, train_fraction, min_events, session, raw_entries=None,
    ):
        if not stored:
            return note("Load a target and this feature on Setup first.", "warn"), "", no_update
        if feature not in stored.get("features", []):
            return note(
                f"{FEATURE_LABELS.get(feature, feature)} is not in the loaded panel. "
                "Add it on Setup, then Load.", "warn"
            ), "", no_update
        try:
            work.start(session, "dis", "Preparing loaded panel and discovery grid")
            target = stored["target"]
            rows = stored["rows"]
            frame = pl.DataFrame({
                "ts": pl.Series([row["ts"] for row in rows], dtype=pl.Utf8),
                **{col: pl.Series([row[col] for row in rows], dtype=pl.Float64)
                   for col in dict.fromkeys([target, feature, *stored['legs'], *stored.get('weight_columns', {}).values()])},
            }).with_columns(pl.col("ts").str.to_date())
            backend = "CPU · batched gate statistics"
            started = time.perf_counter()
            frame, results = dislocation_scan(
                frame, target=target, feature=feature,
                fit_on=fit_on or ["changes"], device="cpu",
                beta_lookbacks=beta_lbs or DISLOCATION_BETA_LBS,
                residual_lookbacks=residual_lbs or DISLOCATION_RESIDUAL_LBS,
                normalization_lookbacks=norm_lbs or DISLOCATION_NORM_LBS,
                thresholds=thresholds or DISLOCATION_THRESHOLDS, raw_thresholds=raw_entries or RAW_THRESHOLDS,
                horizons=horizons or DISLOCATION_HORIZONS,
                gate_names=gates or [],
                gate_windows=gate_windows or [126, 252, 504],
                signal_kind=signal_kind, train_fraction=float(train_fraction),
                trade_legs=stored['legs'], weight_columns=stored.get('weight_columns'),
                progress=lambda done, total, message: work.update(session, "dis", message, done, total),
            )
            elapsed_s = time.perf_counter() - started
        except Exception as exc:
            work.update(session, "dis", f"Failed: {exc}", 1, 1)
            return html.Div([
                note(f"{type(exc).__name__}: {exc}", "bad"),
                html.Pre(traceback.format_exc(), style={"fontSize": 10, "color": DIM,
                                                        "whiteSpace": "pre-wrap"}),
            ]), "", no_update
        info = {
            "backend": backend,
            "elapsed_s": elapsed_s,
            "cells_per_second": len(results) / max(elapsed_s, 1e-9),
            "train_fraction": train_fraction, "target": target,
        }
        info.update(target_definition(stored))
        info.update(result_source='Fresh discovery', data_period=f"{frame['ts'].min()} to {frame['ts'].max()}")
        status = note(
            f"Completed · {backend} · {len(results):,} cells in {elapsed_s:.2f}s",
            "good",
        )
        work.update(session, "dis", "Saving all discovery cells and ranking the discovery period")
        saved = save_run("discovery", frame, results, dict(feature=feature,
            grid=grid_spec(fit_on or ["changes"], beta_lbs or DISLOCATION_BETA_LBS,
                residual_lbs or DISLOCATION_RESIDUAL_LBS, norm_lbs or DISLOCATION_NORM_LBS,
                thresholds or DISLOCATION_THRESHOLDS, horizons or DISLOCATION_HORIZONS,
                gates or [], gate_windows or [126, 252, 504], signal_kind, train_fraction, raw_entries or RAW_THRESHOLDS),
            input_sha256=input_hash(stored, feature),
            signal_kind=signal_kind, min_events=min_events,
            panel_id=stored["panel_id"],
            gate_min_history=126, **info))
        view, board_records = dislocation_view(results, feature, info, int(min_events or 30), frame)
        split_date = str(frame["ts"][int(len(frame) * float(train_fraction))]) if float(train_fraction) < 1 else None
        board = {"target": target, "feature": feature, "rows": board_records,
                 "target_definition": target_definition(stored),
                 "panel_id": stored["panel_id"], "split_date": split_date, "run_path": saved,
                 "weight_columns": stored.get("weight_columns", {})}
        work.update(session, "dis", f"Completed · {len(results):,} cells · saved {saved}", 1, 1)
        return view, status, board

    @app.callback(
        Output("dis-candidate", "data"),
        Output("bt-candidate", "children"),
        Output("bt-entry-zs", "value"),
        Input({"type": "dis-pick", "index": ALL}, "n_clicks"),
        State("dis-board", "data"),
        State("bt-entry-zs", "value"),
        prevent_initial_call=True,
    )
    def _pick_dislocation_candidate(n_clicks, board, entry_zs=None):
        # Dash fires this the moment the row buttons first appear (all zero
        # clicks), before any real click. Only a genuine click should freeze
        # a candidate.
        if not board or not any(n_clicks or []):
            return no_update, no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update, no_update, no_update
        rows = board["rows"]
        idx = triggered["index"]
        if idx >= len(rows):
            return no_update, no_update, no_update
        row = rows[idx]
        candidate = {
            "target": board["target"], "feature": board["feature"],
            "fit_on": row["fit_on"], "beta_lb": int(row["beta_lb"]),
            "residual_lb": (
                None if row["residual_lb"] is None else int(row["residual_lb"])
            ),
            "norm_lb": int(row["norm_lb"]), "gate": row["gate"],
            "gate_bucket": row["gate_bucket"], "gate_window": row["gate_window"],
            "signal_kind": row.get("signal_kind", "normalized"),
            "entry_z": row["entry_z"], "panel_id": board["panel_id"],
            "split_date": board.get("split_date"), "discovery_run": board.get("run_path"),
            "archive_id": board.get("archive_id"),
            "target_definition": board.get("target_definition", {}),
            "weight_columns": board.get("weight_columns", {}),
        }
        gate_desc = (
            f"{candidate['gate']}:{candidate['gate_bucket']} "
            f"(window={candidate['gate_window']})"
            if candidate["gate"] != "(none)" else "ungated"
        )
        label = (
            f"{target_label(candidate.get('target_definition') or {'target': candidate['target']})} vs "
            f"{FEATURE_LABELS.get(candidate['feature'], candidate['feature'])} · "
            f"{candidate['signal_kind']} ({'target units; bp for rates' if candidate['signal_kind'] == 'raw' else 'standard deviations'}) · {candidate['fit_on']} · beta_lb={candidate['beta_lb']} · "
            f"resid_lb={candidate['residual_lb'] or '—'} · "
            f"norm_lb={candidate['norm_lb']} · {gate_desc}"
        )
        return candidate, note(f"Frozen: {label}", "good"), (entry_zs if entry_zs and len(entry_zs) > 1 else [row["entry_z"]])

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
        State("bt-half-lives", "value"), State("bt-lag", "value"), State("research-session", "data"),
        prevent_initial_call=True,
        running=[
            (Output("bt-run", "children"), "Running backtest…", "Run backtest grid"),
            (Output("bt-run", "disabled"), True, False),
        ],
    )
    def _run_backtest_grid(
        _n, candidate, stored, target, custom, entry_zs, exit_styles,
        time_stops, bands, revert_fracs, stop_bp, cost_bp, half_lives, lag, session,
    ):
        if not candidate:
            return note("Choose Backtest on a discovery row first.", "warn"), "", no_update
        if candidate.get("archive_id"):
            try:
                meta, snapshot, _ = load_run(candidate['archive_id'], include_results=False)
                if (not meta.get('legs') or meta.get('weighting') not in {'fixed', 'beta'}
                        or (meta.get('weighting') == 'beta' and not meta.get('weight_columns'))):
                    return note("This legacy run lacks saved trade weights. View its results, but rerun discovery to enable reliable exit tests.", "warn"), "", no_update
                stored = dict(target=meta['target'], features=[meta['feature']], legs=meta['legs'],
                    weighting=meta.get('weighting'), beta_lookback=meta.get('beta_lookback'),
                    beta_dependent=meta.get('beta_dependent'), weight_columns=meta.get('weight_columns', {}),
                    panel_id=candidate['panel_id'], rows=snapshot.with_columns(pl.col('ts').cast(pl.Utf8)).to_dicts())
            except Exception as exc:
                return note(f"Cannot open saved snapshot: {exc}", "bad"), "", no_update
        if not stored:
            return note("Load a target on Setup first.", "warn"), "", no_update
        if candidate.get("panel_id") != stored.get("panel_id"):
            return note("The loaded panel changed. Run discovery and select a row from the new panel.", "warn"), "", no_update
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
                "half_life_frac": half_lives,
            }.items()
            if style in (exit_styles or []) and values
        }
        if not exit_params or not entry_zs:
            return note(
                "Choose at least one entry threshold and one exit style.", "warn",
            ), "", no_update
        try:
            work.start(session, "bt", "Building the frozen candidate signal and all exit combinations")
            trade = TradeDef(stored["target"], stored["legs"])
            rows = stored["rows"]
            needed = list(dict.fromkeys([trade.name, candidate["feature"], *trade.legs, *candidate.get("weight_columns", {}).values()]))
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
                stop_losses=stop_bp or [None], execution_lag=int(lag),
                progress=lambda done, total: work.update(session, "bt", f"Completed full backtest {done}/{total} · exits, stops, costs and trade logs", done, total),
            )
            elapsed_s = time.perf_counter() - started
        except Exception as exc:
            work.update(session, "bt", f"Failed: {exc}", 1, 1)
            return html.Div([
                note(f"{type(exc).__name__}: {exc}", "bad"),
                html.Pre(traceback.format_exc(), style={"fontSize": 10, "color": DIM,
                                                        "whiteSpace": "pre-wrap"}),
            ]), "", no_update
        status = note(f"Completed · {len(grid):,} cells in {elapsed_s:.2f}s", "good")
        work.update(session, "bt", "Saving every configuration's trades, equity and annual results")
        saved = save_run("exits", data, grid, dict(candidate=candidate,
            legs=stored["legs"], weighting=stored.get("weighting"),
            beta_lookback=stored.get('beta_lookback'), beta_dependent=stored.get('beta_dependent'),
            weight_columns=stored.get('weight_columns', {}), execution_lag=lag,
            cost_bps=cost_bp, stop_losses=stop_bp), selected["runs"])
        work.update(session, "bt", f"Completed · saved {saved}", len(grid), len(grid))
        # Every configuration's trades/equity/periods were just written to
        # saved (trades.parquet etc, one row per config_id) by save_run. Do
        # not also duplicate that same data -- every cell's full equity
        # curve, across up to thousands of cells -- into this Store; that
        # payload has to be JSON-serialized and shipped to the browser on
        # every run, which is what made this callback appear to hang after
        # logging "Completed". _inspect reads a single config's detail back
        # off disk instead.
        return backtest_grid_view(grid, selected), status, {"rows": grid.to_dicts(), "run_path": saved}

    @app.callback(
        Output("panel-out", "children"),
        Output("research-level-data", "data"),
        Output("target", "value"),
        Input("load", "n_clicks"),
        Input("fill", "n_clicks"),
        State("target", "value"), State("custom", "value"),
        State("weighting", "value"), State("beta-lb", "value"),
        State("beta-dependent", "value"),
        State("features", "value"), State("start", "value"),
        State("invert-feature", "value"),
        State("research-level-window", "data"),
        State("research-session", "data"),
        running=[(Output("load", "disabled"), True, False), (Output("fill", "disabled"), True, False)],
    )
    def _load(
        _n, _fill_n, target, custom, weighting, beta_lb, beta_dependent, features, start,
        invert_feature, chart_window, session,
    ):
        try:
            work.start(session, "load", "Loading market history for the target and selected features")
            selected_target = target
            trade = resolve_target(selected_target, custom)
            panel = build_panel(trade, features or [], start=start or START,
                                weighting=weighting or "fixed",
                                beta_lookback=int(beta_lb or BETA_LOOKBACK),
                                beta_dependent=(beta_dependent or None) if weighting == "beta" else None,
                                progress=lambda message: work.update(session, "load", message))
            work.update(session, "load", "Computing coverage and relationship diagnostics", 2, 4)
            diag = diagnostics(panel)
        except Exception as exc:
            work.update(session, "load", f"Failed: {exc}", 1, 1)
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
        weight_columns = {}
        if panel.weighting == "beta" and len(trade.legs) > 1:
            dep = dependent_leg(trade, panel.beta_dependent)
            scale = float(trade.legs[dep])
            weight_columns = {leg: f"held_weight_{leg}" for leg in trade.legs}
            chart_data = chart_data.with_columns(*[
                (pl.lit(scale) if leg == dep else -scale * pl.col(f"w_{leg}")).alias(col)
                for leg, col in weight_columns.items()
            ])
        if weight_cols:
            dependent = dependent_leg(trade, panel.beta_dependent)
            scale = float(trade.legs[dependent])
            weight_priors = {
                col: -float(trade.legs[col.removeprefix("w_")]) / scale
                for col in weight_cols
            }
        else:
            weight_priors = {}
        work.update(session, "load", "Rendering charts and filling the research panel", 3, 4)
        view = panel_view(
            panel, trade, diag, chart_window or DEFAULT_CHART_WINDOW,
            "invert" in (invert_feature or []),
        )
        try:
            save_preferences(selected_target, chart_features, custom)
        except OSError as exc:
            view = html.Div([note(f"Panel loaded, but Setup preferences could not be saved: {exc}", "warn"), view])
        work.update(session, "load", "Completed · setup available in discovery", 4, 4)
        return view, {
            "target": trade.name,
            "legs": trade.legs, "weighting": panel.weighting,
            "weight_columns": weight_columns,
            "beta_dependent": dependent_leg(trade, panel.beta_dependent) if panel.weighting == 'beta' else None,
            "beta_lookback": panel.beta_lookback,
            "panel_id": uuid4().hex,
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
        Input("invert-feature", "value"),
        State("research-level-data", "data"),
        State("research-level-window", "data"),
        prevent_initial_call=True,
    )
    def _resize_level(*args):
        *clicks, invert_feature, chart_data, current = args
        inversion_changed = ctx.triggered_id == "invert-feature"
        # Adding a freshly loaded chart also adds its buttons. Dash may invoke
        # this callback at that point with every n_clicks still zero; choosing
        # the first input would silently reset the user's window to 1M.
        if not inversion_changed and not any(clicks):
            return no_update, no_update, current, *[no_update] * len(WINDOW_PRESETS)
        selected = (current or DEFAULT_CHART_WINDOW) if inversion_changed else ctx.triggered_id.removeprefix("research-level-window-")
        if not chart_data:
            return no_update, no_update, current, *[no_update] * len(WINDOW_PRESETS)
        target = chart_data["target"]
        features = chart_data.get("features", [])
        invert_features = "invert" in (invert_feature or [])
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
            if weight_cols and not inversion_changed
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
