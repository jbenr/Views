"""Research app.

    python -m research.app            # http://localhost:8052
    python -m research.app --port N
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import re
import threading
import time
import traceback
from pathlib import Path
from uuid import uuid4

import polars as pl
import numpy as np
from stats import fit_lr
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from dashboard.charts import (
    WINDOW_PRESETS,
    coverage_chart,
    curve_fit_chart,
    hedge_weights_chart,
    level_chart,
    regression_scatter_chart,
    residual_bars_chart,
)
from research.curve_pca import PCA_TENORS, PCA_WINDOWS, pca_curve

from research.panel import (
    BETA_LOOKBACK,
    CATALOG,
    START,
    PACKAGE_LEGS,
    TRADEABLE_LEGS,
    beta_package,
    structure_definition,
    VOLS,
    YIELDS,
    build_panel,
    dependent_leg,
    diagnostics,
    is_derived,
    parse_derived,
    resolve_target,
)
from research.dislocation import BOARD_ROWS_SAVED, RANK_COLUMNS, RANK_RULES, discovery_compact, placebo_scan
from backtest.lab import REGIME_GATE_BUCKETS
from research.dislocation_backtest import detail_for, exit_cells, run_vector_grid, signal_frame, _entry_gate
from backtest.validation import event_overlap_diagnostics
from research.artifacts import append_details, save_run
from research.saved_runs import (board_rules, grid_spec, input_hash, list_exit_runs, list_runs, compare_run, load_board,
                                 load_run, run_path)
from research.saved_runs import target_definition, definition_match, target_label
from research import artifacts, jobs, progress as work
from research.preferences import load_controls, load_preferences, save_controls, save_preferences
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
    "SOFR OIS": ["sofr2", "sofr5", "sofr10", "sofr20", "sofr30"],
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
    "sofr5": "5Y SOFR OIS",
    "sofr10": "10Y SOFR OIS",
    "sofr20": "20Y SOFR OIS",
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

# Treasury curves read best by their names; single non-Treasury series by their labels.
TARGET_LABELS = {name: FEATURE_LABELS[name] if name in FEATURE_LABELS and name not in YIELDS else name
                 for name in CATALOG}


def _target_group(name: str) -> tuple[int, str]:
    """(order, group) of a catalog target, so the list reads Treasuries, curves, flies, then the rest."""
    legs = CATALOG[name].legs
    if name.startswith("swsp"):
        return 3, "Swap spread"
    if name.startswith("be"):
        return 4, "Breakeven"
    if name.startswith("real"):
        return 5, "TIPS"
    return {1: (0, "Treasury"), 2: (1, "Curve"), 3: (2, "Fly")}[len(legs)]


# Grouped, then by tenor (every number in the name, so 2s10s comes before 10s30s).
TARGET_OPTIONS = [
    {"label": f"{_target_group(n)[1]} · {TARGET_LABELS[n]}", "value": n}
    for n in sorted(CATALOG, key=lambda n: (_target_group(n)[0],
                                            [int(x) for x in re.findall(r"\d+", n)]))
]

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
    if isinstance(control, dcc.Dropdown):
        if getattr(control, "multi", False):
            control.clearable = True
        options = getattr(control, "options", [])
        if options and all(isinstance(o.get("value"), (int, float)) for o in options):
            control.options = sorted(options, key=lambda o: o["value"])
    control.style = {"fontFamily": "Arial, Helvetica, sans-serif", "fontSize": 12,
                     **(getattr(control, "style", None) or {})}
    return html.Div([html.Span(label, style=LABEL), control],
                    id=f"{control.id}-field", className="research-field",
                    style={"marginBottom": 12, **({"display": "none"} if control.id in {"custom", "beta-lb"} else {})})


def note(text: str, tone: str = "dim") -> html.Div:
    colour = {"dim": DIM, "warn": ORANGE, "bad": C0, "good": C1}[tone]
    return html.Div(text, style={"fontSize": 11, "color": colour,
                                 "marginTop": 6, "lineHeight": 1.5})


TABLE_HEIGHT = 560  # about 20 rows under a sticky header; the rest scrolls


def info(summary: str, *text: str) -> html.Details:
    """Explanatory text folded behind a one-line summary."""
    return html.Details([html.Summary(summary), *[note(t, "dim") for t in text]],
                        className="research-info")


def section(title: str, children: list, open: bool = True) -> html.Details:
    """A collapsible group of controls."""
    return html.Details([html.Summary(title), html.Div(children, className="research-section-body")],
                        open=open, className="research-section")


def table(df, title=None, **kwargs) -> html.Div:
    """House table that shows about 20 rows and scrolls for the rest."""
    return table_div(df, title=title, max_rows=kwargs.pop("max_rows", 2000), max_height=TABLE_HEIGHT, **kwargs)


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


EXIT_CHOICES = {
    "time-stops": ([5, 10, 20, 40, 60], [3, 5, 10, 15, 20, 30, 40, 60, 80, 100]),
    "bands": ([0.0, 0.25], [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 2., 3., 5., 10., 15., 20.]),
    "revert-fracs": ([0.25, 0.5, 0.75, 1.0], [0.25, 0.5, 0.75, 1.0]),
    "half-lives": ([0.5, 0.75, 1.0, 1.5, 2.0, 3.0], [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]),
    "caps": ([0.0], [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]),
    "signal-stops": ([0.0], [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]),
    "stops": ([15.0], [0.0, 10.0, 15.0, 25.0, 40.0, 60.0]),
}


def exit_controls(prefix: str, styles: list, overrides: dict | None = None) -> list:
    """The exit menu shared by discovery and Trade mechanics.

    Same styles, parameters, caps and stops in both, so a discovery row's exit
    is one Trade mechanics can reproduce and refine exactly. 0 means none for
    caps and stops.
    """
    overrides = overrides or {}

    def multi(label, key):
        value, choices = overrides.get(key, EXIT_CHOICES[key])
        return field(label, dcc.Dropdown(
            id={("dis", "time-stops"): "dis-horizons", ("bt", "stops"): "bt-stop"}.get((prefix, key), f"{prefix}-{key}"),
            value=value, multi=True, clearable=False,
            options=[{"label": "none" if v == 0 and key in ("caps", "signal-stops", "stops") else str(v), "value": v}
                     for v in choices]))

    return [
        field("exit styles", dcc.Checklist(
            id=f"{prefix}-exit-styles", value=styles,
            options=[
                {"label": " time stop (bars)", "value": "time"},
                {"label": " residual band (signal units)", "value": "band"},
                {"label": " reversion fraction", "value": "revert_frac"},
                {"label": " entry half-life multiple", "value": "half_life_frac"},
            ],
            labelStyle={"display": "block", "fontSize": 12, "marginBottom": 4, "color": TEXT},
            inputStyle={"marginRight": 5})),
        multi("time stops (bars)", "time-stops"),
        multi("exit bands (signal units)", "bands"),
        multi("reversion fractions", "revert-fracs"),
        multi("half-life multiples", "half-lives"),
        multi("half-life cap on band / reversion exits (× entry half-life)", "caps"),
        multi("signal stop (signal units beyond entry)", "signal-stops"),
        multi("hard stops (bp)", "stops"),
        info("About the exits",
             "Time exits hold a fixed number of bars. Band exits leave when the signal comes back inside the band "
             "(0 = back to equilibrium). Reversion fraction leaves once that share of the entry dislocation has "
             "reverted. Half-life multiple holds for that multiple of the half-life estimated at entry.",
             "The half-life cap bounds band and reversion exits at that multiple of the entry half-life, so a trade "
             "that never reverts cannot sit forever. The signal stop leaves when the dislocation extends that far "
             "beyond where it was entered (the relationship breaking); hard stops are a P&L loss in bp."),
    ]


def dislocation_tab() -> html.Div:
    """Discovery by real backtests, then trade-mechanics testing for one selected row."""
    def multi(label, id_, value, choices):
        return field(label, dcc.Dropdown(
            id=id_, value=value, multi=True, clearable=False,
            options=[{"label": str(v), "value": v} for v in sorted(set(value) | set(choices))]))

    lookbacks = [20, 40, 60, 100, 130, 140, 190, 252, 360, 410, 504]
    controls = [
        section("Signal model", [
            field("signals", dcc.Dropdown(id="dis-signal", value=["normalized"], multi=True, clearable=False,
                options=[{"label": "Residual / rolling std", "value": "normalized"},
                         {"label": "OU z-score", "value": "ou_z"},
                         {"label": "Raw residual (target units)", "value": "raw"}])),
            field("regression basis", dcc.Checklist(
                id="dis-fit-on", value=["changes", "levels"],
                options=[{"label": label, "value": value} for value, label in DISLOCATION_FIT_BASES.items()],
                labelStyle={"display": "block", "fontSize": 12, "marginBottom": 4, "color": TEXT},
                inputStyle={"marginRight": 5})),
            multi("regression beta lookbacks", "dis-beta-lbs", DISLOCATION_BETA_LBS, lookbacks),
            multi("residual windows (changes basis)", "dis-residual-lbs", DISLOCATION_RESIDUAL_LBS, lookbacks),
            multi("normalization / OU lookbacks", "dis-norm-lbs", DISLOCATION_NORM_LBS, lookbacks),
            info("About the signals and bases",
                 "All signals start from the same residual r. Scaled residual = r / rolling std(r). "
                 "OU z = (r - fitted OU equilibrium) / rolling std(r). Raw uses r directly in target units.",
                 "Changes fits daily moves and accumulates the unexplained part over the residual window. "
                 "Levels uses the level-regression residual, so it has no residual window (shown as —).",
                 "For raw signals the normalization / OU window only affects OU gates, so raw cells repeat across it."),
        ]),
        section("Entries", [
            multi("entry thresholds (z)", "dis-thresholds", DISLOCATION_THRESHOLDS, [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]),
            field("raw entry thresholds (target units)", dcc.Dropdown(
                id="dis-raw-thresholds", value=RAW_THRESHOLDS, multi=True, clearable=False,
                options=[{"label": str(v), "value": v} for v in [.5, 1., 2., 3., 5., 10., 15., 20., 25., 40., 50.]])),
            field("minimum discovery-period trades", dcc.Input(id="dis-min-events", type="number", value=30, min=5, step=5, style=INPUT)),
            info("How discovery trades",
                 "Every cell is a real backtest: enter on the first threshold crossing, hold one position at a time, "
                 "and leave by the exit rule chosen under Exits, with the fills, costs and entry-frozen beta weights "
                 "below. Pick a row to carry its exact trade into Trade mechanics and refine it there."),
        ]),
        section("Exits", exit_controls("dis", ["time"], {
            "time-stops": (DISLOCATION_HORIZONS, EXIT_CHOICES["time-stops"][1]),
            "bands": ([0.0], EXIT_CHOICES["bands"][1]),
            "revert-fracs": ([0.5], EXIT_CHOICES["revert-fracs"][1]),
            "half-lives": ([1.0], EXIT_CHOICES["half-lives"][1]),
            "stops": ([0.0], EXIT_CHOICES["stops"][1]),
        })),
        section("Gates", [
            field("gate conditions", dcc.Dropdown(
                id="dis-gates", value=DEFAULT_DISLOCATION_GATES, multi=True, clearable=True,
                options=[{"label": label, "value": name} for name, label in DISLOCATION_GATES.items()])),
            html.Div(style={"display": "flex", "gap": 7, "marginTop": -6, "marginBottom": 12}, children=[
                html.Button("All", id="dis-gates-all", n_clicks=0, className="ref-btn", style=btn_style()),
                html.Button("None", id="dis-gates-none", n_clicks=0, className="ref-btn", style=btn_style()),
                html.Span("None = ungated only", style={"fontSize": 10, "color": DIM, "alignSelf": "center"}),
            ]),
            field("gate percentile lookbacks", dcc.Dropdown(
                id="dis-gate-windows", value=[126, 252, 504], multi=True, clearable=False,
                options=[{"label": str(v), "value": v} for v in [126, 252, 504, 756, 1260, 1764]])),
            info("About gates",
                 "Each checked condition is ranked against its own trailing history (causal percentiles) and "
                 "adds 12 regime buckets per lookback. Ungated cells are always included. Percentile gates need "
                 "126 valid observations, so lookbacks must be at least 126."),
        ], open=False),
        section("Costs & execution", [
            field("round-trip cost (bp)", dcc.Dropdown(id="dis-cost", value=0.1, clearable=False,
                options=[{"label": str(v), "value": v} for v in [0.0, 0.1, 0.25, 0.5, 1.0]])),
            field("execution", dcc.Dropdown(id="dis-lag", value=1, clearable=False,
                options=[{"label": "Next observation (signal known before fill)", "value": 1},
                         {"label": "Same observation (optimistic diagnostic)", "value": 0}])),
        ]),
        section("Sample & validation", [
            field("discovery period", dcc.Dropdown(id="dis-train", value=0.7, clearable=False,
                options=[{"label": "First 70%; hold out later 30%", "value": 0.7},
                         {"label": "First 80%; hold out later 20%", "value": 0.8},
                         {"label": "Full history (exploratory)", "value": 1.0}])),
            field("cross-validation blocks", dcc.Dropdown(id="dis-cv", value=0, clearable=False,
                options=[{"label": "Off", "value": 0}, *[{"label": f"{k} blocks", "value": k} for k in (3, 5, 8, 10)]])),
            field("rank by", dcc.Dropdown(id="dis-rank", value="family", clearable=False,
                options=[{"label": label, "value": value} for value, label in RANK_RULES.items()])),
            field("placebo runs", dcc.Dropdown(id="dis-placebo", value=0, clearable=False,
                options=[{"label": "Off", "value": 0}, *[{"label": str(k), "value": k} for k in (5, 10, 25, 50)]])),
            info("About validation",
                 "Ranking only ever uses the discovery period; the held-out later period is reported, never ranked on.",
                 "Cross-validation splits the discovery period into blocks, dropping the first longest-holding-period "
                 "bars after each boundary so no trade is scored in a block it was not opened in. Each cell gets its "
                 "block Sharpes; the selection checks then ask whether picking the top cell on some blocks picks a "
                 "good cell on a block it never saw (out-of-fold, and walk-forward using earlier blocks only).",
                 "Placebo runs rerun the whole grid with the feature's daily changes circularly shifted, which keeps "
                 "its behaviour but breaks any link to the target. If the real top score does not beat the "
                 "placebos' top scores, the search is finding noise. Each placebo costs one full discovery run."),
        ]),
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": 8, "alignItems": "center", "margin": "4px 0 6px"}, children=[
            html.Button("Select all sweep options", id="dis-select-all", n_clicks=0, className="ref-btn", style=btn_style()),
        ]),
        info("What Select all does",
             "Selects every discovery and exit-grid option. Keeps target, feature, data split, costs, validation and "
             "execution unchanged, and does not start a run."),
        html.Div(id="dis-grid-size"),
        html.Button("Run discovery", id="dis-run", n_clicks=0,
                    className="ref-btn", style={**btn_style(primary=True), "width": "100%", "marginTop": 6}),
        html.Div(id="dis-run-info"),
        section("Saved discovery runs", [
            dcc.Checklist(id="dis-saved-other", value=[], options=[{"label": " Include other / unverified target definitions", "value": "show"}],
                          style={"fontSize": 11, "marginBottom": 8}),
            field("same target definition / feature", dcc.Dropdown(id="dis-saved-run", options=[],
                  placeholder="Load a panel to find saved runs")),
            html.Div(id="dis-saved-summary"),
            html.Button("Open saved run", id="dis-saved-open", n_clicks=0, className="ref-btn", style=btn_style()),
            note("Opens the original snapshot. Run discovery always starts a fresh run.", "dim"),
        ], open=False),
        dcc.Store(id="dis-saved-query"),
    ]
    return remember_controls(html.Div(style={"padding": "18px 24px"}, children=[
        html.Div([
            html.Div(id="dis-context", style={"flex": "1 1 auto", "minWidth": 0}),
            field("feature (several loaded: choose one)", dcc.Dropdown(id="dis-feature", value=None, clearable=False,
                                                                      options=[], style={"width": 260})),
        ], className="research-banner"),
        heading("dislocation discovery"),
        html.Div(style={"display": "grid", "gridTemplateColumns": "320px minmax(0, 1fr)",
                        "gap": 26, "alignItems": "start"}, children=[
            html.Div(controls, className="research-controls"),
            loading_panel("dis-out", "dis", "searching"),
        ]),
        html.Div(style={"borderTop": f"1px solid {BORDER}", "marginTop": 24,
                        "paddingTop": 18}, children=[
            heading("trade mechanics"),
            html.Div(style={"display": "grid", "gridTemplateColumns": "320px minmax(0, 1fr)",
                            "gap": 26, "alignItems": "start"}, children=[
                dislocation_backtest_controls(),
                html.Div([loading_panel("bt-out", "bt", "backtesting"), html.Div(id="bt-detail")]),
            ]),
        ]),
        dcc.Store(id="dis-board"),
        dcc.Store(id="dis-sort"),
        dcc.Store(id="dis-candidate"),
        dcc.Store(id="bt-grid"),
        dcc.Store(id="controls-saved"),
        dcc.Store(id="dis-rendered"),
        dcc.Store(id="bt-rendered"),
        dcc.Store(id="dis-ready"),
        dcc.Store(id="bt-ready"),
    ]), load_controls())


REMEMBERED = [
    "dis-signal", "dis-fit-on", "dis-beta-lbs", "dis-residual-lbs", "dis-norm-lbs", "dis-thresholds",
    "dis-raw-thresholds", "dis-min-events", "dis-exit-styles", "dis-horizons", "dis-bands", "dis-revert-fracs",
    "dis-half-lives", "dis-caps", "dis-signal-stops", "dis-stops", "dis-gates", "dis-gate-windows", "dis-cost",
    "dis-lag", "dis-train", "dis-cv", "dis-rank", "dis-placebo",
    "bt-entry-zs", "bt-exit-styles", "bt-time-stops", "bt-bands", "bt-revert-fracs", "bt-half-lives", "bt-caps",
    "bt-signal-stops", "bt-stop", "bt-cost", "bt-lag",
    "weighting", "beta-lb", "custom-kind", "custom-leg-1", "custom-leg-2", "custom-leg-3",
    "fv-pca-tenors", "fv-pca-start", "fv-pca-window", "fv-pca-factors",
]


def remember_controls(tree, saved: dict):
    """Give each REMEMBERED control its last saved value, if that value is still offered.

    Values no longer among a control's options are dropped; a multi-select
    left empty, or a single choice that is gone, keeps its default. The
    walk sets values in place and returns the tree.
    """
    def visit(node):
        if not hasattr(node, "to_plotly_json"):
            return
        key = getattr(node, "id", None)
        if isinstance(key, str) and key in REMEMBERED and key in saved:
            value, options = saved[key], getattr(node, "options", None)
            if options is None:  # free inputs (e.g. minimum trades)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    node.value = value
            else:
                allowed = [o["value"] if isinstance(o, dict) else o for o in options]
                if isinstance(node.value, list) and isinstance(value, list):
                    kept = [v for v in value if v in allowed]
                    if kept or not value:
                        node.value = kept
                elif not isinstance(node.value, list) and value in allowed:
                    node.value = value
        children = getattr(node, "children", None)
        for child in children if isinstance(children, (list, tuple)) else [children]:
            visit(child)
    visit(tree)
    return tree


def dislocation_backtest_controls() -> html.Div:
    """Controls deliberately limited to execution choices, not model refitting."""
    def multi(label, id_, value, choices):
        return field(label, dcc.Dropdown(id=id_, value=value, multi=True, clearable=False,
                                         options=[{"label": str(v), "value": v} for v in choices]))

    return html.Div(className="research-controls", children=[
        html.Div(id="bt-candidate", children=note(
            "Pick a discovery row's Backtest button to freeze its relationship and gate.", "dim")),
        html.Button("Select all", id="bt-select-all", n_clicks=0, className="ref-btn",
                    style={**btn_style(), "marginBottom": 8}),
        section("Entries", [
            multi("entry thresholds (selected signal units)", "bt-entry-zs", [0.5],
                  [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 5., 10., 15., 20., 25., 40., 50.]),
        ]),
        section("Exits & stops", exit_controls("bt", ["time", "band", "revert_frac", "half_life_frac"])),
        section("Costs & execution", [
            field("round-trip cost (bp)", dcc.Dropdown(
                id="bt-cost", value=0.1, clearable=False,
                options=[{"label": str(v), "value": v} for v in [0.0, 0.1, 0.25, 0.5, 1.0]])),
            field("execution", dcc.Dropdown(id="bt-lag", value=1, clearable=False,
                options=[{"label": "Next observation (signal known before fill)", "value": 1},
                         {"label": "Same observation (optimistic diagnostic)", "value": 0}])),
        ]),
        html.Button("Run backtest grid", id="bt-run", n_clicks=0,
                    className="ref-btn", style={**btn_style(primary=True), "width": "100%"}),
        html.Div(id="bt-run-info"),
        section("Saved backtest grids", [
            field("saved grid", dcc.Dropdown(id="bt-saved-run", options=[], placeholder="No saved grids yet")),
            html.Button("Open saved grid", id="bt-saved-open", n_clicks=0, className="ref-btn", style=btn_style()),
            note("Reopens a grid's saved results; inspecting a configuration reruns it exactly.", "dim"),
        ], open=False),
    ])


# ---- setup tab --------------------------------------------------------------


def _valid_derived(name: str) -> bool:
    try:
        return parse_derived(name).name == name
    except ValueError:
        return False


CUSTOM_LEG_ROLES = {"spread": ["short leg (−1)", "long leg (+1)"],
                    "fly": ["wing (−1)", "belly (+2)", "wing (−1)"]}


def custom_builder() -> html.Div:
    """Pick a structure and its legs; the custom definition below is written from them."""
    legs = [{"label": f"{leg} · {FEATURE_LABELS.get(leg, leg)}", "value": leg} for leg in TRADEABLE_LEGS]
    return html.Div([
        field("custom structure", dcc.RadioItems(
            id="custom-kind", value="fly", inline=True,
            options=[{"label": " spread", "value": "spread"}, {"label": " fly", "value": "fly"},
                     {"label": " free basket (type below)", "value": "basket"}],
            labelStyle={"marginRight": 14, "fontSize": 12, "color": TEXT}, inputStyle={"marginRight": 4})),
        *[html.Div([html.Span(role, id=f"custom-leg-{i}-label", style=LABEL),
                    dcc.Dropdown(id=f"custom-leg-{i}", value=default, clearable=False, options=legs,
                                 style={"fontFamily": "Arial, Helvetica, sans-serif", "fontSize": 12})],
                   id=f"custom-leg-{i}-box", className="research-field", style={"marginBottom": 12})
          for i, (role, default) in enumerate(zip(CUSTOM_LEG_ROLES["fly"], ["5y", "7y", "10y"]), start=1)],
        html.Div(note("Treasuries, SOFR swaps and swap spreads can all be legs. Beta weighting hedges the belly "
                      "(or the long leg) with fitted betas on the others.", "dim"), style={"marginBottom": 12}),
    ], id="custom-builder", style={"display": "none"})


def controls() -> html.Div:
    preferences = load_preferences(CATALOG, {name for names in FEATURE_GROUPS.values() for name in names},
                                   DEFAULT_TARGET, DEFAULT_FEATURES, extra_feature=_valid_derived)
    derived = [f for f in preferences['features'] if is_derived(f)]
    preferences['features'] = [f for f in preferences['features'] if not is_derived(f)]
    return html.Div(children=[
        field("target", dcc.Dropdown(
            id="target", value=preferences['target'], clearable=False,
            options=TARGET_OPTIONS
                    + [{"label": "custom spread / fly / basket...", "value": "custom"}],
            style={"fontSize": 12})),
        custom_builder(),
        field("custom definition (name = leg:weight, ...)", dcc.Input(
            id="custom", type="text", value=preferences['custom'], placeholder="5s7s10s = 5y:-1, 7y:2, 10y:-1",
            debounce=True, style=INPUT)),
        html.Div(id="custom-error"),
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
        field("derived features (residual / beta-weighted)", dcc.Input(
            id="derived-features", type="text", value="; ".join(derived), debounce=True, style=INPUT,
            placeholder="10y ~ 2y; swsp10 ~ 10y; 20y ~ 10y + 30y @ 63")),
        note("Each is the dependent series' move left over after a rolling changes-beta hedge (default 126d), "
             "using yesterday's betas and accumulated into a level. '10y ~ 2y' is beta-weighted 2s10s.", "dim"),
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
    return remember_controls(html.Div(style={"padding": "18px 24px"}, children=[
        heading("panel setup"),
        html.Div(style={"display": "grid",
                        "gridTemplateColumns": "300px minmax(0, 1fr)",
                        "gap": 26, "alignItems": "start"}, children=[
            controls(),
            html.Div([loading_panel("panel-out", "load", "loading"), scatter_section()]),
        ]),
        dcc.Store(id="research-level-data"),
        dcc.Store(id="research-level-window", data=DEFAULT_CHART_WINDOW),
    ]), load_controls())


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


SCATTER_WINDOWS = [63, 126, 252, 504]


def scatter_section() -> html.Div:
    """Controls and chart for the target-vs-feature regression scatter (filled once a panel loads)."""
    def choice(id_, label, value, options, width):
        return html.Div(field(label, dcc.Dropdown(id=id_, value=value, clearable=False, options=options,
                                                  style={"width": width})), style={"flex": "0 0 auto"})
    return html.Div([
        heading("regression scatter"),
        html.Div([
            choice("scatter-feature", "feature", None, [], 220),
            choice("scatter-basis", "basis", "levels", [
                {"label": "Levels (target vs feature)", "value": "levels"},
                {"label": "Daily changes (Δ target vs Δ feature)", "value": "changes"}], 260),
            choice("scatter-window", "latest regression window", 126,
                   [{"label": f"{w} days", "value": w} for w in SCATTER_WINDOWS], 170),
            choice("scatter-past", "past", "previous", [
                {"label": "Previous window (same length)", "value": "previous"},
                {"label": "All earlier history", "value": "all"},
                {"label": "None", "value": "none"}], 230),
        ], style={"display": "flex", "gap": 16, "flexWrap": "wrap", "alignItems": "flex-end"}),
        html.Div(id="scatter-out"),
    ], style={"marginTop": 18, "paddingTop": 14, "borderTop": f"1px solid {BORDER}",
              "maxWidth": RESEARCH_CHART_MAX_WIDTH})


def scatter_fits(data: pl.DataFrame, target: str, feature: str, basis: str, window: int, past: str) -> dict:
    """The scatter's numbers: the latest ``window`` days and an older stretch, each with its OLS fit,
    and today's residual against the latest fit (also in residual standard deviations)."""
    frame = data.select("ts", target, feature).drop_nulls().sort("ts")
    if basis == "changes":
        frame = frame.with_columns(pl.col(target).diff(), pl.col(feature).diff()).drop_nulls()
    latest = frame.tail(window)
    older = frame.head(max(len(frame) - window, 0))
    deeper = {"previous": older.tail(window), "all": older, "none": older.head(0)}[past]
    fits = {"latest": fit_lr(latest[feature], latest[target]), "past": fit_lr(deeper[feature], deeper[target])}
    now = latest.tail(1).row(0, named=True) if len(latest) else None
    residual = (now[target] - (fits["latest"]["alpha"] + fits["latest"]["beta"] * now[feature])) if now else np.nan
    std = fits["latest"]["resid_std"]
    return dict(rows=len(frame), latest=latest, deeper=deeper, fits=fits, now=now, residual=residual,
                z=residual / std if std else np.nan)


def regression_scatter(data: pl.DataFrame, target: str, feature: str, basis: str, window: int,
                       past: str) -> html.Div:
    """Target against feature: the latest ``window`` days with their fitted line, an older
    stretch in grey with its own line, and today's point and residual against the latest fit."""
    s = scatter_fits(data, target, feature, basis, window, past)
    if s["rows"] < window + 3:
        return note(f"Only {s['rows']} usable observations; the latest window needs {window}.", "warn")
    latest, deeper, fits, now = s["latest"], s["deeper"], s["fits"], s["now"]
    units = "daily change" if basis == "changes" else "level"
    name = FEATURE_LABELS.get(feature, feature)

    def fit_label(which, fit):
        return f"{which} fit: β {fit['beta']:.3f}, R² {fit['r2']:.2f}"
    png = regression_scatter_chart(
        (latest[feature].to_numpy(), latest[target].to_numpy()),
        (deeper[feature].to_numpy(), deeper[target].to_numpy()), fits, (now[feature], now[target]),
        title=f"{target} vs {name} · {units}s", x_title=f"{name} ({units})", y_title=f"{target} ({units})",
        zero_lines=basis == "changes",
        labels={"latest": f"latest {window}d",
                "latest_fit": fit_label("latest", fits["latest"]),
                "past": "past" if len(deeper) else "",
                "past_fit": fit_label("past", fits["past"]),
                "now": "latest point", "extend_latest": past == "previous"})
    periods = f"Latest: {latest['ts'].min()} to {latest['ts'].max()} (n={fits['latest']['n']})."
    if len(deeper):
        periods += f" Past: {deeper['ts'].min()} to {deeper['ts'].max()} (n={fits['past']['n']})."
    summary = (f"Latest point is {s['residual']:+.2f} {'bp ' if units == 'level' else ''}"
               f"{'above' if s['residual'] >= 0 else 'below'} the latest {window}d line ({s['z']:+.1f} residual std).")
    if np.isfinite(fits["past"]["beta"]):
        summary += f" β moved {fits['past']['beta']:.3f} → {fits['latest']['beta']:.3f} from the past to the latest window."
    return html.Div([
        html.Img(src=f"data:image/png;base64,{png}", style={"width": "100%", "border": f"1px solid {BORDER}"}),
        note(periods, "dim"),
        note(summary, "dim"),
    ])


# ---- the load callback's output ---------------------------------------------


def latest_hedge(panel, trade) -> dict[str, float] | None:
    """Latest held weights of a beta-weighted target, per leg (dependent leg keeps its weight)."""
    if panel.weighting != "beta" or not panel.weight_cols:
        return None
    dependent = dependent_leg(trade, panel.beta_dependent)
    scale = float(trade.legs[dependent])
    last = panel.data.select(panel.weight_cols).drop_nulls()
    if not len(last):
        return None
    row = last.row(-1, named=True)
    return {leg: scale if leg == dependent else -scale * row[f"w_{leg}"] for leg in trade.legs}


def summary_bar(panel, trade, aligned_n: int, diag: dict) -> html.Div:
    hedge = latest_hedge(panel, trade)
    legs = "  ".join(f"{w:+.2f}·{leg}" if hedge else f"{w:+g}·{leg}" for leg, w in (hedge or trade.legs).items())
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
        stat_block("legs · latest β" if hedge else "legs", legs),
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


BOARD_ROWS = BOARD_ROWS_SAVED  # every saved row of a board is shown; the table scrolls
# Board columns that have their own saved top rows: clicking one loads that board.
SORT_RULES = {**{column: rule for rule, column in RANK_COLUMNS.items()}, "trades_per_year": "n_trades"}


def exit_label(row: dict) -> str:
    """One discovery or grid row's exit, compactly: 'band 0 cap 2xHL sig 1 stop 15'."""
    text = f"{row.get('exit_style', 'time')} {row.get('exit_param', row.get('horizon'))}"
    for key, name in (("half_life_cap", "cap {}xHL"), ("signal_stop", "sig {}"), ("stop_loss_bps", "stop {}")):
        if row.get(key):
            text += " " + name.format(row[key])
    return text


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
        score = ("ic", row.get("ic")) if "ic" in row else ("sharpe", row.get("sharpe"))
        score_label = f"{score[1]:.3f}" if score[1] is not None else "n/a"
        label = (
            f"#{i + 1}  {score[0]}={score_label}  {row.get('signal_kind', 'normalized')}  {row['fit_on']}  beta_lb={row['beta_lb']}  "
            f"resid_lb={row['residual_lb'] if row['residual_lb'] is not None else '—'}  "
            f"norm_lb={row['norm_lb']}  {gate_desc}"
            + ("" if "ic" in row else f"  exit={exit_label(row)}")
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


def _display_board(board: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    shown = board.with_row_index("#", offset=1).with_columns(
        pl.col("residual_lb").cast(pl.Utf8).fill_null("—"))
    return shown.select([c for c in cols if c in shown.columns])


def run_stats(run_info: dict, feature: str, extra: list) -> html.Div:
    stats = [
        stat_block("result target", target_label(run_info)),
        stat_block("result source", run_info.get("result_source", "Fresh discovery")),
        stat_block("result data period", run_info.get("data_period", "Not recorded")),
        stat_block("feature", FEATURE_LABELS.get(feature, feature)),
        *extra,
        stat_block("elapsed", f"{run_info.get('elapsed_s', 0):.2f}s"),
    ]
    return html.Div(stats, style={"display": "flex", "gap": 28, "flexWrap": "wrap", "marginBottom": 14,
                                  "paddingBottom": 12, "borderBottom": f"1px solid {BORDER}"})


def validation_view(checks: list[dict], placebo: list[dict], real_score: float | None, rank_by: str) -> html.Div:
    """Whether picking the top of this board is likely to mean anything."""
    parts = []
    if checks:
        frame = pl.DataFrame(checks)
        lines = []
        for method, group in frame.group_by("method", maintain_order=True):
            pct = group["chosen_held_out_percentile"].mean()
            chosen = group["chosen_held_out_sharpe"].mean()
            median = group["median_config_held_out_sharpe"].mean()
            verdict = ("generalises: the chosen cell beats most cells on blocks it never saw" if pct >= 0.75 else
                       "does not generalise: the chosen cell lands near or below the typical cell on unseen blocks")
            lines.append(note(f"{method[0]}: picking the top cell {verdict} · average held-out Sharpe "
                              f"{chosen:.2f} vs median cell {median:.2f} · average percentile {pct:.0%}",
                              "good" if pct >= 0.75 else "warn"))
        parts += [html.Div("SELECTION CHECKS", style={"fontWeight": "bold", "fontSize": 11, "color": "#333"}),
                  *lines,
                  html.Details([html.Summary("Per-block detail"),
                                table(frame.drop("config_index").to_pandas(), float_fmt=",.2f")],
                               className="research-info")]
    if placebo:
        scores = np.array([p["best_score"] for p in placebo])
        beaten = int(np.sum(scores >= real_score)) if real_score is not None else len(scores)
        p_value = (1 + beaten) / (1 + len(scores))
        parts += [
            html.Div("PLACEBO", style={"fontWeight": "bold", "fontSize": 11, "color": "#333", "marginTop": 10}),
            note(f"Real top score {real_score:.3f} ({RANK_RULES[rank_by]}) vs placebo top scores: median "
                 f"{np.median(scores):.3f}, max {scores.max():.3f}. {beaten} of {len(scores)} placebos matched or beat it "
                 f"(p ≈ {p_value:.2f}).", "good" if p_value <= 0.05 else "warn"),
            html.Details([html.Summary("Placebo runs"), table(pl.DataFrame(placebo).to_pandas(), float_fmt=",.3f")],
                         className="research-info"),
        ]
    if not parts:
        return html.Div()
    return html.Div(parts, style={"marginBottom": 14, "paddingBottom": 12, "borderBottom": f"1px solid {BORDER}"})


def _with_exit_columns(results: pl.DataFrame) -> pl.DataFrame:
    """Backtest runs saved before exits joined discovery held for ``horizon`` bars."""
    if "exit_style" in results.columns:
        return results
    return results.with_columns(
        pl.lit("time").alias("exit_style"), pl.col("horizon").cast(pl.Float64).alias("exit_param"),
        *[pl.lit(None, dtype=pl.Float64).alias(c) for c in ("half_life_cap", "signal_stop", "stop_loss_bps")])


def backtest_discovery_view(
    board: pl.DataFrame, feature: str, run_info: dict, min_trades: int = 30,
    rank_by: str = "family", checks: list | None = None, placebo: list | None = None,
    cells: int | None = None, eligible: int | None = None, sort_by: str | None = None,
    available=(), real_score: float | None = None,
) -> tuple[html.Div, list[dict]]:
    """Discovery board of real backtests, ranked on the discovery period only.

    ``board`` is already ranked (see ``rank_board`` / ``discovery_compact``)
    by ``sort_by`` (default ``rank_by``, the run's selection rule); only its
    saved top rows exist, so no full grid is ever needed here. Columns whose
    rule is in ``available`` load their own board when their header is clicked.
    ``real_score`` is the selection rule's top score, for the placebo test.
    """
    shown = sort_by or rank_by
    board = _with_exit_columns(board).head(BOARD_ROWS)
    if real_score is None and shown == rank_by and len(board):
        real_score = float(board["rank_score"][0])
    cols = ["#", "signal_kind", "fit_on", "beta_lb", "residual_lb", "norm_lb", "entry_z", "exit_style", "exit_param",
            "half_life_cap", "signal_stop", "stop_loss_bps", "gate", "gate_bucket", "gate_window",
            "family_median_sharpe", "sharpe", "cv_mean_sharpe", "cv_worst_sharpe", "cv_positive_share", "n_trades",
            "trades_per_year", "avg_holding_days", "win_rate", "pnl_bps", "later_sharpe", "later_pnl_bps", "later_trades"]
    display = _display_board(board, cols).rename({"entry_z": "entry", "half_life_cap": "hl_cap"})
    records = board.to_dicts()
    view = html.Div([
        run_stats(run_info, feature, [
            stat_block("cells backtested", f"{cells:,}" if cells is not None else "not recorded"),
            stat_block(f"cells with ≥{min_trades} trades", f"{eligible:,}" if eligible is not None else "not recorded"),
            stat_block("sorted by", RANK_RULES[shown]),
            *([stat_block("selection rule", RANK_RULES[rank_by])] if shown != rank_by else []),
            stat_block("cost · execution", f"{run_info.get('cost_bps', 0)} bp · "
                       f"{'next bar' if run_info.get('execution_lag', 1) else 'same bar'}"),
        ]),
        validation_view(checks or [], placebo or [], real_score, rank_by),
        info("How to read this board",
             "Each row is a real backtest of one model, gate, entry and holding period. Sharpe, trades, win rate and "
             "P&L are for the discovery period. Blank caps and stops mean none. Family median is the median Sharpe "
             "across model lookbacks for the same signal, basis, entry, exit and gate.",
             f"The run saved the top {BOARD_ROWS} cells by each of: {', '.join(RANK_RULES[r] for r in RANK_RULES if r in available)}. "
             "Clicking one of those column headers loads that column's own top cells, best first; clicking it again "
             "flips the order. Other columns only reorder the rows shown.",
             "Later columns are the held-out period: they are never ranked on, and choosing rows by them turns that "
             "period into research data."),
        table(display.to_pandas(), title=f"backtest discovery board · top {len(board)} by {RANK_RULES[shown]}",
              max_rows=BOARD_ROWS, float_fmt=",.3f", sortable=True,
              header_props={column: {"data-board-rule": rule, "data-board-store": "dis-sort"}
                            for column, rule in SORT_RULES.items() if rule in available},
              sorted_by=(RANK_COLUMNS[shown], "desc")),
        note("Pick a row's Backtest button to freeze its relationship and gate "
             "for exact trade-mechanics testing below.", "dim"),
        board_row_picker(records, enabled=run_info.get("can_backtest", True)),
    ])
    return view, records


def dislocation_view(
    results: pl.DataFrame, feature: str, run_info: dict, min_events: int = 30,
    data: pl.DataFrame | None = None,
) -> tuple[html.Div, list[dict]]:
    """Legacy IC board, kept so saved IC discovery runs still open."""
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
    cols = [
        "#", "signal_kind", "signal_units", "fit_on", "beta_lb", "residual_lb", "norm_lb", "entry_z", "horizon", "gate",
        "gate_bucket", "gate_window", "positive_windows_30plus", "model_family_median_ic", "ic", "hit_rate",
        "n_obs", "events_per_year", "later_ic", "later_event_hit_rate", "later_events",
        "n_non_overlapping", "overlap_fraction",
    ]
    display = _display_board(board, cols).rename({"hit_rate": "event_hit_rate", "entry_z": "entry_threshold"})
    view = html.Div([
        run_stats(run_info, feature, [
            stat_block("cells tested", f"{len(results):,}"),
            stat_block(f"cells with ≥{min_events} events", f"{len(valid):,}"),
            stat_block("gated cells", f"{len(gated):,}"),
            stat_block("scoring", "IC (legacy)"),
        ]),
        info("How to read this legacy IC board",
             "Saved before discovery switched to real backtests. Ranked by median discovery IC across tested model "
             "lookbacks, then individual IC. Event hit rate is the fraction of threshold-crossing events with a "
             "favourable forward move, not winning trades. Forward windows can overlap. Later-period columns "
             "evaluate the same candidates; repeatedly choosing from them turns that period into research data."),
        table(display.to_pandas(), title="IC discovery board", max_rows=40, float_fmt=",.3f", sortable=True),
        note("Pick a row's Backtest button to freeze its relationship and gate "
             "for exact trade-mechanics testing below.", "dim"),
        board_row_picker(board_records, enabled=run_info.get("can_backtest", True)),
    ])
    return view, board_records


def banner_fact(label: str, value) -> html.Div:
    """One labelled fact in the strip across the top of a tab."""
    return html.Div([html.Span(label, style=LABEL),
                     html.Span(value, style={"fontSize": 13, "color": TEXT, "fontWeight": "bold"})])


def construction(stored: dict) -> str:
    """How the loaded target is built from its legs, in one line."""
    legs = stored.get("legs") or {}
    if stored.get("weighting") == "beta":
        held = stored.get("weight_columns") or {}
        last = next((row for row in reversed(stored.get("rows") or [])
                     if held and all(row.get(col) is not None for col in held.values())), None)
        if last:
            weights = "  ".join(f"{last[col]:+.2f}·{leg}" for leg, col in held.items())
            return f"beta-weighted · {weights} (latest β) · {stored.get('beta_lookback', '?')}d, held daily"
        dependent = stored.get("beta_dependent") or next((leg for leg, w in legs.items() if w > 0), "?")
        hedges = " + ".join(f"β×{leg}" for leg in legs if leg != dependent)
        return f"beta-weighted · {dependent} vs {hedges} · {stored.get('beta_lookback', '?')}d β, held daily"
    return "fixed · " + ", ".join(f"{leg} {weight:+g}" for leg, weight in legs.items())


GAIN, LOSS, FLAT = (42, 120, 214), (227, 73, 72), (240, 239, 236)  # diverging blue / red, gray midpoint
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _diverging(value, scale: float) -> dict | None:
    """Cell fill for P&L on a symmetric scale; dark fills switch to white ink."""
    if value is None or not isinstance(value, (int, float)) or np.isnan(value) or scale <= 0:
        return None
    weight = min(abs(value) / scale, 1.0)
    pole = GAIN if value > 0 else LOSS
    rgb = [round(f + (p - f) * weight) for f, p in zip(FLAT, pole)]
    return {"background": f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", "color": "#FFFFFF" if weight > 0.6 else TEXT,
            "borderBottom": "2px solid #FFFFFF", "borderRight": "2px solid #FFFFFF"}


def pnl_heatmap(equity: pl.DataFrame, periods: pl.DataFrame) -> html.Div:
    """Monthly P&L by year, coloured by sign and size, with the full-year total."""
    monthly = (equity.with_columns(pl.col("ts").cast(pl.Date))
               .group_by(pl.col("ts").dt.year().alias("year"), pl.col("ts").dt.month().alias("month"))
               .agg(pl.col("pnl_bps").sum()))
    grid = monthly.pivot(on="month", index="year", values="pnl_bps").rename(
        {str(m): MONTHS[m - 1] for m in range(1, 13)}, strict=False)
    grid = grid.with_columns(*[pl.lit(None, dtype=pl.Float64).alias(m) for m in MONTHS if m not in grid.columns])
    if len(periods):
        grid = grid.join(periods.select("year", pl.col("pnl_bps").alias("full_year"), "active_days"),
                         on="year", how="left")
    else:
        grid = grid.with_columns(pl.sum_horizontal(MONTHS).alias("full_year"), pl.lit(None).alias("active_days"))
    grid = grid.sort("year")
    values = MONTHS + ["full_year"]
    month_scale = float(np.nanmax(np.abs(grid.select(MONTHS).to_numpy().astype(float)))) if len(grid) else 0.0
    year_scale = float(np.nanmax(np.abs(grid["full_year"].to_numpy().astype(float)))) if len(grid) else 0.0
    # Cells show formatted text (blank where a month has no data); raw numbers
    # ride along unrendered as raw_<col> so the fill still keys off the value.
    shown = grid.with_columns(
        *[pl.col(c).alias(f"raw_{c}") for c in values],
        *[pl.col(c).map_elements(lambda v: f"{v:,.1f}", return_dtype=pl.Utf8).fill_null("").alias(c) for c in values],
        pl.col("year").cast(pl.Utf8), pl.col("active_days").cast(pl.Utf8).fill_null(""),
    )

    def colour(column, _value, row):
        if column in MONTHS:
            return _diverging(row[f"raw_{column}"], month_scale)
        if column == "full_year":
            return {**(_diverging(row["raw_full_year"], year_scale) or {}), "fontWeight": "bold"}
        return None

    return table(shown.to_pandas(), title="P&L by month and year (bp) · blue gains, red losses; full year on its own scale",
                 columns=["year", *MONTHS, "full_year", "active_days"], cell_style=colour,
                 headers={"full_year": "full year", "active_days": "active days"})


def _png_img(png: str) -> html.Img:
    return html.Img(src=f"data:image/png;base64,{png}", style={"width": "100%", "border": f"1px solid {BORDER}",
                                                              "marginBottom": 14})


def pca_view(result: dict, window: int, n_components: int) -> html.Div:
    """Today's curve against the curve its factors imply, and each tenor's gap in bp and in z-score."""
    table, curve = result["table"], result["curve"]
    explained = "  ·  ".join(f"PC{i + 1} {v:.1%}" for i, v in enumerate(result["explained"]))
    tenors = table["tenor"].to_list()
    as_of = result["as_of"]
    return html.Div([
        html.Div([
            stat_block("as of", str(as_of)),
            stat_block("fit", f"{window}d trailing, point-in-time" if window else "whole sample (in-sample)"),
            stat_block("history", f"{curve['ts'][0]} → {curve['ts'][-1]} · {len(curve):,} days"),
            stat_block(f"variance explained · {n_components} factors", explained),
        ], style={"display": "flex", "gap": 28, "flexWrap": "wrap", "marginBottom": 14, "paddingBottom": 12,
                  "borderBottom": f"1px solid {BORDER}"}),
        _png_img(curve_fit_chart(table["years"].to_list(), table["actual_bp"].to_list(), table["fair_bp"].to_list(),
                                 f"Treasury curve vs PCA-implied curve · {as_of}")),
        html.Div([
            html.Div(_png_img(residual_bars_chart(tenors, table["residual_bp"].to_list(),
                                                  "residual · actual − PCA-implied (bp)", "bp")),
                     style={"flex": "1 1 380px", "minWidth": 0}),
            html.Div(_png_img(residual_bars_chart(tenors, table["z"].to_list(),
                                                  "residual z-score (vs its own history)", "z", bands=(1.0, 2.0))),
                     style={"flex": "1 1 380px", "minWidth": 0}),
        ], style={"display": "flex", "gap": 18, "flexWrap": "wrap"}),
        note("Blue: the yield is above the PCA-implied curve (cheap against the rest of the curve). Red: below it "
             "(rich). z divides today's gap by that tenor's residual standard deviation over the history shown; "
             "dotted and dashed lines mark ±1 and ±2.", "dim"),
    ])


def trade_distribution(pnl: np.ndarray) -> dcc.Graph:
    """Histogram of closed-trade P&L; bins below zero red, above blue."""
    pnl = pnl[np.isfinite(pnl)]
    # zero is always a bin edge, so no bar mixes winners and losers
    width = max(float(pnl.max() - pnl.min()), 1e-9) / min(40, max(10, len(pnl) // 4))
    edges = np.arange(np.floor(pnl.min() / width), np.ceil(pnl.max() / width) + 1) * width
    counts, edges = np.histogram(pnl, bins=edges if len(edges) > 1 else 1)
    centres = (edges[:-1] + edges[1:]) / 2
    colours = [f"rgb{LOSS}" if c < 0 else f"rgb{GAIN}" for c in centres]
    return dcc.Graph(figure={
        "data": [{"type": "bar", "x": centres.tolist(), "y": counts.tolist(), "width": float(edges[1] - edges[0]) * 0.92,
                  "marker": {"color": colours}, "hovertemplate": "%{x:.1f} bp: %{y} trades<extra></extra>"}],
        "layout": {"title": {"text": f"Trade return distribution · {len(pnl)} closed trades · mean {pnl.mean():.1f} bp, "
                                     f"median {np.median(pnl):.1f} bp", "font": {"size": 13}},
                   "xaxis": {"title": {"text": "trade P&L (bp)"}, "zeroline": True},
                   "yaxis": {"title": {"text": "trades"}},
                   "height": 300, "bargap": 0, "margin": {"l": 55, "r": 20, "t": 45, "b": 45}},
    }, config={"displayModeBar": False})


INSPECT_CHOICES = 500


def backtest_grid_view(grid: pl.DataFrame, selected: dict) -> html.Div:
    """Every requested entry/exit cell for the frozen candidate.

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
        "config_id", "signal_kind", "signal_units", "entry_z", "exit_style", "exit_param", "half_life_cap", "signal_stop",
        "stop_loss_bps", "sharpe", "total_pnl_bps",
        "n_trades", "trade_win_rate", "avg_pnl_per_trade_bps", "median_pnl_bps",
        "avg_holding_days", "time_in_market_pct", "max_drawdown_bps", "open_trades",
        "earlier_sharpe", "later_sharpe", "earlier_pnl_bps", "later_pnl_bps",
        "closed_pnl_bps",
    ]
    ordered = grid.sort(rank_metric, descending=True)
    return html.Div([
        html.Div(stats, style={"display": "flex", "gap": 28, "flexWrap": "wrap",
                               "marginBottom": 14, "paddingBottom": 12,
                               "borderBottom": f"1px solid {BORDER}"}),
        info("How this grid is computed",
             "Every cell runs through the vectorised engine (backtest/vector.py), which tests hold to the exact "
             "Engine trade for trade. Inspecting a configuration reruns it through Engine itself and reports any disagreement, "
             "with its median trade P&L. Legs are re-marked and transaction cost applied. Stops are evaluated on "
             "observations, not intraday fills. Ranking uses the earlier period when a split is selected. Later "
             "daily P&L includes positions carried across the split. This is a chronological diagnostic, not a "
             "complete walk-forward validation."),
        table(ordered.select([c for c in cols if c in ordered.columns]).to_pandas(),
              title="backtest grid", max_rows=60, float_fmt=",.2f", sortable=True),
        # A select-all grid has 100k+ configurations; listing them all sent tens
        # of MB to the browser and froze the page. Offer the best few hundred.
        field(f"inspect configuration (top {min(INSPECT_CHOICES, len(ordered)):,} of {len(ordered):,} by {rank_metric})",
              dcc.Dropdown(id="bt-inspect", value=str(best["config_id"]), options=[
                  {"label": f"#{r['config_id']} · entry {r['entry_z']} · {exit_label(r)}", "value": str(r["config_id"])}
                  for r in ordered.head(INSPECT_CHOICES).iter_rows(named=True)])),
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
                selected_style=SELECTED_TAB_STYLE, children=fair_value_tab()),
    ])], className="research-bench")


SUB_TAB_STYLE = {**TAB_STYLE, "padding": "7px 14px", "fontSize": 11}
SUB_TAB_SELECTED = {**SUB_TAB_STYLE, "background": "#FFFFFF", "color": TEXT, "borderTop": f"2px solid {ORANGE}"}


def fv_placeholder(title: str, what: str, plan: list[str], data: list[str]) -> html.Div:
    """A Fair Value sub-tab that is planned but not built: what it will answer and what it needs."""
    return html.Div([
        html.Div([html.Div(title.upper(), style=HEADING), note(what, "dim")], className="research-banner",
                 style={"display": "block"}),
        html.Div([
            html.Div([html.Span("plan", style=LABEL),
                      html.Ul([html.Li(p, style={"fontSize": 12, "color": TEXT, "marginBottom": 6}) for p in plan])]),
            html.Div([html.Span("data", style=LABEL),
                      html.Ul([html.Li(d, style={"fontSize": 12, "color": TEXT, "marginBottom": 6}) for d in data])]),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": 32, "maxWidth": 1100}),
    ])


def pca_controls() -> html.Div:
    return html.Div([
        field("tenors", dcc.Checklist(
            id="fv-pca-tenors", value=PCA_TENORS,
            options=[{"label": f" {t}" + (" (from 2020)" if t == "20y" else " (from 2009)" if t == "7y" else ""),
                      "value": t} for t in sorted(YIELDS, key=lambda t: int(t[:-1]))],
            labelStyle={"display": "block", "fontSize": 12, "marginBottom": 3, "color": TEXT},
            inputStyle={"marginRight": 5})),
        field("history from", dcc.Dropdown(id="fv-pca-start", value="2010-01-01", clearable=False, options=[
            {"label": "2000", "value": "2000-01-01"}, {"label": "2010", "value": "2010-01-01"},
            {"label": "2015", "value": "2015-01-01"}, {"label": "2020", "value": "2020-01-01"}])),
        field("fit window (point-in-time)", dcc.Dropdown(id="fv-pca-window", value=504, clearable=False, options=[
            *[{"label": f"{w} days, trailing", "value": w} for w in PCA_WINDOWS],
            {"label": "whole sample (in-sample, uses the future)", "value": 0}])),
        field("factors kept", dcc.Dropdown(id="fv-pca-factors", value=3, clearable=False, options=[
            {"label": "2 (level, slope)", "value": 2}, {"label": "3 (level, slope, curvature)", "value": 3},
            {"label": "4", "value": 4}])),
        html.Button("Run PCA", id="fv-pca-run", n_clicks=0, className="ref-btn",
                    style={**btn_style(primary=True), "width": "100%", "marginTop": 4, "marginBottom": 14}),
        info("How the fair value is built",
             "Each day, PCA is fitted on the trailing window of curve levels (only data known that day), and the day's "
             "curve is rebuilt from the kept factors. That rebuilt curve is the PCA-implied curve; the residual is "
             "actual minus implied.",
             "z is today's residual over the standard deviation of that tenor's residual history.",
             "The 20y only exists from 2020 and the 7y from 2009; including them shortens the history. Untick the "
             "20y for a longer one."),
    ], className="research-controls")


def pca_tab() -> html.Div:
    return html.Div([
        html.Div([html.Div("PCA · CURVE FAIR VALUE", style=HEADING), note(
            "Level, slope and curvature explain almost all of the curve. Each tenor's gap to the curve they "
            "imply is where it is rich or cheap against the rest of the curve.",
            "dim")], className="research-banner", style={"display": "block"}),
        html.Div(style={"display": "grid", "gridTemplateColumns": "300px minmax(0, 1fr)", "gap": 26,
                        "alignItems": "start"}, children=[
            pca_controls(),
            dcc.Loading(html.Div(id="fv-pca-out"), type="dot"),
        ]),
        dcc.Store(id="fv-pca-settings"),
    ])


def fair_value_tab() -> html.Div:
    return html.Div(style={"padding": "14px 24px"}, children=[
        html.Div(id="fair-value-context", style={"display": "none"}),  # Setup's Fill tabs still writes here
        dcc.Tabs(id="fv-tabs", value="pca", children=[
            dcc.Tab(label="PCA", value="pca", style=SUB_TAB_STYLE, selected_style=SUB_TAB_SELECTED,
                    children=html.Div(pca_tab(), style={"paddingTop": 14})),
            dcc.Tab(label="Term Premium", value="tp", style=SUB_TAB_STYLE, selected_style=SUB_TAB_SELECTED,
                    children=html.Div(fv_placeholder(
                        "term premium",
                        "How much of the long end is compensation for duration risk rather than the expected path "
                        "of rates, built from our own curve rather than a published model.",
                        ["A structural model: the 2s10s30s fly regressed on the residual of 2s vs 10s (the part of "
                         "10s the front end does not explain), to be specified together.",
                         "Rolling multi-factor fit with stationarity, error-correction speed and coefficient "
                         "stability (FairValueStudy), so the residual is a tradeable fair-value gap."],
                        ["Generic yields 2y–30y and on-the-run bonds: in the database.",
                         "No published term premium (ACM / Kim-Wright) stored; not needed for our own model."]),
                        style={"paddingTop": 14})),
            dcc.Tab(label="Inflation", value="infl", style=SUB_TAB_STYLE, selected_style=SUB_TAB_SELECTED,
                    children=html.Div(fv_placeholder(
                        "inflation",
                        "Are breakevens rich or cheap against what usually drives them: oil, the dollar and "
                        "nominal yields?",
                        ["Breakeven fair value from oil, the dollar and nominals (FairValueStudy), per tenor.",
                         "Breakeven curve (5s10s, 10s30s) and TIPS vs inflation-swap basis."],
                        ["Breakevens and TIPS 5/10/30y (generics and by CUSIP), inflation swaps 1–30y from 2004, "
                         "oil, gasoline, metals, the dollar: in the database.",
                         "Missing: CPI and other economic releases, so no 'vs CPI trend' yet."]),
                        style={"paddingTop": 14})),
            dcc.Tab(label="Positioning", value="pos", style=SUB_TAB_STYLE, selected_style=SUB_TAB_SELECTED,
                    children=html.Div(fv_placeholder(
                        "positioning",
                        "Who is long or short Treasury futures (dealers, asset managers, leveraged funds), and how "
                        "extreme that is against history.",
                        ["Net positions by trader type and tenor (TU, FV, TY, UXY, US, WN), in contracts and DV01, "
                         "as z-scores and percentiles; crowding against curve and level moves."],
                        ["md.cftc (CFTC Traders in Financial Futures) has Treasury futures only for Jul 2010 – Dec "
                         "2012 and Feb 2026 onward; nothing in 2013–2025. Needs a backfill from the CFTC's "
                         "historical files before z-scores mean anything.",
                         "Contract code 020604 is the old long bond in 2010–12 but the Ultra bond in 2026."]),
                        style={"paddingTop": 14})),
        ]),
    ])


MECHANICS_EXITS = ["bt-exit-styles", "bt-time-stops", "bt-bands", "bt-revert-fracs",
                   "bt-half-lives", "bt-caps", "bt-signal-stops", "bt-stop"]


def freeze_candidate(board: dict, idx: int, entry_zs=None, current: dict | None = None) -> tuple:
    """Freeze one board row for Trade mechanics.

    Returns the candidate, its label, the entry thresholds, every exit control
    in MECHANICS_EXITS order, cost and fill timing. The row's own exit,
    cost and execution are added to what is already selected, so the grid
    always contains the exact trade the discovery row described.
    """
    current = current or {}
    row = board["rows"][idx]
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
        f"norm_lb={candidate['norm_lb']} · {gate_desc} · discovery exit {exit_label(row)}"
    )
    style = row.get("exit_style", "time")  # IC rows: the forward horizon as a time stop
    param = row["exit_param"] if "exit_style" in row else float(row["horizon"])
    values = {key: list(current.get(key) or []) for key in MECHANICS_EXITS}
    values["bt-exit-styles"] = list(dict.fromkeys([*values["bt-exit-styles"], style]))
    control = {"time": "bt-time-stops", "band": "bt-bands", "revert_frac": "bt-revert-fracs",
               "half_life_frac": "bt-half-lives"}[style]
    for key, value in ((control, param), ("bt-caps", row.get("half_life_cap") or 0.0),
                       ("bt-signal-stops", row.get("signal_stop") or 0.0), ("bt-stop", row.get("stop_loss_bps") or 0.0)):
        values[key] = sorted({*values[key], int(value) if key == "bt-time-stops" else float(value)})
    cost, lag = board.get("cost_bps"), board.get("execution_lag")
    return (candidate, note(f"Frozen: {label}", "good"),
            entry_zs if entry_zs and len(entry_zs) > 1 else [row["entry_z"]],
            *[values[key] for key in MECHANICS_EXITS],
            no_update if cost is None else cost, no_update if lag is None else lag)


JOB_SESSION = "jobs"  # progress for background jobs is shared by every page, so reloads see it


def _job_paths(settings: dict) -> None:
    """In a worker process, save where the app that launched it saves (tests point this elsewhere)."""
    if settings.get("runs_dir"):
        artifacts.RUNS = Path(settings["runs_dir"])


def discovery_job(stored: dict, feature: str, settings: dict) -> dict:
    """One discovery run, start to saved result. Runs as a background job.

    ``settings`` holds the validated discovery controls. Returns the saved
    run's path and a one-line summary; the page renders the result from disk.
    """
    def progress(done, total, message):
        work.update(JOB_SESSION, "dis", message, done, total)

    s = settings
    _job_paths(s)
    work.start(JOB_SESSION, "dis", "Preparing loaded panel and discovery grid")
    try:
        target = stored["target"]
        rows = stored["rows"]
        frame = pl.DataFrame({
            "ts": pl.Series([row["ts"] for row in rows], dtype=pl.Utf8),
            **{col: pl.Series([row[col] for row in rows], dtype=pl.Float64)
               for col in dict.fromkeys([target, feature, *stored['legs'], *stored.get('weight_columns', {}).values()])},
        }).with_columns(pl.col("ts").str.to_date())
        scan = dict(
            target=target, legs=stored["legs"], weight_columns=stored.get("weight_columns") or None,
            fit_on=s["fit_on"], beta_lookbacks=s["beta_lbs"], residual_lookbacks=s["residual_lbs"],
            normalization_lookbacks=s["norm_lbs"], thresholds=s["thresholds"], raw_thresholds=s["raw_entries"],
            exit_params=s["exit_params"], stop_losses=s["stops"] or [None], half_life_caps=s["caps"] or [None],
            signal_stops=s["signal_stops"] or [None], gate_names=s["gates"], gate_windows=s["gate_windows"],
            signal_kind=s["signal_kind"], train_fraction=float(s["train_fraction"]),
            cost_bps=float(s["cost"] or 0.0), execution_lag=int(s["lag"]), cv_folds=int(s["cv"] or 0),
        )
        started = time.perf_counter()
        # Only each cell's ranking numbers are kept while backtesting; every rank
        # rule's top rows are rebuilt in full at the end and saved as boards.
        found = discovery_compact(frame, feature=feature, progress=progress, selection_rule=s["rank_by"],
                                  selection_min_trades=s["min_trades"], checkpoints=jobs.JOBS / "checkpoints", **scan)
        frame, checks, cells = found["frame"], found["checks"], found["cells"]
        scan_s = time.perf_counter() - started
        placebo = []
        if s["placebos"]:
            progress(0, 0, f"Starting {int(s['placebos'])} placebo runs: each reruns all {cells:,} cells "
                           f"with the feature scrambled, about {scan_s:,.0f}s each "
                           f"(~{scan_s * int(s['placebos']) / 60:,.0f} min in total)")
            placebo = placebo_scan(frame, feature=feature, placebos=int(s["placebos"]), rank_by=s["rank_by"],
                                   min_trades=s["min_trades"], progress=progress,
                                   checkpoints=jobs.JOBS / "checkpoints", **scan)
        elapsed_s = time.perf_counter() - started
        info = {
            "backend": "Vectorised backtests (backtest/vector.py)",
            "elapsed_s": elapsed_s,
            "cells_per_second": cells / max(elapsed_s, 1e-9),
            "train_fraction": s["train_fraction"], "target": target,
            "cost_bps": float(s["cost"] or 0.0), "execution_lag": int(s["lag"]),
        }
        info.update(target_definition(stored))
        info.update(result_source='Fresh discovery', data_period=f"{frame['ts'].min()} to {frame['ts'].max()}")
        progress(0, 0, f"Saving {len(found['boards'])} ranked boards (top rows per rank rule and minimum trades)")
        saved = save_run("discovery", frame, None, dict(feature=feature,
            grid=grid_spec(s["fit_on"], s["beta_lbs"], s["residual_lbs"], s["norm_lbs"], s["thresholds"],
                           s["horizons"], s["gates"], s["gate_windows"], s["signal_kind"], s["train_fraction"],
                           s["raw_entries"], scoring="backtest", cost=s["cost"], lag=s["lag"], cv_folds=s["cv"],
                           exits=s["exits"]),
            input_sha256=input_hash(stored, feature),
            signal_kind=s["signal_kind"], min_events=s["min_trades"], scoring="backtest", rank_by=s["rank_by"],
            cv_folds=int(s["cv"] or 0), selection_checks=checks, placebo=placebo,
            panel_id=stored["panel_id"], gate_min_history=126,
            board_counts={"cells": cells, "eligible": {str(k): v for k, v in found["eligible"].items()}},
            **info), boards=found["boards"])
    except Exception as exc:
        work.update(JOB_SESSION, "dis", f"Failed: {exc}", 1, 1)
        raise
    summary = (f"{cells:,} backtested cells" + (f" + {len(placebo)} placebo runs" if placebo else "")
               + f" in {elapsed_s:.1f}s")
    work.update(JOB_SESSION, "dis", f"Completed · {summary} · saved {saved}", 1, 1)
    return {"run_path": saved, "summary": summary}


def render_discovery(run_id: str, min_events=None, fresh: bool = False, query: dict | None = None,
                     sort_by: str | None = None):
    """A saved discovery run as (view, status note, board), for fresh jobs and reopened runs alike.

    The board is built from the run's own snapshot, so picking a row and
    testing exits works whether or not a panel is loaded on this page.
    ``sort_by`` shows another saved angle's board (default: the run's rank rule).
    """
    meta, frame, _ = load_run(run_id, include_results=False)
    info = dict(meta, data_period=f"{frame['ts'].min()} to {frame['ts'].max()}",
                can_backtest=bool(meta.get('legs')) and meta.get('weighting') in {'fixed', 'beta'}
                    and (meta.get('weighting') != 'beta' or bool(meta.get('weight_columns'))),
                elapsed_s=meta.get('elapsed_s', 0), cells_per_second=meta.get('cells_per_second', 0))
    if not fresh:
        info.update(backend='Saved results (no scan)', result_source=f'Archive {run_id}')
    if meta.get('scoring') == 'backtest':
        wanted = int(min_events or meta.get('min_events') or 30)
        rank_by = meta.get('rank_by', 'family')
        available = board_rules(run_id)
        shown = sort_by if sort_by in available else rank_by
        board, used, cells, eligible = load_board(run_id, shown, wanted)
        real_score = None
        if shown != rank_by and meta.get('placebo') and rank_by in available:
            top = load_board(run_id, rank_by, wanted, 1)[0]
            real_score = float(top['rank_score'][0]) if len(top) else None
        view, records = backtest_discovery_view(
            board, meta['feature'], info, used, rank_by, meta.get('selection_checks'), meta.get('placebo'),
            cells=cells, eligible=eligible, sort_by=shown, available=available, real_score=real_score)
        if used != wanted:
            view = html.Div([note(f"This run saved boards for set minimum-trade levels; showing ≥{used} trades "
                                  f"(closest to the {wanted} requested).", "warn"), view])
    else:
        _, _, results = load_run(run_id)
        view, records = dislocation_view(results, meta['feature'], info, int(min_events or 30), frame)
    fraction = float(meta.get('train_fraction', 1))
    split = str(frame['ts'][int(len(frame)*fraction)]) if fraction < 1 else None
    board = dict(target=meta['target'], feature=meta['feature'], rows=records,
        panel_id='archive:'+run_id, split_date=split, run_path=str(run_path(run_id)),
        archive_id=run_id, weight_columns=meta.get('weight_columns', {}),
        target_definition=target_definition(meta),
        cost_bps=meta.get('cost_bps'), execution_lag=meta.get('execution_lag'), fresh=fresh)
    if fresh:
        text = f"Completed · {meta.get('board_counts', {}).get('cells', 0):,} backtested cells in {info['elapsed_s']:.1f}s · saved as {run_id}"
    else:
        text = (f"Saved result target: {target_label(meta)}. Opened run {run_id}: {frame['ts'].min()} to {frame['ts'].max()} · "
                "Original settings/data; not a new scan. "
                + ("Exit tests use this saved snapshot and current engine code." if info['can_backtest']
                   else "View only: saved trade weights are incomplete; exit testing is disabled."))
    tone = 'good'
    if query and definition_match(meta, query.get('target_definition', {})) != 'match':
        warning = f"Historical comparison only: this is NOT a verified match for Current Setup ({target_label(query['target_definition'])})."
        view = html.Div([note(warning, 'warn'), view])
        text, tone = warning + ' ' + text, 'warn'
    return view, note(text, tone), board


def mechanics_job(stored: dict, candidate: dict, settings: dict) -> dict:
    """One trade-mechanics grid, start to saved result. Runs as a background job."""
    s = settings
    _job_paths(s)
    work.start(JOB_SESSION, "bt", "Building the frozen candidate signal and all exit combinations")
    try:
        trade = TradeDef(stored["target"], stored["legs"])
        rows = stored["rows"]
        needed = list(dict.fromkeys([trade.name, candidate["feature"], *trade.legs, *candidate.get("weight_columns", {}).values()]))
        columns = {"ts": pl.Series([row["ts"] for row in rows], dtype=pl.Utf8)}
        columns.update({col: pl.Series([row[col] for row in rows], dtype=pl.Float64) for col in needed})
        data = pl.DataFrame(columns).with_columns(pl.col("ts").str.to_date())
        started = time.perf_counter()
        work.update(JOB_SESSION, "bt", "Running every cell through the vectorised engine; exact Engine detail for the best")
        grid, selected, yearly, _ = run_vector_grid(
            data, trade, candidate, entry_zs=s["entry_zs"], exit_params=s["exit_params"],
            round_trip_cost_bps=s["cost"] or 0.0, stop_losses=s["stops"] or [None], execution_lag=int(s["lag"]),
            half_life_caps=s["caps"] or [None], signal_stops=s["signal_stops"] or [None],
        )
        elapsed_s = time.perf_counter() - started
        work.update(JOB_SESSION, "bt", "Saving every configuration's results and yearly P&L, and the best one's trades")
        saved = save_run("exits", data, grid, dict(candidate=candidate,
            legs=stored["legs"], weighting=stored.get("weighting"),
            beta_lookback=stored.get('beta_lookback'), beta_dependent=stored.get('beta_dependent'),
            weight_columns=stored.get('weight_columns', {}), execution_lag=s["lag"],
            cost_bps=s["cost"], stop_losses=s["stops"], half_life_caps=s["caps"], signal_stops=s["signal_stops"],
            engine="vector", rank_metric=selected["rank_metric"], best_config_id=selected["metrics"]["config_id"],
            elapsed_s=elapsed_s), selected["runs"])
        yearly.write_parquet(Path(saved) / "yearly.parquet")
    except Exception as exc:
        work.update(JOB_SESSION, "bt", f"Failed: {exc}", 1, 1)
        raise
    summary = f"{len(grid):,} cells in {elapsed_s:.2f}s"
    work.update(JOB_SESSION, "bt", f"Completed · {summary} · saved {saved}", len(grid), len(grid))
    return {"run_path": saved, "summary": summary}


def render_mechanics(saved: str | Path):
    """A saved trade-mechanics grid as (view, status note, grid store)."""
    path = Path(saved)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    grid = pl.read_parquet(path / "results.parquet")
    candidate = meta.get("candidate", {})
    rank_metric = meta.get("rank_metric") or ("earlier_sharpe" if candidate.get("split_date") else "sharpe")
    best_id = meta.get("best_config_id") or grid.sort(rank_metric, descending=True)["config_id"][0]
    best = grid.filter(pl.col("config_id") == str(best_id)).row(0, named=True)
    text = (f"Completed · {len(grid):,} cells"
            + (f" in {meta['elapsed_s']:.2f}s" if meta.get("elapsed_s") is not None else "")
            + f" · {candidate.get('target', '')} vs {candidate.get('feature', '')} · saved as {path.name}")
    return (backtest_grid_view(grid, {"metrics": best, "rank_metric": rank_metric}), note(text, "good"),
            # The browser gets where the grid is, not its rows: inspect reads one row from disk.
            {"run_path": str(path), "discovery_run": candidate.get("discovery_run"), "cells": len(grid),
             "target_definition": candidate.get("target_definition"), "feature": candidate.get("feature")})


_RENDERS: dict = {}
_RENDERS_LOCK = threading.Lock()


def render_once(key, render):
    """Render a finished job's result once; concurrent requests for it wait and share it.

    Several open tabs (or a reload mid-render) used to each start their own
    render of the same result at the same time.
    """
    with _RENDERS_LOCK:
        entry = _RENDERS.get(key)
        owner = entry is None
        if owner:
            entry = _RENDERS[key] = {"done": threading.Event(), "result": None, "error": None}
    if not owner:
        entry["done"].wait()
        if entry["error"] is not None:
            raise entry["error"]
        return entry["result"]
    try:
        entry["result"] = render()
    except Exception as exc:
        entry["error"] = exc
        with _RENDERS_LOCK:
            _RENDERS.pop(key, None)
        raise
    finally:
        entry["done"].set()
    with _RENDERS_LOCK:
        for old in [k for k, v in _RENDERS.items() if v["done"].is_set()][:-8]:
            _RENDERS.pop(old)
    return entry["result"]


def show_job(kind: str, job_id: str, min_events=None) -> tuple:
    """(view, status note, store) for one finished job: its result, or why it has none."""
    record = jobs.get(job_id)
    if not record or record["status"] != "done":
        return (job_failure(record or {"error": "Job record not found."}),
                note(f"Job {(record or {}).get('status', 'missing')}: {(record or {}).get('error')}", "bad"), no_update)
    try:
        if kind == "dis":
            run = Path(record["result"]["run_path"]).name
            return render_once(("dis", job_id, min_events), lambda: render_discovery(run, min_events, fresh=True))
        return render_once(("bt", job_id), lambda: render_mechanics(record["result"]["run_path"]))
    except Exception as exc:
        return job_failure(dict(error=f"Cannot show result: {exc}", traceback=traceback.format_exc())), no_update, no_update


def job_failure(record: dict) -> html.Div:
    return html.Div([
        note(f"{record.get('error') or 'Job did not finish.'}", "bad"),
        html.Pre(record.get("traceback") or "", style={"fontSize": 10, "color": DIM, "whiteSpace": "pre-wrap"}),
    ])


def register_callbacks(app) -> None:
    sweep_controls = ["dis-signal", "dis-fit-on", "dis-beta-lbs", "dis-residual-lbs",
        "dis-norm-lbs", "dis-thresholds", "dis-horizons", "dis-gates", "dis-gate-windows", "dis-raw-thresholds",
        "dis-exit-styles", "dis-bands", "dis-revert-fracs", "dis-half-lives", "dis-caps", "dis-signal-stops", "dis-stops",
        "bt-entry-zs", *MECHANICS_EXITS]

    @app.callback(*[Output(control, "value", allow_duplicate=True) for control in sweep_controls],
        Input("dis-select-all", "n_clicks"),
        *[State(control, "options") for control in sweep_controls], prevent_initial_call=True)
    def _select_all_sweeps(_clicks, *options):
        return [[option['value'] for option in choices if not option.get('disabled', False)]
                for choices in options]

    mechanics_controls = ["bt-entry-zs", *MECHANICS_EXITS]

    @app.callback(*[Output(control, "value", allow_duplicate=True) for control in mechanics_controls],
        Input("bt-select-all", "n_clicks"),
        *[State(control, "options") for control in mechanics_controls], prevent_initial_call=True)
    def _select_all_mechanics(_clicks, *options):
        # Entries, exits and stops only: cost and execution stay as chosen.
        return [[option['value'] for option in choices] for choices in options]

    grid_inputs = ["dis-signal", "dis-fit-on", "dis-beta-lbs", "dis-residual-lbs", "dis-norm-lbs", "dis-thresholds",
                   "dis-raw-thresholds", "dis-gates", "dis-gate-windows", "dis-exit-styles", "dis-horizons", "dis-bands",
                   "dis-revert-fracs", "dis-half-lives", "dis-caps", "dis-signal-stops", "dis-stops", "dis-cv", "dis-placebo"]

    @app.callback(Output("controls-saved", "data"), *[Input(control, "value") for control in REMEMBERED],
                  prevent_initial_call=True)
    def _remember_controls(*values):
        # Any change, by hand, Select all or a picked row, becomes the next page's starting point.
        try:
            save_controls(dict(zip(REMEMBERED, values)))
        except OSError:
            return no_update
        return True

    @app.callback(Output("dis-grid-size", "children"), *[Input(control, "value") for control in grid_inputs])
    def _grid_size(signals, bases, beta, residual, norm, entries, raw_entries, gates, windows, styles, times,
                   bands, reverts, half_lives, caps, signal_stops, stops, cv, placebos):
        signals = [signals] if isinstance(signals, str) else signals or []
        per_kind = len(beta or [])*len(norm or [])*(
            (len(residual or []) if 'changes' in (bases or []) else 0)
            + (1 if 'levels' in (bases or []) else 0))
        gate_cells = 1 + len(gates or [])*len(windows or [])*len(REGIME_GATE_BUCKETS)
        params = {"time": times, "band": bands, "revert_frac": reverts, "half_life_frac": half_lives}
        exits = {s: v for s, v in params.items() if s in (styles or []) and v}
        per_model = sum(len(exit_cells((raw_entries if s == 'raw' else entries) or [], exits, stops or [None],
                                       caps or [None], signal_stops or [None])) for s in signals)
        cells = per_kind*per_model*gate_cells
        # ~200k backtests/s and ~200 bytes per saved cell (plus CV block sums) on this machine
        seconds = cells / 200_000
        gigabytes = cells * (200 + 16 * (cv or 0)) / 1e9

        def span(sec):
            return f"{sec / 3600:.1f} h" if sec >= 3600 else f"{sec / 60:.0f} min" if sec >= 60 else f"{sec:.0f} s"
        text = (f"{per_kind*len(signals):,} models · {cells:,} backtested discovery cells · "
                f"about {span(seconds)} and {gigabytes:.1f} GB")
        if placebos:
            text += f" · {placebos} placebos rerun the whole grid: about {span(seconds * (1 + placebos))} in total"
        warning = gigabytes > 8
        return note(text + (" · Very large: narrow the selections before running." if warning else "."),
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
                  Output("beta-advanced", "style"), Output("custom-builder", "style"),
                  Input("target", "value"), Input("weighting", "value"))
    def _weight_controls(target, weighting):
        show, hide = {"marginBottom": 12}, {"display": "none"}
        return (show if target == "custom" else hide, show if weighting == "beta" else hide,
                show if weighting == "beta" else hide, {} if target == "custom" else hide)

    @app.callback(*[Output(f"custom-leg-{i}-box", "style") for i in (1, 2, 3)],
                  *[Output(f"custom-leg-{i}-label", "children") for i in (1, 2, 3)],
                  Input("custom-kind", "value"))
    def _custom_roles(kind):
        # A spread uses two legs, a fly three; a free basket is typed, so no pickers.
        roles = CUSTOM_LEG_ROLES.get(kind, [])
        return (*[{"marginBottom": 12} if i < len(roles) else {"display": "none"} for i in range(3)],
                *[roles[i] if i < len(roles) else "" for i in range(3)])

    @app.callback(Output("custom", "value"), Output("custom-error", "children"),
                  Input("custom-kind", "value"), *[Input(f"custom-leg-{i}", "value") for i in (1, 2, 3)],
                  prevent_initial_call=True)
    def _custom_definition(kind, *legs):
        # The builder writes the definition; a free basket is typed by hand instead.
        if kind not in CUSTOM_LEG_ROLES:
            return no_update, ""
        try:
            return structure_definition(kind, list(legs[:len(CUSTOM_LEG_ROLES[kind])])), ""
        except ValueError as exc:
            return no_update, note(str(exc), "warn")

    @app.callback(Output("custom", "value", allow_duplicate=True), Input("target", "value"),
                  State("custom", "value"), State("custom-kind", "value"),
                  *[State(f"custom-leg-{i}", "value") for i in (1, 2, 3)], prevent_initial_call=True)
    def _custom_on_select(target, current, kind, *legs):
        # Choosing custom fills an empty definition from the builder; one already typed is kept.
        if target != "custom" or (current or "").strip() or kind not in CUSTOM_LEG_ROLES:
            return no_update
        try:
            return structure_definition(kind, list(legs[:len(CUSTOM_LEG_ROLES[kind])]))
        except ValueError:
            return no_update

    @app.callback(Output("weighting", "options"), Output("weighting", "value", allow_duplicate=True),
                  Input("target", "value"), State("weighting", "value"), prevent_initial_call="initial_duplicate")
    def _beta_only_for_packages(target, weighting):
        # Beta weighting needs a second leg. Swap spreads and breakevens get
        # theirs (swap vs Treasury, nominal vs TIPS); an outright has none, so
        # the option is disabled.
        package = PACKAGE_LEGS.get(target)
        single = target in CATALOG and len(CATALOG[target].legs) < 2 and not package
        label = (f" beta-weighted ({package[0]} vs β × {package[1]})"
                 if package else " beta-weighted" + (" (needs 2+ legs)" if single else ""))
        options = [{"label": " fixed", "value": "fixed"},
                   {"label": label, "value": "beta", "disabled": single}]
        return options, "fixed" if single and weighting == "beta" else no_update

    for prefix in ("dis", "bt"):
        time_id = "dis-horizons" if prefix == "dis" else "bt-time-stops"

        @app.callback(*[Output(f"{name}-field", "style") for name in
                        (time_id, f"{prefix}-bands", f"{prefix}-revert-fracs", f"{prefix}-half-lives", f"{prefix}-caps")],
                      Input(f"{prefix}-exit-styles", "value"))
        def _exit_controls(styles):
            show, hide = {"marginBottom": 12}, {"display": "none"}
            styles = styles or []
            return [*[show if style in styles else hide for style in ("time", "band", "revert_frac", "half_life_frac")],
                    show if {"band", "revert_frac"} & set(styles) else hide]

    @app.callback(Output("fv-pca-out", "children"), Output("fv-pca-settings", "data"),
                  Input("fv-pca-run", "n_clicks"), Input("bench-tabs", "value"),
                  State("fv-pca-tenors", "value"), State("fv-pca-start", "value"), State("fv-pca-window", "value"),
                  State("fv-pca-factors", "value"), State("fv-pca-settings", "data"))
    def _run_pca(clicks, tab, tenors, start, window, factors, shown):
        # Runs on the button, and once when the Fair Value tab is first opened.
        if not clicks and (tab != "fv" or shown):
            return no_update, no_update
        tenors = [t for t in (tenors or []) if t in YIELDS]
        if len(tenors) < 3:
            return note("Choose at least three tenors.", "warn"), None
        settings = dict(tenors=tenors, start=start, window=int(window or 0), factors=int(factors or 3))
        try:
            result = pca_curve(tenors, start, settings["window"] or None, settings["factors"])
        except Exception as exc:
            return note(f"PCA failed: {type(exc).__name__}: {exc}", "bad"), None
        return pca_view(result, settings["window"], settings["factors"]), settings

    @app.callback(Output("scatter-feature", "options"), Output("scatter-feature", "value"),
                  Input("research-level-data", "data"), State("scatter-feature", "value"))
    def _scatter_features(stored, current):
        if not stored:
            return [], None
        features = [f for f in stored.get("features", []) if f != stored["target"]]
        return ([{"label": FEATURE_LABELS.get(f, f), "value": f} for f in features],
                current if current in features else (features[0] if features else None))

    @app.callback(Output("scatter-out", "children"), Input("research-level-data", "data"),
                  Input("scatter-feature", "value"), Input("scatter-basis", "value"),
                  Input("scatter-window", "value"), Input("scatter-past", "value"))
    def _scatter(stored, feature, basis, window, past):
        if not stored:
            return note("Load a panel to see the target against its feature.", "dim")
        if not feature:
            return note("Add a feature on the left and Load to see the regression scatter.", "dim")
        rows = stored["rows"]
        data = pl.DataFrame({"ts": [r["ts"] for r in rows],
                             **{c: pl.Series([r.get(c) for r in rows], dtype=pl.Float64) for c in (stored["target"], feature)}})
        return regression_scatter(data, stored["target"], feature, basis, int(window), past)

    @app.callback(Output("dis-feature", "options"), Output("dis-feature", "value"),
                  Output("dis-context", "children"), Output("fill-out", "children"),
                  Output("relative-value-context", "children"), Output("fair-value-context", "children"),
                  Output("dis-feature-field", "style"),
                  Input("research-level-data", "data"), State("dis-feature", "value"))
    def _fill(stored, current):
        hidden = {"display": "none"}
        if not stored:
            return [], None, note("Load a target and feature on Setup, then Fill tabs.", "dim"), "", "", "", hidden
        features = [f for f in stored.get("features", []) if f != stored["target"]]
        options = [{"label": FEATURE_LABELS.get(f, f), "value": f} for f in features]
        chosen = current if current in features else (features[0] if features else None)
        dates = [row["ts"] for row in stored["rows"]]
        headline = html.Div([
            html.Div([
                banner_fact("target", stored["target"]),
                banner_fact("construction", construction(stored)),
                banner_fact("feature", FEATURE_LABELS.get(chosen, chosen) if len(features) == 1 else
                            (f"{len(features)} loaded · choose one" if features else "none: add one on Setup")),
                banner_fact("data", f"{min(dates)} to {max(dates)} · {len(dates):,} observations"),
            ], style={"display": "flex", "gap": 32, "flexWrap": "wrap", "alignItems": "flex-start"}),
            note("Discovery backtests every signal model, entry, exit and gate on the discovery period and saves the "
                 "best cells from several angles. Freeze a row to test its trade mechanics exactly below.", "dim"),
        ])
        context = note(f"Current Setup: {target_label(stored)} · {len(stored['rows']):,} observations · "
                       f"features: {', '.join(features) or 'none selected'}")
        return (options, chosen, headline,
                note("Loaded setup carried into discovery. Choose a discovery row to fill trade mechanics.", "good"),
                context, context, {"marginBottom": 12} if len(features) > 1 else hidden)

    # The invalidation callbacks reset what is *frozen* (the candidate) but never
    # wipe the backtest-grid area. Wiping it raced its own inspect update: the
    # update arrived for a component that was gone, the page threw, and Dash
    # stopped applying the updates queued behind it (e.g. Setup's controls).
    @app.callback(
        Output("dis-out", "children", allow_duplicate=True), Output("dis-board", "data", allow_duplicate=True),
        Output("dis-candidate", "data", allow_duplicate=True), Output("bt-candidate", "children", allow_duplicate=True),
        Output("bt-out", "children", allow_duplicate=True), Output("bt-grid", "data", allow_duplicate=True),
        Output("bt-detail", "children", allow_duplicate=True),
        Input("research-level-data", "data"), State("dis-board", "data"), State("bt-grid", "data"),
        prevent_initial_call=True,
    )
    def _invalidate_panel(stored, board, grid):
        """Loading a setup clears shown results that belong to a different target or feature.

        Results for the same target definition and a loaded feature stay. The
        inspect detail lives outside the grid area, so clearing the grid cannot
        leave an inspect update with nowhere to land.
        """
        if not stored:
            return (no_update,) * 7
        current = target_definition(stored)
        loaded = set(stored.get("features", []))

        def stale(shown):
            return bool(shown) and not (definition_match(shown.get("target_definition") or {}, current) == "match"
                                        and shown.get("feature") in loaded)

        def was(shown):
            return (f"{target_label(shown.get('target_definition') or {'target': shown.get('target', 'another target')})}"
                    f" vs {FEATURE_LABELS.get(shown.get('feature'), shown.get('feature') or 'another feature')}")
        dis = bt = (no_update,) * 3
        candidate = no_update
        if stale(board):
            dis = (note(f"The board shown was for {was(board)}; run discovery for the loaded setup.", "dim"), None, None)
            candidate = note("Choose Backtest on a discovery row to fill trade mechanics.")
        if stale(grid):
            bt = (note(f"The grid shown was for {was(grid)}.", "dim"), None, "")
        return (*dis, candidate, *bt)

    @app.callback(
        Output("dis-candidate", "data", allow_duplicate=True),
        Output("bt-candidate", "children", allow_duplicate=True),
        Input("dis-board", "data"), State("bt-grid", "data"), State("dis-candidate", "data"),
        prevent_initial_call=True,
    )
    def _invalidate_candidate_for_board(board, grid, candidate=None):
        # A new board un-freezes the candidate; a grid already shown stays on screen.
        # Re-sorting the same run's board keeps what was frozen from it.
        if board and ((grid and grid.get("discovery_run") == board.get("run_path"))
                      or (candidate and candidate.get("discovery_run") == board.get("run_path"))):
            return (no_update,) * 2
        return None, note("Choose Backtest on a row from this discovery run.", "dim")

    @app.callback(*[Output(f"{name}-progress", "children") for name in ("load", "dis", "bt")],
                  Input("research-progress-poll", "n_intervals"), State("research-session", "data"))
    def _progress(_tick, session):
        # Jobs run in worker processes, which report progress through a file.
        return [progress_view(work.snapshot(session, name) if name == "load" else jobs.progress_of(name), caption)
                for name, caption in (("load", "loading"), ("dis", "searching"), ("bt", "backtesting"))]

    @app.callback(Output("bt-detail", "children"), Input("bt-inspect", "value"), State("bt-grid", "data"))
    def _inspect(config_id, stored):
        # The grid's run saves exact Engine detail for its best cell only. Any
        # other cell is rerun through Engine on first inspection and added to
        # the run, so its trades are exact and saved like the best cell's.
        saved_dir = Path((stored or {}).get("run_path", "")) if stored else None
        if not saved_dir or not (saved_dir / "metadata.json").is_file():
            return note("Run a backtest grid to inspect trades.")
        cid = str(config_id)
        results = saved_dir / "results.parquet"
        found = (pl.scan_parquet(results).filter(pl.col("config_id") == cid).collect().to_dicts()
                 if results.is_file() else [])
        row = found[0] if found else None
        equity_file = saved_dir / "equity.parquet"
        equity = (pl.read_parquet(equity_file).filter(pl.col("config_id") == cid)
                  if equity_file.is_file() else pl.DataFrame())
        detail = None
        if equity.is_empty():
            if row is None:
                return note("No such configuration in this grid.")
            try:
                meta = json.loads((saved_dir / "metadata.json").read_text(encoding="utf-8"))
                candidate = meta["candidate"]
                detail = detail_for(
                    pl.read_parquet(saved_dir / "data.parquet"), TradeDef(candidate["target"], meta["legs"]),
                    candidate, config_id=cid, entry_z=row["entry_z"], exit_style=row["exit_style"],
                    exit_param=row["exit_param"], stop_loss_bps=row["stop_loss_bps"],
                    round_trip_cost_bps=meta.get("cost_bps") or 0.0, execution_lag=int(meta["execution_lag"]),
                    half_life_cap=row.get("half_life_cap"), signal_stop=row.get("signal_stop"),
                )
                append_details(saved_dir, {cid: detail})
            except Exception as exc:
                return note(f"Exact Engine rerun failed: {type(exc).__name__}: {exc}", "bad")
            equity = pl.read_parquet(equity_file).filter(pl.col("config_id") == cid)
        parity = None
        if row is not None and detail is not None:
            gap = max(abs(detail["metrics"]["sharpe"] - row["sharpe"]),
                      abs(detail["metrics"]["total_pnl_bps"] - row["total_pnl_bps"]))
            same = gap < 1e-6 and detail["metrics"]["n_trades"] == row["n_trades"]
            parity = note(f"Exact Engine rerun {'matches' if same else 'DISAGREES WITH'} the vectorised grid "
                          f"(max |difference| in Sharpe/P&L {gap:.1e}; trades {detail['metrics']['n_trades']} vs {row['n_trades']}).",
                          "good" if same else "bad")
        periods_file = saved_dir / "periods.parquet"
        periods = (pl.read_parquet(periods_file).filter(pl.col("config_id") == cid).drop("config_id")
                   if periods_file.is_file() else pl.DataFrame())
        trades_file = saved_dir / "trades.parquet"
        trades = (pl.read_parquet(trades_file).filter(pl.col("config_id") == cid).drop("config_id")
                  if trades_file.is_file() else pl.DataFrame())
        median = note(f"Median closed-trade P&L: {trades['pnl_bps'].median():,.2f} bps", "dim") if len(trades) else None
        scale = float(np.nanpercentile(np.abs(trades["pnl_bps"].to_numpy()), 95)) if len(trades) else 0.0
        return html.Div([
            parity, median,
            dcc.Graph(figure={"data": [{"x": equity["ts"].to_list(),
                                       "y": equity["cumulative_pnl"].to_list(),
                                       "type": "scatter", "mode": "lines", "name": "Net P&L"}],
                              "layout": {"title": {"text": "Selected configuration · cumulative P&L", "font": {"size": 13}},
                                         "yaxis": {"title": {"text": "bp"}},
                                         "height": 320, "margin": {"l": 55, "r": 20, "t": 45, "b": 40}}}),
            pnl_heatmap(equity, periods),
            html.Div([
                html.Div(trade_distribution(trades["pnl_bps"].to_numpy()), style={"flex": "2 1 420px", "minWidth": 0}),
                html.Div(table(trades.group_by("exit_reason").agg(
                    pl.len().alias("trades"), pl.col("pnl_bps").mean().alias("mean_pnl_bps")).to_pandas(),
                    title="Exit reasons"), style={"flex": "1 1 260px", "minWidth": 0}),
            ], style={"display": "flex", "gap": 24, "flexWrap": "wrap", "alignItems": "flex-start"}) if len(trades) else None,
            table(trades.to_pandas(), title="Closed trades · MAE/MFE measured on observations · P&L blue gains, red losses",
                  cell_style=lambda column, value, _row: _diverging(value, scale) if column == "pnl_bps" else None)
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
        Input("dis-cost", "value"), Input("dis-lag", "value"), Input("dis-cv", "value"),
    )
    def _find_saved(stored, feature, bases, beta, residual, norm, entries, horizons,
                    gates, windows, signal, train, _board, selected, raw_entries=None, show_other=None,
                    cost=0.1, lag=1, cv=0):
        if not stored or feature not in stored.get('features', []):
            return [], None, None
        request = grid_spec(bases or ['changes'], beta or DISLOCATION_BETA_LBS,
            residual or DISLOCATION_RESIDUAL_LBS, norm or DISLOCATION_NORM_LBS,
            entries or DISLOCATION_THRESHOLDS, horizons or DISLOCATION_HORIZONS,
            gates or [], windows or [126, 252, 504], signal, train, raw_entries or RAW_THRESHOLDS,
            scoring="backtest", cost=cost, lag=lag, cv_folds=cv)
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
            return render_discovery(run_id, min_events, fresh=False, query=query)
        except Exception as exc:
            return no_update, note(f'Cannot open saved run: {exc}', 'bad'), no_update

    @app.callback(
        Output("dis-out", "children", allow_duplicate=True),
        Output("dis-board", "data", allow_duplicate=True),
        Input("dis-sort", "data"), State("dis-board", "data"),
        State("dis-min-events", "value"), State("dis-saved-query", "data"),
        prevent_initial_call=True,
    )
    def _sort_board(sort, board, min_events, query):
        # A header click on a column with its own saved board shows that board.
        if not sort or not board or not board.get("archive_id"):
            return no_update, no_update
        try:
            view, _status, new_board = render_discovery(board["archive_id"], min_events, fresh=board.get("fresh", False),
                                                        query=query, sort_by=sort.get("rule"))
        except Exception as exc:
            return note(f"Cannot load that board: {exc}", "bad"), no_update
        return view, new_board

    @app.callback(
        Output("dis-run-info", "children", allow_duplicate=True),
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
        State("dis-cost", "value"), State("dis-lag", "value"), State("dis-cv", "value"),
        State("dis-rank", "value"), State("dis-placebo", "value"),
        State("dis-exit-styles", "value"), State("dis-bands", "value"), State("dis-revert-fracs", "value"),
        State("dis-half-lives", "value"), State("dis-caps", "value"), State("dis-signal-stops", "value"),
        State("dis-stops", "value"),
        prevent_initial_call=True,
    )
    def _run_dislocation(
        _n, stored, feature, fit_on, beta_lbs, residual_lbs, norm_lbs, thresholds,
        horizons, gates, gate_windows, signal_kind, train_fraction, min_events, session, raw_entries=None,
        cost=0.1, lag=1, cv=0, rank_by="family", placebos=0, exit_styles=None, bands=None, revert_fracs=None,
        half_lives=None, caps=None, signal_stops=None, stops=None,
    ):
        if jobs.running("dis"):
            return note("A discovery job is already running; its progress is shown on the right.", "warn")
        if not stored:
            return note("Load a target and this feature on Setup first.", "warn")
        if feature not in stored.get("features", []):
            return note(
                f"{FEATURE_LABELS.get(feature, feature)} is not in the loaded panel. "
                "Add it on Setup, then Load.", "warn"
            )
        rank_by = rank_by or "family"
        if rank_by.startswith("cv") and not cv:
            return note("CV ranking needs cross-validation blocks: choose a block count under Sample & validation.", "warn")
        min_trades = int(min_events or 30)
        exit_params = {style: values for style, values in {
            "time": horizons or DISLOCATION_HORIZONS, "band": bands, "revert_frac": revert_fracs,
            "half_life_frac": half_lives}.items() if style in (exit_styles or ["time"]) and values}
        if not exit_params:
            return note("Choose at least one exit style with at least one parameter under Exits.", "warn")
        exits = dict(styles=sorted(exit_params), **{style: sorted(v) for style, v in exit_params.items()},
                     half_life_caps=sorted(caps or [0.0]), signal_stops=sorted(signal_stops or [0.0]),
                     stops=sorted(stops or [0.0]))
        settings = dict(
            fit_on=fit_on or ["changes"], beta_lbs=beta_lbs or DISLOCATION_BETA_LBS,
            residual_lbs=residual_lbs or DISLOCATION_RESIDUAL_LBS, norm_lbs=norm_lbs or DISLOCATION_NORM_LBS,
            thresholds=thresholds or DISLOCATION_THRESHOLDS, raw_entries=raw_entries or RAW_THRESHOLDS,
            horizons=horizons or DISLOCATION_HORIZONS, gates=gates or [], gate_windows=gate_windows or [126, 252, 504],
            signal_kind=signal_kind, train_fraction=train_fraction, min_trades=min_trades, cost=cost, lag=lag,
            cv=cv, rank_by=rank_by, placebos=placebos, exit_params=exit_params, exits=exits, caps=caps,
            signal_stops=signal_stops, stops=stops)
        try:
            # A worker process: a native crash can only end the job, and it is relaunched to resume.
            jobs.submit("dis", f"{stored['target']} vs {feature} discovery", "research.app:discovery_job",
                        stored, feature, dict(settings, runs_dir=str(artifacts.RUNS)))
        except jobs.JobRunning:
            return note("A discovery job is already running; its progress is shown on the right.", "warn")
        return note("Discovery started as a background job. It keeps running if you reload or leave this page; "
                    "the result appears here when it finishes.", "dim")

    @app.callback(
        Output("dis-candidate", "data"),
        Output("bt-candidate", "children"),
        Output("bt-entry-zs", "value"),
        *[Output(control, "value", allow_duplicate=True) for control in MECHANICS_EXITS],
        Output("bt-cost", "value"),
        Output("bt-lag", "value"),
        Input({"type": "dis-pick", "index": ALL}, "n_clicks"),
        State("dis-board", "data"),
        State("bt-entry-zs", "value"),
        *[State(control, "value") for control in MECHANICS_EXITS],
        prevent_initial_call=True,
    )
    def _pick_dislocation_candidate(n_clicks, board, entry_zs=None, *exits):
        # Dash fires this the moment the row buttons first appear (all zero
        # clicks), before any real click. Only a genuine click should freeze
        # a candidate.
        outputs = 5 + len(MECHANICS_EXITS)
        if not board or not any(n_clicks or []):
            return (no_update,) * outputs
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or triggered["index"] >= len(board["rows"]):
            return (no_update,) * outputs
        return freeze_candidate(board, triggered["index"], entry_zs, dict(zip(MECHANICS_EXITS, exits)))

    @app.callback(
        Output("bt-run-info", "children", allow_duplicate=True),
        Input("bt-run", "n_clicks"),
        State("dis-candidate", "data"),
        State("research-level-data", "data"),
        State("target", "value"), State("custom", "value"),
        State("bt-entry-zs", "value"), State("bt-exit-styles", "value"),
        State("bt-time-stops", "value"), State("bt-bands", "value"),
        State("bt-revert-fracs", "value"), State("bt-stop", "value"),
        State("bt-cost", "value"),
        State("bt-half-lives", "value"), State("bt-lag", "value"), State("research-session", "data"),
        State("bt-caps", "value"), State("bt-signal-stops", "value"),
        prevent_initial_call=True,
    )
    def _run_backtest_grid(
        _n, candidate, stored, target, custom, entry_zs, exit_styles,
        time_stops, bands, revert_fracs, stop_bp, cost_bp, half_lives, lag, session, caps=None, signal_stops=None,
    ):
        if jobs.running("bt"):
            return note("A backtest grid is already running; its progress is shown on the right.", "warn")
        if not candidate:
            return note("Choose Backtest on a discovery row first.", "warn")
        if candidate.get("archive_id"):
            try:
                meta, snapshot, _ = load_run(candidate['archive_id'], include_results=False)
                if (not meta.get('legs') or meta.get('weighting') not in {'fixed', 'beta'}
                        or (meta.get('weighting') == 'beta' and not meta.get('weight_columns'))):
                    return note("This legacy run lacks saved trade weights. View its results, but rerun discovery to enable reliable exit tests.", "warn")
                stored = dict(target=meta['target'], features=[meta['feature']], legs=meta['legs'],
                    weighting=meta.get('weighting'), beta_lookback=meta.get('beta_lookback'),
                    beta_dependent=meta.get('beta_dependent'), weight_columns=meta.get('weight_columns', {}),
                    panel_id=candidate['panel_id'], rows=snapshot.with_columns(pl.col('ts').cast(pl.Utf8)).to_dicts())
            except Exception as exc:
                return note(f"Cannot open saved snapshot: {exc}", "bad")
        if not stored:
            return note("Load a target on Setup first.", "warn")
        if candidate.get("panel_id") != stored.get("panel_id"):
            return note("The loaded panel changed. Run discovery and select a row from the new panel.", "warn")
        if candidate["target"] != stored["target"]:
            return note(
                "Loaded target has changed since discovery ran; rerun discovery "
                "for the current target.", "warn",
            )
        if candidate["feature"] not in stored.get("features", []):
            return note(
                f"{FEATURE_LABELS.get(candidate['feature'], candidate['feature'])} is "
                "not in the loaded panel. Add it on Setup, then Load.", "warn",
            )
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
            )
        settings = dict(entry_zs=entry_zs, exit_params=exit_params, cost=cost_bp, stops=stop_bp, lag=lag,
                        caps=caps, signal_stops=signal_stops)
        try:
            jobs.submit("bt", f"{candidate['target']} vs {candidate['feature']} trade mechanics",
                        "research.app:mechanics_job", stored, candidate, dict(settings, runs_dir=str(artifacts.RUNS)))
        except jobs.JobRunning:
            return note("A backtest grid is already running; its progress is shown on the right.", "warn")
        return note("Backtest grid started as a background job; the result appears here when it finishes.", "dim")

    @app.callback(
        Output("dis-ready", "data"), Output("bt-ready", "data"),
        Output("dis-run", "disabled"), Output("dis-run", "children"),
        Output("bt-run", "disabled"), Output("bt-run", "children"),
        Input("research-progress-poll", "n_intervals"),
        State("dis-rendered", "data"), State("bt-rendered", "data"),
        State("dis-ready", "data"), State("bt-ready", "data"),
        State("dis-run", "disabled"), State("bt-run", "disabled"),
        prevent_initial_call=True,
    )
    def _job_status(_tick, dis_shown, bt_shown, dis_ready, bt_ready, dis_was_busy, bt_was_busy):
        """The cheap per-tick check: Run buttons, and which finished job each result area should show.

        It never outputs to the result areas themselves. Doing that on every
        tick made Dash dim them as 'loading' each time: the flashing.
        """
        out = []
        for kind, label, shown, ready, was_busy in (("dis", "Run discovery", dis_shown, dis_ready, dis_was_busy),
                                                    ("bt", "Run backtest grid", bt_shown, bt_ready, bt_was_busy)):
            record = jobs.latest(kind)
            busy = bool(record) and record["status"] == "running"
            new = (record["id"] if record and record["status"] in jobs.FINISHED and record["id"] not in (shown, ready)
                   else no_update)
            changed = busy != bool(was_busy)
            out.append((new, busy if changed else no_update,
                        ("Running… (background job)" if busy else label) if changed else no_update))
        (d_new, d_busy, d_label), (b_new, b_busy, b_label) = out
        return d_new, b_new, d_busy, d_label, b_busy, b_label

    @app.callback(
        Output("dis-out", "children", allow_duplicate=True),
        Output("dis-run-info", "children", allow_duplicate=True),
        Output("dis-board", "data", allow_duplicate=True),
        Output("dis-rendered", "data"),
        Input("dis-ready", "data"), State("dis-min-events", "value"),
        prevent_initial_call=True,
    )
    def _show_discovery_job(job_id, min_events):
        # Runs once per finished job, so the discovery area only redraws when there is something new.
        if not job_id:
            return (no_update,) * 4
        return (*show_job("dis", job_id, min_events), job_id)

    @app.callback(
        Output("bt-out", "children", allow_duplicate=True),
        Output("bt-run-info", "children", allow_duplicate=True),
        Output("bt-grid", "data", allow_duplicate=True),
        Output("bt-rendered", "data"),
        Input("bt-ready", "data"),
        prevent_initial_call=True,
    )
    def _show_grid_job(job_id):
        if not job_id:
            return (no_update,) * 4
        return (*show_job("bt", job_id), job_id)

    @app.callback(Output("bt-saved-run", "options"), Input("bt-grid", "data"), Input("bench-tabs", "value"))
    def _saved_grids(_grid, _tab):
        return [{"label": m["label"], "value": m["path"]} for m in list_exit_runs()]

    @app.callback(
        Output("bt-out", "children", allow_duplicate=True),
        Output("bt-run-info", "children", allow_duplicate=True),
        Output("bt-grid", "data", allow_duplicate=True),
        Input("bt-saved-open", "n_clicks"), State("bt-saved-run", "value"),
        prevent_initial_call=True,
    )
    def _open_saved_grid(_n, saved):
        if not saved:
            return no_update, note("Choose a saved grid first.", "warn"), no_update
        try:
            return render_mechanics(saved)
        except Exception as exc:
            return no_update, note(f"Cannot open saved grid: {exc}", "bad"), no_update

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
        State("invert-feature", "value"), State("derived-features", "value"),
        State("research-level-window", "data"),
        State("research-session", "data"),
        running=[(Output("load", "disabled"), True, False), (Output("fill", "disabled"), True, False)],
    )
    def _load(
        _n, _fill_n, target, custom, weighting, beta_lb, beta_dependent, features, start,
        invert_feature, derived, chart_window, session,
    ):
        try:
            work.start(session, "load", "Loading market history for the target and selected features")
            selected_target = target
            trade = resolve_target(selected_target, custom)
            fallback = None
            if weighting == "beta":
                trade = beta_package(trade)  # swap spreads: swap vs beta x Treasury
            if weighting == "beta" and len(trade.legs) < 2:
                weighting, fallback = "fixed", (
                    f"{trade.name} is one series with no second leg to hedge, so it loaded with fixed weighting.")
            features = list(dict.fromkeys([
                *(features or []),
                *(parse_derived(spec).name for spec in (derived or "").split(";") if spec.strip()),
            ]))
            panel = build_panel(trade, features, start=start or START,
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
        if fallback:
            view = html.Div([note(fallback, "warn"), view])
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


CRASH_LOG = Path(__file__).parent / "data" / "logs" / "crashes.log"


def record_native_crashes(path: Path = CRASH_LOG):
    """Write every thread's Python stack to ``path`` if the process dies natively.

    A crash inside compiled code (access violation, abort) kills the server
    with no traceback in the terminal; this log shows what each thread was
    running at that moment. Returns the open file, which must stay open.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    log = open(path, "a", encoding="utf-8")
    log.write(f"\n=== research server started {time.strftime('%Y-%m-%d %H:%M:%S')} · pid {os.getpid()} ===\n")
    log.flush()
    faulthandler.enable(file=log, all_threads=True)
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description="research bench")
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    crash_log = record_native_crashes()  # noqa: F841 - faulthandler writes to it until exit
    print(f"  Native crashes are recorded in {CRASH_LOG}")
    run(build_app(), port=args.port, host=args.host)


if __name__ == "__main__":
    main()
