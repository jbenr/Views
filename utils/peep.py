#!/usr/bin/env python3
"""peep.py — interactive DB explorer for the Views research database.

Run:
    python utils/peep.py [--port 8050]
    Open http://localhost:8050
"""
from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path

logging.getLogger("werkzeug").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dash import Dash, Input, Output, State, ALL, ctx, dash_table, dcc, html, no_update

from utils.helpers import _connect
from utils.viz import Viz, PlotlyViz, _range_window

# ── Viz renderer (no server — just the matplotlib render engine) ──────────────
_vr = object.__new__(PlotlyViz)
Viz.__init__(_vr)
plt.switch_backend("Agg")

RANGE_LABELS = ["1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y", "10Y", "ALL"]

# ── Palette — matches viz.py light theme ──────────────────────────────────────
BG     = "#FFFFFF"
PANEL  = "#F5F5F5"
BAR    = "#EBEBEB"
BORDER = "#DCDCDC"
TEXT   = "#333333"
DIM    = "#888888"
ACCENT = "#2980B9"   # Viz COLORS[3]
ORANGE = "#D35400"   # Viz COLORS[0]

_TEXT_TYPES = {"text", "character varying", "character", "USER-DEFINED"}
_NUM_TYPES  = {"double precision", "numeric", "real", "integer", "bigint", "smallint", "float"}

_SKIP_SCHEMAS = {
    "pg_catalog", "information_schema",
    "_timescaledb_internal", "_timescaledb_catalog",
    "_timescaledb_config", "_timescaledb_cache",
    "timescaledb_information", "timescaledb_experimental",
}

# ── DB helpers ────────────────────────────────────────────────────────────────

def db_tables() -> list[str]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT table_schema||'.'||table_name FROM information_schema.tables"
            " WHERE table_type='BASE TABLE' ORDER BY 1"
        )
        return [r[0] for r in cur.fetchall() if r[0].split(".")[0] not in _SKIP_SCHEMAS]


def db_cols(table: str) -> list[tuple[str, str]]:
    s, t = table.split(".", 1)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns"
            " WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
            (s, t),
        )
        return cur.fetchall()


def db_query(table: str, d_from: str, d_to: str, limit: int) -> pd.DataFrame:
    cols = db_cols(table)
    has_ts = any(c[0] == "ts" for c in cols)
    clauses, params = [], []
    if has_ts:
        if d_from:
            clauses.append('"ts">=%s'); params.append(d_from)
        if d_to:
            clauses.append('"ts"<=%s'); params.append(d_to)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    order = 'ORDER BY "ts" DESC' if has_ts else ""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {table} {where} {order} LIMIT %s", params + [limit])
        rows = cur.fetchall()
        col_names = [d.name for d in cur.description]
    df = pd.DataFrame(rows, columns=col_names)
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype(str)
        elif df[c].dtype == object:
            df[c] = df[c].fillna("").astype(str)
    return df


def _num_cols(cols: list[tuple]) -> list[str]:
    return [n for n, t in cols if t in _NUM_TYPES]


# ── Per-column filter helpers ─────────────────────────────────────────────────

def _col_kind(col: str, series: pd.Series) -> str:
    """Return 'date', 'numeric', 'categorical', or 'text' for a column."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    col_l = col.lower()
    if col_l == "ts" or col_l.endswith("_date") or col_l.endswith("_at"):
        return "date"
    return "categorical" if series.nunique() <= 100 else "text"


def _col_width(col: str, kind: str) -> int:
    if kind == "date":       return 115
    if kind == "numeric":    return 125
    if col == "status":      return 75
    if col in ("tenor",):    return 100
    if col in ("asset_class", "security_type"): return 115
    if col in ("cusip", "ticker", "source"):    return 110
    return 140


# ── Chart rendering ───────────────────────────────────────────────────────────

def _render_to_png(traces: dict[str, pd.Series], start, end, title=None) -> str | None:
    clean = {}
    for k, v in traces.items():
        s = v.dropna()
        if s.empty:
            continue
        if s.index.duplicated().any():
            s = s.groupby(level=0).last()
        clean[k] = s
    if not clean:
        return None

    merged = pd.concat(clean, axis=1)
    merged.index = pd.DatetimeIndex(merged.index)

    def render_fn(fig, ax, s, e):
        subset = merged.loc[s:e]
        for i, (label, series) in enumerate(subset.items()):
            color = _vr.colors[i % len(_vr.colors)]
            clean_s = series.dropna()
            if clean_s.empty:
                continue
            ax.plot(clean_s.index, clean_s, color=color, linewidth=1.5, label=label)
        _vr._style_ax(ax)
        _vr._legend(ax)
        _vr._format_dates(ax, s, e)
        ax.set_xlim(s, e)
        fig.subplots_adjust(bottom=0.15)

    return _vr._render_png(merged, render_fn, start, end, title=title)


# ── App ───────────────────────────────────────────────────────────────────────

TABLES = db_tables()
_ASSETS = Path(__file__).parent / "assets"
app = Dash(__name__, suppress_callback_exceptions=True, assets_folder=str(_ASSETS))
app.title = "peep"

_label = lambda txt: html.Span(
    txt, style={"color": DIM, "fontSize": 11, "marginRight": 4, "fontWeight": "bold",
                 "letterSpacing": "0.05em", "textTransform": "uppercase"}
)

def _dd(id_, placeholder, multi=False, width=260):
    return dcc.Dropdown(
        id=id_, placeholder=placeholder, multi=multi, clearable=True,
        style={"width": width, "fontSize": 12},
    )

_btn_base = {
    "minWidth": "38px", "height": "22px", "padding": "0 6px",
    "fontSize": "10px", "border": f"1px solid {BORDER}",
    "borderRadius": "2px", "background": PANEL, "color": TEXT,
    "cursor": "pointer", "fontFamily": "Arial, Helvetica, sans-serif",
}
_btn_active = {**_btn_base, "background": "#F1C40F", "color": "#000", "borderColor": "#B7950B", "fontWeight": "bold"}


app.layout = html.Div(
    style={"background": BG, "minHeight": "100vh",
           "fontFamily": "Arial, Helvetica, sans-serif", "color": TEXT},
    children=[

    # ── header bar ────────────────────────────────────────────────────────────
    html.Div(style={
        "background": PANEL, "borderBottom": f"2px solid {BORDER}",
        "padding": "10px 20px", "display": "flex", "alignItems": "center", "gap": 16,
    }, children=[
        html.Span("peep", style={"fontSize": 18, "fontWeight": "bold", "color": ORANGE,
                                   "letterSpacing": 3, "marginRight": 8}),
        dcc.Dropdown(
            id="tbl-select",
            options=[{"label": t, "value": t} for t in TABLES],
            value=TABLES[0] if TABLES else None,
            clearable=False,
            style={"width": 300, "fontSize": 12},
        ),
        _label("limit"),
        dcc.Input(id="limit-input", type="number", value=2000, min=1, max=100000,
                  style={"width": 72, "fontSize": 12, "padding": "3px 6px",
                         "border": f"1px solid {BORDER}", "borderRadius": 3}),
        _label("from"),
        dcc.Input(id="date-from", type="text", placeholder="YYYY-MM-DD", debounce=True,
                  style={"width": 105, "fontSize": 12, "padding": "3px 6px",
                         "border": f"1px solid {BORDER}", "borderRadius": 3}),
        _label("to"),
        dcc.Input(id="date-to", type="text", placeholder="YYYY-MM-DD", debounce=True,
                  style={"width": 105, "fontSize": 12, "padding": "3px 6px",
                         "border": f"1px solid {BORDER}", "borderRadius": 3}),
        html.Span(id="row-count",
                  style={"color": DIM, "fontSize": 11, "marginLeft": "auto", "fontStyle": "italic"}),
    ]),

    # ── filter row + table in one scroll container ────────────────────────────
    html.Div(style={"overflowX": "auto", "padding": "0 20px"}, children=[
        html.Div(id="filter-row", style={
            "display": "flex", "flexWrap": "nowrap", "gap": 0,
            "background": PANEL, "borderTop": f"1px solid {BORDER}",
            "borderLeft": f"1px solid {BORDER}", "borderRight": f"1px solid {BORDER}",
        }),
        html.Div(id="table-wrap"),
    ]),

    # ── plot controls ─────────────────────────────────────────────────────────
    html.Div(style={
        "background": PANEL, "borderTop": f"1px solid {BORDER}",
        "borderBottom": f"1px solid {BORDER}",
        "padding": "8px 20px", "display": "flex", "alignItems": "center", "gap": 14,
        "marginTop": 16,
    }, children=[
        _label("y"),  _dd("y-col",     "y-axis",   width=170),
        _label("x"),  _dd("x-col",     "x-axis",   width=140),
        _label("by"), _dd("color-col", "color by", width=170),
        html.Div(style={"marginLeft": "auto", "display": "flex", "gap": 3}, children=[
            html.Button(
                lbl,
                id={"type": "rng-btn", "label": lbl},
                n_clicks=0,
                style=(_btn_active if lbl == "ALL" else _btn_base),
            )
            for lbl in RANGE_LABELS
        ]),
    ]),

    # ── chart ─────────────────────────────────────────────────────────────────
    html.Div(style={"padding": "14px 20px 20px"}, children=[
        html.Img(id="main-chart-img",
                 style={"maxWidth": "100%", "display": "block",
                        "border": f"1px solid {BORDER}"}),
    ]),

    # ── stores ────────────────────────────────────────────────────────────────
    dcc.Store(id="data-store"),
    dcc.Store(id="cols-store"),
    dcc.Store(id="range-store", data="ALL"),
    dcc.Store(id="filtered-store"),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("y-col", "options"), Output("x-col", "options"), Output("color-col", "options"),
    Output("y-col", "value"),   Output("x-col", "value"),   Output("color-col", "value"),
    Output("cols-store", "data"),
    Input("tbl-select", "value"),
)
def on_table_change(table):
    if not table:
        return [], [], [], None, None, None, None
    cols = db_cols(table)
    ncols = _num_cols(cols)
    all_names = [c[0] for c in cols]
    text_cols = [n for n, t in cols if t in _TEXT_TYPES]
    has_ts = "ts" in all_names
    y_opts  = [{"label": c, "value": c} for c in ncols]
    x_names = (["ts"] + [n for n in all_names if n != "ts"]) if has_ts else all_names
    x_opts  = [{"label": c, "value": c} for c in x_names]
    cb_opts = [{"label": c, "value": c} for c in text_cols]
    return (
        y_opts, x_opts, cb_opts,
        ncols[0] if ncols else None,
        "ts" if has_ts else (all_names[0] if all_names else None),
        text_cols[0] if text_cols else None,
        json.dumps(all_names),
    )


@app.callback(
    Output("data-store",  "data"),
    Output("row-count",   "children"),
    Input("tbl-select",   "value"),
    Input("date-from",    "value"),
    Input("date-to",      "value"),
    Input("limit-input",  "value"),
)
def load_data(table, d_from, d_to, limit):
    if not table:
        return None, ""
    limit = int(limit or 2000)
    df = db_query(table, d_from or "", d_to or "", limit)
    count = f"{len(df):,} rows"
    if len(df) == limit:
        count += " (limit)"
    return df.to_json(date_format="iso", orient="split"), count


_inp = lambda **kw: {
    "fontSize": 11, "padding": "3px 6px", "width": "100%", "boxSizing": "border-box",
    "border": f"1px solid {BORDER}", "borderRadius": 2, "background": BG, "color": TEXT,
    "fontFamily": "Arial, Helvetica, sans-serif",
    **kw,
}

@app.callback(
    Output("filter-row", "children"),
    Input("data-store", "data"),
)
def gen_filter_row(data_json):
    if not data_json:
        return []
    df = pd.read_json(io.StringIO(data_json), orient="split")
    items = []
    for col in df.columns:
        kind  = _col_kind(col, df[col])
        width = _col_width(col, kind)
        cell_style = {
            "width": f"{width}px", "minWidth": f"{width}px", "maxWidth": f"{width}px",
            "padding": "5px 6px", "borderRight": f"1px solid {BORDER}",
            "boxSizing": "border-box", "overflow": "hidden",
        }

        if kind == "categorical":
            vals = sorted(df[col].dropna().unique().tolist(), key=str)[:500]
            ctrl = dcc.Dropdown(
                id={"type": "col-filter", "col": col},
                options=[{"label": str(v), "value": str(v)} for v in vals],
                value=None, multi=True, placeholder="all", clearable=True,
                style={"fontSize": 11},
            )
        elif kind == "date":
            ctrl = dcc.Input(
                id={"type": "col-filter", "col": col},
                type="text", placeholder="e.g. 2026-06", debounce=True,
                style=_inp(),
            )
        elif kind == "numeric":
            ctrl = html.Div([
                dcc.Input(id={"type": "col-filter-min", "col": col},
                          type="number", placeholder="min", debounce=True,
                          style=_inp(width="50%")),
                dcc.Input(id={"type": "col-filter-max", "col": col},
                          type="number", placeholder="max", debounce=True,
                          style=_inp(width="50%", borderLeft="none")),
            ], style={"display": "flex"})
        else:
            ctrl = dcc.Input(
                id={"type": "col-filter", "col": col},
                type="text", placeholder="search…", debounce=True,
                style=_inp(),
            )

        items.append(html.Div([
            html.Div(col, style={"fontSize": 9, "color": DIM, "textTransform": "uppercase",
                                  "letterSpacing": "0.05em", "marginBottom": 2}),
            ctrl,
        ], style=cell_style))
    return items


@app.callback(
    Output("table-wrap",     "children"),
    Output("filtered-store", "data"),
    Input("data-store", "data"),
    Input({"type": "col-filter",     "col": ALL}, "value"),
    Input({"type": "col-filter-min", "col": ALL}, "value"),
    Input({"type": "col-filter-max", "col": ALL}, "value"),
    State({"type": "col-filter",     "col": ALL}, "id"),
    State({"type": "col-filter-min", "col": ALL}, "id"),
    State({"type": "col-filter-max", "col": ALL}, "id"),
)
def render_table(data_json, fvals, min_vals, max_vals, fids, min_ids, max_ids):
    if not data_json:
        return html.Div("select a table",
                        style={"color": DIM, "padding": 20, "fontStyle": "italic"}), None
    df = pd.read_json(io.StringIO(data_json), orient="split")

    # categorical / date / text filters
    for val, id_ in zip(fvals or [], fids or []):
        col = id_["col"]
        if not val or col not in df.columns:
            continue
        kind = _col_kind(col, df[col])
        if kind == "categorical":
            df = df[df[col].astype(str).isin([str(v) for v in val])]
        else:
            # date prefix or text contains
            df = df[df[col].astype(str).str.contains(str(val), case=False, na=False)]

    # numeric min/max filters
    for val, id_ in zip(min_vals or [], min_ids or []):
        col = id_["col"]
        if val is not None and col in df.columns:
            df = df[pd.to_numeric(df[col], errors="coerce") >= float(val)]
    for val, id_ in zip(max_vals or [], max_ids or []):
        col = id_["col"]
        if val is not None and col in df.columns:
            df = df[pd.to_numeric(df[col], errors="coerce") <= float(val)]

    filtered_json = df.to_json(date_format="iso", orient="split")

    # per-column widths to match filter row
    col_widths = [(_col_width(c, _col_kind(c, df[c])) if c in df.columns else 130)
                  for c in df.columns]
    width_conditional = [
        {"if": {"column_id": col}, "width": f"{w}px", "minWidth": f"{w}px", "maxWidth": f"{w}px"}
        for col, w in zip(df.columns, col_widths)
    ]

    return dash_table.DataTable(
        id="main-table",
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=30,
        page_action="native",
        sort_action="native",
        sort_mode="multi",
        filter_action="none",
        cell_selectable=False,
        style_table={
            "overflowX": "unset",
            "border": f"1px solid {BORDER}",
            "minWidth": f"{sum(col_widths)}px",
        },
        style_header={
            "backgroundColor": PANEL, "color": TEXT, "fontWeight": "bold",
            "fontSize": 11, "textTransform": "uppercase", "letterSpacing": "0.05em",
            "border": f"1px solid {BORDER}", "padding": "7px 8px", "cursor": "pointer",
        },
        style_cell={
            "backgroundColor": BG, "color": TEXT, "border": f"1px solid {BORDER}",
            "fontSize": 12, "padding": "5px 8px",
            "fontFamily": "'Courier New', 'Lucida Console', monospace",
            "textOverflow": "ellipsis", "overflow": "hidden",
        },
        style_cell_conditional=width_conditional,
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#FAFAFA"},
            {"if": {"state": "active"}, "backgroundColor": BG, "border": f"1px solid {BORDER}"},
        ],
    ), filtered_json


@app.callback(
    Output("range-store", "data"),
    Output({"type": "rng-btn", "label": ALL}, "style"),
    Input({"type": "rng-btn", "label": ALL}, "n_clicks"),
    State({"type": "rng-btn", "label": ALL}, "id"),
    prevent_initial_call=True,
)
def on_range_click(clicks, ids):
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        return no_update, [no_update] * len(ids)
    label = trig["label"]
    styles = [(_btn_active if d["label"] == label else _btn_base) for d in ids]
    return label, styles


@app.callback(
    Output("main-chart-img", "src"),
    Input("filtered-store", "data"),
    Input("y-col",          "value"),
    Input("x-col",          "value"),
    Input("color-col",      "value"),
    Input("range-store",    "data"),
)
def render_chart(data_json, y_col, x_col, color_col, range_label):
    if not data_json or not y_col or not x_col:
        return None

    df = pd.read_json(io.StringIO(data_json), orient="split")
    if y_col not in df.columns or x_col not in df.columns:
        return None

    try:
        df[x_col] = pd.to_datetime(df[x_col], format="mixed")
    except Exception:
        return None

    try:
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    except Exception:
        return None

    df = df.dropna(subset=[x_col, y_col]).sort_values(x_col)
    if df.empty:
        return None

    traces: dict[str, pd.Series] = {}
    if color_col and color_col in df.columns:
        for grp, sub in df.groupby(color_col, sort=True):
            traces[str(grp)] = sub.set_index(x_col)[y_col]
    else:
        traces[y_col] = df.set_index(x_col)[y_col]

    full_idx = pd.DatetimeIndex(sorted({x for s in traces.values() for x in s.index}))
    dummy = pd.DataFrame(index=full_idx)
    start, end = _range_window(dummy, range_label or "ALL")

    png = _render_to_png(traces, start, end, title=y_col)
    return f"data:image/png;base64,{png}" if png else None


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"  peep  →  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
