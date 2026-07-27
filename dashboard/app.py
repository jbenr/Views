"""Live signal dashboard -- trading overview plus signal deep dives.

The Live Overview tab groups trusted signals by traded target and answers the
desk questions first: is a position open, what does the latest reading say,
is its gate open, and how has the frozen backtest behaved?

The Signal Deep Dive tab selects one promoted signal at a time and exposes the
full chart, gate, PnL, and trade-history card. Two buttons drive its data:

    Re-pull data       fresh Strategy.load_data(), cached to disk.
                        No ledger write.
    Re-run analysis     Strategy.compute() on the cached data, logged as
                        one timestamped row in the signal ledger.

No background loop or auto-refresh -- nothing changes until you click. Per-card
time-frame buttons, the trade-table pager, and a "snap chart to view" zoom
button also trigger a re-render, but touch neither the DB nor the ledger --
they just re-slice/re-page what's already cached.

    python -m dashboard.registry --promote book.curve.tens_10s30s
    mamba run -n 2s10s python -m dashboard.app
    open http://127.0.0.1:8052

See README.md for the full workflow.
"""

from __future__ import annotations

import argparse
import re

import dash
import pandas as pd
from dash import Input, Output, State, ctx, dash_table, dcc, html

from dashboard import runner
from dashboard.charts import (
    DEFAULT_WINDOW,
    WINDOW_PRESETS,
    gate_chart,
    level_chart,
    pnl_chart,
    signal_chart,
)
from dashboard.ledger import SignalLedger
from dashboard.registry import LiveRegistry
from utils.research_app import (
    BORDER, C0, C1, DIM, ORANGE, PANEL, TEXT, make_app, run, stat_block,
)

REGISTRY = LiveRegistry()
LEDGER = SignalLedger()

TRADES_PER_PAGE = 6
SNAP_LEFT_BUFFER_BDAYS = 5

TRADE_TABLE_COLS = [
    "entry_date", "exit_date", "direction",
    "entry_level", "exit_level", "target_lvl", "expected_return_bps",
    "pnl_bps", "entry_half_life", "bars_held", "exit_reason",
]
TRADE_TABLE_HEADERS = {
    "entry_date": "Entry", "exit_date": "Exit", "direction": "Dir",
    "entry_level": "Entry Lvl", "exit_level": "Exit Lvl",
    "target_lvl": "Target Lvl", "expected_return_bps": "Exp Ret (bps)",
    "pnl_bps": "Net PnL (bps)", "entry_half_life": "Half-life (d)",
    "bars_held": "Held (d)", "exit_reason": "Exit Reason",
}
TRADE_TABLE_ROUND = {
    "entry_level": 2, "exit_level": 2, "target_lvl": 2,
    "expected_return_bps": 1, "pnl_bps": 1, "entry_half_life": 1,
}

OVERVIEW_COLUMNS = [
    ("name", "Signal"),
    ("feature", "Feature"),
    ("position", "Position"),
    ("reading", "Latest Reading"),
    ("signal_value", "Live Signal"),
    ("entry_threshold", "Entry Threshold"),
    ("gate_status", "Gate"),
    ("live_pnl_bps", "Net PnL (bps)"),
    ("sharpe", "Sharpe"),
    ("n_trades", "Trades"),
    ("hit_rate", "Hit Rate (%)"),
    ("max_drawdown_bps", "Max DD (bps)"),
    ("data_asof", "Data As-Of"),
]


def _slug(signal_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", signal_id.lower()).strip("-")


def _display_name(row: dict) -> str:
    """Canonical user-facing signal name: traded target first, feature second.
    Always suffixed with a label -- curated variants use their own, sweep-top
    promotions fall back to "default" -- so every row has the same shape."""
    base = f"{row['target']}_{row['feature']}"
    variant_label = row.get("variant_label") or "default"
    return f"{base} · {variant_label}"


def _row_id(row: dict) -> str:
    """Registry identity, with module fallback for legacy/test rows."""
    return row.get("signal_id") or row["module"]


def _exit_summary(row: dict) -> str:
    style = row["exit_style"]
    param = float(row["exit_param"])
    if style == "revert_frac":
        return (
            f"exit=revert_frac={param:g} "
            f"(after {param:.0%} of entry signal reverts toward zero)"
        )
    if style == "half_life_frac":
        return f"exit=half_life_frac={param:g} ({param:g}× entry half-life)"
    if style == "band":
        units = "bps" if row["entry_signal"] == "residual" else "z"
        return f"exit=band={param:g}{units} (inside ±{param:g}{units})"
    return f"exit={style}={param:g}"


def _param_summary(row: dict) -> str:
    bits = [f"{row['entry_signal']} beta_lb={row['beta_lb']}"]
    if row.get("ou_lb"):
        bits.append(f"ou_lb={row['ou_lb']}")
    bits.append(f"entry={row['entry_threshold']:g}")
    bits.append(_exit_summary(row))
    if row.get("stop_loss_bps") is not None:
        bits.append(f"hard stop={float(row['stop_loss_bps']):g}bps")
    if row.get("gate") and row["gate"] != "(none)":
        bits.append(f"gate={row['gate']}")
    return "  ·  ".join(bits)


def _fnum(value, fmt: str, suffix: str = "") -> str:
    """Format a possibly-missing/NaN metric, else '—'."""
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if value != value:  # NaN
        return "—"
    return format(value, fmt) + suffix


def _round(value, ndigits: int):
    """Round a possibly-missing/NaN metric, keeping it numeric so sortable
    table columns still sort correctly (unlike _fnum's formatted strings)."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return round(value, ndigits)


def _live_pnl_bps(trades, open_entry: dict | None) -> float:
    """Realized PnL from closed trades plus unrealized PnL from the open
    position, if any -- the running live total since promotion."""
    total = 0.0
    if trades is not None and not trades.is_empty():
        total += float(trades["pnl_bps"].sum())
    if open_entry is not None:
        total += float(open_entry["pnl_bps"])
    return total


def _trade_page_rows(trades, page: int) -> list[dict]:
    """5-row page slice, formatted for dash_table.DataTable's `data` prop."""
    sliced = trades.slice(page, TRADES_PER_PAGE)
    rows = []
    for row in sliced.to_dicts():
        out = {}
        for col in TRADE_TABLE_COLS:
            val = row.get(col)
            if col in ("entry_date", "exit_date") and val is not None:
                val = str(val)
            elif col in TRADE_TABLE_ROUND and val is not None:
                val = round(val, TRADE_TABLE_ROUND[col])
            out[col] = val
        rows.append(out)
    return rows


def _open_trade_row(open_entry: dict) -> dict:
    """Format the live open position as a table row, same shape as
    _trade_page_rows() output -- pinned above the paginated closed trades."""
    out = {}
    for col in TRADE_TABLE_COLS:
        val = open_entry.get(col)
        if col == "exit_date":
            out[col] = "open"
        elif col == "entry_date" and val is not None:
            out[col] = str(val)
        elif col in TRADE_TABLE_ROUND and val is not None:
            out[col] = round(val, TRADE_TABLE_ROUND[col])
        else:
            out[col] = val
    return out


def _visible_trade_range(trades, open_entry: dict | None, page: int, data_asof):
    """Date span covering exactly what's currently shown in the trade table --
    the open row (page 0 only) plus the current page of closed trades."""
    dates = []
    if trades is not None and not trades.is_empty():
        for row in trades.slice(page, TRADES_PER_PAGE).to_dicts():
            dates.append(row["entry_date"])
            dates.append(row["exit_date"])
    if open_entry is not None and page == 0:
        dates.append(open_entry["entry_date"])
        dates.append(data_asof)
    dates = [d for d in dates if d is not None]
    if not dates:
        return None
    start = pd.Timestamp(min(dates)) - pd.offsets.BDay(SNAP_LEFT_BUFFER_BDAYS)
    return start, max(dates)


def _gate_status(state: dict) -> str:
    params = state["params"]
    if params.get("gate") is None:
        return "none"
    last = state["last"]
    percentile = last.get("gate_percentile")
    if percentile is None or percentile != percentile:
        return "warming up"
    status = "open" if bool(last.get("gate_allow")) else "closed"
    pct = round(float(percentile) * 100)
    suffix = "th" if 10 <= pct % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(
        pct % 10, "th"
    )
    return f"{status} · {pct}{suffix} pct"


def _overview_snapshot(row: dict) -> dict:
    """One desk-level status row for a promoted signal."""
    base = {
        "signal_id": _row_id(row),
        "module": row["module"],
        "target": row["target"],
        "name": _display_name(row),
        "feature": row["feature"],
        "position": "UNAVAILABLE",
        "reading": "—",
        "signal_value": "—",
        "entry_threshold": "—",
        "gate_status": "—",
        "live_pnl_bps": None,
        "sharpe": _round(row.get("sharpe"), 2),
        "n_trades": _round(row.get("n_trades"), 0),
        "hit_rate": _round(row.get("hit_rate") * 100 if row.get("hit_rate") is not None else None, 1),
        "max_drawdown_bps": _round(row.get("max_drawdown_bps"), 1),
        "data_asof": "—",
    }
    try:
        signal_id = _row_id(row)
        state = runner.compute_signal(signal_id)
        trades, open_entry = runner.trade_history(signal_id, state)
    except RuntimeError as exc:
        return {**base, "reading": str(exc)}

    params = state["params"]
    signal_value = state["last"].get("signal")
    units = "bps" if params["entry_signal"] == "residual" else "z"
    threshold = float(params["entry_threshold"])
    reading_str = (
        f"{float(signal_value):+.2f}{units}"
        if signal_value is not None and signal_value == signal_value
        else "warming up"
    )
    entry_threshold_str = f"±{threshold:g}{units}"
    position = (
        f"{open_entry['direction'].upper()} OPEN"
        if open_entry is not None
        else "FLAT"
    )
    return {
        **base,
        "position": position,
        "reading": state["fired"].upper(),
        "signal_value": reading_str,
        "entry_threshold": entry_threshold_str,
        "gate_status": _gate_status(state).upper(),
        "live_pnl_bps": round(_live_pnl_bps(trades, open_entry), 1),
        "data_asof": str(state["data_asof"]),
    }


def _overview_table(target: str, rows: list[dict]) -> html.Div:
    snapshots = [_overview_snapshot(row) for row in rows]
    return html.Div(
        style={"marginBottom": 22},
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "baseline",
                    "gap": 10,
                    "marginBottom": 7,
                },
                children=[
                    html.Span(
                        target,
                        style={
                            "fontSize": 15,
                            "fontWeight": "bold",
                            "color": ORANGE,
                            "textTransform": "uppercase",
                        },
                    ),
                    html.Span(
                        f"{len(rows)} trusted signal{'s' if len(rows) != 1 else ''}",
                        style={"fontSize": 11, "color": DIM},
                    ),
                ],
            ),
            dash_table.DataTable(
                columns=[{"name": label, "id": key} for key, label in OVERVIEW_COLUMNS],
                data=snapshots,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "backgroundColor": "#FFFFFF",
                    "color": TEXT,
                    "border": f"1px solid {BORDER}",
                    "fontSize": 11,
                    "padding": "7px 9px",
                    "textAlign": "center",
                    "whiteSpace": "nowrap",
                },
                style_header={
                    "backgroundColor": "#EDEDED",
                    "fontWeight": "bold",
                    "border": f"1px solid {BORDER}",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": '{position} = "LONG OPEN"', "column_id": "position"},
                        "color": C1,
                        "fontWeight": "bold",
                    },
                    {
                        "if": {"filter_query": '{position} = "SHORT OPEN"', "column_id": "position"},
                        "color": C0,
                        "fontWeight": "bold",
                    },
                    {
                        "if": {"filter_query": '{reading} = "LONG"', "column_id": "reading"},
                        "color": C1,
                        "fontWeight": "bold",
                    },
                    {
                        "if": {"filter_query": '{reading} = "SHORT"', "column_id": "reading"},
                        "color": C0,
                        "fontWeight": "bold",
                    },
                    {
                        "if": {"filter_query": '{gate_status} contains "CLOSED"', "column_id": "gate_status"},
                        "color": C0,
                    },
                    {
                        "if": {"filter_query": "{live_pnl_bps} > 0", "column_id": "live_pnl_bps"},
                        "color": C1,
                        "fontWeight": "bold",
                    },
                    {
                        "if": {"filter_query": "{live_pnl_bps} < 0", "column_id": "live_pnl_bps"},
                        "color": C0,
                        "fontWeight": "bold",
                    },
                ],
            ),
        ],
    )


def _overview_content(rows: list[dict]) -> list:
    if not rows:
        return [
            html.Div(
                "No signals promoted yet.",
                style={"color": DIM, "padding": "24px 0"},
            )
        ]
    targets = sorted({row["target"] for row in rows})
    return [
        _overview_table(
            target,
            sorted(
                [row for row in rows if row["target"] == target],
                key=_display_name,
            ),
        )
        for target in targets
    ]


def _trade_table_block(trades, page: int, open_entry: dict | None, slug: str) -> html.Div:
    n = 0 if trades is None or trades.is_empty() else len(trades)
    if n == 0 and open_entry is None:
        return html.Div("no closed trades yet", style={"color": DIM, "padding": "8px 0"})

    rows = ([_open_trade_row(open_entry)] if open_entry is not None and page == 0 else [])
    rows += _trade_page_rows(trades, page) if n else []
    lo, hi = (page + 1, min(page + TRADES_PER_PAGE, n)) if n else (0, 0)

    return html.Div(
        style={"marginTop": 14},
        children=[
            dash_table.DataTable(
                columns=[{"name": TRADE_TABLE_HEADERS[c], "id": c} for c in TRADE_TABLE_COLS],
                data=rows,
                style_table={"overflowX": "auto"},
                style_cell={
                    "backgroundColor": PANEL, "color": TEXT,
                    "border": f"1px solid {BORDER}", "fontSize": 11,
                    "padding": "4px 8px", "textAlign": "center",
                },
                style_header={
                    "backgroundColor": "#EDEDED", "fontWeight": "bold",
                    "border": f"1px solid {BORDER}",
                },
                style_data_conditional=[
                    {
                        "if": {
                            "filter_query": '{direction} = "long"',
                            "column_id": "direction",
                        },
                        "color": C1,
                        "fontWeight": "bold",
                    },
                    {
                        "if": {
                            "filter_query": '{direction} = "short"',
                            "column_id": "direction",
                        },
                        "color": C0,
                        "fontWeight": "bold",
                    },
                    {"if": {"filter_query": "{pnl_bps} > 0", "column_id": "pnl_bps"},
                     "color": C1, "fontWeight": "bold"},
                    {"if": {"filter_query": "{pnl_bps} < 0", "column_id": "pnl_bps"},
                     "color": C0, "fontWeight": "bold"},
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": 8, "alignItems": "center", "marginTop": 8},
                children=[
                    html.Button("< prev", id=f"trades-prev-{slug}", n_clicks=0,
                                style=_btn_style()),
                    html.Button("next >", id=f"trades-next-{slug}", n_clicks=0,
                                style=_btn_style()),
                    html.Button("Snap chart to view", id=f"snap-{slug}", n_clicks=0,
                                style=_btn_style()),
                    html.Span(
                        f"showing {lo}-{hi} of {n} closed trades" if n else "no closed trades yet",
                        style={"fontSize": 11, "color": DIM},
                    ),
                ],
            ),
        ],
    )


def _card_sections(
    signal_id: str,
    state: dict | None = None,
    trades=None,
    open_entry: dict | None = None,
    window: str = DEFAULT_WINDOW,
    page: int = 0,
    date_range: tuple | None = None,
) -> tuple[html.Div, html.Div]:
    row = REGISTRY.get(signal_id)
    if row is None:
        return (
            html.Div(),
            html.Div(f"{signal_id}: no longer promoted", style={"color": DIM}),
        )

    ledger_row = LEDGER.latest(signal_id)
    error = None
    if state is None:
        # initial non-callback render at build_app() time -- callback-driven
        # renders always pass a precomputed state/trades pair down.
        try:
            state = runner.compute_signal(signal_id)
            trades, open_entry = runner.trade_history(signal_id, state)
        except RuntimeError as exc:
            error = str(exc)

    live_pnl = _live_pnl_bps(trades, open_entry) if state and not error else None
    stats = [
        stat_block("data as-of", str(state["data_asof"]) if state else "—"),
        stat_block("last analysis run", ledger_row["run_ts"] if ledger_row else "never"),
        stat_block(
            "reading",
            state["fired"] if state else "—",
            alert=bool(state and state["fired"] not in ("flat", "flat (gated)")),
        ),
        *(
            [stat_block("gate", _gate_status(state))]
            if state and state["params"].get("gate") is not None
            else []
        ),
        stat_block("live pnl", _fnum(live_pnl, "+.1f", " bps"),
                    alert=bool(live_pnl and live_pnl < 0)),
        stat_block("params", _param_summary(row)),
    ]
    last = state["last"] if state else {}
    backtest_stats = [
        stat_block("sharpe", _fnum(row.get("sharpe"), ".2f")),
        stat_block("n trades", _fnum(row.get("n_trades"), ".0f")),
        stat_block("hit rate", _fnum(row.get("hit_rate"), ".0%")),
        stat_block("max drawdown", _fnum(row.get("max_drawdown_bps"), "+.1f", " bps")),
        stat_block("half-life", _fnum(last.get("half_life"), ".1f", "d")),
        stat_block("r²", _fnum(last.get("r2"), ".2f")),
        stat_block("beta", _fnum(last.get("beta"), "+.3f")),
    ]

    if error:
        body = [html.Div(error, style={"color": ORANGE, "padding": "12px 0"})]
    else:
        window_bars = WINDOW_PRESETS.get(window, WINDOW_PRESETS[DEFAULT_WINDOW])
        level_png = level_chart(
            state["data"], state["strategy"].target,
            trades=trades, open_entry=open_entry,
            window_bars=window_bars, date_range=date_range,
        )
        sig_png = signal_chart(
            state["data"], state["signal_frame"],
            state["params"]["entry_signal"], state["params"]["entry_threshold"],
            window_bars=window_bars, date_range=date_range, fired=state["fired"],
        )
        gate_png = gate_chart(
            state["data"],
            state["signal_frame"],
            state["params"].get("gate"),
            window_bars=window_bars,
            date_range=date_range,
        )
        pnl_png = pnl_chart(
            state["equity_curve"],
            window_bars=window_bars,
            date_range=date_range,
        )
        chart_pngs = [level_png, sig_png]
        if gate_png is not None:
            chart_pngs.append(gate_png)
        chart_pngs.append(pnl_png)
        body = [
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": 10},
                children=[
                    html.Img(
                        src=f"data:image/png;base64,{png}",
                        style={"width": "100%", "border": f"1px solid {BORDER}"},
                    )
                    for png in chart_pngs
                ],
            ),
            _trade_table_block(trades, page, open_entry, _slug(signal_id)),
        ]

    summary = html.Div([
        html.Div(stats, style={"display": "flex", "gap": 28, "flexWrap": "wrap",
                                "padding": "8px 2px 8px"}),
        html.Div(backtest_stats, style={"display": "flex", "gap": 28, "flexWrap": "wrap",
                                         "padding": "0 2px 14px",
                                         "borderTop": f"1px solid {BORDER}",
                                         "marginTop": 4, "paddingTop": 8}),
    ])
    return summary, html.Div(body)


def _btn_style(primary: bool = False) -> dict:
    return {
        "padding": "6px 14px", "fontSize": 12, "cursor": "pointer",
        "border": f"1px solid {ORANGE if primary else BORDER}",
        "background": ORANGE if primary else "#FFFFFF",
        "color": "#FFFFFF" if primary else TEXT,
        "borderRadius": 3,
    }


def _card(row: dict) -> html.Div:
    module = row["module"]
    signal_id = _row_id(row)
    slug = _slug(signal_id)
    summary, analysis = _card_sections(signal_id)
    return html.Div(
        style={"border": f"1px solid {BORDER}", "background": PANEL,
               "padding": "14px 18px", "marginBottom": 18},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "baseline", "gap": 12,
                       "marginBottom": 4},
                children=[
                    html.Span(_display_name(row), style={"fontSize": 15, "fontWeight": "bold",
                                                         "color": ORANGE}),
                    html.Span(f"{row['target']} ~ {row['feature']}",
                              style={"fontSize": 12, "color": DIM}),
                    html.Span(module, style={"fontSize": 11, "color": DIM,
                                              "marginLeft": "auto", "fontStyle": "italic"}),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": 8, "marginBottom": 4,
                       "alignItems": "center", "flexWrap": "wrap"},
                children=[
                    html.Button("Re-pull data", id=f"pull-{slug}", n_clicks=0,
                                style=_btn_style()),
                    html.Button("Re-run analysis", id=f"run-{slug}", n_clicks=0,
                                style=_btn_style(primary=True)),
                    dcc.Store(id=f"window-{slug}", data=DEFAULT_WINDOW),
                    dcc.Store(id=f"snap-range-{slug}", data=None),
                    dcc.Store(id=f"trades-page-{slug}", data=0),
                ],
            ),
            html.Div(id=f"card-summary-{slug}", children=summary),
            html.Div(
                style={
                    "display": "flex",
                    "gap": 8,
                    "marginBottom": 8,
                    "alignItems": "center",
                    "flexWrap": "wrap",
                },
                children=[
                    html.Span(
                        "Chart window",
                        style={
                            "fontSize": 10,
                            "color": DIM,
                            "textTransform": "uppercase",
                            "marginRight": 2,
                        },
                    ),
                    *[
                        html.Button(key, id=f"window-{slug}-{key}", n_clicks=0,
                                    style=_btn_style(primary=(key == DEFAULT_WINDOW)))
                        for key in WINDOW_PRESETS
                    ],
                ],
            ),
            html.Div(id=f"card-analysis-{slug}", children=analysis),
        ],
    )


def build_app() -> dash.Dash:
    reg_df = REGISTRY.list()
    rows = (
        []
        if reg_df.is_empty()
        else reg_df.sort("target", "family", "name").to_dicts()
    )
    selected_signal = _row_id(rows[0]) if rows else None
    tab_style = {
        "padding": "10px 18px",
        "fontSize": 12,
        "fontWeight": "bold",
        "background": PANEL,
        "border": f"1px solid {BORDER}",
        "color": DIM,
    }
    selected_tab_style = {
        **tab_style,
        "background": "#FFFFFF",
        "color": ORANGE,
        "borderTop": f"3px solid {ORANGE}",
    }

    overview_tab = html.Div(
        style={"padding": "18px 24px"},
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": 14,
                },
                children=[
                    html.Div(
                        [
                            html.Div(
                                "LIVE TRADING OVERVIEW",
                                style={"fontSize": 14, "fontWeight": "bold"},
                            ),
                            html.Div(
                                "Grouped by traded target · positions are exact-engine state",
                                style={"fontSize": 11, "color": DIM, "marginTop": 2},
                            ),
                        ]
                    ),
                    html.Button(
                        "Refresh live status",
                        id="refresh-overview",
                        n_clicks=0,
                        style={**_btn_style(primary=True), "marginLeft": "auto"},
                    ),
                ],
            ),
            dcc.Loading(
                html.Div(id="overview-content", children=_overview_content(rows)),
                type="dot",
                color=ORANGE,
            ),
        ],
    )

    deep_dive_tab = html.Div(
        style={"padding": "18px 24px"},
        children=(
            [
                html.Div(
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": 14,
                        "marginBottom": 14,
                    },
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    "SIGNAL DEEP DIVE",
                                    style={"fontSize": 14, "fontWeight": "bold"},
                                ),
                                html.Div(
                                    "Charts, causal gate state, exact PnL, and trade history",
                                    style={"fontSize": 11, "color": DIM, "marginTop": 2},
                                ),
                            ]
                        ),
                        dcc.Dropdown(
                            id="deep-dive-signal",
                            options=[
                                {
                                    "label": _display_name(row),
                                    "value": _row_id(row),
                                }
                                for row in rows
                            ],
                            value=selected_signal,
                            clearable=False,
                            style={"width": 360, "marginLeft": "auto", "fontSize": 12},
                        ),
                    ],
                ),
                *[
                    html.Div(
                        id=f"deep-card-{_slug(_row_id(row))}",
                        style={
                            "display": "block"
                            if _row_id(row) == selected_signal
                            else "none"
                        },
                        children=_card(row),
                    )
                    for row in rows
                ],
            ]
            if rows
            else [
                html.Div(
                    "No signals promoted yet -- "
                    "python -m dashboard.registry --promote <module>",
                    style={"color": DIM, "padding": "24px 0"},
                )
            ]
        ),
    )

    body = html.Div(
        children=[
            dcc.Tabs(
                id="dashboard-tabs",
                value="live-overview",
                children=[
                    dcc.Tab(
                        label="Signal",
                        value="live-overview",
                        style=tab_style,
                        selected_style=selected_tab_style,
                        children=overview_tab,
                    ),
                    dcc.Tab(
                        label="Dig",
                        value="signal-deep-dive",
                        style=tab_style,
                        selected_style=selected_tab_style,
                        children=deep_dive_tab,
                    ),
                ],
            )
        ]
    )

    app = make_app(
        title="LIVE",
        subtitle="trusted live signals · overview and exact-engine deep dives",
        data_info=f"{len(rows)} live signal{'s' if len(rows) != 1 else ''}",
        sliders=[],
        body=body,
    )

    window_keys = list(WINDOW_PRESETS)

    if rows:
        app.callback(
            *[
                Output(f"deep-card-{_slug(_row_id(row))}", "style")
                for row in rows
            ],
            Input("deep-dive-signal", "value"),
        )(
            lambda selected: [
                {
                    "display": "block"
                    if _row_id(row) == selected
                    else "none"
                }
                for row in rows
            ]
        )

        app.callback(
            Output("overview-content", "children"),
            Input("refresh-overview", "n_clicks"),
            prevent_initial_call=True,
        )(lambda *_clicks: _overview_content(rows))

    for row in rows:
        signal_id = _row_id(row)
        slug = _slug(signal_id)
        window_prefix = f"window-{slug}-"

        def _update(
            _pull,
            _run,
            *rest,
            signal_id=signal_id,
            slug=slug,
            window_prefix=window_prefix,
        ):
            *_window_clicks, _prev, _next, _snap, window, snap_range, page = rest
            trigger = ctx.triggered_id or ""
            no_btn_styles = [dash.no_update] * len(window_keys)
            try:
                if trigger.startswith("pull-"):
                    runner.pull_data(signal_id)
                    page = 0
                elif trigger.startswith("run-"):
                    runner.run_analysis(signal_id)
                    page = 0
                elif trigger.startswith(window_prefix):
                    window = trigger[len(window_prefix):]
                    snap_range = None
                    page = 0
                state = runner.compute_signal(signal_id)
                trades, open_entry = runner.trade_history(signal_id, state)
            except RuntimeError as exc:
                err = html.Div(str(exc), style={"color": ORANGE, "padding": "12px 0"})
                return (html.Div(), err, 0, window, snap_range, *no_btn_styles)

            n = 0 if trades is None or trades.is_empty() else len(trades)
            last_page = max(0, (n - 1) // TRADES_PER_PAGE * TRADES_PER_PAGE) if n else 0
            if trigger == f"trades-prev-{slug}":
                page = max(0, page - TRADES_PER_PAGE)
            elif trigger == f"trades-next-{slug}":
                page = min(page + TRADES_PER_PAGE, last_page)
            page = min(page, last_page)

            if trigger == f"snap-{slug}":
                rng = _visible_trade_range(trades, open_entry, page, state["data_asof"])
                if rng is not None:
                    snap_range = [str(rng[0]), str(rng[1])]

            date_range = None
            if snap_range:
                date_range = (pd.Timestamp(snap_range[0]), pd.Timestamp(snap_range[1]))

            summary, analysis = _card_sections(
                signal_id,
                state=state,
                trades=trades,
                open_entry=open_entry,
                window=window,
                page=page,
                date_range=date_range,
            )
            btn_styles = [_btn_style(primary=(k == window)) for k in window_keys]
            return summary, analysis, page, window, snap_range, *btn_styles

        app.callback(
            Output(f"card-summary-{slug}", "children"),
            Output(f"card-analysis-{slug}", "children"),
            Output(f"trades-page-{slug}", "data"),
            Output(f"window-{slug}", "data"),
            Output(f"snap-range-{slug}", "data"),
            *[Output(f"window-{slug}-{k}", "style") for k in window_keys],
            Input(f"pull-{slug}", "n_clicks"),
            Input(f"run-{slug}", "n_clicks"),
            *[Input(f"window-{slug}-{k}", "n_clicks") for k in window_keys],
            Input(f"trades-prev-{slug}", "n_clicks"),
            Input(f"trades-next-{slug}", "n_clicks"),
            Input(f"snap-{slug}", "n_clicks"),
            State(f"window-{slug}", "data"),
            State(f"snap-range-{slug}", "data"),
            State(f"trades-page-{slug}", "data"),
            prevent_initial_call=True,
        )(_update)

    return app


app = build_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    run(app, port=args.port, host=args.host)
