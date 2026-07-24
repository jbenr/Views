"""Live signal dashboard -- one card per promoted signal.

Static on load: shows whatever was last pulled / analyzed. Two buttons per
card drive everything else:

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
from dashboard.charts import DEFAULT_WINDOW, WINDOW_PRESETS, level_chart, signal_chart
from dashboard.ledger import SignalLedger
from dashboard.registry import LiveRegistry
from utils.research_app import (
    BORDER, C0, C1, DIM, ORANGE, PANEL, TEXT, make_app, run, stat_block,
)

REGISTRY = LiveRegistry()
LEDGER = SignalLedger()

TRADES_PER_PAGE = 5

TRADE_TABLE_COLS = [
    "entry_date", "exit_date", "direction",
    "entry_level", "exit_level", "target_lvl", "expected_return_bps",
    "pnl_bps", "entry_half_life", "bars_held", "exit_reason",
]
TRADE_TABLE_HEADERS = {
    "entry_date": "Entry", "exit_date": "Exit", "direction": "Dir",
    "entry_level": "Entry Lvl", "exit_level": "Exit Lvl",
    "target_lvl": "Target Lvl", "expected_return_bps": "Exp Ret (bps)",
    "pnl_bps": "PnL (bps)", "entry_half_life": "Half-life (d)",
    "bars_held": "Held (d)", "exit_reason": "Exit Reason",
}
TRADE_TABLE_ROUND = {
    "entry_level": 2, "exit_level": 2, "target_lvl": 2,
    "expected_return_bps": 1, "pnl_bps": 1, "entry_half_life": 1,
}


def _slug(module: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", module.lower()).strip("-")


def _param_summary(row: dict) -> str:
    bits = [f"{row['entry_signal']} beta_lb={row['beta_lb']}"]
    if row.get("ou_lb"):
        bits.append(f"ou_lb={row['ou_lb']}")
    bits.append(f"entry={row['entry_threshold']:g}")
    bits.append(f"exit={row['exit_style']}/{row['exit_param']:g}")
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
    the pinned open row plus the current page of closed trades."""
    dates = []
    if trades is not None and not trades.is_empty():
        for row in trades.slice(page, TRADES_PER_PAGE).to_dicts():
            dates.append(row["entry_date"])
            dates.append(row["exit_date"])
    if open_entry is not None:
        dates.append(open_entry["entry_date"])
        dates.append(data_asof)
    dates = [d for d in dates if d is not None]
    if not dates:
        return None
    return min(dates), max(dates)


def _trade_table_block(trades, page: int, open_entry: dict | None, slug: str) -> html.Div:
    n = 0 if trades is None or trades.is_empty() else len(trades)
    if n == 0 and open_entry is None:
        return html.Div("no closed trades yet", style={"color": DIM, "padding": "8px 0"})

    rows = ([_open_trade_row(open_entry)] if open_entry is not None else [])
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
                    {"if": {"filter_query": '{direction} = "long"'}, "color": C1},
                    {"if": {"filter_query": '{direction} = "short"'}, "color": C0},
                    {"if": {"filter_query": "{pnl_bps} > 0", "column_id": "pnl_bps"},
                     "color": C1, "fontWeight": "bold"},
                    {"if": {"filter_query": "{pnl_bps} < 0", "column_id": "pnl_bps"},
                     "color": C0, "fontWeight": "bold"},
                    {"if": {"filter_query": '{exit_reason} = "active"'}, "fontStyle": "italic"},
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


def _card_body(
    module: str,
    state: dict | None = None,
    trades=None,
    open_entry: dict | None = None,
    window: str = DEFAULT_WINDOW,
    page: int = 0,
    date_range: tuple | None = None,
) -> html.Div:
    row = REGISTRY.get(module)
    if row is None:
        return html.Div(f"{module}: no longer promoted", style={"color": DIM})

    ledger_row = LEDGER.latest(module)
    error = None
    if state is None:
        # initial non-callback render at build_app() time -- callback-driven
        # renders always pass a precomputed state/trades pair down.
        try:
            state = runner.compute_signal(module)
            trades, open_entry = runner.trade_history(module, state)
        except RuntimeError as exc:
            error = str(exc)

    stats = [
        stat_block("data as-of", str(state["data_asof"]) if state else "—"),
        stat_block("last analysis run", ledger_row["run_ts"] if ledger_row else "never"),
        stat_block(
            "reading",
            state["fired"] if state else "—",
            alert=bool(state and state["fired"] not in ("flat", "flat (gated)")),
        ),
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
        body = [
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": 10},
                children=[
                    html.Img(src=f"data:image/png;base64,{level_png}",
                              style={"width": "100%", "border": f"1px solid {BORDER}"}),
                    html.Img(src=f"data:image/png;base64,{sig_png}",
                              style={"width": "100%", "border": f"1px solid {BORDER}"}),
                ],
            ),
            _trade_table_block(trades, page, open_entry, _slug(module)),
        ]

    return html.Div([
        html.Div(stats, style={"display": "flex", "gap": 28, "flexWrap": "wrap",
                                "padding": "8px 2px 8px"}),
        html.Div(backtest_stats, style={"display": "flex", "gap": 28, "flexWrap": "wrap",
                                         "padding": "0 2px 14px",
                                         "borderTop": f"1px solid {BORDER}",
                                         "marginTop": 4, "paddingTop": 8}),
        *body,
    ])


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
    slug = _slug(module)
    return html.Div(
        style={"border": f"1px solid {BORDER}", "background": PANEL,
               "padding": "14px 18px", "marginBottom": 18},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "baseline", "gap": 12,
                       "marginBottom": 4},
                children=[
                    html.Span(row["name"], style={"fontSize": 15, "fontWeight": "bold",
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
                    *[
                        html.Button(key, id=f"window-{slug}-{key}", n_clicks=0,
                                    style=_btn_style(primary=(key == DEFAULT_WINDOW)))
                        for key in WINDOW_PRESETS
                    ],
                    dcc.Store(id=f"window-{slug}", data=DEFAULT_WINDOW),
                    dcc.Store(id=f"snap-range-{slug}", data=None),
                    dcc.Store(id=f"trades-page-{slug}", data=0),
                ],
            ),
            html.Div(id=f"card-body-{slug}", children=_card_body(module)),
        ],
    )


def build_app() -> dash.Dash:
    reg_df = REGISTRY.list()
    rows = [] if reg_df.is_empty() else reg_df.sort("family", "name").to_dicts()

    body = html.Div(
        style={"padding": "18px 24px"},
        children=(
            [html.Div(
                "No signals promoted yet -- "
                "python -m dashboard.registry --promote <module>",
                style={"color": DIM, "padding": "24px 0"},
            )]
            if not rows else [_card(row) for row in rows]
        ),
    )

    app = make_app(
        title="LIVE",
        subtitle="promoted signals -- data as-of / last analysis run / current reading",
        data_info=f"{len(rows)} live signal{'s' if len(rows) != 1 else ''}",
        sliders=[],
        body=body,
    )

    window_keys = list(WINDOW_PRESETS)

    for row in rows:
        module = row["module"]
        slug = _slug(module)
        window_prefix = f"window-{slug}-"

        def _update(_pull, _run, *rest, module=module, slug=slug, window_prefix=window_prefix):
            *_window_clicks, _prev, _next, _snap, window, snap_range, page = rest
            trigger = ctx.triggered_id or ""
            no_btn_styles = [dash.no_update] * len(window_keys)
            try:
                if trigger.startswith("pull-"):
                    runner.pull_data(module)
                    page = 0
                elif trigger.startswith("run-"):
                    runner.run_analysis(module)
                    page = 0
                elif trigger.startswith(window_prefix):
                    window = trigger[len(window_prefix):]
                    snap_range = None
                    page = 0
                state = runner.compute_signal(module)
                trades, open_entry = runner.trade_history(module, state)
            except RuntimeError as exc:
                err = html.Div(str(exc), style={"color": ORANGE, "padding": "12px 0"})
                return (err, 0, window, snap_range, *no_btn_styles)

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

            body = _card_body(module, state=state, trades=trades, open_entry=open_entry,
                               window=window, page=page, date_range=date_range)
            btn_styles = [_btn_style(primary=(k == window)) for k in window_keys]
            return body, page, window, snap_range, *btn_styles

        app.callback(
            Output(f"card-body-{slug}", "children"),
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
