"""Live Treasury futures quotes on the Market tab, from the Portfolio tab's IBKR connection."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
from dash import Input, Output, dcc, html

from dashboard.portfolio import BUSY_COLOUR, GAIN, LOSS, SESSION
from dashboard.sheet import grid_table, heading
from utils.formatting import format_tsy_price

COLUMNS = {"root": "", "contract": "Contract", "bid_size": "Bid sz", "bid": "Bid", "ask": "Ask",
           "ask_size": "Ask sz", "last": "Last", "change": "Chg (32nds)", "volume": "Volume",
           "time": "Tick (ET)"}


def _price(value, tick):
    return "—" if value is None else format_tsy_price(value, is_ticks=False, tick_32nds=tick)


def _size(value):
    return "—" if value is None else f"{value:,.0f}"


def _table(rows):
    records = []
    for r in rows:
        stamp = dt.datetime.fromisoformat(r["time"]).astimezone(ZoneInfo("America/New_York")) if r["time"] else None
        records.append({
            "root": r["root"], "contract": r["contract"], "bid": _price(r["bid"], r["tick"]),
            "ask": _price(r["ask"], r["tick"]), "last": _price(r["last"], r["tick"]),
            "change": "—" if r["change"] is None else f"{r['change'] * 32:+.2f}",
            "bid_size": _size(r["bid_size"]), "ask_size": _size(r["ask_size"]), "volume": _size(r["volume"]),
            "time": f"{stamp:%H:%M:%S}" if stamp else "—", "_change": r["change"],
        })

    def style(column, _value, row):
        if column in ("root", "contract"):
            return {"textAlign": "left", **({"fontWeight": "bold"} if column == "root" else {})}
        if column == "change" and row["_change"]:
            return {"color": GAIN if row["_change"] > 0 else LOSS}
        return None

    return grid_table(pd.DataFrame(records), columns=list(COLUMNS), headers=COLUMNS, cell_style=style)


def _content():
    if SESSION.state != "connected":
        return [html.Div("Not connected: click Connect IBKR on the Portfolio tab for live futures quotes.",
                         className="sd-note")]
    quotes = SESSION.quotes()
    rows = quotes["rows"]
    notes = []
    if quotes["note"]:
        notes.append(html.Div(quotes["note"], role="status", className="sd-note", style={"color": BUSY_COLOUR}))
    if any(r["delayed"] for r in rows):
        notes.append(html.Div("Delayed quotes: IBKR is not serving this session real-time CBOT data.",
                              className="sd-note", style={"color": BUSY_COLOUR}))
    if not rows:
        return notes or [html.Div("Subscribing to futures quotes…", className="sd-note")]
    return [*notes, _table(rows),
            html.Div("IBKR's lead contract for each future · prices in 32nds · change against the prior settlement",
                     className="sd-note", style={"marginTop": 6})]


def layout():
    return html.Div([
        html.Div([
            heading("TREASURY FUTURES (IBKR)", "streaming · read-only"),
            html.Span(id="futures-activity", className="sd-futures-activity",
                      title="Waiting for next table refresh", role="img",
                      **{"aria-label": "Quote table refresh indicator"}),
        ], className="sd-futures-heading"),
        html.Div(id="futures-content", children=_content()),
        dcc.Interval(id="futures-poll", interval=1000),
    ], style={"marginBottom": 20})


def register_callbacks(app):
    @app.callback(Output("futures-content", "children"), Input("futures-poll", "n_intervals"),
                  prevent_initial_call=True,
                  running=[
                      (Output("futures-activity", "className"),
                       "sd-futures-activity sd-futures-activity--updating", "sd-futures-activity"),
                      (Output("futures-activity", "title"),
                       "Updating quote table", "Waiting for next table refresh"),
                  ])
    def update(_tick):
        return _content()
