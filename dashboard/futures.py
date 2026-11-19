"""Live Treasury futures quotes on the Market tab, from the Portfolio tab's IBKR connection."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
from dash import Input, Output, dcc, html

from dashboard.portfolio import BUSY_COLOUR, GAIN, LOSS, SESSION
from dashboard.sheet import grid_table, heading
from execution.quotes import SPREADS
from utils.formatting import format_tsy_price

COLUMNS = {"root": "", "contract": "Contract", "bid_size": "Bid sz", "bid": "Bid", "ask": "Ask",
           "ask_size": "Ask sz", "last": "Last", "change": "Chg (32nds)", "volume": "Volume",
           "time": "Tick (ET)"}
SPREAD_COLUMNS = {"name": "", "legs": "Legs", "month": "Month", "bid_size": "Bid sz", "bid": "Bid", "ask": "Ask",
                  "ask_size": "Ask sz", "last": "Last", "volume": "Volume", "time": "Tick (ET)"}


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


def _stamp(time):
    return f"{dt.datetime.fromisoformat(time).astimezone(ZoneInfo('America/New_York')):%H:%M:%S}" if time else "—"


def _spread_table(rows):
    def points(value):
        return "—" if value is None else f"{value * 32:+.3f}"

    records = [{"name": r["name"], "legs": r["legs"], "month": r["month"], "bid": points(r["bid"]),
                "ask": points(r["ask"]), "last": points(r["last"]), "bid_size": _size(r["bid_size"]),
                "ask_size": _size(r["ask_size"]), "volume": _size(r["volume"]), "time": _stamp(r["time"])}
               for r in rows]

    def style(column, _value, _row):
        if column in ("name", "legs", "month"):
            return {"textAlign": "left", **({"fontWeight": "bold"} if column == "name" else {})}
        return None

    return grid_table(pd.DataFrame(records), columns=list(SPREAD_COLUMNS), headers=SPREAD_COLUMNS, cell_style=style)


def _spread_content(streaming):
    """Every listed spread while the fold is open; closing it stops their streams."""
    if SESSION.state != "connected":
        return [html.Div("Not connected.", className="sd-note")] if streaming else []
    try:
        SESSION.set_spreads(list(SPREADS) if streaming else [])
    except Exception as exc:
        return [html.Div(f"Spread quotes unavailable ({type(exc).__name__}: {exc}).", className="sd-note",
                         style={"color": BUSY_COLOUR})]
    rows = SESSION.spread_quotes()
    return [_spread_table(rows),
            html.Div("One exchange instrument per spread at the exchange's leg ratio · buying it buys the first leg "
                     "and sells the second · prices in 32nds", className="sd-note", style={"marginTop": 6})] if rows else []


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
        # the spreads stream only while this fold is open; each click on its caret flips it
        html.Details([
            html.Summary(["TREASURY SPREADS (IBKR)",
                          html.Span("exchange-listed · open to stream, close to stop", className="sd-h-note")],
                         id="spread-toggle", n_clicks=0),
            html.Div(id="spread-content", className="sd-fold-body"),
        ], className="sd-fold"),
        dcc.Interval(id="futures-poll", interval=1000),
    ], style={"marginBottom": 20})


def register_callbacks(app):
    @app.callback(Output("futures-content", "children"), Output("spread-content", "children"),
                  Input("futures-poll", "n_intervals"), Input("spread-toggle", "n_clicks"),
                  prevent_initial_call=True,
                  running=[
                      (Output("futures-activity", "className"),
                       "sd-futures-activity sd-futures-activity--updating", "sd-futures-activity"),
                      (Output("futures-activity", "title"),
                       "Updating quote table", "Waiting for next table refresh"),
                  ])
    def update(_tick, clicks):
        return _content(), _spread_content(bool((clicks or 0) % 2))  # the fold starts closed
