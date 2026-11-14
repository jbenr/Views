"""Read-only IBKR portfolio: the broker's account, positions, orders and fills as plain tables."""
from decimal import Decimal, InvalidOperation

import pandas as pd
from dash import Input, Output, State, ctx, dcc, html, no_update

from dashboard.sheet import grid_table, heading, kv, kv_range
from execution.journal import TradeJournal
from execution.portfolio import fetch_snapshot
from execution.ibkr import IBKRConfig
from execution.grouping import group_positions
from execution.session import BUSY, Session
from utils.research_app import shimmer_loader

SESSION = Session()  # the dashboard's one broker connection; it dies with the server process


ACCOUNT_STATS = [
    ("NetLiquidation", "net liquidation"),
    ("InitMarginReq", "initial margin"),
    ("MaintMarginReq", "maintenance margin"),
]
FORWARD_STATS = [
    ("FullInitMarginReq", "full initial margin"),
    ("FullMaintMarginReq", "full maintenance margin"),
    ("LookAheadInitMarginReq", "look-ahead initial margin"),
    ("LookAheadMaintMarginReq", "look-ahead maintenance margin"),
]
POSITION_COLS = {"symbol": "Symbol", "quantity": "Qty", "avg_cost": "Avg cost", "current_price": "Price",
                 "unrealized_pnl": "Unrealized P&L", "currency": "Ccy", "activity": "Last trade (UTC)"}
ORDER_COLS = {"symbol": "Symbol", "action": "Side", "quantity": "Qty", "status": "Status", "filled": "Filled",
              "remaining": "Remaining", "avg_fill_price": "Avg fill", "strategy": "Signal"}
FILL_COLS = {"time": "Time", "symbol": "Symbol", "side": "Side", "quantity": "Qty", "price": "Price",
             "strategy": "Signal"}
GAIN, LOSS, BUSY_COLOUR = "#27AE60", "#E74C3C", "#D35400"


def _account_value(balances, tag):
    values = []
    for row in balances:
        if row.get("tag") != tag:
            continue
        try:
            value = Decimal(str(row["value"]))
        except (InvalidOperation, ValueError, KeyError):
            continue
        if not value.is_finite() or abs(value) > Decimal("1e100"):
            continue
        if tag == "Cushion":
            values.append(f"{value * 100:,.1f}%")
        else:
            currency = row.get("currency") or ""
            values.append(f"{value:,.2f} {currency}".strip())
    return " / ".join(dict.fromkeys(values)) or "—"


def _account_range(balances, stats):
    return kv_range([kv(label, _account_value(balances, tag)) for tag, label in stats])


def _number(value, precision):
    return f"{value:,.{precision}f}" if value is not None else "—"


def _text_left(columns):
    """Cell style: the named columns read as text (left), the rest as numbers."""
    def style(column, _value, _row):
        return {"textAlign": "left"} if column in columns else None
    return style


def _records_table(rows, columns):
    """Broker rows as a gridded table of the chosen columns; a plain line when there are none."""
    if not rows:
        return html.Div("none", className="sd-note")
    frame = pd.DataFrame([{c: "" if r.get(c) is None else str(r.get(c)) for c in columns} for r in rows])
    return grid_table(frame, columns=list(columns), headers=columns,
                      cell_style=_text_left({"symbol", "action", "status", "strategy", "time", "side"}))


def _positions_table(groups):
    """Every position in one table; the trade-activity group it was inferred into is a column."""
    rows = []
    for stamp, positions in groups:
        for p in positions:
            rows.append({
                "symbol": p["symbol"], "quantity": p["quantity"], "avg_cost": _number(p.get("avg_cost"), 4),
                "current_price": _number(p.get("current_price"), 4),
                "unrealized_pnl": _number(p.get("unrealized_pnl"), 2), "currency": p.get("currency", ""),
                "activity": f"{stamp:%Y-%m-%d %H:%M}" if stamp else "—", "_pnl": p.get("unrealized_pnl"),
            })
    if not rows:
        return html.Div("none", className="sd-note")

    def style(column, _value, row):
        if column in ("symbol", "currency", "activity"):
            return {"textAlign": "left", **({"fontWeight": "bold"} if column == "symbol" else {})}
        if column == "unrealized_pnl" and row["_pnl"]:
            return {"color": GAIN if row["_pnl"] > 0 else LOSS, "fontWeight": "bold"}
        return None

    return grid_table(pd.DataFrame(rows), columns=list(POSITION_COLS), headers=POSITION_COLS, cell_style=style)


def _content():
    journal = TradeJournal()
    snapshot = journal.latest_snapshot()
    if not snapshot:
        return [html.Div("No broker snapshot yet. Start TWS / IB Gateway and refresh.", className="sd-note")]
    balances = snapshot.get("balances", [])
    refs = {r["order_ref"]: r["signal_id"] for r in journal.records("intent")}

    def attributed(records):
        return [dict(r, strategy=refs.get(r.get("order_ref"), "unattributed")) for r in records]

    fills = journal.saved_fills(snapshot["account"]) + snapshot.get("fills", [])
    return [
        *[html.Div(w, role="status", className="sd-note", style={"color": "#D35400"})
          for w in snapshot.get("warnings", [])],
        kv_range([kv("account", snapshot["account"]), kv("snapshot", str(snapshot["as_of"])[:19].replace("T", " "),
                                                         "UTC")]),
        heading("ACCOUNT"),
        _account_range(balances, ACCOUNT_STATS),
        html.Details([html.Summary("FULL AND LOOK-AHEAD MARGIN"),
                      html.Div([_account_range(balances, FORWARD_STATS),
                                html.Div("Full figures exclude intraday credits; look-ahead figures reflect the next "
                                         "margin change.", className="sd-note")], className="sd-fold-body")],
                     className="sd-fold sd-fold--quiet"),
        heading("POSITIONS", "P&L is IBKR's unrealized P&L at refresh · average cost and price in the same units"),
        _positions_table(group_positions(snapshot["positions"], fills)),
        heading("OPEN ORDERS"),
        _records_table(attributed(snapshot["orders"]), ORDER_COLS),
        heading("RECENT COMPLETED ORDERS"),
        _records_table(attributed(snapshot["completed_orders"]), ORDER_COLS),
        heading("RECENT FILLS"),
        _records_table(attributed(snapshot["fills"]), FILL_COLS),
        html.Div("Last trade is the latest reported activity for the position, grouped within 2 minutes: an inferred "
                 "group, not a confirmed opening trade or strategy. Broker history is limited to what TWS returns.",
                 className="sd-note", style={"marginTop": 8}),
    ]


def _connection(status):
    """The connect button's label, the status line beside it, and whether to keep polling."""
    state, detail = status["state"], status["detail"]
    if state == "connected":
        mode = "paper" if status["account"].startswith("D") else "live"
        text = " · ".join(filter(None, ["connected", status["account"], mode, detail]))
        return "Disconnect", html.Span(text, style={"color": GAIN}), False
    if state in BUSY:
        return "Cancel", html.Span(detail, style={"color": BUSY_COLOUR}), False
    if state == "failed":
        return "Connect IBKR", html.Span(f"connection failed · {detail}", style={"color": LOSS}), True
    return "Connect IBKR", " · ".join(filter(None, ["not connected", detail])), True


def layout():
    config = IBKRConfig.from_env()
    status = SESSION.status()
    label, text, idle = _connection(status)
    return html.Div([
        heading("BROKER ACCOUNT (IBKR)", f"read-only · order submission disabled · {config.host}:{config.port} "
                                         f"client {config.client_id}"),
        html.Div(className="sd-bar", children=[
            html.Button(label, id="ibkr-connect", className="ref-btn sd-btn"),
            html.Button("Refresh IBKR", id="ibkr-refresh", className="ref-btn sd-btn"),
            html.Div(text, id="ibkr-status", role="status", className="sd-note"),
            html.Div(id="ibkr-message", role="status", className="sd-note"),
        ]),
        # ticks only while a connection is up or being made; the snapshot stamp it last drew
        dcc.Interval(id="ibkr-poll", interval=2000, disabled=idle),
        dcc.Store(id="ibkr-drawn", data=status["snapshot"]),
        dcc.Loading(
            html.Div(id="ibkr-content", children=_content()),
            custom_spinner=shimmer_loader(image="guy.png"),
            overlay_style={"visibility": "visible"},
            parent_className="dashboard-portfolio-loader",
        ),
    ])


def register_callbacks(app):
    @app.callback(Output("ibkr-connect", "children"), Output("ibkr-status", "children"),
                  Output("ibkr-poll", "disabled"), Output("ibkr-drawn", "data"),
                  Input("ibkr-connect", "n_clicks"), Input("ibkr-poll", "n_intervals"),
                  State("ibkr-drawn", "data"), prevent_initial_call=True)
    def connection(_connect, _tick, drawn):
        if ctx.triggered_id == "ibkr-connect":
            if SESSION.state == "connected" or SESSION.state in BUSY:
                SESSION.disconnect()
            else:
                SESSION.connect()
        status = SESSION.status()
        # a new snapshot stamp redraws the tables below; the tick alone never does
        return *_connection(status), status["snapshot"] if status["snapshot"] != drawn else no_update

    @app.callback(Output("ibkr-content", "children"), Output("ibkr-message", "children"),
                  Input("ibkr-refresh", "n_clicks"), Input("ibkr-drawn", "data"), prevent_initial_call=True)
    def update(_refresh, _drawn):
        if ctx.triggered_id != "ibkr-refresh":
            return _content(), ""
        try:
            if SESSION.state == "connected":
                SESSION.refresh()
                message = "Snapshot refreshed."
            else:
                TradeJournal().save_snapshot(fetch_snapshot())
                message = "Snapshot refreshed. Broker connection closed."
            return _content(), message
        except Exception as exc:
            return _content(), f"Refresh/action failed ({type(exc).__name__}): {exc}. Previous snapshot retained."
