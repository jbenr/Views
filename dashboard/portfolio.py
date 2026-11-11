"""Read-only IBKR portfolio: the broker's account, positions, orders and fills as plain tables."""
from decimal import Decimal, InvalidOperation

import pandas as pd
from dash import Input, Output, dcc, html

from dashboard.sheet import grid_table, heading, kv, kv_range
from execution.journal import TradeJournal
from execution.portfolio import fetch_snapshot
from execution.ibkr import IBKRConfig
from execution.grouping import group_positions


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
GAIN, LOSS = "#27AE60", "#E74C3C"


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
                                                         "UTC · disconnected after refresh")]),
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


def layout():
    config = IBKRConfig.from_env()
    return html.Div([
        heading("BROKER ACCOUNT (IBKR)", f"read-only · order submission disabled · {config.host}:{config.port} "
                                         f"client {config.client_id}"),
        html.Div(className="sd-bar", children=[
            html.Button("Refresh IBKR", id="ibkr-refresh", className="ref-btn sd-btn"),
            html.Div(id="ibkr-message", role="status", className="sd-note"),
        ]),
        dcc.Loading(html.Div(id="ibkr-content", children=_content())),
    ])


def register_callbacks(app):
    @app.callback(Output("ibkr-content", "children"), Output("ibkr-message", "children"),
                  Input("ibkr-refresh", "n_clicks"), prevent_initial_call=True)
    def update(_refresh):
        try:
            journal = TradeJournal()
            journal.save_snapshot(fetch_snapshot())
            message = "Snapshot refreshed. Broker connection closed."
            return _content(), message
        except Exception as exc:
            return _content(), f"Refresh/action failed ({type(exc).__name__}): {exc}. Previous snapshot retained."
