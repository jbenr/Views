"""Read-only IBKR portfolio tab."""
from decimal import Decimal, InvalidOperation

from dash import Input, Output, dcc, html

from execution.journal import TradeJournal
from execution.portfolio import fetch_snapshot
from execution.ibkr import IBKRConfig
from execution.grouping import group_positions
from utils.research_app import stat_block


ACCOUNT_STATS = [
    ("InitMarginReq", "Initial margin"),
    ("MaintMarginReq", "Maintenance margin"),
]
FORWARD_STATS = [
    ("FullInitMarginReq", "Full initial margin"),
    ("FullMaintMarginReq", "Full maintenance margin"),
    ("LookAheadInitMarginReq", "Look-ahead initial margin"),
    ("LookAheadMaintMarginReq", "Look-ahead maintenance margin"),
]


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


def _account_summary(balances):
    def grid(stats):
        return html.Div([stat_block(label, _account_value(balances, tag)) for tag, label in stats],
                        style={"display": "flex", "flexWrap": "wrap", "gap": "24px",
                               "margin": "16px 0"})
    return html.Div([
        html.Div([html.Div("Net liquidation"),
                  html.Div(_account_value(balances, "NetLiquidation"),
                           style={"fontSize": "28px", "fontWeight": "bold", "fontFamily": "monospace"})],
                 style={"marginTop": "16px"}),
        grid(ACCOUNT_STATS),
        html.Details([html.Summary("Full and look-ahead margin"), grid(FORWARD_STATS),
                      html.P("Full figures exclude intraday credits; look-ahead figures reflect the next margin change.")]),
    ])


def _table(rows):
    if not rows:
        return html.P("No records.")
    cols = list(rows[0])
    return html.Div(html.Table([
        html.Thead(html.Tr([html.Th(c.replace('_', ' ')) for c in cols])),
        html.Tbody([html.Tr([html.Td(str(r.get(c, '')), style={"padding": "6px 12px"})
                            for c in cols]) for r in rows]),
    ]), style={"overflowX": "auto"})


def _positions_table(rows):
    formatted = []
    for row in rows:
        def number(key, precision=2):
            value = row.get(key)
            return f"{value:,.{precision}f}" if value is not None else "—"
        pnl = row.get("unrealized_pnl")
        formatted.append({
            "symbol": row["symbol"], "quantity": row["quantity"],
            "avg cost": number("avg_cost", 4), "current price": number("current_price", 4),
            "unrealized P&L": html.Span(number("unrealized_pnl"), style={
                "color": "#4caf88" if pnl is not None and pnl > 0 else
                         "#ef6b73" if pnl is not None and pnl < 0 else "inherit"}),
            "currency": row.get("currency", ""), "strategy": row.get("strategy", "Unattributed"),
        })
    if not formatted:
        return html.P("No positions.")
    cols = list(formatted[0])
    return html.Div(html.Table([
        html.Thead(html.Tr([html.Th(c) for c in cols])),
        html.Tbody([html.Tr([html.Td(r[c], style={"padding": "6px 12px"}) for c in cols])
                    for r in formatted]),
    ]), style={"overflowX": "auto"})


def _content():
    journal = TradeJournal()
    snapshot = journal.latest_snapshot()
    items = []
    if snapshot:
        items += [html.P(w, role="status") for w in snapshot.get("warnings", [])]
        items += [html.H4(f"Account {snapshot['account']}"),
                  html.P(f"Last successful snapshot: {snapshot['as_of']} · disconnected after refresh"),
                  html.P("Broker history is limited to what TWS returns. Refresh for current statuses. "
                         "Average cost and current price use the same price units. P&L is IBKR’s unrealized P&L at refresh.")]
        items.append(_account_summary(snapshot.get("balances", [])))
        for key, label in [("positions", "Positions"),
                           ("orders", "Open orders"), ("completed_orders", "Recent completed orders"),
                           ("fills", "Recent fills")]:
            records = snapshot[key]
            if key in ("orders", "completed_orders", "fills"):
                refs = {r["order_ref"]: r["signal_id"] for r in journal.records("intent")}
                records = [dict(r, strategy=refs.get(r.get("order_ref"), "Unattributed")) for r in records]
            items.append(html.H4(label))
            if key == "positions":
                fills = journal.saved_fills(snapshot["account"]) + snapshot.get("fills", [])
                groups = group_positions(records, fills)
                if any(stamp is not None for stamp, _ in groups):
                    items.append(html.P("Grouped by latest reported trade activity within 2 minutes; times shown in UTC. "
                                        "These are inferred groups, not confirmed opening trades or strategies."))
                for stamp, positions in groups:
                    heading = (f"Trade activity · {stamp:%Y-%m-%d %H:%M:%S} UTC"
                               if stamp else "Trade time unavailable")
                    items += [html.H5(heading), _positions_table(positions)]
                if not records:
                    items.append(html.P("No positions."))
            else:
                items.append(_table(records))
    else:
        items.append(html.P("No broker snapshot yet. Start TWS / IB Gateway and refresh."))
    return items


def layout():
    config = IBKRConfig.from_env()
    return html.Div([
        html.H3("IBKR Portfolio"),
        html.P("Read-only connection · order submission disabled · refresh on demand"),
        html.P(f"TWS / Gateway endpoint: {config.host}:{config.port} · client ID {config.client_id}"),
        html.Button("Refresh IBKR portfolio", id="ibkr-refresh"),
        html.Div(id="ibkr-message", role="status"),
        dcc.Loading(html.Div(id="ibkr-content", children=_content())),
    ], style={"padding": "20px"})


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
