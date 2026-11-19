"""The signals' side of the Portfolio tab: one summary row per signal that is in a trade
or calling for one, then a section per signal -- its size and P&L as modelled and as
futures, the legs that express its package, and how close it is to each exit. A signal
calling for a trade has an enter button: it opens an order ticket whose legs go to IBKR as
limit orders, on a paper account only."""
from __future__ import annotations

import datetime as dt

import pandas as pd
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from dashboard import runner
from dashboard.portfolio import SESSION
from dashboard.sheet import grid_table, heading, usd
from execution.journal import TradeJournal
from execution.quotes import FUTURES
from execution.ticket import ROOTS, contract_legs, futures_dv01, package_dv01, yield_legs
from oven.engine import hedgeable
from research.basis_basket import FACE
from research.panel import CATALOG
from utils.basis import load_futures
from utils.formatting import format_tsy_price

SUMMARY_HEADERS = {"signal": "Signal", "package": "Package", "status": "Status", "entry": "Entry",
                   "pnl_bps": "P&L bp", "fut_usd": "P&L $", "dv01_held": "DV01 $/bp", "exp_bps": "Exp bp",
                   "captured": "Captured", "held": "Days", "nearest": "Nearest exit"}
SPEC_HEADERS = {"spec": "", "entry": "Entry", "now": "Live"}
LEG_HEADERS = {"action": "Side", "quantity": "Qty", "contract": "Contract", "exact": "Exact", "dv01": "DV01 each",
               "risk": "DV01 model", "held": "DV01 actual", "entry": "Entry px", "last": "Last px", "pnl": "P&L $"}
EXIT_HEADERS = {"rule": "Exit rule", "now": "Now", "exits_at": "Exits at", "progress": "Progress"}
NEAR_EXIT = 0.75  # an exit rule this far along is flagged
TICKS = {root: tick for root, tick in FUTURES.values()}  # each contract's price tick in 32nds
GAIN, LOSS, WARN = "#27AE60", "#E74C3C", "#D35400"
TOTAL_RULE = {"borderTop": "2px solid #333333", "fontWeight": "bold"}
_dv01_cache: dict[dt.date, dict] = {}
_close_cache: dict[tuple, dict] = {}


def _dv01() -> dict:
    """One basket pricing per day per server process; a failed one is retried next load."""
    today = dt.date.today()
    if today not in _dv01_cache:
        _dv01_cache.clear()
        _dv01_cache[today] = futures_dv01()
    return _dv01_cache[today]


def _closes(start: dt.date) -> dict[str, dict[dt.date, float]]:
    """Daily closes per futures contract from `start`; one load per day per start date."""
    key = (dt.date.today(), start)
    if key not in _close_cache:
        _close_cache.clear()
        frame = load_futures(list(ROOTS.values()), start.isoformat())
        _close_cache[key] = {name: dict(zip(group["ts"], group["fut_px"]))
                             for (name,), group in frame.group_by("contract")}
    return _close_cache[key]


def package(row: dict, read: dict) -> dict | None:
    """What the signal holds or wants: side, status, hedge and the beta that hedge is
    frozen at. None when it is flat."""
    state, held = read["state"], read["open_entry"]
    if held is not None:
        side, status, on = held["direction"], f"open since {held['entry_date']}", held["entry_date"]
    elif state["fired"] in ("long", "short"):
        side, status, on = state["fired"], "enter", None
    else:
        return None
    strategy, data = state["strategy"], state["data"]
    hedge, beta = None, 0.0
    if getattr(strategy, "setup", {}).get("basis") == "portfolio" and hedgeable(strategy.feature):
        betas = data.filter(data["ts"] <= on) if on is not None else data
        hedge, beta = strategy.feature, float(betas["beta_changes"][-1])
    target = row["target"]
    name = f"{target} {'−' if beta >= 0 else '+'} {abs(beta):.2f}×{hedge}" if hedge else target
    return dict(target=target, side=1 if side == "long" else -1, status=status, hedge=hedge, beta=beta,
                name=f"{side} {name}")


def size(trade: dict, usd_per_bp: float, dv01: dict | None, closes: dict, marks: dict) -> None:
    """Add the trade's futures legs and what owning them since entry has made: legs,
    held_dv01, leak, model_pnl, fut_pnl and slip (dollars), or why there are no legs.
    A leg is marked at its live price in `marks` (by contract) when there is one, else its last close."""
    pack, held = trade["pack"], trade["held"]
    trade.update(legs=None, why=None, held_dv01=None, leak=None, model_pnl=None, fut_pnl=None, slip=None)
    if dv01 is None:
        trade["why"] = "Futures DV01 unavailable: the legs cannot be sized."
        return
    if pack["target"] not in CATALOG or any(t not in ROOTS for t in CATALOG[pack["target"]].legs):
        trade["why"] = "No futures expression for this package."
        return
    legs = contract_legs(yield_legs(pack["target"], pack["side"], pack["hedge"], pack["beta"]), usd_per_bp, dv01)
    for leg in legs:
        history = closes.get(leg["contract"]) or {}
        leg["last_px"] = marks.get(leg["contract"]) or (history[max(history)] if history else None)
        leg["entry_px"] = history.get(held["entry_date"]) if held is not None else None
        leg["pnl"] = None
        if leg["signed"] is not None and leg["entry_px"] is not None:
            # a price point on one contract is 1% of its face
            leg["pnl"] = leg["signed"] * (leg["last_px"] - leg["entry_px"]) * FACE.get(leg["root"], 100_000) / 100.0
    trade["legs"], trade["held_dv01"] = legs, package_dv01(legs)
    if trade["held_dv01"] is not None:
        trade["leak"] = sum(leg["residual"] for leg in legs)
    if held is not None:
        trade["model_pnl"] = float(held["pnl_bps"]) * usd_per_bp
        if all(leg["pnl"] is not None for leg in legs):
            trade["fut_pnl"] = sum(leg["pnl"] for leg in legs)
            trade["slip"] = trade["fut_pnl"] - trade["model_pnl"]


def _summary(trades: list[dict]) -> html.Div:
    """One row per signal, and their total."""
    records = []
    for trade in trades:
        held, rules = trade["held"], trade["rules"]
        enter = html.Button("enter", id={"type": "trade-enter", "signal": trade["id"]}, n_clicks=0,
                            className="ref-btn sd-btn sd-status-btn", title="Open the order ticket")
        record = dict(signal=trade["name"], package=trade["pack"]["name"],
                      status="open" if held is not None else enter,
                      entry=str(held["entry_date"]) if held is not None else "—",
                      pnl_bps="—", fut_usd="—", exp_bps="—", captured="—", held="—", nearest="—",
                      dv01_held="—" if trade["held_dv01"] is None else f"${trade['held_dv01']:,.0f}",
                      _pnl=None, _fut=None, _near=False, _total=False)
        if held is not None:
            pnl, expected = float(held["pnl_bps"]), held.get("expected_return_bps")
            nearest = max(rules, key=lambda r: r["progress"], default=None)
            record.update(
                pnl_bps=f"{pnl:+.1f}", fut_usd=usd(trade["fut_pnl"]), held=str(held["bars_held"]),
                exp_bps=f"{expected:+.1f}" if expected is not None else "—",
                captured=f"{pnl / expected:.0%}" if expected else "—",
                nearest=f"{nearest['rule']} · {max(nearest['progress'], 0.0):.0%}" if nearest else "—",
                _pnl=pnl, _fut=trade["fut_pnl"],
                _near=bool(nearest and nearest["progress"] >= NEAR_EXIT))
        records.append(record)

    def total(key):
        return sum(r[key] for r in records if r[key] is not None)

    records.append(dict(signal="Total", package="", status="", entry="", pnl_bps=f"{total('_pnl'):+.1f}",
                        fut_usd=usd(total("_fut")), dv01_held="", exp_bps="", captured="", held="", nearest="",
                        _pnl=total("_pnl"), _fut=total("_fut"), _near=False, _total=True))
    signed = {"pnl_bps": "_pnl", "fut_usd": "_fut"}

    def style(column, _value, record):
        rule = TOTAL_RULE if record["_total"] else {}
        if column in ("signal", "package", "status", "entry", "nearest"):
            flag = {"color": WARN, "fontWeight": "bold"} if column == "nearest" and record["_near"] else {}
            return {"textAlign": "left", **({"fontWeight": "bold"} if column == "signal" else {}), **flag, **rule}
        if column in signed and record[signed[column]] is not None:
            return {"color": GAIN if record[signed[column]] >= 0 else LOSS, "fontWeight": "bold", **rule}
        return rule or None

    return grid_table(pd.DataFrame(records), columns=list(SUMMARY_HEADERS), headers=SUMMARY_HEADERS,
                      cell_style=style)


def _legs(trade: dict) -> html.Div:
    """The package as futures orders, one row per leg, and their net."""
    if trade["legs"] is None:
        return html.Div(trade["why"], className="sd-note")

    def price(leg, key):
        return "—" if leg[key] is None else format_tsy_price(leg[key], is_ticks=False, tick_32nds=TICKS[leg["root"]])

    records = []
    for leg in trade["legs"]:
        priced = leg["dv01"] is not None
        records.append(dict(
            action=leg["action"] or "—", quantity=str(leg["quantity"]) if priced else "—",
            contract=leg["contract"] or f"{leg['root']} (no DV01)",
            exact=f"{leg['exact']:+.2f}" if priced else "—", dv01=f"{leg['dv01']:,.1f}" if priced else "—",
            risk=usd(leg["risk"]), held=usd(leg["held"]), entry=price(leg, "entry_px"), last=price(leg, "last_px"),
            pnl=usd(leg["pnl"]), _action=leg["action"], _pnl=leg["pnl"], _total=False))
    held = [leg["held"] for leg in trade["legs"]]
    records.append(dict(action="Net", quantity="", contract="", exact="", dv01="",
                        risk=usd(sum(leg["risk"] for leg in trade["legs"])),
                        held=usd(None if None in held else sum(held)), entry="", last="", pnl=usd(trade["fut_pnl"]),
                        _action=None, _pnl=trade["fut_pnl"], _total=True))

    def style(column, _value, record):
        rule = TOTAL_RULE if record["_total"] else {}
        if column == "action" and record["_action"]:
            return {"textAlign": "left", "color": GAIN if record["_action"] == "BUY" else LOSS, "fontWeight": "bold"}
        if column == "pnl" and record["_pnl"] is not None:
            return {"color": GAIN if record["_pnl"] >= 0 else LOSS, "fontWeight": "bold", **rule}
        return {**({"textAlign": "left"} if column in ("action", "contract") else {}), **rule} or None

    return grid_table(pd.DataFrame(records), columns=list(LEG_HEADERS), headers=LEG_HEADERS, cell_style=style)


def _exits(rules: list[dict]) -> html.Div:
    """Each exit rule of an open trade: where it stands and where it fires."""
    records = [dict(rule=r["rule"], now=r["now"], exits_at=r["exits_at"],
                    progress=f"{max(r['progress'], 0.0):.0%}", _near=r["progress"] >= NEAR_EXIT) for r in rules]

    def style(column, _value, record):
        flag = {"color": WARN, "fontWeight": "bold"} if column in ("rule", "progress") and record["_near"] else {}
        return {**({"textAlign": "left"} if column != "progress" else {}), **flag} or None

    return grid_table(pd.DataFrame(records), columns=list(EXIT_HEADERS), headers=EXIT_HEADERS, cell_style=style)


def _num(value, fmt: str = ".2f", unit: str = "") -> str:
    """A possibly missing or NaN number, else '—'."""
    if value is None or value != value:
        return "—"
    return f"{float(value):{fmt}}{unit}"


def _specs(trade: dict) -> html.Div:
    """The signal's own numbers, at entry and now: date, level, signal, fair value,
    half-life, expected return, and what the trade has made. A signal only calling
    for a trade has no entry column."""
    held, live, data = trade["held"], trade["live"], trade["state"]["data"]
    oven = hasattr(trade["state"]["strategy"], "setup")  # an oven signal reads a residual in bp, a book one a z
    now = {"date": str(data["ts"][-1]), "level (bp)": _num(live.get("level")),
           "signal": _num(live.get("resid"), "+.2f", "bp") if oven else _num(live.get("z"), "+.2f", "z"),
           "fair value (bp)": _num(live.get("fair")), "half-life (days)": _num(live.get("half_life"), ".1f"),
           "expected return (bp)": _num(live.get("exp_return"), "+.1f")}
    entry = {}
    if held is not None:
        if oven:
            fired = data.filter(data["ts"] == held["entry_date"])
            signal = _num(fired["resid"][0] if len(fired) and "resid" in data.columns else None, "+.2f", "bp")
        else:
            signal = _num(held.get("entry_signal"), "+.2f", "z")
        entry = {"date": str(held["entry_date"]), "level (bp)": _num(held.get("entry_level")), "signal": signal,
                 "fair value (bp)": _num(held.get("target_lvl")),
                 "half-life (days)": _num(held.get("entry_half_life"), ".1f"),
                 "expected return (bp)": _num(held.get("expected_return_bps"), "+.1f")}
        now.update({"P&L (bp)": _num(held["pnl_bps"], "+.1f"), "P&L model ($)": usd(trade["model_pnl"]),
                    "P&L futures ($)": usd(trade["fut_pnl"]), "slippage ($)": usd(trade["slip"]),
                    "days held": str(held["bars_held"])})
    records = [dict(spec=spec, entry=entry.get(spec, ""), now=value) for spec, value in now.items()]
    signed = {"P&L (bp)": (held or {}).get("pnl_bps"), "P&L model ($)": trade["model_pnl"],
              "P&L futures ($)": trade["fut_pnl"], "slippage ($)": trade["slip"]}

    def style(column, _value, record):
        if column == "spec":
            return {"textAlign": "left"}
        if column == "now" and signed.get(record["spec"]) is not None:
            return {"color": GAIN if signed[record["spec"]] >= 0 else LOSS, "fontWeight": "bold"}
        return None

    columns = ["spec", "entry", "now"] if held is not None else ["spec", "now"]
    return grid_table(pd.DataFrame(records), columns=columns, headers=SPEC_HEADERS, cell_style=style)


def _section(trade: dict) -> list:
    """One signal: its numbers at entry and now, its legs, its exits."""
    pack = trade["pack"]
    out = [heading(trade["name"].upper(), f"{pack['name']} · {pack['status']}"), _specs(trade)]
    out.append(html.Div(_legs(trade), style={"marginTop": 8}))
    if trade["rules"]:
        out.append(html.Div(_exits(trade["rules"]), style={"marginTop": 8}))
    return out


def _marks() -> dict:
    """Each futures contract's price from the IBKR stream while it is connected: the mid,
    else the last trade. Keyed by the name the legs use (ZNZ6 streams as TYZ6)."""
    marks = {}
    if SESSION.state == "connected":
        for quote in SESSION.quotes()["rows"]:
            bid, ask, last = quote["bid"], quote["ask"], quote["last"]
            price = (bid + ask) / 2 if bid and ask and bid > 0 and ask > 0 else last
            if price and price > 0:
                marks[quote["root"] + quote["contract"][-2:]] = price
    return marks


def sized(rows: list[dict], usd_per_bp: float, label) -> tuple[list[dict], dict | None, dict, dict]:
    """Every signal in a trade or calling for one, with its futures legs: (trades, the
    contracts' DV01, their closes, their live marks). label(row) names a signal the way
    the rest of the page does."""
    marks, trades = _marks(), []
    for row in rows:
        signal_id = row.get("signal_id") or row["module"]
        try:
            read = runner.analysis(signal_id)
        except RuntimeError:
            continue
        pack = package(row, read)
        if pack is not None:
            trades.append(dict(id=signal_id, module=row["module"], name=label(row), pack=pack,
                               held=read["open_entry"], rules=read.get("exits") or [], live=read["live"],
                               state=read["state"]))
    try:  # database asleep: the legs say so, the rest of the page stands
        dv01 = _dv01()
        entries = [t["held"]["entry_date"] for t in trades if t["held"] is not None]
        closes = _closes(min(entries, default=dt.date.today()) - dt.timedelta(days=7))
    except Exception:
        dv01, closes = None, {}
    for trade in trades:
        size(trade, usd_per_bp, dv01, closes, marks)
    return trades, dv01, closes, marks


def content(rows: list[dict], usd_per_bp: float, label) -> list:
    trades, dv01, closes, marks = sized(rows, usd_per_bp, label)
    log = _order_log({row.get("signal_id") or row["module"]: label(row) for row in rows})
    if not trades:
        return [heading("SIGNALS"), html.Div("No signal is in a trade or calling for one.", className="sd-note"), *log]
    in_trade = sum(t["held"] is not None for t in trades)
    asof = max(d["asof"] for d in dv01.values()) if dv01 else "—"
    last_close = max((day for history in closes.values() for day in history), default="—")
    marked = (f"futures marked from IBKR at {dt.datetime.now():%H:%M:%S}" if marks
              else f"futures marked at the {last_close} close (connect IBKR, then Refresh IBKR, for live)")
    return [
        heading("SIGNALS", f"{in_trade} open · {len(trades) - in_trade} to enter · {marked} · sized "
                           f"at ${usd_per_bp:,.0f} per bp of each package (assumed)"),
        _summary(trades),
        *[part for trade in trades for part in _section(trade)],
        html.Div("P&L bp is the signal's own trade at its latest close. P&L $ is what owning the legs' whole "
                 "contracts has made since the entry-date close, marked at IBKR's mid as of the last Refresh IBKR; "
                 "no fills are involved. Enter opens an order ticket for a paper account. DV01 $/bp is the size "
                 "those contracts carry: the legs' actual DV01 projected onto the package's weights. Entry is the "
                 "signal's reading on the day it fired and Live its latest close; live expected return is for a new "
                 f"entry. P&L model is P&L bp at ${usd_per_bp:,.0f} per bp, and slippage is futures less model: the "
                 "futures track their deliverables, not the yields the signal is built on, they carry and roll, "
                 "their DV01 drifts, and whole contracts miss the exact ratio. A leg's DV01 model and actual are "
                 "dollars gained per bp rise in its yield: buy futures to be short a yield, sell to be long it. "
                 "2y TU · 5y FV · 7y TY · 10y UXY · 20y US · 30y WN. DV01 each is every deliverable's DV01 / "
                 f"conversion factor weighted by its probability of being CTD, baskets as of {asof}, applied to "
                 "the whole holding period. The hedge beta is frozen at entry. Exit rules run 0% at entry to 100% "
                 f"when they fire, orange from {NEAR_EXIT:.0%}.",
                 className="sd-note", style={"marginTop": 10}),
        *log,
    ]


# ── the order ticket ─────────────────────────────────────────────────────────

TICKET_HEADERS = {"action": "Side", "quantity": "Qty", "contract": "Contract", "bid": "Bid", "ask": "Ask",
                  "limit": "Limit (decimal)", "dv01": "DV01 actual"}
SENT_HEADERS = {"sent": "Sent (UTC)", "signal": "Signal", "action": "Side", "quantity": "Qty", "contract": "Contract",
                "bid": "Bid at send", "ask": "Ask at send", "limit": "Limit", "order_id": "Order id",
                "status": "Status at send", "message": "IBKR"}
HIDDEN, SHOWN = {"display": "none"}, {"display": "flex"}


def _sent_table(records: list[dict], signal: bool) -> html.Div:
    """Orders as sent: each leg's limit beside the bid and ask it was sent into, in 32nds."""
    def price(value, root):
        return "—" if not value else format_tsy_price(value, is_ticks=False, tick_32nds=TICKS.get(root, 1))

    columns = [c for c in SENT_HEADERS if signal or c != "signal"]
    frame = pd.DataFrame([{
        "sent": (r.get("sent") or "")[:23].replace("T", " "), "signal": r.get("signal", ""), "action": r["action"],
        "quantity": str(int(r["quantity"])), "contract": r["contract"], "bid": price(r.get("bid"), r["root"]),
        "ask": price(r.get("ask"), r["root"]), "limit": price(r["limit"], r["root"]),
        "order_id": str(r["order_id"]), "status": r["status"], "message": r.get("message") or ""} for r in records])

    def style(column, _value, row):
        if column == "action":
            return {"textAlign": "left", "color": GAIN if row["action"] == "BUY" else LOSS, "fontWeight": "bold"}
        return {"textAlign": "left"} if column in ("sent", "signal", "contract", "status", "message") else None

    return grid_table(frame, columns=columns, headers=SENT_HEADERS, cell_style=style)


def _order_log(label_of: dict) -> list:
    """Every submission the dashboard has made, newest first: the audit trail."""
    records = [dict(order, signal=label_of.get(sub["signal_id"], sub["signal_id"]))
               for sub in TradeJournal().records("orders") for order in sub["orders"]]
    if not records:
        return []
    return [heading("ORDER LOG", "every order sent from here, newest first · bid and ask are IBKR's book the "
                                 "instant each leg was sent · prices in 32nds"),
            _sent_table(records, signal=True)]


def order_layout() -> html.Div:
    """The order ticket: hidden until a signal's enter button fills and shows it."""
    return html.Div([
        dcc.Store(id="order-ticket"),
        html.Div(id="order-modal", className="sd-modal", style=HIDDEN, children=html.Div([
            html.Div(id="order-body"),
            html.Div(className="sd-bar", children=[
                html.Button("Submit to IBKR", id="order-submit", className="ref-btn sd-btn", disabled=True),
                html.Button("Close", id="order-close", className="ref-btn sd-btn"),
            ]),
            html.Div(id="order-result", role="status"),
        ], className="sd-modal-box")),
    ])


def _ticket(trade: dict) -> tuple[list, list[dict]]:
    """The ticket's body for one trade, and the legs it would send. Each leg's limit starts
    at the far touch -- the ask to buy, the bid to sell -- where the stream is quoting."""
    quotes = {q["root"]: q for q in SESSION.quotes()["rows"]} if SESSION.state == "connected" else {}
    legs, records = [], []
    for leg in [leg for leg in trade["legs"] or [] if leg["signed"]]:
        quote = quotes.get(leg["root"]) or {}
        if quote and quote["contract"][-2:] != leg["contract"][-2:]:
            quote = {}  # the stream is on another month than the one the legs are sized on
        bid, ask = quote.get("bid"), quote.get("ask")
        touch = ask if leg["action"] == "BUY" else bid
        tick = TICKS[leg["root"]]

        def price(value):
            return "—" if not value or value <= 0 else format_tsy_price(value, is_ticks=False, tick_32nds=tick)

        records.append(dict(
            action=leg["action"], contract=leg["contract"], bid=price(bid), ask=price(ask), dv01=usd(leg["held"]),
            quantity=dcc.Input(id={"type": "order-qty", "leg": len(legs)}, type="number", min=1, step=1,
                               value=leg["quantity"]),
            limit=dcc.Input(id={"type": "order-limit", "leg": len(legs)}, type="number", step=tick / 32.0,
                            value=touch if touch and touch > 0 else None)))
        legs.append(dict(root=leg["root"], action=leg["action"]))

    def style(column, _value, record):
        if column == "action":
            return {"textAlign": "left", "color": GAIN if record["action"] == "BUY" else LOSS, "fontWeight": "bold"}
        return {"textAlign": "left"} if column == "contract" else None

    pack = trade["pack"]
    if not legs:
        return [heading(trade["name"].upper(), pack["name"]),
                html.Div(trade["why"] or "No legs to send.", className="sd-note")], []
    if SESSION.can_trade():
        where = html.Div(f"Submit sends {len(legs)} separate day limit orders to paper account {SESSION.account}, "
                         "one per row, back to back: not a spread order, so one leg can fill without the others. "
                         "Each limit starts at the far touch and can be changed · limits in decimal points.",
                         className="sd-note", style={"marginTop": 6})
    else:
        where = html.Div("Orders are only sent to a connected paper account: connect IBKR on a paper port first.",
                         className="sd-note", style={"marginTop": 6, "color": WARN})
    return [heading(trade["name"].upper(), f"{pack['name']} · order ticket"),
            grid_table(pd.DataFrame(records), columns=list(TICKET_HEADERS), headers=TICKET_HEADERS, cell_style=style),
            where], legs


def register_callbacks(app, rows, usd_per_bp: float, label) -> None:
    """rows() is the registry's signals now; usd_per_bp and label as in content()."""

    @app.callback(Output("order-modal", "style"), Output("order-body", "children"), Output("order-ticket", "data"),
                  Output("order-result", "children"), Output("order-submit", "disabled"),
                  Input({"type": "trade-enter", "signal": ALL}, "n_clicks"), Input("order-close", "n_clicks"),
                  prevent_initial_call=True)
    def show(_enters, _close):
        trigger = ctx.triggered_id
        if trigger == "order-close":
            return HIDDEN, no_update, None, "", True
        # a redrawn table makes new buttons, which fire with no click behind them
        if not isinstance(trigger, dict) or not ctx.triggered[0]["value"]:
            return (no_update,) * 5
        trade = next((t for t in sized(rows(), usd_per_bp, label)[0] if t["id"] == trigger["signal"]), None)
        if trade is None:
            return (no_update,) * 5
        body, legs = _ticket(trade)
        ticket = dict(signal_id=trade["id"], module=trade["module"], asof=str(trade["state"]["data"]["ts"][-1]),
                      legs=legs)
        return SHOWN, body, ticket, "", not (legs and SESSION.can_trade())

    @app.callback(Output("order-result", "children", allow_duplicate=True),
                  Output("order-submit", "disabled", allow_duplicate=True),
                  Input("order-submit", "n_clicks"), State("order-ticket", "data"),
                  State({"type": "order-qty", "leg": ALL}, "value"),
                  State({"type": "order-limit", "leg": ALL}, "value"), prevent_initial_call=True)
    def send(clicks, ticket, quantities, limits):
        if not clicks or not ticket:
            return no_update, no_update
        try:
            if None in quantities or None in limits or len(quantities) != len(ticket["legs"]):
                raise ValueError("every leg needs a quantity and a limit price")
            legs = [dict(leg, quantity=quantity, limit=limit)
                    for leg, quantity, limit in zip(ticket["legs"], quantities, limits)]
            journal = TradeJournal()
            intent = journal.flag(ticket["signal_id"], ticket["module"], ticket["asof"])
            sent = SESSION.submit(legs, f"views:{intent}")  # the tag ties orders and fills back to the signal
            journal.save_orders(intent, ticket["signal_id"], sent)
        except Exception as exc:
            return html.Div(f"Not sent ({type(exc).__name__}): {exc}", className="sd-note",
                            style={"color": LOSS, "marginTop": 8}), False
        return [html.Div(f"Sent {len(sent)} limit orders to {SESSION.account}, one per leg. They are in the order "
                         "log and under Open orders on the Portfolio tab.", className="sd-note",
                         style={"color": GAIN, "margin": "8px 0 4px"}),
                _sent_table(sent, signal=False)], True
