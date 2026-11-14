"""Trade tickets on the Portfolio tab: each signal that is in a trade, or calling for one,
as the futures orders that would express it. Preview only -- nothing is submitted."""
from __future__ import annotations

import datetime as dt

import pandas as pd
from dash import html

from dashboard import runner
from dashboard.sheet import grid_table, heading, usd
from execution.ticket import ROOTS, contract_legs, futures_dv01, yield_legs
from oven.engine import hedgeable
from research.panel import CATALOG

HEADERS = {"signal": "Signal", "package": "Package", "status": "Status", "action": "Side", "quantity": "Qty",
           "contract": "Contract", "exact": "Exact", "dv01": "DV01 each", "risk": "Leg $/bp", "residual": "Unfilled $/bp"}
BUY, SELL, WARN = "#27AE60", "#E74C3C", "#D35400"
_dv01_cache: dict[dt.date, dict] = {}


def _dv01() -> dict:
    """One basket pricing per day per server process; a failed one is retried next load."""
    today = dt.date.today()
    if today not in _dv01_cache:
        _dv01_cache.clear()
        _dv01_cache[today] = futures_dv01()
    return _dv01_cache[today]


def package(row: dict, read: dict) -> dict | None:
    """What the signal holds or wants: side, status, hedge and the beta that hedge is
    frozen at. None when it is flat."""
    state, held = read["state"], read["open_entry"]
    if held is not None:
        side, status, on = held["direction"], f"in trade since {held['entry_date']}", held["entry_date"]
    elif state["fired"] in ("long", "short"):
        side, status, on = state["fired"], "new entry", None
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


def content(rows: list[dict], usd_per_bp: float, label) -> list:
    """label(row) names a signal the way the rest of the page does."""
    title = heading("TRADE TICKETS", f"preview · ${usd_per_bp:,.0f} per bp of each package · nothing is submitted")
    wanted = []
    for row in rows:
        try:
            read = runner.analysis(row.get("signal_id") or row["module"])
        except RuntimeError:
            continue
        pack = package(row, read)
        if pack is not None:
            wanted.append((label(row), pack))
    if not wanted:
        return [title, html.Div("No signal is in a trade or calling for one.", className="sd-note")]
    try:
        dv01 = _dv01()
    except Exception as exc:  # database asleep: say so, the rest of the page stands
        return [title, html.Div(f"Futures DV01 unavailable ({type(exc).__name__}: {exc}).", className="sd-note",
                                style={"color": WARN})]
    records = []
    for name, pack in wanted:
        if pack["target"] not in CATALOG or any(t not in ROOTS for t in CATALOG[pack["target"]].legs):
            records.append(dict(signal=name, package=pack["name"], status=pack["status"], action="—", quantity="—",
                                contract="no futures expression", exact="—", dv01="—", risk="—", residual="—",
                                _action=None, _first=True))
            continue
        legs = contract_legs(yield_legs(pack["target"], pack["side"], pack["hedge"], pack["beta"]), usd_per_bp, dv01)
        for i, leg in enumerate(legs):
            priced = leg["dv01"] is not None
            records.append(dict(
                signal=name if i == 0 else "", package=pack["name"] if i == 0 else "",
                status=pack["status"] if i == 0 else "", action=leg["action"] or "—",
                quantity=str(leg["quantity"]) if priced else "—",
                contract=leg["contract"] or f"{leg['root']} (no DV01)",
                exact=f"{leg['exact']:+.2f}" if priced else "—", dv01=f"{leg['dv01']:,.1f}" if priced else "—",
                risk=usd(leg["risk"]), residual=usd(leg["residual"]) if priced else "—",
                _action=leg["action"], _first=i == 0))

    def style(column, _value, record):
        rule = {"borderTop": "1px solid #333333"} if record["_first"] else {}
        if column in ("signal", "package", "status", "contract"):
            return {"textAlign": "left", **({"fontWeight": "bold"} if column == "signal" else {}), **rule}
        if column == "action" and record["_action"]:
            return {"textAlign": "left", "color": BUY if record["_action"] == "BUY" else SELL, "fontWeight": "bold",
                    **rule}
        return rule or None

    asof = max(d["asof"] for d in dv01.values()) if dv01 else "—"
    return [title, grid_table(pd.DataFrame(records), columns=list(HEADERS), headers=HEADERS, cell_style=style),
            html.Div(f"Buy futures to be short a yield, sell to be long it. DV01 each: every deliverable's "
                     f"DV01 / conversion factor weighted by its probability of being CTD, baskets as of {asof} "
                     "(research app, Basis tab). 2y TU · 5y FV · 7y TY · 10y UXY · 20y US · 30y WN. The hedge "
                     "beta is frozen at entry. Exact is contracts before rounding; unfilled is the leg's risk "
                     "whole contracts leave.",
                     className="sd-note", style={"marginTop": 6})]
