"""Live signal dashboard -- trading overview plus signal deep dives.

The Live Overview tab groups trusted signals by traded target and answers the
desk questions first: is a position open, what does the latest reading say,
is its gate open, and how has the frozen backtest behaved?

The Signal Deep Dive tab selects one promoted signal at a time and exposes the
full chart, gate, PnL, and trade-history card. One button drives its data:

    Ref     fresh Strategy.load_data() cached to disk, then Strategy.compute()
            on it, logged as one timestamped row in the signal ledger.

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
import datetime as dt
import json
import re

import dash
import numpy as np
import pandas as pd
import polars as pl
from dash import ALL, MATCH, Input, Output, Patch, State, ctx, dcc, html

import utils.helpers
from dashboard import runner
from dashboard import markets
from dashboard import portfolio
from dashboard.charts import (
    chart_image,
    dual_theme_charts,
    DEFAULT_WINDOW,
    WINDOW_PRESETS,
    gate_chart,
    input_chart,
    level_chart,
    pnl_chart,
    return_distribution_chart,
    signal_chart,
)
from dashboard.ledger import SignalLedger
from dashboard.params import (
    exit_short, gate_short as _gate_rule, input_label, setting_rows, short_label,
    signal_label as _display_name,
)
from dashboard.registry import LiveRegistry
from dashboard.sheet import grid_table as _grid_table, heading as _heading, kv as _kv, kv_range as _range, usd as _usd
from stats import CONF_SHRINK
from research.app import _diverging, note, pnl_heatmap, regression_scatter, table
from utils.viz import table_div
from utils.research_app import (
    BORDER, C0, C1, DIM, FOLDER_TAB, FOLDER_TAB_SELECTED, FOLDER_TABS, ORANGE, TEXT, make_app, run,
    shimmer_loader,
)

REGISTRY = LiveRegistry()
LEDGER = SignalLedger()

TRADES_PER_PAGE = 6  # fewest trade rows a page holds; it grows to the monthly heatmap's years
# the Deep Dive's charts stand one to a row, so each is drawn (width, height in inches)
# for the row it fills: wide and short on a desktop, near square on a phone, where a
# wide chart scaled down to the screen leaves its text too small to read
WIDE_CHART = (18, 4.2)
PHONE_CHART = (5.6, 3.9)
PHONE_MAX_WIDTH = 900  # px; narrower viewports get PHONE_CHART
# Dollars per basis point of a signal's P&L: the size every signal is assumed to trade
# at, since no trade is wired to a signal yet. An assumption, not a measurement: about
# what the account's 10s30s steepener (8 TN against 4 UB) carried per bp when it was set.
USD_PER_BP = 750.0
SNAP_LEFT_BUFFER_BDAYS = 5

# the Oven's trade table: its columns, short lower-case headers and colours (oven.app)
TRADE_TABLE_COLS = [
    "entry_date", "exit_date", "direction",
    "entry_level", "exit_level", "target_lvl", "pnl_bps", "expected_return_bps", "expected_bars",
    "conf", "conf_score", "entry_half_life", "bars_held", "exit_reason",
]
TRADE_TABLE_HEADERS = {
    "entry_date": "entry", "exit_date": "exit", "direction": "dir",
    "entry_level": "entry lvl", "exit_level": "exit lvl",
    "target_lvl": "target lvl", "expected_return_bps": "exp ret", "expected_bars": "exp bars",
    "conf_score": "conf score", "entry_half_life": "half-life",
    "bars_held": "held", "exit_reason": "exit",
}
TRADE_TABLE_ROUND = {
    "entry_level": 2, "exit_level": 2, "target_lvl": 2, "expected_return_bps": 2, "expected_bars": 1,
    "pnl_bps": 2, "conf_score": 2, "entry_half_life": 1,
}
TRADE_TEXT_COLS = {"entry_date", "exit_date", "exit_reason"}

def _slug(signal_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", signal_id.lower()).strip("-")


def _pid(kind: str, signal_id) -> dict:
    """A card component's pattern-matching id. One set of MATCH callbacks then serves
    every card, so a signal promoted after the server started works on a browser refresh."""
    return {"type": kind, "sid": signal_id}


def _row_id(row: dict) -> str:
    """Registry identity, with module fallback for legacy/test rows."""
    return row.get("signal_id") or row["module"]


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


def _live_pnl_bps(trades, open_entry: dict | None) -> float:
    """Realized PnL from closed trades plus unrealized PnL from the open
    position, if any, over the full model history."""
    total = 0.0
    if trades is not None and not trades.is_empty():
        total += float(trades["pnl_bps"].sum())
    if open_entry is not None:
        total += float(open_entry["pnl_bps"])
    return total


def _format_trade_rows(trades) -> list[dict]:
    """Dashboard-safe, display-ready rows for the full closed-trade log.

    Keeping these compact rows in a dcc.Store lets the pager swap a slice
    without re-running the backtest that produced the log.
    """
    rows = []
    for row in trades.to_dicts():
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


def _visible_trade_range(trades, open_entry: dict | None, page: int, data_asof, size: int = TRADES_PER_PAGE):
    """Date span covering exactly what's currently shown in the trade table --
    the open row (first, on page 0) plus the closed trades on the current page."""
    dates = []
    shift = int(open_entry is not None)  # the open row takes the first slot, so closed rows sit one later
    if trades is not None and not trades.is_empty():
        first = max(0, page - shift)
        for row in trades.slice(first, page + size - shift - first).to_dicts():
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


def _position_label(state: dict, open_entry: dict | None) -> str:
    """One column for what the book holds and what today says.

    An open trade wins; otherwise report whether the signal is calling for one
    and, if it is, why nothing happened -- a fired-but-gated signal is the case
    a desk most needs to see, and it read as a bare "FLAT" before.
    """
    if open_entry is not None:
        return f"{open_entry['direction'].upper()} OPEN"
    signal = state["last"].get("signal")
    threshold = float(state["params"]["entry_threshold"])
    if signal is None or signal != signal:
        return "WARMING UP"
    if signal <= -threshold:
        side = "LONG"
    elif signal >= threshold:
        side = "SHORT"
    else:
        return "FLAT"
    return f"{side} GATED" if state["fired"] == "flat (gated)" else f"{side} SIGNAL"


def _gate_status(state: dict) -> str:
    params = state["params"]
    if params.get("gate") is None:
        return "—"
    last = state["last"]
    percentile = last.get("gate_percentile")
    if percentile is None or percentile != percentile:
        return "warming up"
    status = "open" if bool(last.get("gate_allow")) else "closed"
    pct = round(float(percentile) * 100)
    suffix = "th" if 10 <= pct % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(
        pct % 10, "th"
    )
    return f"{status} {pct}{suffix}"


LOCAL_TZ = dt.datetime.now().astimezone().tzinfo


def _local(when) -> pd.Timestamp:
    """Any timestamp as local wall-clock. The ledger stores UTC and file
    mtimes are naive local; the desk reads both in local time."""
    when = pd.Timestamp(when)
    return when.tz_convert(LOCAL_TZ) if when.tzinfo else when.tz_localize(LOCAL_TZ)


def _zone(when: pd.Timestamp) -> str:
    """Windows names the zone "Eastern Daylight Time" where the desk writes
    "EDT"; initials recover the abbreviation platforms that already report
    one leave alone."""
    name = when.strftime("%Z")
    parts = name.split()
    return "".join(part[0] for part in parts).upper() if len(parts) > 1 else name


def _clock_parts(
    when, same_day_as=None, always_date: bool = False
) -> tuple[str, str]:
    """A timestamp split into (date, time), so a caller can join them on one
    line or stack them on two.

    The date comes back empty when it falls on `same_day_as` (default: today)
    and would therefore be redundant; `always_date` forces it on for readouts
    that stand alone, with no neighbouring date to read them against.
    """
    when = _local(when)
    reference = (
        dt.date.today() if same_day_as is None else pd.Timestamp(same_day_as).date()
    )
    stamp = when.strftime("%I:%M:%S%p").lstrip("0").lower()
    dated = always_date or when.date() != reference
    return (f"{when:%Y-%m-%d}" if dated else ""), f"{stamp} {_zone(when)}"


def _clock(when, same_day_as=None, always_date: bool = False) -> str:
    """One-line timestamp as a human reads one: "2026-07-29 9:47:50pm EDT"."""
    day, stamp = _clock_parts(when, same_day_as, always_date)
    return f"{day} {stamp}".strip()


def _stacked(top: str, bottom: str):
    """Stat-block value laid out over two lines, falling back to one when
    there is no second line to show."""
    return [top, html.Br(), bottom] if bottom else top


def _db_asof(signal_id: str, data_asof) -> tuple[str, str, bool]:
    """The bar this card is drawn on, and when that data landed in the DB.

    Returns (bar, written, stale) as separate parts so a one-line table cell
    and a stacked stat block can each lay them out their own way.

    The two halves answer different questions, and conflating them is what
    makes a cold pipeline look like a broken app:

      bar / stale  is this CACHE behind the DB?   -> fix by clicking Ref
      written      is the DB itself behind now?   -> fix by running --live2

    Ref re-reads Postgres and never touches Bloomberg, so it can only ever
    make the page as current as the last pull. The write age is therefore the
    number that says whether clicking Ref can help at all. Stale (the orange
    flag) stays reserved for the case where it can: a newer bar, or the same
    bar rewritten since this cache was pulled -- the intraday case, since the
    live upserts move today's row without the date ever changing.
    """
    fresh = runner.db_freshness(signal_id)
    if fresh is None:
        return str(data_asof), "db unreachable", False
    db_bar = pd.Timestamp(fresh["last_ts"]).date()
    stale = db_bar > pd.Timestamp(data_asof).date()
    pulled_at = runner.cache_written_at(signal_id)
    if not stale and pulled_at is not None:
        stale = pd.Timestamp(fresh["last_written"]) > pd.Timestamp(pulled_at)
    written = f"db {_clock(fresh['last_written'], same_day_as=data_asof)}"
    return str(data_asof), written, stale


def _short_asof(bar: str, written: str) -> str:
    """'2026-10-29' + 'db 6:32:37pm EDT' as '10-29 · 6:32pm': the bar's month-day and
    the DB write time to the minute. A write on another day keeps its own date."""
    when = re.sub(r":\d\d(?=[ap]m)", "", written.removeprefix("db "))  # drop seconds
    when = re.sub(r"\s+[A-Z]{2,5}$", "", when)  # drop the zone
    when = re.sub(r"^\d{4}-(\d\d-\d\d)", r"\1", when)
    return f"{str(bar)[5:]} · {'db down' if written == 'db unreachable' else when}"


def _overview_snapshot(row: dict) -> dict:
    """One desk-level status row for a promoted signal."""
    base = {
        "signal_id": _row_id(row),
        "module": row["module"],
        "target": row["target"],
        "name": _display_name(row),
        "position": "UNAVAILABLE",
        "model": "—",
        "input_now": "—",
        "fair": "—",
        "conf": "—",
        "exp_usd": "—",
        "exp_work": "",
        "reading": "—",
        "gate_rule": "—",
        "gate_status": "—",
        "live_pnl_bps": None,
        "data_asof": "—",
        "data_stale": 0,
        "details": {},
        "reading_active": False,
        "reading_value": None,
        "exp_value": None,
    }
    try:
        signal_id = _row_id(row)
        read = runner.analysis(signal_id)
    except RuntimeError as exc:
        return {**base, "reading": str(exc)}
    state, trades, open_entry = read["state"], read["trades"], read["open_entry"]

    asof_bar, asof_written, asof_stale = _db_asof(signal_id, state["data_asof"])
    params = state["params"]
    signal_value = state["last"].get("signal")
    units = "bps" if params["entry_signal"] == "residual" else "z"
    threshold = float(params["entry_threshold"])
    reading_str = (
        f"{float(signal_value):+.2f}{units} / ±{threshold:g}{units}"
        if signal_value is not None and signal_value == signal_value
        else "warming up"
    )
    return {
        **base,
        **_signal_facts(row, state, read["live"]),
        "target": _trade_target(row, state),
        "details": _overview_details(row, state, read["live"], open_entry),
        "reading_active": bool(signal_value is not None and abs(float(signal_value)) >= threshold),
        "reading_value": float(signal_value) if signal_value is not None and np.isfinite(signal_value) else None,
        "position": _position_label(state, open_entry),
        "reading": reading_str,
        "gate_rule": _gate_rule(params),
        "gate_status": _gate_status(state).upper(),
        "live_pnl_bps": round(_live_pnl_bps(trades, open_entry), 1),
        # compact for the table: the target header above it carries the full stamp
        "data_asof": _short_asof(asof_bar, asof_written),
        # not a displayed column -- drives the stale conditional style below
        "data_stale": int(asof_stale),
    }


def _model_text(row: dict, state: dict) -> str:
    """What the signal is built from, in one cell: the target (named by the section
    header above the row) against beta x the input, and what is read off that -- the
    lookback and basis for an oven signal, the residual or its OU z for a book one."""
    feature = state["strategy"].feature
    beta = state["last"].get("beta")
    beta = f"{float(beta):+.2f}" if beta is not None and beta == beta else "β"
    if row.get("kind") == "oven":
        setup = state["strategy"].setup
        basis = {"portfolio": "port"}.get(setup["basis"], setup["basis"])
        return f"~ {beta}×{feature} · {setup['lookback']}d {basis}"
    kind = "resid" if state["params"]["entry_signal"] == "residual" else "OU z"
    return f"~ {beta}×{feature} · {kind}"


def _trade_target(row: dict, state: dict) -> str:
    """Name the traded package, keeping a model input distinct from an actual hedge."""
    from oven.engine import hedgeable

    target = row["target"]
    strategy = state.get("strategy")
    if hasattr(strategy, "setup") and hedgeable(strategy.feature):
        return f"{target} − β×{strategy.feature}"
    return target


def _overview_details(row: dict, state: dict, live: dict | None, held: dict | None) -> dict:
    """Short label/value pairs grouped under the overview's seven columns."""
    from research.panel import CATALOG

    params, strategy = state["params"], state.get("strategy")
    setup = getattr(strategy, "setup", {})
    target = CATALOG.get(row["target"])
    legs = " ".join(f"{weight:+g}×{leg}" for leg, weight in target.legs.items()) if target else row["target"]
    position = [("held", f"{held.get('bars_held', '—')} bars"),
                ("since", str(held.get("entry_date", "—"))[:10]),
                ("P&L", _fnum(held.get("pnl_bps"), "+.1f", "bp"))] if held else [("held", "none")]
    construction = [("legs", legs)]
    if setup:
        from oven.engine import hedgeable
        hedged = hedgeable(strategy.feature)
        construction += [("hedge", f"β×{strategy.feature}" if hedged else "none"),
                         ("β fit", f"Δ · {setup['hedge_lb']} bars") if hedged else ("input", strategy.feature),
                         ("signal", f"{setup['basis']} · {setup['lookback']} bars")]
        data = state.get("data")
        if hedged and data is not None and "beta_changes" in data.columns:
            construction.insert(2, ("β now", _fnum(data["beta_changes"][-1], "+.2f")))
        exits = [("revert", f"{setup['revert']:.0%}")] if setup.get("revert") else []
        exits += [("max hold", f"{setup['max_hold']} bars")]
        if setup.get("hl_mult"):
            exits.append(("time", f"{setup['hl_mult']:g}× HL"))
        if setup.get("stop"):
            exits.append(("stop", f"+{setup['stop']:g}σ"))
    else:
        construction += [("hedge", "none"), ("model input", row.get("feature", "—"))]
        style, value = params.get("exit_style"), params.get("exit_param")
        exits = []
        if value is not None:
            value = float(value)
            exits = [("exit", f"revert {value:.0%}" if style == "revert_frac" else
                      f"{value:g}× HL" if style == "half_life_frac" else
                      f"±{value:g}{'bp' if params['entry_signal'] == 'residual' else 'z'}" if style == "band" else str(style))]
    if params.get("stop_loss_bps"):
        exits.append(("loss stop", f"{params['stop_loss_bps']:g}bp"))
    expected = _expected(state, live) if live else None
    estimate = [("net", _fnum(expected["bps"], "+.2f", "bp")),
                ("horizon", _fnum(live.get("exp_bars"), ".0f", " bars") if setup else "OU mean"),
                ("cost", f"{getattr(strategy, 'transaction_cost_bps', 0):g}bp")] if expected else [("net", "—")]
    if expected and setup:
        resid = abs(float(live["resid"]))
        gross = expected["bps"] + float(getattr(strategy, "transaction_cost_bps", 0))
        estimate.insert(1, ("recovery", f"{gross / resid:.0%} of {resid:.2f}bp" if resid else "—"))
    conf = (live or {}).get("confidence", {})
    return dict(position=position, target=construction, reading=exits or [("exit", "—")],
                expected=estimate, confidence=[("trades", str(conf.get("n_past", "—")))])


def _signal_facts(row: dict, state: dict, live: dict | None) -> dict:
    """An overview row's what-it-is and where-it-stands cells: model, the input's value
    now, the target's fair value and today's confidence (``live``, runner.live_read's).
    '—' where the signal's state lacks them -- a decoration on the row must never break it."""
    try:
        feature = state["strategy"].feature
        expected = _expected(state, live)
        return {
            "model": _model_text(row, state),
            "input_now": f"{float(state['data'][feature][-1]):.1f}",  # the input is named in Built From
            "fair": f"{live['fair']:.1f}" if live["fair"] == live["fair"] else "—",
            "conf": live["confidence"]["label"],
            **({"exp_usd": _usd(expected["usd"]), "exp_value": expected["usd"], "exp_work": expected["work"]} if expected else {}),
        }
    except (KeyError, AttributeError, TypeError):  # TypeError: no live read (None)
        return {}


def _trade_cell_style(pnl_scale: float):
    """The Oven's trade colours: direction, P&L filled on a scale of the 95th percentile
    |P&L|, the open position flagged; plus confidence at entry."""
    def style(column: str, value, _row: dict) -> dict | None:
        if column == "exit_reason" and value == "active":
            return {"textAlign": "left", "color": ORANGE, "fontWeight": "bold"}
        if column in TRADE_TEXT_COLS:
            return {"textAlign": "left"}
        if column == "direction":
            return {"color": C1 if value == "long" else C0, "fontWeight": "bold"}
        if column == "conf" and value in CONF_COLOURS:
            return {"color": CONF_COLOURS[value], "fontWeight": "bold"}
        if column == "pnl_bps":
            return _diverging(value, pnl_scale)
        return None
    return style


def _tone(position: str) -> str:
    """A position label's colour family: 'long' or 'short' for a trade that is open or
    called for, 'gated' for one the gate is holding back, 'quiet' for everything else
    (flat, warming up, unavailable)."""
    for token, tone in (("GATED", "gated"), ("LONG", "long"), ("SHORT", "short")):
        if token in position:
            return tone
    return "quiet"


def _gate_colour(status: str) -> str:
    status = status.upper()
    return C0 if "CLOSED" in status else C1 if "OPEN" in status else DIM


def _expected(state: dict, live: dict | None) -> dict | None:
    """What an entry at today's reading is expected to make: {bps, usd, work}, or None
    when the model has no expectation yet. ``bps`` is runner.live_read's, net of cost;
    ``usd`` is that at USD_PER_BP; ``work`` spells the arithmetic out.

    An oven signal expects an OU recovery of its residual at the entry half-life, up to
    its expected exit day; a book signal expects the residual all the way back to its
    OU mean, with no clock on it."""
    if live is None or live["exp_return"] != live["exp_return"]:
        return None
    bps = float(live["exp_return"])
    cost = float(getattr(state["strategy"], "transaction_cost_bps", 0.0))
    gross, resid = bps + cost, abs(float(live["resid"]))
    if hasattr(state["strategy"], "setup"):
        share = gross / resid if resid else float("nan")
        work = (f"|resid| {resid:.2f}bp × {share:.0%} recovered by day {live['exp_bars']:.0f} "
                f"(half-life {live['half_life']:.1f}d) − {cost:g}bp cost")
    else:
        work = f"|OU mean − resid| {gross:.2f}bp, all the way back − {cost:g}bp cost"
    return dict(bps=bps, usd=bps * USD_PER_BP,
                work=f"{work} = {bps:+.2f}bp × ${USD_PER_BP:,.0f}/bp = {_usd(bps * USD_PER_BP)}")


def _signal_row(snap: dict) -> html.Details:
    """An expandable signal with compact details aligned under each sortable column."""
    tone = _tone(snap["position"])
    stale = bool(snap["data_stale"])  # the db holds a newer bar, or has rewritten this one, since the pull
    reading, separator, threshold = snap["reading"].partition(" / ")
    head = html.Summary(className="sd-row-head", children=[
        html.Span(snap["position"], className=f"sd-pos sd-pos--{tone}"),
        html.Span(snap["target"], className="sd-target", title=snap["target"]),
        html.Span([
            html.Span(reading, className="sd-reading--active" if snap.get("reading_active") else ""),
            separator + threshold,
        ], className="sd-reading"),
        html.Span(snap["exp_usd"], className="sd-reading"),
        html.Span(snap["conf"], className="sd-conf", style={"color": CONF_COLOURS.get(snap["conf"], TEXT)}),
        html.Span(snap["gate_status"], className="sd-gate", style={"color": _gate_colour(snap["gate_status"])}),
        html.Span([snap["name"], *([html.Span("STALE", className="sd-stale")] if stale else [])],
                  className="sd-name"),
    ])
    detail = snap.get("details", {})

    def cell(pairs):
        return html.Div([html.Div([html.Span(label, className="sd-detail-label"), html.Span(value)])
                         for label, value in pairs], className="sd-detail-cell")

    body = html.Div(className="sd-row-body sd-row-columns", children=[
        html.Div(),
        cell(detail.get("position", [])),
        cell(detail.get("target", [])),
        cell(detail.get("reading", [])),
        cell(detail.get("expected", [])),
        cell(detail.get("confidence", [])),
        cell([("rule", snap["gate_rule"])]),
        html.Div([
            cell([("as of", snap["data_asof"]), *([("data", "stale · click Ref")] if stale else [])]),
            html.Button("Deep Dive →", id=_pid("open-deep", snap["signal_id"]), className="ref-btn sd-btn"),
        ], className="sd-detail-cell"),
    ])
    sort_values = [snap["position"], snap["target"], snap.get("reading_value"), snap.get("exp_value"),
                   {"LOW": 0, "MED": 1, "HIGH": 2}.get(snap["conf"]), snap["gate_status"], snap["name"]]
    return html.Details([head, body], className="sd-row", **{"data-sort-values": json.dumps(sort_values)})


def _row_legend(key: str) -> html.Div:
    """The header row over a section's signals; its first cell sits over the rows' +/- marker."""
    labels = ["Position", "Target", "Reading / entry", "Exp $", "Conf", "Gate now", "Signal"]
    return html.Div([html.Span(), *[
        html.Span(html.Button([label, html.Span(className="viz-sort-arrow", **{"aria-hidden": "true"})],
                             className="sd-sort", **{"data-sort-column": str(i)}),
                  role="columnheader", **{"aria-sort": "none"})
        for i, label in enumerate(labels)
    ]], className="sd-row-legend")


def _refresh_all_data(rows: list[dict]) -> None:
    """Fresh Strategy.load_data() for every promoted module, deduped -- named
    variants share one cache file so each module is only pulled once."""
    for module in {row["module"] for row in rows}:
        runner.pull_data(module)


def _held_text(direction: str | None) -> str:
    """'LONG*' for a held position; the star marks an example (see markets.HELD)."""
    if not direction:
        return ""
    return direction.upper() + ("*" if markets.HELD_IS_EXAMPLE else "")


def _held_style(direction: str | None) -> dict:
    return {"color": C1 if direction == "long" else C0, "fontWeight": "bold", "fontSize": 12}


HEADLINE_COLUMNS = [("group", ""), ("name", ""), ("last", "Last"), ("d1", "1d"), ("m1", "1m"), ("ytd", "YTD"),
                    ("pos", "Pos")]


def _headline_cell_style(column: str, value, row: dict) -> dict | None:
    """Moves coloured by sign; a held line's name and position stand out; a rule over
    each group's first row."""
    rule = {"borderTop": f"1px solid {TEXT}"} if row["group"] else {}
    if column in ("group", "name"):
        return {"textAlign": "left", "fontWeight": "bold", **rule,
                **({"color": ORANGE} if column == "name" and row.get("pos") else {})}
    if column == "pos" and value:
        return {**_held_style("long" if value.startswith("LONG") else "short"), **rule}
    if column in ("d1", "m1", "ytd") and value != "—":
        return {"color": C0 if value.startswith("-") else C1 if value.strip("+0.") else TEXT, **rule}
    return rule or None


def _market_content() -> list:
    """Yields, curves and flies as one table: last level (yields in %, the rest in bps)
    and the 1d, 1m and YTD moves in bps. Read from the cache a Ref fills."""
    frame, pulled = markets.cached()
    if frame is None or frame.is_empty():
        return [html.Div("Market data not loaded yet: click Ref.", className="sd-note")]
    lines = markets.board(frame)
    rows = []
    for title, key, last_fmt in (("YIELDS (%)", "yields", ".3f"), ("CURVES", "curves", ".1f"),
                                 ("FLIES", "flies", ".1f")):
        rows += [{"group": title if i == 0 else "", "name": r["name"], "last": _fnum(r["last"], last_fmt),
                  **{k: _fnum(r[k], "+.1f") for k in ("d1", "m1", "ytd")}, "pos": _held_text(r["held"])}
                 for i, r in enumerate(lines[key])]
    stamp = f"bar {markets.asof(frame)} · pulled {_clock(pulled)} · changes in bps"
    if markets.HELD_IS_EXAMPLE and markets.HELD:
        stamp += " · * example position: no trades are wired to the dashboard yet"
    return [
        _grid_table(pd.DataFrame(rows), columns=[k for k, _ in HEADLINE_COLUMNS], headers=dict(HEADLINE_COLUMNS),
                    cell_style=_headline_cell_style),
        html.Div(stamp, className="sd-note", style={"marginTop": 6}),
    ]


# the overview's sections, in order; a target that is none of these is grouped by its family
STRUCTURES = {"outright": "OUTRIGHT DURATION", "curve": "CURVES", "fly": "FLIES"}
SECTION_NOUN = {"outright": "Tenor", "curve": "Curve", "fly": "Fly"}  # the target column's header


def _structure(row: dict) -> str:
    """What a signal trades, read off its target's name: '10y' an outright, '10s30s' a
    curve, '7s10s30s' a fly; anything else falls back to the registry family."""
    target = row["target"]
    if re.fullmatch(r"\d+[my]", target):
        return "outright"
    if re.fullmatch(r"(\d+s){2,3}", target):
        return "curve" if target.count("s") == 2 else "fly"
    return row.get("family") or "other"


def _overview_content(rows: list[dict]) -> list:
    """Each section's signals as one table."""
    if not rows:
        return [
            html.Div(
                "No signals promoted yet.",
                style={"color": DIM, "padding": "24px 0"},
            ),
        ]
    targets = sorted({row["target"] for row in rows})
    by_target = {
        target: sorted(
            [row for row in rows if row["target"] == target], key=_display_name
        )
        for target in targets
    }
    runner.prime_freshness([_row_id(row) for row in rows])  # every signal's db stamp in one query
    snapshots = {
        target: [_overview_snapshot(row) for row in group]
        for target, group in by_target.items()
    }
    # outrights, curves, flies, then anything else, each section's targets in order
    sections: dict[str, list[str]] = {}
    for target in targets:
        sections.setdefault(_structure(by_target[target][0]), []).append(target)
    order = [*[k for k in STRUCTURES if k in sections], *sorted(k for k in sections if k not in STRUCTURES)]
    out = []
    for key in order:
        rows_in = [snap for t in sections[key] for snap in snapshots[t]]
        firing = sum(_tone(snap["position"]) != "quiet" for snap in rows_in)
        out.append(html.Div([
            _heading(STRUCTURES.get(key, key.upper()), f"{firing} of {len(rows_in)} on"),
            html.Div([_row_legend(key), *[_signal_row(snap) for snap in rows_in]], className="sd-sheet"),
        ], className="sd-signal-section"))
    return [*out, html.Div(f"Exp $: new entry, net of cost · assumed ${USD_PER_BP:,.0f}/bp · "
                           "β: input hedge, fixed at entry · holding periods in trading bars.",
                           className="sd-note", style={"marginTop": 8})]


EXIT_HEADERS = {"target": "Target", "signal": "Signal", "rule": "Exit rule", "now": "Now", "exits_at": "Exits at",
                "progress": "Of the way"}
POSITION_HEADERS = {"target": "Target", "signal": "Signal", "dir": "Dir", "since": "Since", "held": "Days",
                    "pnl_bps": "P&L bp", "pnl_usd": "P&L $", "exp_bps": "Exp bp", "exp_usd": "Exp $",
                    "captured": "Captured", "nearest": "Nearest exit"}
NEAR_EXIT = 0.75  # an exit rule this far along is flagged


def _model_positions(rows: list[dict]) -> list:
    """The trades the signals are in, and how close each is to getting out.

    Two tables. Positions: one row per signal in a trade -- P&L in bps and in dollars at
    USD_PER_BP, what the trade expected at entry, the share of that captured, and its
    nearest exit. Exits: one row per exit rule of each -- where it stands, where it
    fires, and how far along it is (runner.exit_status)."""
    positions, exits = [], []
    for row in rows:
        try:
            read = runner.analysis(_row_id(row))
        except RuntimeError:
            continue
        held = read["open_entry"]
        if held is None:
            continue
        name = {"target": _trade_target(row, read["state"]), "signal": _display_name(row)}
        pnl, expected = float(held["pnl_bps"]), held.get("expected_return_bps")
        rules = read.get("exits") or []
        nearest = max(rules, key=lambda r: r["progress"], default=None)
        positions.append({
            **name, "dir": held["direction"], "since": str(held["entry_date"]), "held": str(held["bars_held"]),
            "pnl_bps": f"{pnl:+.1f}", "pnl_usd": _usd(pnl * USD_PER_BP),
            "exp_bps": _fnum(expected, "+.1f"), "exp_usd": _usd(None if expected is None else expected * USD_PER_BP),
            "captured": f"{pnl / expected:.0%}" if expected else "—",
            "nearest": f"{nearest['rule']} · {max(nearest['progress'], 0.0):.0%}" if nearest else "—",
            "_near": bool(nearest and nearest["progress"] >= NEAR_EXIT), "_pnl": pnl,
        })
        exits += [{**name, "rule": r["rule"], "now": r["now"], "exits_at": r["exits_at"],
                   "progress": f"{max(r['progress'], 0.0):.0%}", "_near": r["progress"] >= NEAR_EXIT} for r in rules]
    if not positions:
        return [_heading("MODEL POSITIONS"), html.Div("No signal is in a trade.", className="sd-note")]

    def style(column, value, record):
        if column in ("target", "signal", "rule", "now", "nearest", "since"):
            flag = {"color": ORANGE, "fontWeight": "bold"} if column in ("nearest", "rule") and record["_near"] else {}
            return {"textAlign": "left", **({"fontWeight": "bold"} if column == "target" else {}), **flag}
        if column == "dir":
            return {"textAlign": "left", "color": C1 if value == "long" else C0, "fontWeight": "bold"}
        if column in ("pnl_bps", "pnl_usd"):
            return {"color": C1 if record["_pnl"] >= 0 else C0, "fontWeight": "bold"}
        if column == "progress" and record["_near"]:
            return {"color": ORANGE, "fontWeight": "bold"}
        return None

    total = sum(p["_pnl"] for p in positions)
    return [
        _heading("MODEL POSITIONS", f"{len(positions)} signals in a trade · P&L {total:+.1f}bp · "
                                    f"{_usd(total * USD_PER_BP)} at ${USD_PER_BP:,.0f} per bp (assumed size)"),
        _grid_table(pd.DataFrame(positions), columns=list(POSITION_HEADERS), headers=POSITION_HEADERS,
                    cell_style=style),
        _heading("HOW CLOSE TO EXIT", f"each rule of each open trade · 0% at entry, 100% when it fires · "
                                      f"orange from {NEAR_EXIT:.0%}"),
        _grid_table(pd.DataFrame(exits), columns=list(EXIT_HEADERS), headers=EXIT_HEADERS, cell_style=style),
        html.Div("Exp bp is what the trade expected at its entry; Captured is P&L as a share of it. These are the "
                 "signals' own trades, not the broker's fills.", className="sd-note", style={"marginTop": 6}),
    ]


def _trade_page_label(n: int, page: int, size: int = TRADES_PER_PAGE) -> str:
    """The Oven's pager caption: which rows are on screen, and which page. ``n`` counts
    every row, the open position (listed first) included."""
    if not n:
        return "no closed trades yet"
    lo, hi = page + 1, min(page + size, n)
    return f"rows {lo:,}–{hi:,} of {n:,} · page {page // size + 1} of {(n - 1) // size + 1} · newest first"


def _trade_page_size(equity: pl.DataFrame) -> int:
    """One trade row per year of the monthly P&L heatmap beside it, so the two stand
    the same height without scrolling (as on the Oven's setup card)."""
    years = _active_years(equity)
    return max(TRADES_PER_PAGE, len(years))


def _trade_table_block(
    closed_rows: list[dict], page: int, open_row: dict | None, size: int = TRADES_PER_PAGE
) -> html.Div:
    """One visible trade page from already-formatted trade rows, as the Oven's trade
    table: its headers, two decimals, P&L filled. Columns this signal never fills
    (target lvl on an oven signal, exp bars on a book one) are left out."""
    n = len(closed_rows)
    if n == 0 and open_row is None:
        return html.Div("no closed trades yet", style={"color": DIM, "padding": "8px 0"})

    every = closed_rows + ([open_row] if open_row is not None else [])
    columns = [c for c in TRADE_TABLE_COLS if any(r.get(c) is not None for r in every)]
    pnls = np.abs([r["pnl_bps"] for r in every if isinstance(r.get("pnl_bps"), (int, float))])
    scale = float(np.nanpercentile(pnls, 95)) if len(pnls) else 0.0
    rows = every[-1:] + closed_rows if open_row is not None else closed_rows  # the open position first
    rows = rows[page:page + size]
    frame = pd.DataFrame(rows).reindex(columns=columns)
    frame = frame.astype(object).where(frame.notna(), "")
    return table(frame,  # untitled: the card's TRADES heading stands over it
                 columns=columns, headers=TRADE_TABLE_HEADERS, max_rows=len(rows), float_fmt=",.2f",
                 cell_style=_trade_cell_style(scale))


def _trade_table_section(
    closed_rows: list[dict], page: int, open_row: dict | None, signal_id: str, size: int = TRADES_PER_PAGE
) -> html.Div:
    """A fixed control row around the table area that the pager replaces.

    Keeping all buttons outside ``trade-table-*`` matters: replacing a Dash
    callback input can trigger dependent callbacks as it is reinserted. The
    pager therefore only replaces rows and its caption, never chart controls.
    """
    table = html.Div(
        _trade_table_block(closed_rows, page, open_row, size),
        id=_pid("trade-table", signal_id),
    )
    n = len(closed_rows)
    if n == 0 and open_row is None:
        return table
    small = {**_btn_style(), "padding": "2px 8px", "fontSize": 11}  # the Oven's pager buttons
    return html.Div([
        table,
        html.Span(_trade_page_label(n + (open_row is not None), page, size), id=_pid("trades-status", signal_id),
                  className="oven-pager-status"),
        html.Div(className="oven-pager", children=[
            html.Button("<<", id=_pid("trades-first", signal_id), n_clicks=0, className="ref-btn",
                        title="First trade page", style=small),
            html.Button("< prev", id=_pid("trades-prev", signal_id), n_clicks=0, className="ref-btn", style=small),
            html.Button("next >", id=_pid("trades-next", signal_id), n_clicks=0, className="ref-btn", style=small),
            html.Button(">>", id=_pid("trades-last", signal_id), n_clicks=0, className="ref-btn",
                        title="Last trade page", style=small),
            html.Button("Snap chart to view", id=_pid("snap", signal_id), n_clicks=0, className="ref-btn",
                        style=small),
        ]),
    ])


@dual_theme_charts
def _card_sections(
    signal_id: str,
    state: dict | None = None,
    trades=None,
    open_entry: dict | None = None,
    window: str = DEFAULT_WINDOW,
    page: int = 0,
    date_range: tuple | None = None,
    read_live: dict | None = None,
    phone: bool = False,
) -> tuple[html.Div, html.Div, dict | None, html.Div, html.Div]:
    row = REGISTRY.get(signal_id)
    if row is None:
        return (
            html.Div(),
            html.Div(f"{signal_id}: no longer promoted", style={"color": DIM}),
            None, html.Div(), html.Div(),
        )

    ledger_row = LEDGER.latest(signal_id)
    error = None
    live = None
    if state is None:
        # the card's first render; callback-driven renders pass the same read down
        try:
            read = runner.analysis(signal_id)
            state, trades, open_entry, live = read["state"], read["trades"], read["open_entry"], read["live"]
        except RuntimeError as exc:
            error = str(exc)
    elif read_live is not None:
        live = read_live

    live_pnl = _live_pnl_bps(trades, open_entry) if state and not error else None
    if live is None and state and not error:
        live = runner.live_read(state, trades, open_entry)
    if live is not None:
        trades = live["trades"]  # with each trade's confidence at its entry
    trade_data = {
        "closed": _format_trade_rows(trades) if trades is not None else [],
        "open": _open_trade_row(open_entry) if open_entry is not None else None,
        "size": _trade_page_size(state["equity_curve"]) if state and not error else TRADES_PER_PAGE,
    }
    asof_bar, asof_written, asof_stale = (
        _db_asof(signal_id, state["data_asof"]) if state else ("—", "", False)
    )
    small = lambda label, value, alert=False: html.Span(  # noqa: E731 -- one-line meta facts
        [html.Span(f"{label} "), value], style={"color": ORANGE if alert else TEXT})
    stamp = html.Div(className="sd-stamp", children=[
        small("data", f"{asof_bar} · {asof_written}", asof_stale),
        small("last run", _clock(ledger_row["run_ts"], always_date=True) if ledger_row else "never"),
    ])
    # how the strategy has done: its backtest on today's history
    stats = _backtest_stats(state["equity_curve"], trades) if state and not error else None
    record = _track_record(state if not error else None, stats, row, trades)

    if error:
        now_panel = html.Div()
        chart_body = html.Div(error, style={"color": ORANGE, "padding": "12px 0"})
        months = extra = html.Div()
    else:
        months = _monthly_pnl(state["equity_curve"])
        closed = trades is not None and not trades.is_empty()
        extra = html.Div([
            html.Div([_scatters(state)], className="sd-deep-view sd-deep-signal"),
            html.Div([
                *([_trade_mechanics(state, trades)] if closed else []),
                *([_confidence_section(state["equity_curve"], trades, live["confidence"])]
                  if closed and live is not None else []),
                *([_regime_section(state["equity_curve"], trades, runner.regime_states())] if closed else []),
            ], className="sd-deep-view sd-deep-performance"),
        ])
        now_panel = _now_panel(state, live, open_entry, live_pnl)
        window_bars = WINDOW_PRESETS.get(window, WINDOW_PRESETS[DEFAULT_WINDOW])
        span = dict(window_bars=window_bars, date_range=date_range, size=PHONE_CHART if phone else WIDE_CHART)
        target = state["strategy"].target
        # chart what the signal is built on: the hedged portfolio for a portfolio-basis
        # signal, else the outright target -- and beside it, always the input
        traded = "portfolio" if getattr(state["strategy"], "setup", {}).get("basis") == "portfolio" else target
        data = state["data"].with_columns(pl.Series("fair", _fair_series(state, traded)))
        signal_png = signal_chart(
            state["data"], state["signal_frame"],
            state["params"]["entry_signal"], state["params"]["entry_threshold"],
            fired=state["fired"], **span,
        )
        marks, held_mark = _on_series(trades, open_entry, state["data"], traded) if traded != target             else (trades, open_entry)
        level_png = level_chart(data, traded, trades=marks, open_entry=held_mark, overlays=["fair"], **span)
        second = input_chart(state["data"], state["strategy"].feature, **span)
        gate_png = gate_chart(
            state["data"], state["signal_frame"], state["params"].get("gate"),
            gate_window=state["params"].get("gate_window"), **span,
        )
        pnl_png = pnl_chart(state["equity_curve"], **span)
        # one to a row, in the order the trade is read: the signal, what it trades against
        # fair value, its input, its gate, then what it has made
        chart_body = html.Div(
            className="sd-charts",
            children=[html.Div([
                chart_image(png) for png in pngs if png is not None
            ], className=f"sd-charts sd-deep-view sd-deep-{view}") for view, pngs in (
                ("signal", [signal_png, level_png, second, gate_png]),
                ("performance", [pnl_png, return_distribution_chart(trades, size=span["size"])]))],
        )

    summary = html.Div([
        stamp,
        html.Div([_heading("SIGNAL"), now_panel], className="sd-deep-view sd-deep-signal"),
        html.Div([_heading("PERFORMANCE", "model replay on current cached history"), *record],
                 className="sd-deep-view sd-deep-performance"),
        html.Div([_heading("DETAILS"), _details_panel(row)], className="sd-deep-view sd-deep-details"),
    ])
    return summary, chart_body, trade_data, months, extra


def _fair_series(state: dict, series: str) -> np.ndarray:
    """The model's fair value for the series the signal is built on, bar by bar: that
    series less its residual (plus the OU mean a book signal reverts to)."""
    sig = state["signal_frame"]
    fair = state["data"][series].to_numpy().astype(float) - sig["resid"].to_numpy().astype(float)
    if "ou_mean" in sig.columns:
        fair = fair + sig["ou_mean"].fill_null(0.0).to_numpy().astype(float)
    return fair


def _on_series(trades, open_entry: dict | None, data: pl.DataFrame, series: str):
    """Trade markers moved onto another series (the portfolio): each entry and exit at
    that series' level on its date, since the logged levels are the target's."""
    level = data.select(pl.col("ts").cast(pl.Date), pl.col(series))
    if trades is not None and not trades.is_empty():
        trades = (trades.drop("entry_level", "exit_level")
                  .join(level.rename({"ts": "entry_date", series: "entry_level"}), on="entry_date", how="left")
                  .join(level.rename({"ts": "exit_date", series: "exit_level"}), on="exit_date", how="left"))
    if open_entry is not None:
        at = level.filter(pl.col("ts") == pl.lit(open_entry["date"]).cast(pl.Date))[series]
        open_entry = {**open_entry, "level": float(at[0]) if len(at) else open_entry["level"]}
    return trades, open_entry


def _backtest_stats(equity: pl.DataFrame, trades: pl.DataFrame | None, *, after: dt.date | None = None) -> dict | None:
    """The Oven's strategy stats from this signal's own daily P&L and closed trades on
    the cached history, scored from its first entry (the warm-up before says nothing);
    the halves split those days down the middle."""
    if equity.is_empty():
        return None
    scored = equity.with_columns(pl.col("ts").cast(pl.Date)).sort("ts")
    if after is not None:
        scored = scored.filter(pl.col("ts") > after)
        if trades is not None and not trades.is_empty():
            trades = trades.filter(pl.col("entry_date").cast(pl.Date) > after)
    elif trades is not None and not trades.is_empty():
        scored = scored.filter(pl.col("ts") >= trades["entry_date"].cast(pl.Date).min())
    else:
        active = scored.filter(pl.col("pnl_bps") != 0)
        if active.is_empty():
            return None
        scored = scored.filter(pl.col("ts") >= active["ts"].min())
    if scored.is_empty():
        return None
    daily = scored["pnl_bps"].to_numpy().astype(float)

    def sharpe(d):
        return d.mean() / d.std() * np.sqrt(252) if len(d) > 1 and d.std() > 0 else np.nan

    half = len(daily) // 2
    cum = np.cumsum(daily)
    total = float(cum[-1])
    max_dd = float((cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]).min())
    years = len(daily) / 252
    pnl = trades["pnl_bps"] if trades is not None and not trades.is_empty() else pl.Series([], dtype=pl.Float64)
    losses = float(pnl.filter(pnl <= 0).sum())
    return dict(
        sharpe=sharpe(daily), sharpe_h1=sharpe(daily[:half]), sharpe_h2=sharpe(daily[half:]),
        split=scored["ts"][half], total_pnl=total, pnl_per_year=total / years, n_trades=len(pnl),
        trades_per_year=len(pnl) / years, hit_rate=float((pnl > 0).mean()) if len(pnl) else np.nan,
        avg_trade=float(pnl.mean()) if len(pnl) else np.nan,
        median_trade=float(pnl.median()) if len(pnl) else np.nan,
        profit_factor=float(pnl.filter(pnl > 0).sum()) / abs(losses) if losses else np.nan,
        max_drawdown=max_dd, calmar=(total / years) / abs(max_dd) if max_dd else np.nan,
        time_in_market=float((daily != 0).mean()),
        best_trade_share=float(pnl.max()) / total if total > 0 and len(pnl) else np.nan,
        first=scored["ts"][0], last=scored["ts"][-1], bars=len(daily),
    )


def _promotion_day(row: dict) -> dt.date | None:
    value = row.get("promoted_at")
    if not value:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York")
    return stamp.date()


def _track_record(state: dict | None, stats: dict | None, row: dict, trades=None) -> list:
    """Compare replay periods with the frozen promotion snapshot."""
    promoted = _promotion_day(row)
    since = _backtest_stats(state["equity_curve"], trades, after=promoted) if state and promoted else None
    frozen = {**row, "max_drawdown": row.get("max_drawdown_bps")}
    metrics = [
        ("Sharpe", "sharpe", ".2f", ""), ("P&L", "total_pnl", "+,.2f", " bp"),
        ("P&L / year", "pnl_per_year", "+,.2f", " bp"), ("Closed trades", "n_trades", ".0f", ""),
        ("Hit rate", "hit_rate", ".0%", ""), ("Avg trade", "avg_trade", "+.2f", " bp"),
        ("Median trade", "median_trade", "+.2f", " bp"), ("Max drawdown", "max_drawdown", ".2f", " bp"),
        ("Profit factor", "profit_factor", ".2f", ""), ("Calmar", "calmar", ".2f", ""),
        ("Trades / year", "trades_per_year", ".1f", ""),
        ("Sharpe · first half", "sharpe_h1", ".2f", ""),
        ("Sharpe · second half", "sharpe_h2", ".2f", ""),
        ("Best trade / total P&L", "best_trade_share", ".0%", ""),
        ("Days with P&L", "time_in_market", ".0%", ""),
    ]
    records = [dict(metric=label,
                    history=_fnum((stats or {}).get(key), fmt, suffix),
                    since=_fnum((since or {}).get(key), fmt, suffix),
                    promotion=_fnum(frozen.get(key), fmt, suffix)) for label, key, fmt, suffix in metrics]
    period = lambda s: f"{s['first']} – {s['last']} ({s['bars']} bars)" if s else "—"
    return [
        _grid_table(pd.DataFrame(records), columns=["metric", "history", "since", "promotion"],
                    headers={"metric": "Metric", "history": "Full history", "since": "Since promotion",
                             "promotion": "At promotion"}),
        html.Div(f"Full history: {period(stats)} · Since promotion: {period(since)}", className="sd-note"),
        html.Div(f"Promotion: {promoted or 'not recorded'}. Since-promotion P&L starts on the next trading bar "
                 "and includes carried positions; trade statistics count only trades entered afterward and closed. "
                 "Both periods are model replays, not broker returns. At promotion is the saved backtest snapshot.",
                 className="sd-note", style={"maxWidth": 850, "marginTop": 4}),
    ]


def _details_panel(row: dict) -> html.Div:
    """Provenance and frozen settings, separate from today's signal and outcomes."""
    return _range([
        _kv("promoted", _clock(row["promoted_at"], always_date=True) if row.get("promoted_at") else "not recorded"),
        _kv("source", row.get("selection_source") or "—"),
        _kv("rationale", row.get("rationale") or "not recorded"),
        _kv("target", row["target"]), _kv("model input", row["feature"]),
        *[_kv(label, value) for label, value in setting_rows(row)],
        _kv("module", row["module"]),
    ])


def _active_years(equity: pl.DataFrame) -> pl.DataFrame:
    """P&L and active days per year, from the first year the signal was in the market."""
    years = (equity.select(pl.col("ts").cast(pl.Date), "pnl_bps")
             .group_by(pl.col("ts").dt.year().alias("year"))
             .agg(pl.col("pnl_bps").sum(), (pl.col("pnl_bps") != 0).sum().alias("active_days")).sort("year"))
    first = years.filter(pl.col("active_days") > 0)["year"].min()
    return years.clear() if first is None else years.filter(pl.col("year") >= first)


def _monthly_pnl(equity: pl.DataFrame) -> html.Div:
    """The Oven's monthly P&L heatmap, from the first year the signal was in the market."""
    years = _active_years(equity)
    if years.is_empty():
        return html.Div()
    daily = equity.select(pl.col("ts").cast(pl.Date), "pnl_bps")
    return pnl_heatmap(daily.filter(pl.col("ts").dt.year() >= years["year"].min()), years)


def _scatters(state: dict) -> html.Div:
    """The Oven's levels and changes scatters of the target on its input, fitted over the
    signal's own lookback with the stretch before it in grey."""
    target, feature = state["strategy"].target, state["strategy"].feature
    if feature not in state["data"].columns:
        return html.Div()
    lookback = int(state["params"]["beta_lb"])
    return _section("REGRESSION FIT", [html.Div(
        className="sd-charts sd-charts--narrow",
        children=[regression_scatter(state["data"], target, feature, basis, lookback, "previous")
                  for basis in ("levels", "changes")],
    )], f"{target} on {feature}: levels and daily changes")


def _exit_reasons(equity: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """Per exit reason, plus 'all': trades, avg and total P&L, its share of the total, hit
    rate, bars held and the Sharpe of holding only those trades (as _performance)."""
    total = trades["pnl_bps"].sum()
    days, daily = _scored_days(equity, trades)
    groups = [("all", trades)] + [(r, trades.filter(pl.col("exit_reason") == r))
                                  for r in trades["exit_reason"].value_counts(sort=True)["exit_reason"]]
    return pl.DataFrame([dict(reason=r, trades=len(t), avg_trade=t["pnl_bps"].mean(), total_pnl=t["pnl_bps"].sum(),
                              share_of_pnl=t["pnl_bps"].sum() / total if total else None,
                              hit_rate=(t["pnl_bps"] > 0).mean(), avg_held=t["bars_held"].mean(),
                              sharpe=_performance(days, daily, t, total)["sharpe"])
                         for r, t in groups])


def _trade_mechanics(state: dict, trades: pl.DataFrame) -> html.Div:
    """The Oven's trade mechanics: where the P&L came from, by how each trade ended."""
    target = state["strategy"].target
    lines = [f"Median closed-trade P&L: {trades['pnl_bps'].median():,.2f} bps"]
    if "portfolio" in state["data"].columns:  # an oven signal: the target hedged by its input
        level = state["data"].select(pl.col("ts").cast(pl.Date), pl.col(target))
        legs = (trades.join(level.rename({"ts": "entry_date", target: "y0"}), on="entry_date", how="left")
                .join(level.rename({"ts": "exit_date", target: "y1"}), on="exit_date", how="left")
                .with_columns((pl.when(pl.col("direction") == "long").then(1).otherwise(-1)
                               * (pl.col("y1") - pl.col("y0"))).alias("target_leg")))
        target_leg = legs["target_leg"].sum()
        lines.append(f"of total P&L, the target leg made {target_leg:+,.1f} bps and the hedge leg and costs "
                     f"{trades['pnl_bps'].sum() - target_leg:+,.1f} bps")
    return _section("TRADE MECHANICS", [
        note(" · ".join(lines), "dim"),
        table(_shown(_exit_reasons(state["equity_curve"], trades)), title="exit reasons",
              float_fmt=",.2f",
              headers={"avg_trade": "avg trade", "total_pnl": "total pnl", "hit_rate": "hit",
                       "avg_held": "avg held", "share_of_pnl": "share of pnl"},
              cell_style=lambda col, value, row: (
                  {"textAlign": "left", **({"fontWeight": "bold"} if value == "all" else {})}
                  if col == "reason" else None)),
    ], "where the P&L came from, by how each trade ended")


CONF_COLOURS = {"HIGH": C1, "LOW": C0, "MED": TEXT, "—": DIM}


def _now_panel(state: dict, live: dict, open_entry: dict | None, live_pnl: float | None) -> html.Div:
    """Today, as one label | value | note range, in the order it is read: what the book
    holds or the signal calls for, the reading against its entry threshold, the open
    position, what an entry now is expected to make, level against fair value,
    half-life, walk-forward confidence and what drives it, and the gate."""
    params = state["params"]
    units = "bps" if params["entry_signal"] == "residual" else "z"
    reading = live["resid"] if units == "bps" else live["z"]
    position = _position_label(state, open_entry)
    tone_colour = {"long": C1, "short": C0, "gated": ORANGE}.get(_tone(position))
    conf = live["confidence"]

    if open_entry is not None:
        held = open_entry
        progress = f"{held['bars_held']}d" + (f" of ~{held['expected_bars']:.0f}d" if held.get("expected_bars") else "")
        expected = (f" · expected {held['expected_return_bps']:+.1f} bp at entry"
                    if held.get("expected_return_bps") is not None else "")
        position_row = _kv("open position", f"{held['pnl_bps']:+.1f} bp",
                           f"{held['direction']} since {held['entry_date']} · held {progress}{expected}",
                           C1 if held["pnl_bps"] >= 0 else C0)
    else:
        position_row = _kv("open position", "none")
    if live["exp_return"] == live["exp_return"]:
        over = (f"over ~{live['exp_bars']:.0f}d · {live['bps_per_day']:+.2f} bp/day"
                if live["exp_bars"] == live["exp_bars"] else "to its OU mean")
        risk = f" · risk {live['exp_risk']:.1f} bp" if live["exp_risk"] == live["exp_risk"] else ""
        entry_row = _kv("if entered now", f"{live['exp_return']:+.1f} bp · {_usd(live['exp_return'] * USD_PER_BP)}",
                        f"{over}{risk} · at ${USD_PER_BP:,.0f} per bp")
    else:
        entry_row = _kv("if entered now", "—", "no half-life yet")
    gate = _gate_status(state)
    drivers = conf["drivers"] or ("needs more closed trades" if conf["label"] == "—" else "")
    expected = _expected(state, live)
    return html.Div([_range([
        _kv("position", position, colour=tone_colour),
        _kv("reading", f"{reading:+.2f}{units}", f"entry at ±{float(params['entry_threshold']):g}{units}"),
        position_row,
        _kv("β now", _fnum(state["last"].get("beta"), "+.3f"), "signal model coefficient"),
        _kv("R² now", _fnum(state["last"].get("r2"), ".2f")),
        _kv("level / fair", f"{live['level']:.1f} / {live['fair']:.1f}",
            f"resid {live['resid']:+.1f} bp · z {live['z']:+.2f}"),
        _kv("half-life", f"{live['half_life']:.1f}d"),
        _kv("exit", exit_short(params), "every setting under Details"),
        _kv("loss stop", _fnum(params.get("stop_loss_bps"), "g", "bp")),
        _kv("confidence", conf["label"],
            f"from {conf['n_past']} closed trades",
            CONF_COLOURS.get(conf["label"])),
        *([_kv("gate", gate, _gate_rule(params), _gate_colour(gate))] if params.get("gate") is not None else []),
    ]),
        _section("ENTRY ESTIMATE", [_range([
            entry_row,
            *([_kv("calculation", expected["work"])] if expected else []),
        ])], "hypothetical entry now · net of cost"),
        _section("CONFIDENCE DRIVERS", [html.Div(drivers, className="sd-note")]),
    ])


CONF_LABELS = {**runner.OVEN_CONFIDENCE, **runner.BOOK_CONFIDENCE}  # feature column -> name


def _scored_days(equity: pl.DataFrame, trades: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(dates, daily P&L) from the first entry on: the span every Sharpe here is scored over."""
    first = trades["entry_date"].cast(pl.Date).min()
    daily = equity.select(pl.col("ts").cast(pl.Date), "pnl_bps").filter(pl.col("ts") >= first)
    return daily["ts"].to_numpy(), daily["pnl_bps"].to_numpy().astype(float)


def _performance(days: np.ndarray, daily: np.ndarray, subset: pl.DataFrame, total: float) -> dict:
    """Trades, total and average P&L, share of all P&L, hit rate, and the Sharpe of
    holding only ``subset``'s trades: annualized over every day in ``days``, the
    signal's daily P&L on the days those trades were on and 0 on all others (with all
    trades: the strategy's own Sharpe)."""
    held = np.zeros(len(days), dtype=bool)
    for a, b in subset.select(pl.col("entry_date").cast(pl.Date), pl.col("exit_date").cast(pl.Date)).iter_rows():
        held |= (days >= np.datetime64(a)) & (days <= np.datetime64(b))
    marks = np.where(held, daily, 0.0)
    pnl = subset["pnl_bps"]
    return dict(trades=len(subset), total_pnl=float(pnl.sum()), avg_trade=pnl.mean() if len(pnl) else None,
                hit_rate=float((pnl > 0).mean()) if len(pnl) else None,
                sharpe=marks.mean() / marks.std() * np.sqrt(252) if len(marks) > 1 and marks.std() > 0 else None,
                share_of_pnl=float(pnl.sum()) / total if total else None)


def _confidence_by_tier(days, daily, trades: pl.DataFrame) -> pl.DataFrame:
    """Performance of all trades, then of each confidence tier the trades had at entry;
    '—' is the first trades, before enough had closed to score one."""
    total = float(trades["pnl_bps"].sum())
    rows = [dict(conf="all", **_performance(days, daily, trades, total), avg_score=trades["conf_score"].mean())]
    for tier in ("HIGH", "MED", "LOW", "—"):
        part = trades.filter(pl.col("conf") == tier)
        if len(part):
            rows.append(dict(conf=tier, **_performance(days, daily, part, total), avg_score=part["conf_score"].mean()))
    return pl.DataFrame(rows, infer_schema_length=None)


def _confidence_by_component(days, daily, trades: pl.DataFrame) -> pl.DataFrame:
    """Performance of the scored trades by each confidence feature's tercile at entry
    (cut on the trades closed before it), with that feature's average edge there."""
    total = float(trades["pnl_bps"].sum())
    rows = []
    for col in [c.removeprefix("conf_bucket_") for c in trades.columns if c.startswith("conf_bucket_")]:
        named = False  # the feature's name on its first row only
        for bucket in ("low", "mid", "high"):
            part = trades.filter(pl.col(f"conf_bucket_{col}") == bucket)
            if len(part):
                rows.append(dict(feature="" if named else CONF_LABELS.get(col, col), tercile=bucket,
                                 **_performance(days, daily, part, total), avg_edge=part[f"conf_edge_{col}"].mean()))
                named = True
    return pl.DataFrame(rows, infer_schema_length=None)


def _tercile_trades(trades: pl.DataFrame, c: dict) -> pl.DataFrame:
    """The closed trades in today's tercile of one confidence feature (component ``c``),
    on the same cuts the score used."""
    x = pl.col(c["col"])
    usable = trades.filter(x.is_not_null() & x.is_finite())
    lo, hi = c["cut_lo"], c["cut_hi"]
    return usable.filter({"low": x <= lo, "high": x > hi}.get(c["bucket"], (x > lo) & (x <= hi)))


def _confidence_now(conf: dict, trades: pl.DataFrame, days, daily) -> pl.DataFrame:
    """Today's confidence, built up: per feature, where today's value falls among the
    closed trades and what that tercile made -- its average trade, which the score is
    built from, and for reference its Sharpe beside all trades' (neither Sharpe enters
    the score); then a bottom 'score' row -- the mean edge, the cuts that tier it (in
    the low / high columns), the tier it lands in (in the tercile column) and the
    closed trades it is scored from."""
    total = float(trades["pnl_bps"].sum())
    all_sharpe = _performance(days, daily, trades, total)["sharpe"]
    rows = [dict(feature=c["feature"], now=c["value"], low_to=c["cut_lo"], high_from=c["cut_hi"], tercile=c["bucket"],
                 trades=c["n"], tercile_avg=c["avg"], all_avg=c["past_avg"],
                 tercile_sharpe=_performance(days, daily, _tercile_trades(trades, c), total)["sharpe"],
                 all_sharpe=all_sharpe, spread=c["spread"], weight=c["weight"], edge=c["edge"])
            for c in conf["components"]]
    rows.append(dict(feature="score", low_to=conf["tier_lo"], high_from=conf["tier_hi"], tercile=conf["label"],
                     trades=conf["n_past"], edge=conf["score"]))
    return pl.DataFrame(rows, infer_schema_length=None)


def _subtitle(text: str) -> html.Div:
    """The one line of explanation under a table, kept out of its title."""
    return html.Div(text, style={"fontSize": 10, "color": DIM, "margin": "-2px 0 8px"})


def _shown(frame: pl.DataFrame) -> pd.DataFrame:
    """A frame for the house table, missing values blank rather than 'nan'."""
    pdf = frame.to_pandas()
    return pdf.astype(object).where(pdf.notna(), "")


PERF_HEADERS = {"total_pnl": "total pnl", "avg_trade": "avg trade", "hit_rate": "hit", "share_of_pnl": "share of pnl",
                "avg_score": "avg score", "avg_edge": "avg edge"}
PERF_TEXT = ("conf", "feature", "tercile", "regime", "state")


def _perf_style(frames: list[pl.DataFrame], trades: pl.DataFrame):
    """Colours for the performance tables: Sharpe on one scale across ``frames``, the
    average trade on the trades' own, text left, confidence tiers and today's regime
    state marked."""
    sharpes = pl.concat([f["sharpe"].cast(pl.Float64) for f in frames if len(f)]).drop_nulls()
    sharpe_scale = float(sharpes.abs().max()) if len(sharpes) else 1.0
    avg_scale = float(trades["pnl_bps"].abs().quantile(0.75) or 1.0)

    def style(col, value, row):
        if col in PERF_TEXT:
            colour = {"color": CONF_COLOURS[value], "fontWeight": "bold"} if value in CONF_COLOURS else {}
            return {"textAlign": "left", **colour, **({"fontWeight": "bold"} if value == "all" else {})}
        if col == "now":
            return {"color": ORANGE, "fontWeight": "bold"}
        if col == "sharpe":
            return _diverging(value, sharpe_scale)
        if col == "avg_trade":
            return _diverging(value, avg_scale)
        return None
    return style


def _section(title: str, blocks: list, hint: str = "", quiet: bool = False) -> html.Details:
    """Detail folded behind its title: one line on the card until it is opened. ``hint``
    says what is inside; ``quiet`` drops the box, for a footnote under a panel."""
    return html.Details(className="sd-fold sd-fold--quiet" if quiet else "sd-fold", children=[
        html.Summary([title, *([html.Span(hint, className="sd-h-note")] if hint else [])]),
        html.Div(blocks, className="sd-fold-body"),
    ])


def _confidence_section(equity: pl.DataFrame, trades: pl.DataFrame, conf: dict) -> html.Div:
    """Confidence in tables: today's score built up feature by feature, and how past
    trades did by the tier and by each feature's tercile they had at entry."""
    if "conf" not in trades.columns:
        return html.Div()
    days, daily = _scored_days(equity, trades)
    tiers, parts = _confidence_by_tier(days, daily, trades), _confidence_by_component(days, daily, trades)
    perf_style = _perf_style([tiers, parts], trades)
    blocks = [table(_shown(tiers), title="by confidence at entry", float_fmt=",.2f",
                    headers=PERF_HEADERS, cell_style=perf_style)]
    if len(parts):
        blocks.append(html.Div([table(_shown(parts), title="by feature tercile at entry", float_fmt=",.2f",
                                      headers=PERF_HEADERS, cell_style=perf_style),
                                _subtitle("terciles cut on the trades closed before each entry")]))
    if conf.get("components"):
        now = _confidence_now(conf, trades, days, daily)
        edge_scale = float(now["edge"].abs().max() or 1.0)

        def now_style(col, value, row):
            score_row = row["feature"] == "score"
            line = {"fontWeight": "bold", "borderTop": f"1px solid {TEXT}"} if score_row else {}
            if col == "tercile" and value in CONF_COLOURS:
                return {"textAlign": "left", "color": CONF_COLOURS[value], **line}
            if col in ("feature", "tercile"):
                return {"textAlign": "left", **line}
            if col == "edge":
                return {**(_diverging(value, edge_scale) or {}), **line}
            if col == "tercile_avg" and not score_row:
                return _diverging(value - row["all_avg"], row["spread"])
            if col == "tercile_sharpe" and not score_row and isinstance(value, float) and row["all_sharpe"]:
                return _diverging(value - row["all_sharpe"], abs(row["all_sharpe"]))  # green: beats all trades
            return line or None

        blocks.insert(0, html.Div([
            table(_shown(now), title="today's score", float_fmt=",.2f", cell_style=now_style,
                  headers={"low_to": "low ≤", "high_from": "high >", "tercile_avg": "tercile avg", "all_avg": "all avg",
                           "tercile_sharpe": "tercile sharpe", "all_sharpe": "all sharpe"}),
            _subtitle(f"edge = weight × (tercile avg − all avg) ÷ spread · weight = trades ÷ (trades + {CONF_SHRINK})"
                      f" · score = mean edge, tiered by {conf['tier_from']} · sharpes shown, not scored")]))
    return _section("CONFIDENCE", blocks, "today's score, and how past trades did by tier and feature")


def _regime_rule(name: str, p) -> tuple[str, str, dict]:
    """A research regime's short name, what it measures in a few words, and where each
    of its states sits on that measure (from research.regimes.RegimeParams ``p``, so
    the ranges match the research app)."""
    years = f"{p.vol_rank_window / 252:g}y"
    lo, hi = f"{p.vol_low * 100:.0f}", f"{p.vol_high * 100:.0f}"
    vol = {"low vol": f"< {lo}th pct", "normal vol": f"{lo}–{hi}th", "high vol": f"≥ {hi}th pct"}
    cut = p.cycle_threshold_bp
    infl_lo, infl_hi = p.inflation_low_bp / 100, p.inflation_high_bp / 100
    return {
        "policy": ("Policy cycle", "fed: 6m done + 6m priced",
                   {"cutting": f"< −{cut:g}bp", "on hold": f"−{cut:g} to +{cut:g}bp", "hiking": f"≥ +{cut:g}bp"}),
        "vol": ("Rate vol", f"10y {p.vol_window}d realized, {years} pct", vol),
        "implied": ("Implied vol", f"1m10y normal vol, {years} pct", vol),
        "oil": ("Oil sensitivity", f"{p.oil_window}d corr, 10y vs oil",
                {"none / inverse": f"< {p.oil_low:g}", "weak": f"{p.oil_low:g} to {p.oil_high:g}",
                 "trades with oil": f"≥ {p.oil_high:g}"}),
        "inflation": ("Inflation", "5y inflation swap",
                      {"low inflation": f"< {infl_lo:.2f}%", "anchored": f"{infl_lo:.2f}–{infl_hi:.2f}%",
                       "high inflation": f"≥ {infl_hi:.2f}%"}),
    }[name]


def _regime_live(name: str, today: dict, col: str) -> str:
    """A regime's value today, in its own units."""
    value, pct = today.get(f"{col}_value"), today.get(f"{col}_pct")
    if value is None or value != value:
        return "—"
    rank = f" · {pct * 100:.0f}th pct" if pct is not None and pct == pct else ""
    return {"policy": f"{value:+.0f}bp", "vol": f"{value:.0f}bp/yr{rank}", "implied": f"{value:.0f}bp/yr{rank}",
            "oil": f"{value:+.2f}", "inflation": f"{value / 100:.2f}%"}.get(name, f"{value:.2f}")


def _by_regime(days, daily, trades: pl.DataFrame, states: pl.DataFrame) -> pl.DataFrame:
    """Performance of the trades by each research regime's state on their entry day.

    Each regime is scored over its own span -- from the first day it has a state --
    so one that starts late (implied vol, 2022) is judged only on the trades and days
    it covers: an 'all' row first (every trade entered in its span), then its states,
    each Sharpe over the span's days and each share of P&L of the span's total. The
    regime column names it on the first row and says what it measures on the second;
    the first row also carries when its data starts and its value today; each state
    row says where that state sits on the measure (range) and the share of all days
    since its data starts spent in it (time, whether the signal was on or not). Today's
    state is marked; a state gets a row if it had time, trades or is today's."""
    from research.regimes import REGIMES, STATES, RegimeParams, regime_column

    params = RegimeParams()
    entries = (trades.with_columns(pl.col("entry_date").cast(pl.Date).alias("ts")).sort("ts")
               .join_asof(states.sort("ts"), on="ts", strategy="backward"))
    today = states.tail(1).to_dicts()[0]
    rows = []
    for name in REGIMES:
        col = regime_column(name)
        start = states.filter(pl.col(col).is_not_null())["ts"].min()
        if start is None:
            continue
        label, what, ranges = _regime_rule(name, params)
        in_span = entries.filter(pl.col(col).is_not_null())
        keep = days >= np.datetime64(start)
        span_days, span_daily = days[keep], daily[keep]
        total = float(in_span["pnl_bps"].sum())
        history = states[col].drop_nulls()
        group = [dict(regime=label, data_from=str(start), live=_regime_live(name, today, col), state="all", range="",
                      now="", time=1.0, **_performance(span_days, span_daily, in_span, total))]
        for state in STATES[name]:
            part = in_span.filter(pl.col(col) == state)
            time = float((history == state).mean())
            if not len(part) and not time and today[col] != state:
                continue
            group.append(dict(regime="", data_from="", live="", state=state,
                              range=ranges.get(state, ""), now="●" if today[col] == state else "", time=time,
                              **_performance(span_days, span_daily, part, total)))
        group[1]["regime"] = what  # always there: today's state has a row
        rows += group
    return pl.DataFrame(rows, infer_schema_length=None)


def _regime_section(equity: pl.DataFrame, trades: pl.DataFrame, states: pl.DataFrame | None) -> html.Div:
    """Exploratory: the trades by the macro regime they entered in (research.regimes)."""
    from research.regimes import RegimeParams

    if states is None:
        return _section("REGIMES", [note("Regime states unavailable: click Ref to load them.", "dim")])
    days, daily = _scored_days(equity, trades)
    rows = _by_regime(days, daily, trades, states)
    base = _perf_style([rows], trades)

    def style(col, value, row):
        first = row["state"] == "all"
        divider = {"borderTop": f"1.5px solid {DIM}"} if first else {}  # sets each regime's group apart
        if col == "regime":  # its name bold on the first row, what it measures dim beneath
            return {"textAlign": "left", **({"fontWeight": "bold"} if first else {"color": DIM}), **divider}
        if col in ("data_from", "live", "range"):
            return {"textAlign": "left", **({"color": DIM} if col == "range" else {}), **divider}
        return {**(base(col, value, row) or {}), **divider} or None

    return _section("REGIMES", [html.Div([
        table(_shown(rows), title="by macro regime at entry", float_fmt=",.2f",
              headers={**PERF_HEADERS, "data_from": "data from", "time": "share of time"}, cell_style=style),
        _subtitle(f"state on each trade's entry day (a new state counts after {RegimeParams().confirm} days in a row)"
                  f" · each regime scored over its own span · share of time: all days since data from"
                  f" · sharpe: annualized, 0 on days outside the group's trades · ● today (as of {states['ts'][-1]})")])],
        "the trades by the macro regime they entered in")


def _btn_style(primary: bool = False) -> dict:
    return {
        "padding": "6px 14px", "fontSize": 12, "cursor": "pointer",
        "border": f"1px solid {ORANGE if primary else BORDER}",
        "background": ORANGE if primary else "#FFFFFF",
        "color": "#FFFFFF" if primary else TEXT,
        "borderRadius": 3,
    }


def _dropdown_options(rows: list[dict]) -> list[dict]:
    """Deep-dive picker, grouped by family then ordered by target and input.

    dcc.Dropdown has no optgroups, so the family header is a disabled option --
    it renders as a heading and cannot be chosen. Labels lead with the target
    rather than the feature so the list reads in the order it is sorted; the
    same signal still carries its full name on the card itself.
    """
    options: list[dict] = []
    for family in sorted({r["family"] for r in rows}):
        group = sorted(
            (r for r in rows if r["family"] == family),
            key=lambda r: (r["target"], input_label(r)),
        )
        options.append({
            "label": f"── {family.upper()} ──",
            "value": f"__family__{family}",
            "disabled": True,
        })
        options += [
            {
                "label": f"{r['target']} · {input_label(r)} · "
                         f"{short_label(r)}",
                "value": _row_id(r),
            }
            for r in group
        ]
    return options


def _card(row: dict, phone: bool = False) -> html.Div:
    """Signal, Performance and Details views sharing one refreshed model read.

    Views stay mounted so window controls, paging and callback targets survive switches.
    """
    signal_id = _row_id(row)
    slug = _slug(signal_id)
    summary, chart_body, trade_data, months, extra = _card_sections(signal_id, phone=phone)
    return html.Div(
        className="oven-tight sd-card",  # the Oven's dense house tables (oven.css)
        **{"data-deep-tab": "signal"},
        children=[
            html.Div(
                className="sd-card-head",
                children=[
                    html.Span(_display_name(row), className="sd-card-name"),
                    html.Button("Ref", id=_pid("ref", signal_id), n_clicks=0,
                                className="ref-btn",
                                style=_btn_style(primary=True)),
                    dcc.Store(id=_pid("window", signal_id), data=DEFAULT_WINDOW),
                    dcc.Store(id=_pid("snap-range", signal_id), data=None),
                    dcc.Store(id=_pid("trades-page", signal_id), data=0),
                    dcc.Store(id=_pid("trades-data", signal_id), data=trade_data),
                ],
            ),
            html.Div([
                html.Button(label, type="button", className="sd-deep-tab",
                            **{"data-deep-view": view, "aria-pressed": "true" if view == "signal" else "false"})
                for view, label in (("signal", "Signal"), ("performance", "Performance"), ("details", "Details"))
            ], className="sd-deep-tabs", role="group", **{"aria-label": "Deep Dive view"}),
            html.Div(id=_pid("card-summary", signal_id), children=summary),
            html.Div(_heading("CHARTS"), className="sd-chart-tools"),
            html.Div(
                className="sd-windows sd-chart-tools",
                children=[
                    *[
                        html.Button(key, id=_pid(f"window-{key}", signal_id), n_clicks=0,
                                    style=_btn_style(primary=(key == DEFAULT_WINDOW)))
                        for key in WINDOW_PRESETS
                    ],
                    html.Button(
                        "Copy",
                        id=f"copy-charts-{slug}",
                        n_clicks=0,
                        className="ref-btn copy-charts-btn",
                        title="Copy all visible charts as one image",
                        style=_btn_style(),
                    ),
                    html.Span(
                        id=f"copy-status-{slug}",
                        style={"fontSize": 11, "color": DIM},
                    ),
                ],
            ),
            html.Div(
                id=f"card-analysis-{slug}",
                children=[
                    # the plain id is what research_app.js's Copy reads the images from (one
                    # to a row, as they stand here); the callbacks fill the pattern-id div inside it
                    html.Div(id=f"card-charts-{slug}", **{"data-copy-columns": "1"},
                             children=html.Div(id=_pid("card-charts", signal_id), children=chart_body)),
                    html.Div([_heading("TRADES", "full model history · newest first"),
                    _trade_table_section(
                        trade_data["closed"], 0, trade_data["open"], signal_id, trade_data["size"],
                    ) if trade_data is not None else html.Div()], className="sd-deep-view sd-deep-performance"),
                    html.Div(months, id=_pid("card-months", signal_id), style={"marginTop": 14},
                             className="sd-deep-view sd-deep-performance"),
                    html.Div(extra, id=_pid("card-extra", signal_id)),
                ],
            ),
        ],
    )


def _deep_card(row: dict, phone: bool = False) -> html.Div:
    """One signal's Deep Dive card in the wrapper the dropdown shows and hides."""
    return html.Div(id=_pid("deep-card", _row_id(row)), style={"display": "block"}, children=_card(row, phone))


def _registry_rows() -> list[dict]:
    """The promoted signals as they stand now. Read on every page load and every
    overview Ref -- never once at startup -- so a signal promoted while the server
    runs (the Oven's Send to dashboard, or registry --promote) shows on a refresh."""
    reg_df = REGISTRY.list()
    return [] if reg_df.is_empty() else reg_df.sort("target", "family", "name").to_dicts()


def build_app() -> dash.Dash:
    tab_style, selected_tab_style = FOLDER_TAB, FOLDER_TAB_SELECTED  # shared with the Oven

    def _overview_tab(rows):
        return html.Div(
            className="sd-page",
            children=[
                html.Div(
                    className="sd-bar",
                    children=[
                        html.Div("LIVE SIGNALS", className="sd-bar-title"),
                        html.Button(
                            "Ref",
                            id="refresh-overview",
                            n_clicks=0,
                            className="ref-btn",
                            style=_btn_style(primary=True),
                        ),
                        html.Button("Close all", type="button", className="sd-btn sd-close-all",
                                    title="Collapse all strategy details"),
                    ],
                ),
                dcc.Loading(
                    html.Div(id="overview-content", children=_overview_content(rows)),
                    custom_spinner=shimmer_loader(image="guy.png"),
                    # Keep the table visible beneath the spinner's blurred backdrop.
                    overlay_style={"visibility": "visible"},
                    parent_className="dashboard-overview-loader",
                ),
            ],
        )

    def _deep_dive_tab(rows, phone):
        selected_signal = _row_id(rows[0]) if rows else None
        return html.Div(
            className="sd-page",
            children=(
                [
                    html.Div(
                        className="sd-bar",
                        children=[
                            html.Div("SIGNAL DEEP DIVE", className="sd-bar-title"),
                            dcc.Dropdown(
                                id="deep-dive-signal",
                                options=_dropdown_options(rows),
                                value=selected_signal,
                                clearable=False,
                                className="sd-picker",
                            ),
                        ],
                    ),
                    # only the selected signal's card is built on load; the others are added
                    # the first time they are picked (_add_deep_card) and kept after that
                    dcc.Store(id="deep-dive-built", data=[selected_signal]),
                    html.Div(id="deep-dive-cards", children=[_deep_card(rows[0], phone)]),
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

    def _tabs(phone=False):
        """The real page. Costs one Engine.run() per promoted signal, so it is
        built in a callback rather than in the layout -- see _body. ``phone``
        draws the Deep Dive's charts for a narrow screen."""
        rows = _registry_rows()
        return dcc.Tabs(
            id="dashboard-tabs",
            className="dashboard-tight",
            style=FOLDER_TABS,  # the line the folder tabs sit on
            mobile_breakpoint=0,  # side by side on a phone too, never stacked
            value="live-overview",
            children=[
                dcc.Tab(
                    label="Signal",
                    value="live-overview",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=_overview_tab(rows),
                ),
                dcc.Tab(
                    label="Market", value="market", style=tab_style, selected_style=selected_tab_style,
                    children=html.Div(className="sd-page", children=[
                        html.Div(className="sd-bar", children=[
                            html.Div("TREASURY MARKET", className="sd-bar-title"),
                            html.Button("Ref", id="refresh-market", n_clicks=0, className="ref-btn",
                                        style=_btn_style(primary=True)),
                        ]),
                        html.Div(id="market-content", children=_market_content()),
                    ]),
                ),
                dcc.Tab(
                    label="Deep Dive",
                    value="signal-deep-dive",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=_deep_dive_tab(rows, phone),
                ),
                dcc.Tab(
                    label="Portfolio", value="portfolio", style=tab_style, selected_style=selected_tab_style,
                    # what the signals hold and how close each is to its exit, then the broker's account
                    children=html.Div(className="sd-page", children=[
                        html.Div(id="model-positions", children=_model_positions(rows)),
                        portfolio.layout(),
                    ]),
                ),
            ],
        )

    def _body():
        """A shell that renders instantly; _tabs() fills it from a callback.

        Building the tabs here instead would block the HTTP response for the
        couple of seconds they take, and the browser paints nothing at all
        until the response arrives -- so there is no window in which a loading
        indicator could be shown. Returning a shell first means the page is on
        screen immediately and the wait happens with the spinner visible.

        Content is rebuilt per page load (the boot callback runs each time,
        reading the registry afresh), and each card's components carry
        pattern-matching ids keyed by signal id, which is what lets the MATCH
        callbacks below serve cards for signals promoted after startup. Those
        callbacks reference ids that do not exist until this fills, which is
        legal only because make_app sets suppress_callback_exceptions.

        The boot tick first reports the browser's width ("phone" or "desktop")
        into the viewport store, and that is what triggers the fill: the Deep
        Dive's charts are images, so the server has to know which shape to draw.
        """
        return html.Div([
            dcc.Interval(id="page-boot", interval=60, max_intervals=1),
            dcc.Store(id="viewport"),
            dcc.Loading(
                html.Div(id="page-body"),
                custom_spinner=shimmer_loader(image="guy.png"),
                overlay_style={"visibility": "visible"},
                parent_className="dashboard-page-loader",
            ),
        ])

    app = make_app(title="Model", sliders=[], body=_body, class_name="sd-app", header_controls=[
        html.Button("Dark mode", id="sd-theme-toggle", type="button", className="sd-theme-toggle",
                    **{"aria-pressed": "false", "aria-label": "Toggle dark mode"}),
    ])
    portfolio.register_callbacks(app)

    app.clientside_callback(
        f"function(_n) {{ return window.innerWidth < {PHONE_MAX_WIDTH} ? 'phone' : 'desktop'; }}",
        Output("viewport", "data"),
        Input("page-boot", "n_intervals"),
        prevent_initial_call=True,
    )
    app.callback(
        Output("page-body", "children"),
        Input("viewport", "data"),
        prevent_initial_call=True,
    )(lambda viewport: _tabs(phone=viewport == "phone"))

    def _open_deep(clicks):
        """An overview row's Deep Dive button: switch tabs and pick that signal."""
        if not any(clicks):  # the rows were (re)drawn, nothing was clicked
            return dash.no_update, dash.no_update
        return "signal-deep-dive", ctx.triggered_id["sid"]

    app.callback(
        Output("dashboard-tabs", "value"),
        Output("deep-dive-signal", "value"),
        Input(_pid("open-deep", ALL), "n_clicks"),
        prevent_initial_call=True,
    )(_open_deep)

    window_keys = list(WINDOW_PRESETS)

    # Every callback below is pattern-matched on the signal id rather than registered
    # per promoted signal at startup, so cards built for signals promoted later work too.
    app.callback(
        Output(_pid("deep-card", ALL), "style"),
        Input("deep-dive-signal", "value"),
    )(
        lambda selected: [
            {"display": "block" if out["id"]["sid"] == selected else "none"}
            for out in ctx.outputs_list
        ]
    )

    def _add_deep_card(selected, built, viewport):
        """Build a signal's card the first time it is picked, appended beside the ones
        already built (they keep their state); picking one already built changes nothing
        here, the callback above just shows it."""
        built = built or []
        row = REGISTRY.get(selected) if selected else None
        if row is None or selected in built:
            return dash.no_update, dash.no_update
        cards = Patch()
        cards.append(_deep_card(row, viewport == "phone"))
        return cards, [*built, selected]

    app.callback(
        Output("deep-dive-cards", "children"),
        Output("deep-dive-built", "data"),
        Input("deep-dive-signal", "value"),
        State("deep-dive-built", "data"),
        State("viewport", "data"),
        prevent_initial_call=True,
    )(_add_deep_card)

    def _refresh_overview(*_clicks):
        rows = _registry_rows()
        _refresh_all_data(rows)
        try:
            markets.pull()
        except Exception:  # db down: the market keeps its last pull, as the signals do
            pass
        # the Market and Portfolio tabs read the same pull, so one Ref brings all three up to date
        return _overview_content(rows), _market_content(), _model_positions(rows)

    app.callback(
        Output("overview-content", "children"),
        Output("market-content", "children"),
        Output("model-positions", "children"),
        Input("refresh-overview", "n_clicks"),
        prevent_initial_call=True,
    )(_refresh_overview)

    def _refresh_market(_clicks):
        """The Market tab's own Ref: just the yields and curves, which is quick."""
        try:
            markets.pull()
        except Exception:
            pass
        return _market_content()

    app.callback(
        Output("market-content", "children", allow_duplicate=True),
        Input("refresh-market", "n_clicks"),
        prevent_initial_call=True,
    )(_refresh_market)

    def _update(_ref, *rest):
        *_window_clicks, _snap, window, snap_range, page, viewport = rest
        trigger = ctx.triggered_id or {}
        kind = trigger.get("type", "")
        signal_id = ctx.outputs_list[0]["id"]["sid"]
        no_btn_styles = [dash.no_update] * len(window_keys)
        try:
            if kind == "ref":
                # one click = fresh data, then one logged analysis on it
                runner.pull_data(signal_id)
                runner.run_analysis(signal_id)
                page = 0
            elif kind.startswith("window-"):
                window = kind[len("window-"):]
                snap_range = None
                page = 0
            read = runner.analysis(signal_id)  # recomputed only after a Ref rewrote the data
            state, trades, open_entry = read["state"], read["trades"], read["open_entry"]
        except RuntimeError as exc:
            err = html.Div(str(exc), style={"color": ORANGE, "padding": "12px 0"})
            return (
                html.Div(), err, None,
                dash.no_update, dash.no_update, 0,
                window, snap_range, html.Div(), html.Div(), *no_btn_styles,
            )

        if kind == "snap":
            rng = _visible_trade_range(trades, open_entry, page, state["data_asof"],
                                       _trade_page_size(state["equity_curve"]))
            if rng is not None:
                snap_range = [str(rng[0]), str(rng[1])]

        date_range = None
        if snap_range:
            date_range = (pd.Timestamp(snap_range[0]), pd.Timestamp(snap_range[1]))

        summary, chart_body, trade_data, months, extra = _card_sections(
            signal_id,
            state=state,
            trades=trades,
            open_entry=open_entry,
            window=window,
            page=page,
            date_range=date_range,
            read_live=read["live"],
            phone=viewport == "phone",
        )
        btn_styles = [_btn_style(primary=(k == window)) for k in window_keys]
        if kind == "snap":
            # Snap changes the chart range only. Leaving the trade DOM and
            # page store untouched keeps the viewer on the same rows.
            return (
                summary, chart_body,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                window, snap_range, dash.no_update, dash.no_update, *btn_styles,
            )
        return (
            summary, chart_body, trade_data,
            _trade_table_block(trade_data["closed"], page, trade_data["open"], trade_data["size"]),
            _trade_page_label(len(trade_data["closed"]) + (trade_data["open"] is not None), page,
                              trade_data["size"]), page,
            window, snap_range, months, extra, *btn_styles,
        )

    app.callback(
        Output(_pid("card-summary", MATCH), "children"),
        Output(_pid("card-charts", MATCH), "children"),
        Output(_pid("trades-data", MATCH), "data"),
        Output(_pid("trade-table", MATCH), "children", allow_duplicate=True),
        Output(_pid("trades-status", MATCH), "children", allow_duplicate=True),
        Output(_pid("trades-page", MATCH), "data"),
        Output(_pid("window", MATCH), "data"),
        Output(_pid("snap-range", MATCH), "data"),
        Output(_pid("card-months", MATCH), "children"),
        Output(_pid("card-extra", MATCH), "children"),
        *[Output(_pid(f"window-{k}", MATCH), "style") for k in window_keys],
        Input(_pid("ref", MATCH), "n_clicks"),
        *[Input(_pid(f"window-{k}", MATCH), "n_clicks") for k in window_keys],
        Input(_pid("snap", MATCH), "n_clicks"),
        State(_pid("window", MATCH), "data"),
        State(_pid("snap-range", MATCH), "data"),
        State(_pid("trades-page", MATCH), "data"),
        State("viewport", "data"),
        prevent_initial_call=True,
    )(_update)

    def _page_trades(_first, _prev, _next, _last, trade_data, page):
        """Swap rows only -- chart and backtest work stays off this path."""
        trade_data = trade_data or {"closed": [], "open": None}
        closed_rows = trade_data.get("closed", [])
        open_entry = trade_data.get("open")
        size = trade_data.get("size", TRADES_PER_PAGE)
        n = len(closed_rows) + (open_entry is not None)  # the open position pages in first
        last_page = max(0, (n - 1) // size * size) if n else 0
        page = page or 0
        kind = (ctx.triggered_id or {}).get("type")
        if kind == "trades-first":
            page = 0
        elif kind == "trades-prev":
            page = max(0, page - size)
        elif kind == "trades-next":
            page = min(page + size, last_page)
        elif kind == "trades-last":
            page = last_page
        page = min(page, last_page)
        return (
            _trade_table_block(closed_rows, page, open_entry, size),
            _trade_page_label(n, page, size),
            page,
        )

    app.callback(
        Output(_pid("trade-table", MATCH), "children"),
        Output(_pid("trades-status", MATCH), "children"),
        Output(_pid("trades-page", MATCH), "data", allow_duplicate=True),
        Input(_pid("trades-first", MATCH), "n_clicks"),
        Input(_pid("trades-prev", MATCH), "n_clicks"),
        Input(_pid("trades-next", MATCH), "n_clicks"),
        Input(_pid("trades-last", MATCH), "n_clicks"),
        State(_pid("trades-data", MATCH), "data"),
        State(_pid("trades-page", MATCH), "data"),
        prevent_initial_call=True,
    )(_page_trades)

    return app


app = build_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug tools and in-browser tracebacks. Use --reload "
             "as well to restart the server when Python code changes.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Dev mode: automatically restart on source changes and refresh "
             "the browser when Python, CSS, or JavaScript changes.",
    )
    args = parser.parse_args()
    # This server outlives the WSL VM's idle timeout, and postgres dies with
    # the VM. Hold it up for as long as the dashboard runs; released at exit.
    utils.helpers.hold_wsl()
    app._ra_debug = args.debug
    run(app, port=args.port, host=args.host, reload=args.reload)
