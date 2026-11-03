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
import re

import dash
import numpy as np
import pandas as pd
import polars as pl
from dash import ALL, MATCH, Input, Output, Patch, State, ctx, dcc, html

import utils.helpers
from dashboard import runner
from dashboard.charts import (
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
    exit_label, gate_short as _gate_rule, input_label, short_label,
    signal_label as _display_name,
)
from dashboard.registry import LiveRegistry
from stats import CONF_SHRINK
from research.app import _diverging, note, pnl_heatmap, regression_scatter, table
from utils.viz import column_widths, table_div
from utils.research_app import (
    BORDER, C0, C1, DIM, FOLDER_TAB, FOLDER_TAB_SELECTED, FOLDER_TABS, ORANGE, PANEL, TEXT, make_app, run,
    shimmer_loader, stat_block,
)

REGISTRY = LiveRegistry()
LEDGER = SignalLedger()

TRADES_PER_PAGE = 6  # fewest trade rows a page holds; it grows to the monthly heatmap's years
CHART_BOX = "1260 / 767"  # width / height of the Deep Dive's grid charts: the tallest one's shape
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

OVERVIEW_COLUMNS = [
    # Feature is dropped: the Signal name already leads with it. Position folds
    # in the old "Latest Reading", and Reading pairs the live value with the
    # threshold it is being judged against, so the four state columns become two.
    # Headers and cells are kept short so the table fits the page without a
    # scrollbar; units (bps, %) are the dashboard's defaults and go unstated.
    # What each signal is and where it stands today; how the strategy has performed
    # lives on the Deep Dive.
    ("name", "Signal"),
    ("model", "Built From"),
    ("input_now", "Input"),
    ("fair", "Fair"),
    ("reading", "Reading"),
    ("position", "Position"),
    ("conf", "Conf"),
    ("gate_rule", "Gate"),
    ("gate_status", "Gate Now"),
    ("data_asof", "As Of"),
]


def _slug(signal_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", signal_id.lower()).strip("-")


def _pid(kind: str, signal_id) -> dict:
    """A card component's pattern-matching id. One set of MATCH callbacks then serves
    every card, so a signal promoted after the server started works on a browser refresh."""
    return {"type": kind, "sid": signal_id}


def _row_id(row: dict) -> str:
    """Registry identity, with module fallback for legacy/test rows."""
    return row.get("signal_id") or row["module"]


def _exit_summary(row: dict) -> str:
    return f"exit={exit_label(row)}"


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


def _live_pnl_bps(trades, open_entry: dict | None) -> float:
    """Realized PnL from closed trades plus unrealized PnL from the open
    position, if any -- the running live total since promotion."""
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


CHANGE_WINDOWS = [("1D", 1), ("1W", 7), ("1M", 30)]


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


def _target_snapshot(target: str, rows: list[dict]) -> dict | None:
    """Where the traded curve itself is, and how far it has come: latest level
    plus point changes over standard windows. Read from the freshest cached
    frame among the target's promoted signals -- the same data the signals
    below are computed on, so the header can never disagree with the rows."""
    sources = []
    for row in rows:
        frame = runner.cached_data(_row_id(row))
        if frame is None or target not in frame.columns:
            continue
        frame = frame.select("ts", pl.col(target).alias("level")).drop_nulls()
        if not frame.is_empty():
            sources.append((_row_id(row), frame))
    if not sources:
        return None

    # keep the winning frame's signal_id: its tickers are what dates the level
    signal_id, frame = max(sources, key=lambda pair: pair[1]["ts"][-1])
    ts, level = frame["ts"][-1], float(frame["level"][-1])

    def _change_since(cutoff) -> float | None:
        earlier = frame.filter(pl.col("ts") <= cutoff)
        return None if earlier.is_empty() else level - float(earlier["level"][-1])

    changes = [
        (label, _change_since(ts - dt.timedelta(days=days)))
        for label, days in CHANGE_WINDOWS
    ]
    changes.append(("YTD", _change_since(dt.date(ts.year, 1, 1) - dt.timedelta(days=1))))
    return {"signal_id": signal_id, "ts": ts, "level": level, "changes": changes}


def _delta_chip(label: str, value: float | None) -> html.Div:
    color = TEXT if value is None else (C1 if value > 0 else C0 if value < 0 else TEXT)
    return html.Div(
        style={"display": "flex", "alignItems": "baseline", "gap": 4},
        children=[
            html.Span(label, style={"fontSize": 10, "color": DIM}),
            html.Span(
                _fnum(value, "+.1f"),
                style={"fontSize": 12, "fontFamily": "monospace", "color": color},
            ),
        ],
    )


def _target_header(target: str, rows: list[dict]) -> html.Div:
    snapshot = _target_snapshot(target, rows)
    children = [
        html.Span(
            target,
            style={
                "fontSize": 15,
                "fontWeight": "bold",
                "color": ORANGE,
                "textTransform": "uppercase",
            },
        )
    ]
    if snapshot is not None:
        bar, written, stale = _db_asof(snapshot["signal_id"], snapshot["ts"])
        asof = f"{bar} · {written}"
        children += [
            html.Span(
                f"{snapshot['level']:.1f} bps",
                style={"fontSize": 15, "fontWeight": "bold",
                       "fontFamily": "monospace"},
            ),
            html.Span(asof, style={"fontSize": 11,
                                   "color": ORANGE if stale else DIM}),
            *[_delta_chip(label, value) for label, value in snapshot["changes"]],
        ]
    return html.Div(
        style={"display": "flex", "alignItems": "baseline", "gap": 14,
               "marginBottom": 7},
        children=children,
    )


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
        "reading": "—",
        "gate_rule": "—",
        "gate_status": "—",
        "live_pnl_bps": None,
        "data_asof": "—",
        "data_stale": 0,
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


def _signal_facts(row: dict, state: dict, live: dict | None) -> dict:
    """An overview row's what-it-is and where-it-stands cells: model, the input's value
    now, the target's fair value and today's confidence (``live``, runner.live_read's).
    '—' where the signal's state lacks them -- a decoration on the row must never break it."""
    try:
        feature = state["strategy"].feature
        return {
            "model": _model_text(row, state),
            "input_now": f"{float(state['data'][feature][-1]):.1f}",  # the input is named in Built From
            "fair": f"{live['fair']:.1f}" if live["fair"] == live["fair"] else "—",
            "conf": live["confidence"]["label"],
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


def _overview_cell_style(column: str, value, row: dict) -> dict | None:
    """Semantic colour for one overview cell.

    Replaces the DataTable's style_data_conditional rules one for one. A static
    table has no filter_query equivalent, and on a trading surface direction,
    gate state and P&L sign have to read at a glance rather than be parsed.
    """
    if column == "position":
        for token, colour in (("LONG", C1), ("SHORT", C0), ("GATED", ORANGE)):
            if token in str(value):
                return {"color": colour, "fontWeight": "bold"}
    elif column == "conf" and value in CONF_COLOURS:
        return {"color": CONF_COLOURS[value], "fontWeight": "bold"}
    elif column == "gate_status":
        if "CLOSED" in str(value):
            return {"color": C0}
        if "OPEN" in str(value):
            return {"color": C1}
    elif column == "data_asof" and row.get("data_stale"):
        # the db holds a newer bar, or has rewritten this one, since the pull
        return {"color": ORANGE, "fontWeight": "bold"}
    return None


OVERVIEW_KEYS = [key for key, _label in OVERVIEW_COLUMNS]
OVERVIEW_FLOAT_FMT = ",.4g"  # ",.3f" would print sharpe 0.7 as 0.700


def _overview_table(
    target: str, rows: list[dict], snapshots: list[dict], widths: dict
) -> html.Div:
    """One target's section. Snapshots and widths are passed in rather than
    derived here: the widths have to be measured across every section for the
    columns to line up, and a snapshot costs a full Engine.run()."""
    return html.Div(
        style={"marginBottom": 22},
        children=[
            _target_header(target, rows),
            table_div(
                pd.DataFrame(snapshots),
                columns=OVERVIEW_KEYS,
                headers=dict(OVERVIEW_COLUMNS),
                max_rows=len(snapshots),
                cell_style=_overview_cell_style,
                float_fmt=OVERVIEW_FLOAT_FMT,
                col_widths=widths,
            ),
        ],
    )


def _refresh_all_data(rows: list[dict]) -> None:
    """Fresh Strategy.load_data() for every promoted module, deduped -- named
    variants share one cache file so each module is only pulled once."""
    for module in {row["module"] for row in rows}:
        runner.pull_data(module)


def _overview_content(rows: list[dict]) -> list:
    if not rows:
        return [
            html.Div(
                "No signals promoted yet.",
                style={"color": DIM, "padding": "24px 0"},
            )
        ]
    targets = sorted({row["target"] for row in rows})
    by_target = {
        target: sorted(
            [row for row in rows if row["target"] == target], key=_display_name
        )
        for target in targets
    }
    runner.prime_freshness([_row_id(row) for row in rows])  # every signal's db stamp in one query
    # snapshot everything first: the column widths have to be measured over
    # every section at once, or each table sizes itself and the page reads as
    # a stack of unrelated grids
    snapshots = {
        target: [_overview_snapshot(row) for row in group]
        for target, group in by_target.items()
    }
    widths = column_widths(
        [snap for group in snapshots.values() for snap in group],
        columns=OVERVIEW_KEYS,
        headers=dict(OVERVIEW_COLUMNS),
        float_fmt=OVERVIEW_FLOAT_FMT,
        pad=1,  # the cells carry their own padding (see .dashboard-tight)
    )
    return [
        _overview_table(target, by_target[target], snapshots[target], widths)
        for target in targets
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
    return table(frame, title="trades",
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


def _card_sections(
    signal_id: str,
    state: dict | None = None,
    trades=None,
    open_entry: dict | None = None,
    window: str = DEFAULT_WINDOW,
    page: int = 0,
    date_range: tuple | None = None,
    read_live: dict | None = None,
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
        [html.Span(f"{label} ", style={"color": DIM}), value],
        style={"color": ORANGE if alert else TEXT})
    meta = [
        small("data", f"{asof_bar} · {asof_written}", asof_stale),
        small("last run", _clock(ledger_row["run_ts"], always_date=True) if ledger_row else "never"),
        *([small("gate", _gate_status(state))] if state and state["params"].get("gate") is not None else []),
        small("params", _param_summary(row)),
    ]
    # how the strategy has done: live since promotion, and its backtest on today's history
    stats = _backtest_stats(state["equity_curve"], trades) if state and not error else None
    performance = _stats_rows(state if not error else None, stats, row, live_pnl)

    if error:
        now_panel = html.Div()
        chart_body = html.Div(error, style={"color": ORANGE, "padding": "12px 0"})
        months = extra = html.Div()
    else:
        months = _monthly_pnl(state["equity_curve"])
        closed = trades is not None and not trades.is_empty()
        extra = html.Div([_scatters(state),
                          *([_trade_mechanics(state, trades)] if closed else []),
                          *([_confidence_section(state["equity_curve"], trades, live["confidence"])]
                            if closed and live is not None else []),
                          *([_regime_section(state["equity_curve"], trades, runner.regime_states())]
                            if closed else [])])
        now_panel = _now_panel(state, live, open_entry)
        window_bars = WINDOW_PRESETS.get(window, WINDOW_PRESETS[DEFAULT_WINDOW])
        span = dict(window_bars=window_bars, date_range=date_range)
        target = state["strategy"].target
        # chart what the signal is built on: the hedged portfolio for a portfolio-basis
        # signal, else the outright target -- and beside it, always the input
        traded = "portfolio" if getattr(state["strategy"], "setup", {}).get("basis") == "portfolio" else target
        data = state["data"].with_columns(pl.Series("fair", _fair_series(state, traded)))
        wide = signal_chart(
            state["data"], state["signal_frame"],
            state["params"]["entry_signal"], state["params"]["entry_threshold"],
            fired=state["fired"], wide=True, **span,
        )
        marks, held_mark = _on_series(trades, open_entry, state["data"], traded) if traded != target             else (trades, open_entry)
        level_png = level_chart(data, traded, trades=marks, open_entry=held_mark, overlays=["fair"], **span)
        second = input_chart(state["data"], state["strategy"].feature, **span)
        gate_png = gate_chart(
            state["data"], state["signal_frame"], state["params"].get("gate"),
            gate_window=state["params"].get("gate_window"), **span,
        )
        pnl_png = pnl_chart(state["equity_curve"], **span)
        pngs = [level_png, second, *([gate_png] if gate_png is not None else []), pnl_png,
                return_distribution_chart(trades)]
        img = lambda png, **extra: html.Img(  # noqa: E731
            src=f"data:image/png;base64,{png}", style={"width": "100%", "border": f"1px solid {BORDER}", **extra})
        # every chart in the grid in the same box: they are drawn at slightly different
        # shapes, so each is fitted (never stretched) into the tallest one's
        same_box = dict(aspectRatio=CHART_BOX, objectFit="contain", background="#FFFFFF", display="block")
        chart_body = html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": 10},
            children=[html.Div(img(wide), style={"gridColumn": "1 / span 2"}),
                      *[img(png, **same_box) for png in pngs if png is not None]],
        )

    summary = html.Div([
        now_panel,
        *performance,
        html.Div(meta, style={"display": "flex", "gap": 18, "flexWrap": "wrap", "fontSize": 11,
                              "padding": "4px 2px 10px"}),
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


def _backtest_stats(equity: pl.DataFrame, trades: pl.DataFrame | None) -> dict | None:
    """The Oven's strategy stats from this signal's own daily P&L and closed trades on
    the cached history, scored from its first entry (the warm-up before says nothing);
    the halves split those days down the middle."""
    if trades is None or trades.is_empty() or equity.is_empty():
        return None
    scored = equity.with_columns(pl.col("ts").cast(pl.Date)).filter(pl.col("ts") >= trades["entry_date"].min())
    daily = scored["pnl_bps"].to_numpy().astype(float)

    def sharpe(d):
        return d.mean() / d.std() * np.sqrt(252) if len(d) > 1 and d.std() > 0 else np.nan

    half = len(daily) // 2
    cum = np.cumsum(daily)
    total = float(cum[-1])
    max_dd = float((cum - np.maximum.accumulate(cum)).min())
    years = len(daily) / 252
    pnl = trades["pnl_bps"]
    losses = float(pnl.filter(pnl <= 0).sum())
    return dict(
        sharpe=sharpe(daily), sharpe_h1=sharpe(daily[:half]), sharpe_h2=sharpe(daily[half:]),
        split=scored["ts"][half], total_pnl=total, pnl_per_year=total / years, n_trades=len(pnl),
        trades_per_year=len(pnl) / years, hit_rate=float((pnl > 0).mean()), avg_trade=float(pnl.mean()),
        median_trade=float(pnl.median()),
        profit_factor=float(pnl.filter(pnl > 0).sum()) / abs(losses) if losses else np.nan,
        max_drawdown=max_dd, calmar=total / abs(max_dd) if max_dd else np.nan,
        time_in_market=float((daily != 0).mean()),
        best_trade_share=float(pnl.max()) / total if total > 0 else np.nan,
    )


def _stats_rows(state: dict, stats: dict | None, row: dict, live_pnl) -> list:
    """The Oven card's two stat rows: the model as it stands, then the backtest."""
    stats_row = {"display": "flex", "gap": 28, "flexWrap": "wrap", "padding": "8px 2px 8px"}
    last = state["last"] if state else {}
    model = [
        stat_block("live pnl", _fnum(live_pnl, "+.1f", " bps"), alert=bool(live_pnl and live_pnl < 0)),
        stat_block("β now", _fnum(last.get("beta"), "+.3f")),
        stat_block("r² now", _fnum(last.get("r2"), ".2f")),
        stat_block("sharpe at promotion", _fnum(row.get("sharpe"), ".2f")),
        *([stat_block("halves split", str(stats["split"]))] if stats else []),
    ]
    rows = [html.Div(model, style=stats_row)]
    if stats:
        s = stats
        rows.append(html.Div(style={**stats_row, "borderTop": f"1px solid {BORDER}", "marginTop": 4}, children=[
            stat_block("sharpe", _fnum(s["sharpe"], ".2f")),
            stat_block("halves", f"{_fnum(s['sharpe_h1'], '.2f')} / {_fnum(s['sharpe_h2'], '.2f')}"),
            stat_block("total pnl", f"{s['total_pnl']:+,.0f} bps"),
            stat_block("per year", f"{s['pnl_per_year']:+,.1f} bps"),
            stat_block("n trades", f"{s['n_trades']} · {s['trades_per_year']:.1f}/yr"),
            stat_block("hit rate", f"{s['hit_rate']:.0%}"),
            stat_block("avg / median", f"{s['avg_trade']:+.2f} / {s['median_trade']:+.2f}"),
            stat_block("profit factor", _fnum(s["profit_factor"], ".2f")),
            stat_block("max drawdown", f"{s['max_drawdown']:+,.1f} bps"),
            stat_block("calmar", _fnum(s["calmar"], ".2f")),
            stat_block("time in market", f"{s['time_in_market']:.0%}"),
            stat_block("best trade share", _fnum(s["best_trade_share"], ".0%")),
        ]))
    return rows


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
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr)", "gap": 12,
               "alignItems": "start", "marginTop": 10},
        children=[regression_scatter(state["data"], target, feature, basis, lookback, "previous")
                  for basis in ("levels", "changes")],
    )


def _exit_reasons(trades: pl.DataFrame) -> pl.DataFrame:
    """Per exit reason, plus 'all': trades, avg and total P&L, its share of the total, hit
    rate and bars held."""
    total = trades["pnl_bps"].sum()
    groups = [("all", trades)] + [(r, trades.filter(pl.col("exit_reason") == r))
                                  for r in trades["exit_reason"].value_counts(sort=True)["exit_reason"]]
    return pl.DataFrame([dict(reason=r, trades=len(t), avg_trade=t["pnl_bps"].mean(), total_pnl=t["pnl_bps"].sum(),
                              share_of_pnl=t["pnl_bps"].sum() / total if total else None,
                              hit_rate=(t["pnl_bps"] > 0).mean(), avg_held=t["bars_held"].mean())
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
    return html.Div(style={"marginTop": 18}, children=[
        html.Div("TRADE MECHANICS", style={"fontSize": 12, "fontWeight": "bold", "marginBottom": 6}),
        note(" · ".join(lines), "dim"),
        table(_exit_reasons(trades).to_pandas(), title="exit reasons",
              float_fmt=",.2f",
              headers={"avg_trade": "avg trade", "total_pnl": "total pnl", "hit_rate": "hit",
                       "avg_held": "avg held", "share_of_pnl": "share of pnl"},
              cell_style=lambda col, value, row: (
                  {"textAlign": "left", **({"fontWeight": "bold"} if value == "all" else {})}
                  if col == "reason" else None)),
    ])


CONF_COLOURS = {"HIGH": C1, "LOW": C0, "MED": TEXT, "—": DIM}


def _now_panel(state: dict, live: dict, open_entry: dict | None) -> html.Div:
    """Today, at a glance: what the book holds or the signal calls for, the reading
    against its threshold and fair value, what an entry now is expected to make,
    walk-forward confidence and why, and the open position's progress."""
    params = state["params"]
    units = "bps" if params["entry_signal"] == "residual" else "z"
    reading = live["resid"] if units == "bps" else live["z"]
    position = _position_label(state, open_entry)
    conf = live["confidence"]
    bold = {"fontWeight": "bold"}
    dim = {"color": DIM}

    def line(*parts):
        return html.Div(parts, style={"display": "flex", "gap": 18, "flexWrap": "wrap", "alignItems": "baseline"})

    expected = (
        [html.Span("if entered now", style=dim),
         html.Span(f"{live['exp_return']:+.1f} bp", style=bold),
         *([html.Span(f"over ~{live['exp_bars']:.0f}d ({live['bps_per_day']:+.2f} bp/day)")]
           if live["exp_bars"] == live["exp_bars"] else [html.Span("to its OU mean")]),
         *([html.Span(f"risk {live['exp_risk']:.1f} bp", style=dim)] if live["exp_risk"] == live["exp_risk"] else [])]
        if live["exp_return"] == live["exp_return"] else
        [html.Span("if entered now", style=dim), html.Span("— (no half-life yet)", style=dim)])
    if open_entry is not None:
        held = open_entry
        progress = f"{held['bars_held']}d" + (f" of ~{held['expected_bars']:.0f}" if held.get("expected_bars") else "")
        position_line = [html.Span("open", style=dim),
                         html.Span(f"{held['direction']} since {held['entry_date']}", style=bold),
                         html.Span(f"{held['pnl_bps']:+.1f} bp", style={"color": C1 if held["pnl_bps"] >= 0 else C0,
                                                                         **bold}),
                         html.Span(progress),
                         *([html.Span(f"expected {held['expected_return_bps']:+.1f} bp at entry", style=dim)]
                           if held.get("expected_return_bps") is not None else [])]
    else:
        position_line = [html.Span("open", style=dim), html.Span("no position")]
    return html.Div(
        style={"border": f"1px solid {BORDER}", "background": "#FFFFFF", "padding": "10px 14px",
               "margin": "6px 0 4px", "fontFamily": "monospace", "fontSize": 13,
               "display": "flex", "flexDirection": "column", "gap": 6},
        children=[
            line(html.Span(position, style={"fontSize": 15, **bold, "color": C1 if "LONG" in position
                                            else C0 if "SHORT" in position else TEXT}),
                 html.Span(f"{reading:+.2f}{units} vs ±{float(params['entry_threshold']):g}{units}", style=bold),
                 html.Span(f"resid {live['resid']:+.1f} bp · z {live['z']:+.2f}", style=dim),
                 html.Span(f"level {live['level']:.1f} · fair {live['fair']:.1f}"),
                 html.Span(f"half-life {live['half_life']:.1f}d", style=dim)),
            line(*expected),
            line(html.Span("confidence", style=dim),
                 html.Span(conf["label"], style={**bold, "color": CONF_COLOURS.get(conf["label"], TEXT)}),
                 html.Span(conf["drivers"] or ("needs more closed trades" if conf["label"] == "—" else "")),
                 html.Span(f"from {conf['n_past']} closed trades", style=dim)),
            line(*position_line),
        ],
    )


CONF_LABELS = {**runner.OVEN_CONFIDENCE, **runner.BOOK_CONFIDENCE}  # feature column -> name


def _scored_days(equity: pl.DataFrame, trades: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(dates, daily P&L) from the first entry on: the span every Sharpe here is scored over."""
    first = trades["entry_date"].cast(pl.Date).min()
    daily = equity.select(pl.col("ts").cast(pl.Date), "pnl_bps").filter(pl.col("ts") >= first)
    return daily["ts"].to_numpy(), daily["pnl_bps"].to_numpy().astype(float)


def _performance(days: np.ndarray, daily: np.ndarray, subset: pl.DataFrame, total: float) -> dict:
    """Trades, total and average P&L, share of all P&L, hit rate, and the Sharpe of
    ``subset``'s own trades: annualized, from the signal's daily P&L on only the days
    those trades were on. Days they were not on are left out rather than counted as
    flat, so a group with fewer trades is not marked down for being smaller, and
    groups of any size compare directly (with all trades: their Sharpe in the market)."""
    held = np.zeros(len(days), dtype=bool)
    for a, b in subset.select(pl.col("entry_date").cast(pl.Date), pl.col("exit_date").cast(pl.Date)).iter_rows():
        held |= (days >= np.datetime64(a)) & (days <= np.datetime64(b))
    marks = daily[held]
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


def _section(title: str, blocks: list) -> html.Div:
    return html.Div(style={"marginTop": 18}, children=[
        html.Div(title, style={"fontSize": 12, "fontWeight": "bold", "marginBottom": 6}),
        html.Div(blocks, style={"display": "flex", "flexWrap": "wrap", "gap": "4px 16px", "alignItems": "start"}),
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
    return _section("CONFIDENCE", blocks)


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
    row says where that state sits on the measure (range). Today's state is marked
    (a row even if no trade entered in it)."""
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
        group = [dict(regime=label, data_from=str(start), live=_regime_live(name, today, col), state="all", range="",
                      now="", **_performance(span_days, span_daily, in_span, total))]
        for state in STATES[name]:
            part = in_span.filter(pl.col(col) == state)
            if not len(part) and today[col] != state:
                continue
            group.append(dict(regime="", data_from="", live="", state=state,
                              range=ranges.get(state, ""), now="●" if today[col] == state else "",
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
              headers={**PERF_HEADERS, "data_from": "data from"}, cell_style=style),
        _subtitle(f"state on each trade's entry day (a new state counts after {RegimeParams().confirm} days in a row)"
                  f" · each regime scored over its own span · ● today (as of {states['ts'][-1]})")])])


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


def _card(row: dict) -> html.Div:
    module = row["module"]
    signal_id = _row_id(row)
    slug = _slug(signal_id)
    summary, chart_body, trade_data, months, extra = _card_sections(signal_id)
    return html.Div(
        className="oven-tight",  # the Oven's dense house tables (oven.css)
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
                    html.Button("Ref", id=_pid("ref", signal_id), n_clicks=0,
                                className="ref-btn",
                                style=_btn_style(primary=True)),
                    dcc.Store(id=_pid("window", signal_id), data=DEFAULT_WINDOW),
                    dcc.Store(id=_pid("snap-range", signal_id), data=None),
                    dcc.Store(id=_pid("trades-page", signal_id), data=0),
                    dcc.Store(id=_pid("trades-data", signal_id), data=trade_data),
                ],
            ),
            html.Div(id=_pid("card-summary", signal_id), children=summary),
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
                        style={**_btn_style(), "marginLeft": "auto"},
                    ),
                    html.Span(
                        id=f"copy-status-{slug}",
                        style={"fontSize": 11, "color": DIM, "minWidth": 42},
                    ),
                ],
            ),
            html.Div(
                id=f"card-analysis-{slug}",
                children=[
                    # the plain id is what research_app.js's Copy reads the images from;
                    # the callbacks fill the pattern-id div inside it
                    html.Div(id=f"card-charts-{slug}",
                             children=html.Div(id=_pid("card-charts", signal_id), children=chart_body)),
                    # the trades beside the monthly P&L, as on the Oven's setup card
                    html.Div(
                        className="oven-trades-months",
                        children=[
                            html.Div(_trade_table_section(
                                trade_data["closed"], 0, trade_data["open"], signal_id, trade_data["size"],
                            ) if trade_data is not None else html.Div(), style={"minWidth": 0}),
                            html.Div(months, id=_pid("card-months", signal_id)),
                        ],
                    ),
                    html.Div(extra, id=_pid("card-extra", signal_id)),
                ],
            ),
        ],
    )


def _deep_card(row: dict) -> html.Div:
    """One signal's Deep Dive card in the wrapper the dropdown shows and hides."""
    return html.Div(id=_pid("deep-card", _row_id(row)), style={"display": "block"}, children=_card(row))


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
            style={"padding": "18px 24px"},
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": 12,
                        "marginBottom": 14,
                    },
                    children=[
                        html.Div(
                            "LIVE TRADING OVERVIEW",
                            style={"fontSize": 14, "fontWeight": "bold"},
                        ),
                        html.Button(
                            "Ref",
                            id="refresh-overview",
                            n_clicks=0,
                            className="ref-btn",
                            style=_btn_style(primary=True),
                        ),
                    ],
                ),
                dcc.Loading(
                    html.Div(id="overview-content", children=_overview_content(rows)),
                    custom_spinner=shimmer_loader(image="guy.png", caption="loading"),
                    # keep the stale table visible and dimmed underneath rather
                    # than blanking it: a Ref is a refresh, not a page change
                    overlay_style={"visibility": "visible", "opacity": 0.35},
                    parent_className="dashboard-overview-loader",
                ),
            ],
        )

    def _deep_dive_tab(rows):
        selected_signal = _row_id(rows[0]) if rows else None
        return html.Div(
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
                                "SIGNAL DEEP DIVE",
                                style={"fontSize": 14, "fontWeight": "bold"},
                            ),
                            dcc.Dropdown(
                                id="deep-dive-signal",
                                options=_dropdown_options(rows),
                                value=selected_signal,
                                clearable=False,
                                style={"width": 360, "marginLeft": "auto", "fontSize": 12},
                            ),
                        ],
                    ),
                    # only the selected signal's card is built on load; the others are added
                    # the first time they are picked (_add_deep_card) and kept after that
                    dcc.Store(id="deep-dive-built", data=[selected_signal]),
                    html.Div(id="deep-dive-cards", children=[_deep_card(rows[0])]),
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

    def _tabs():
        """The real page. Costs one Engine.run() per promoted signal, so it is
        built in a callback rather than in the layout -- see _body."""
        rows = _registry_rows()
        return dcc.Tabs(
            id="dashboard-tabs",
            className="dashboard-tight",
            style=FOLDER_TABS,  # the line the folder tabs sit on
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
                    label="Deep Dive",
                    value="signal-deep-dive",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=_deep_dive_tab(rows),
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
        """
        return html.Div([
            dcc.Interval(id="page-boot", interval=60, max_intervals=1),
            dcc.Loading(
                html.Div(id="page-body"),
                custom_spinner=shimmer_loader(image="guy.png", caption="loading"),
                overlay_style={"visibility": "visible"},
                parent_className="dashboard-page-loader",
            ),
        ])

    app = make_app(title="Model", sliders=[], body=_body)

    app.callback(
        Output("page-body", "children"),
        Input("page-boot", "n_intervals"),
    )(lambda _n: _tabs())

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

    def _add_deep_card(selected, built):
        """Build a signal's card the first time it is picked, appended beside the ones
        already built (they keep their state); picking one already built changes nothing
        here, the callback above just shows it."""
        built = built or []
        row = REGISTRY.get(selected) if selected else None
        if row is None or selected in built:
            return dash.no_update, dash.no_update
        cards = Patch()
        cards.append(_deep_card(row))
        return cards, [*built, selected]

    app.callback(
        Output("deep-dive-cards", "children"),
        Output("deep-dive-built", "data"),
        Input("deep-dive-signal", "value"),
        State("deep-dive-built", "data"),
        prevent_initial_call=True,
    )(_add_deep_card)

    def _refresh_overview(*_clicks):
        rows = _registry_rows()
        _refresh_all_data(rows)
        return _overview_content(rows)

    app.callback(
        Output("overview-content", "children"),
        Input("refresh-overview", "n_clicks"),
        prevent_initial_call=True,
    )(_refresh_overview)

    def _update(_ref, *rest):
        *_window_clicks, _snap, window, snap_range, page = rest
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
        help="Flask reloader + in-browser tracebacks. Off by default: the "
             "reloader runs this module in two processes, so every DB probe "
             "and every render happens twice.",
    )
    args = parser.parse_args()
    # This server outlives the WSL VM's idle timeout, and postgres dies with
    # the VM. Hold it up for as long as the dashboard runs; released at exit.
    utils.helpers.hold_wsl()
    app._ra_debug = args.debug
    run(app, port=args.port, host=args.host)
