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
import pandas as pd
import polars as pl
from dash import Input, Output, State, ctx, dcc, html

import utils.helpers
from dashboard import runner
from dashboard.charts import (
    DEFAULT_WINDOW,
    WINDOW_PRESETS,
    gate_chart,
    level_chart,
    pnl_chart,
    signal_chart,
)
from dashboard.ledger import SignalLedger
from dashboard.params import (
    auto_label, exit_label, gate_label as _gate_rule, input_label,
    signal_label as _display_name,
)
from dashboard.registry import LiveRegistry
from utils.viz import column_widths, table_div
from utils.research_app import (
    BORDER, C0, C1, DIM, ORANGE, PANEL, TEXT, make_app, run, shimmer_loader,
    stat_block,
)

REGISTRY = LiveRegistry()
LEDGER = SignalLedger()

TRADES_PER_PAGE = 6
SNAP_LEFT_BUFFER_BDAYS = 5

TRADE_TABLE_COLS = [
    "entry_date", "exit_date", "direction",
    "entry_level", "exit_level", "target_lvl", "expected_return_bps",
    "pnl_bps", "entry_half_life", "bars_held", "exit_reason",
]
TRADE_TABLE_HEADERS = {
    "entry_date": "Entry", "exit_date": "Exit", "direction": "Dir",
    "entry_level": "Entry Lvl", "exit_level": "Exit Lvl",
    "target_lvl": "Target Lvl", "expected_return_bps": "Exp Ret (bps)",
    "pnl_bps": "Net PnL (bps)", "entry_half_life": "Half-life (d)",
    "bars_held": "Held (d)", "exit_reason": "Exit Reason",
}
TRADE_TABLE_ROUND = {
    "entry_level": 2, "exit_level": 2, "target_lvl": 2,
    "expected_return_bps": 1, "pnl_bps": 1, "entry_half_life": 1,
}

OVERVIEW_COLUMNS = [
    # Feature is dropped: the Signal name already leads with it. Position folds
    # in the old "Latest Reading", and Reading pairs the live value with the
    # threshold it is being judged against, so the four state columns become two.
    ("name", "Signal"),
    ("position", "Position"),
    ("reading", "Reading"),
    ("gate_rule", "Gate"),
    ("gate_status", "Gate State"),
    ("live_pnl_bps", "Net PnL (bps)"),
    ("sharpe", "Sharpe"),
    ("n_trades", "Trades"),
    ("hit_rate", "Hit Rate (%)"),
    ("max_drawdown_bps", "Max DD (bps)"),
    ("data_asof", "Data As-Of"),
]


def _slug(signal_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", signal_id.lower()).strip("-")


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


def _round(value, ndigits: int):
    """Round a possibly-missing/NaN metric, keeping it numeric so sortable
    table columns still sort correctly (unlike _fnum's formatted strings)."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return round(value, ndigits)


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


def _visible_trade_range(trades, open_entry: dict | None, page: int, data_asof):
    """Date span covering exactly what's currently shown in the trade table --
    the open row (page 0 only) plus the current page of closed trades."""
    dates = []
    if trades is not None and not trades.is_empty():
        for row in trades.slice(page, TRADES_PER_PAGE).to_dicts():
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
    return f"{status} · {pct}{suffix} pct"


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
        "reading": "—",
        "gate_rule": "—",
        "gate_status": "—",
        "live_pnl_bps": None,
        "sharpe": _round(row.get("sharpe"), 2),
        "n_trades": _round(row.get("n_trades"), 0),
        "hit_rate": _round(row.get("hit_rate") * 100 if row.get("hit_rate") is not None else None, 1),
        "max_drawdown_bps": _round(row.get("max_drawdown_bps"), 1),
        "data_asof": "—",
        "data_stale": 0,
    }
    try:
        signal_id = _row_id(row)
        state = runner.compute_signal(signal_id)
        trades, open_entry = runner.trade_history(signal_id, state)
    except RuntimeError as exc:
        return {**base, "reading": str(exc)}

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
        "position": _position_label(state, open_entry),
        "reading": reading_str,
        "gate_rule": _gate_rule(params),
        "gate_status": _gate_status(state).upper(),
        "live_pnl_bps": round(_live_pnl_bps(trades, open_entry), 1),
        "data_asof": f"{asof_bar} · {asof_written}",
        # not a displayed column -- drives the stale conditional style below
        "data_stale": int(asof_stale),
    }


def _trade_cell_style(column: str, value, row: dict) -> dict | None:
    """Direction and P&L sign, coloured -- the trade table's two read-at-a-glance
    columns. Mirrors the style_data_conditional rules this replaced."""
    if column == "direction":
        return {"color": C1 if value == "long" else C0, "fontWeight": "bold"}
    if column == "pnl_bps" and isinstance(value, (int, float)):
        if value > 0:
            return {"color": C1, "fontWeight": "bold"}
        if value < 0:
            return {"color": C0, "fontWeight": "bold"}
    return None


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
    elif column == "gate_status":
        if "CLOSED" in str(value):
            return {"color": C0}
        if "OPEN" in str(value):
            return {"color": C1}
    elif column == "live_pnl_bps" and isinstance(value, (int, float)):
        if value > 0:
            return {"color": C1, "fontWeight": "bold"}
        if value < 0:
            return {"color": C0, "fontWeight": "bold"}
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
    )
    return [
        _overview_table(target, by_target[target], snapshots[target], widths)
        for target in targets
    ]


def _trade_table_block(
    closed_rows: list[dict], page: int, open_row: dict | None, slug: str
) -> html.Div:
    """One visible trade page from already-formatted trade rows."""
    n = len(closed_rows)
    if n == 0 and open_row is None:
        return html.Div("no closed trades yet", style={"color": DIM, "padding": "8px 0"})

    rows = ([open_row] if open_row is not None and page == 0 else [])
    rows += closed_rows[page:page + TRADES_PER_PAGE]
    lo, hi = (page + 1, min(page + TRADES_PER_PAGE, n)) if n else (0, 0)

    return html.Div(
        style={"marginTop": 14},
        children=[
            table_div(
                pd.DataFrame(rows),
                columns=TRADE_TABLE_COLS,
                headers=TRADE_TABLE_HEADERS,
                max_rows=len(rows),
                cell_style=_trade_cell_style,
                float_fmt=",.4g",
                table_style={"background": "#FFFFFF"},
            ),
            html.Div(
                style={"display": "flex", "gap": 8, "alignItems": "center", "marginTop": 8},
                children=[
                    html.Button("<<", id=f"trades-first-{slug}", n_clicks=0,
                                title="First trade page",
                                style={**_btn_style(), "padding": "3px 6px"}),
                    html.Button("< prev", id=f"trades-prev-{slug}", n_clicks=0,
                                style=_btn_style()),
                    html.Button("next >", id=f"trades-next-{slug}", n_clicks=0,
                                style=_btn_style()),
                    html.Button(">>", id=f"trades-last-{slug}", n_clicks=0,
                                title="Last trade page",
                                style={**_btn_style(), "padding": "3px 6px"}),
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


def _card_sections(
    signal_id: str,
    state: dict | None = None,
    trades=None,
    open_entry: dict | None = None,
    window: str = DEFAULT_WINDOW,
    page: int = 0,
    date_range: tuple | None = None,
) -> tuple[html.Div, html.Div]:
    row = REGISTRY.get(signal_id)
    if row is None:
        return (
            html.Div(),
            html.Div(f"{signal_id}: no longer promoted", style={"color": DIM}),
            None,
        )

    ledger_row = LEDGER.latest(signal_id)
    error = None
    if state is None:
        # initial non-callback render at build_app() time -- callback-driven
        # renders always pass a precomputed state/trades pair down.
        try:
            state = runner.compute_signal(signal_id)
            trades, open_entry = runner.trade_history(signal_id, state)
        except RuntimeError as exc:
            error = str(exc)

    live_pnl = _live_pnl_bps(trades, open_entry) if state and not error else None
    trade_data = {
        "closed": _format_trade_rows(trades) if trades is not None else [],
        "open": _open_trade_row(open_entry) if open_entry is not None else None,
    }
    asof_bar, asof_written, asof_stale = (
        _db_asof(signal_id, state["data_asof"]) if state else ("—", "", False)
    )
    stats = [
        stat_block(
            # bar date on top, DB write time beneath it -- two facts, two lines
            "data as-of", _stacked(asof_bar, asof_written), alert=asof_stale,
        ),
        stat_block(
            "last analysis run",
            _stacked(*_clock_parts(ledger_row["run_ts"], always_date=True))
            if ledger_row
            else "never",
        ),
        stat_block(
            "reading",
            state["fired"] if state else "—",
            alert=bool(state and state["fired"] not in ("flat", "flat (gated)")),
        ),
        *(
            [stat_block("gate", _gate_status(state))]
            if state and state["params"].get("gate") is not None
            else []
        ),
        stat_block("live pnl", _fnum(live_pnl, "+.1f", " bps"),
                    alert=bool(live_pnl and live_pnl < 0)),
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
        gate_png = gate_chart(
            state["data"],
            state["signal_frame"],
            state["params"].get("gate"),
            window_bars=window_bars,
            date_range=date_range,
            gate_window=state["params"].get("gate_window"),
        )
        pnl_png = pnl_chart(
            state["equity_curve"],
            window_bars=window_bars,
            date_range=date_range,
        )
        chart_pngs = [level_png, sig_png]
        if gate_png is not None:
            chart_pngs.append(gate_png)
        chart_pngs.append(pnl_png)
        body = [
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": 10},
                children=[
                    html.Img(
                        src=f"data:image/png;base64,{png}",
                        style={"width": "100%", "border": f"1px solid {BORDER}"},
                    )
                    for png in chart_pngs
                ],
            ),
            html.Div(
                _trade_table_block(
                    trade_data["closed"],
                    page,
                    trade_data["open"],
                    _slug(signal_id),
                ),
                id=f"trade-table-{_slug(signal_id)}",
            ),
        ]

    summary = html.Div([
        html.Div(stats, style={"display": "flex", "gap": 28, "flexWrap": "wrap",
                                "padding": "8px 2px 8px"}),
        html.Div(backtest_stats, style={"display": "flex", "gap": 28, "flexWrap": "wrap",
                                         "padding": "0 2px 14px",
                                         "borderTop": f"1px solid {BORDER}",
                                         "marginTop": 4, "paddingTop": 8}),
    ])
    return summary, html.Div(body), trade_data


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
                         f"{r.get('variant_label') or auto_label(r)}",
                "value": _row_id(r),
            }
            for r in group
        ]
    return options


def _card(row: dict) -> html.Div:
    module = row["module"]
    signal_id = _row_id(row)
    slug = _slug(signal_id)
    summary, analysis, trade_data = _card_sections(signal_id)
    return html.Div(
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
                    html.Button("Ref", id=f"ref-{slug}", n_clicks=0,
                                className="ref-btn",
                                style=_btn_style(primary=True)),
                    dcc.Store(id=f"window-{slug}", data=DEFAULT_WINDOW),
                    dcc.Store(id=f"snap-range-{slug}", data=None),
                    dcc.Store(id=f"trades-page-{slug}", data=0),
                    dcc.Store(id=f"trades-data-{slug}", data=trade_data),
                ],
            ),
            html.Div(id=f"card-summary-{slug}", children=summary),
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
                        html.Button(key, id=f"window-{slug}-{key}", n_clicks=0,
                                    style=_btn_style(primary=(key == DEFAULT_WINDOW)))
                        for key in WINDOW_PRESETS
                    ],
                ],
            ),
            html.Div(id=f"card-analysis-{slug}", children=analysis),
        ],
    )


def build_app() -> dash.Dash:
    reg_df = REGISTRY.list()
    rows = (
        []
        if reg_df.is_empty()
        else reg_df.sort("target", "family", "name").to_dicts()
    )
    selected_signal = _row_id(rows[0]) if rows else None
    tab_style = {
        "padding": "10px 18px",
        "fontSize": 12,
        "fontWeight": "bold",
        "background": PANEL,
        "border": f"1px solid {BORDER}",
        "color": DIM,
    }
    selected_tab_style = {
        **tab_style,
        "background": "#FFFFFF",
        "color": ORANGE,
        "borderTop": f"3px solid {ORANGE}",
    }

    def _overview_tab():
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
                ),
            ],
        )

    def _deep_dive_tab():
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
                    *[
                        html.Div(
                            id=f"deep-card-{_slug(_row_id(row))}",
                            style={
                                "display": "block"
                                if _row_id(row) == selected_signal
                                else "none"
                            },
                            children=_card(row),
                        )
                        for row in rows
                    ],
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
        return dcc.Tabs(
            id="dashboard-tabs",
            value="live-overview",
            children=[
                dcc.Tab(
                    label="Signal",
                    value="live-overview",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=_overview_tab(),
                ),
                dcc.Tab(
                    label="Dig",
                    value="signal-deep-dive",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=_deep_dive_tab(),
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

        Content is still rebuilt per page load (the boot callback runs each
        time), and the component ids inside are identical every call, which is
        what keeps the per-card callbacks below valid. Those callbacks
        reference ids that do not exist until this fills, which is legal only
        because make_app sets suppress_callback_exceptions.
        """
        return html.Div([
            dcc.Interval(id="page-boot", interval=60, max_intervals=1),
            dcc.Loading(
                html.Div(id="page-body"),
                custom_spinner=shimmer_loader(image="guy.png", caption="loading"),
                overlay_style={"visibility": "visible"},
            ),
        ])

    app = make_app(title="Model", sliders=[], body=_body)

    app.callback(
        Output("page-body", "children"),
        Input("page-boot", "n_intervals"),
    )(lambda _n: _tabs())

    window_keys = list(WINDOW_PRESETS)

    if rows:
        app.callback(
            *[
                Output(f"deep-card-{_slug(_row_id(row))}", "style")
                for row in rows
            ],
            Input("deep-dive-signal", "value"),
        )(
            lambda selected: [
                {
                    "display": "block"
                    if _row_id(row) == selected
                    else "none"
                }
                for row in rows
            ]
        )

        def _refresh_overview(*_clicks):
            _refresh_all_data(rows)
            return _overview_content(rows)

        app.callback(
            Output("overview-content", "children"),
            Input("refresh-overview", "n_clicks"),
            prevent_initial_call=True,
        )(_refresh_overview)

    for row in rows:
        signal_id = _row_id(row)
        slug = _slug(signal_id)
        window_prefix = f"window-{slug}-"

        def _update(
            _ref,
            *rest,
            signal_id=signal_id,
            slug=slug,
            window_prefix=window_prefix,
        ):
            *_window_clicks, _snap, window, snap_range, page = rest
            trigger = ctx.triggered_id or ""
            no_btn_styles = [dash.no_update] * len(window_keys)
            try:
                if trigger.startswith("ref-"):
                    # one click = fresh data, then one logged analysis on it
                    runner.pull_data(signal_id)
                    runner.run_analysis(signal_id)
                    page = 0
                elif trigger.startswith(window_prefix):
                    window = trigger[len(window_prefix):]
                    snap_range = None
                    page = 0
                state = runner.compute_signal(signal_id)
                trades, open_entry = runner.trade_history(signal_id, state)
            except RuntimeError as exc:
                err = html.Div(str(exc), style={"color": ORANGE, "padding": "12px 0"})
                return (html.Div(), err, None, 0, window, snap_range, *no_btn_styles)

            if trigger == f"snap-{slug}":
                rng = _visible_trade_range(trades, open_entry, page, state["data_asof"])
                if rng is not None:
                    snap_range = [str(rng[0]), str(rng[1])]

            date_range = None
            if snap_range:
                date_range = (pd.Timestamp(snap_range[0]), pd.Timestamp(snap_range[1]))

            summary, analysis, trade_data = _card_sections(
                signal_id,
                state=state,
                trades=trades,
                open_entry=open_entry,
                window=window,
                page=page,
                date_range=date_range,
            )
            btn_styles = [_btn_style(primary=(k == window)) for k in window_keys]
            return summary, analysis, trade_data, page, window, snap_range, *btn_styles

        app.callback(
            Output(f"card-summary-{slug}", "children"),
            Output(f"card-analysis-{slug}", "children"),
            Output(f"trades-data-{slug}", "data"),
            Output(f"trades-page-{slug}", "data"),
            Output(f"window-{slug}", "data"),
            Output(f"snap-range-{slug}", "data"),
            *[Output(f"window-{slug}-{k}", "style") for k in window_keys],
            Input(f"ref-{slug}", "n_clicks"),
            *[Input(f"window-{slug}-{k}", "n_clicks") for k in window_keys],
            Input(f"snap-{slug}", "n_clicks"),
            State(f"window-{slug}", "data"),
            State(f"snap-range-{slug}", "data"),
            State(f"trades-page-{slug}", "data"),
            prevent_initial_call=True,
        )(_update)

        def _page_trades(
            _first,
            _prev,
            _next,
            _last,
            trade_data,
            page,
            slug=slug,
        ):
            """Swap rows only -- chart and backtest work stays off this path."""
            trade_data = trade_data or {"closed": [], "open": None}
            closed_rows = trade_data.get("closed", [])
            open_entry = trade_data.get("open")
            n = len(closed_rows)
            last_page = max(0, (n - 1) // TRADES_PER_PAGE * TRADES_PER_PAGE) if n else 0
            page = page or 0
            if ctx.triggered_id == f"trades-first-{slug}":
                page = 0
            elif ctx.triggered_id == f"trades-prev-{slug}":
                page = max(0, page - TRADES_PER_PAGE)
            elif ctx.triggered_id == f"trades-next-{slug}":
                page = min(page + TRADES_PER_PAGE, last_page)
            elif ctx.triggered_id == f"trades-last-{slug}":
                page = last_page
            page = min(page, last_page)
            return _trade_table_block(closed_rows, page, open_entry, slug), page

        app.callback(
            Output(f"trade-table-{slug}", "children"),
            Output(f"trades-page-{slug}", "data", allow_duplicate=True),
            Input(f"trades-first-{slug}", "n_clicks"),
            Input(f"trades-prev-{slug}", "n_clicks"),
            Input(f"trades-next-{slug}", "n_clicks"),
            Input(f"trades-last-{slug}", "n_clicks"),
            State(f"trades-data-{slug}", "data"),
            State(f"trades-page-{slug}", "data"),
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
