"""Oven app: bake one target against a boatload of inputs, then hunt for the setups
whose trades pay -- and pay consistently.

    python -m oven                              # http://127.0.0.1:8054
    python -m oven --host raptor --port 8054

Built from the research app's and the dashboard's own parts -- tabs, controls,
house tables, the logo loader, progress log and PNG charts -- so it looks and
reads the same. Two tabs:

  Bake     choose a target, inputs and grid; Bake runs as a background job
           (research.jobs), so a reload doesn't lose it. Every bake is saved
           under oven/data/.
  Results  three paged boards, each opened by an Inspect click on the one
           above: the inputs (each input's best setup), that input's setups,
           and one setup laid out as the dashboard's Dig card -- stats, chart
           window + Copy, the chart grid, the paged trade table with Snap chart
           to view -- followed by the research Dislocation tab's trade
           mechanics (monthly P&L heatmap, return distribution, exit reasons)
           and the Oven's own diagnostics (regression scatters, lookback x
           entry, years, entry-state characteristics and correlations, P&L
           correlation with other inputs' best setups).

Every long table pages: header clicks sort ALL its rows on the server (click
again to flip) and the page buttons step through them.

Each trade holds the portfolio y hedged by x (the changes beta, frozen at
entry); an input that isn't tradable trades y alone. The engine is
oven.engine; this module only lays it out.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from dashboard.charts import (WINDOW_PRESETS, input_chart, level_chart, pnl_chart, residual_bars_chart,
                              return_distribution_chart, series_chart, signal_chart)
from oven import engine
from research import jobs
from research.app import (FEATURE_GROUPS, FEATURE_LABELS, INPUT, SELECTED_TAB_STYLE, TAB_STYLE, TARGET_OPTIONS,
                          _diverging, btn_style, field, heading, info, loading_panel, note, pnl_heatmap,
                          progress_view, regression_scatter, section, table)
from research.panel import CATALOG, build_panel
from utils.research_app import BORDER, C0, C1, DIM, ORANGE, PANEL, make_app, run, shimmer_loader, stat_block

ALL_INPUTS = list(dict.fromkeys(name for names in FEATURE_GROUPS.values() for name in names))
GRID_CONTROLS = {
    "lookbacks": ("signal lookbacks · levels OLS (d)", [21, 63, 126, 252, 504], engine.LOOKBACKS),
    "z_entries": ("entries · normalized z", [1.0, 1.5, 2.0, 2.5, 3.0], engine.Z_ENTRIES),
    "raw_entries": ("entries · raw bps", [3.0, 5.0, 10.0, 15.0, 20.0, 30.0], engine.RAW_ENTRIES),
    "reverts": ("exit · share of frozen residual recovered (0 = no reversion exit)", [0.0, 0.25, 0.5, 0.75, 1.0],
                engine.REVERTS),
    "hl_mults": ("exit · time stop at × entry half-life (0 = off)", [0.0, 0.5, 1.0, 2.0, 3.0], engine.HL_MULTS),
    "max_holds": ("exit · max hold, the hard cap (bars)", [5, 10, 20, 40, 60, 120], engine.MAX_HOLDS),
    "stops": ("exit · stop, entry sigmas wider (0 = none)", [0.0, 0.5, 1.0, 2.0], engine.STOPS),
    "hedge_lookbacks": ("hedge lookback · changes OLS (d)", [21, 63, 126, 252, 504], engine.HEDGE_LOOKBACKS),
}
SETUP_COLS = ["lookback", "hedge_lb", "basis", "scale", "entry", "revert", "max_hold", "stop", "hl_mult"]
BASIS_TEXT = {
    "levels": "levels-fit residual, target − (a + levels β × input)",
    "portfolio": "the traded portfolio's own deviation from its mean",
}
STRATEGY_COLS = ["sharpe", "worst_half_sharpe", "sharpe_h1", "sharpe_h2", "total_pnl", "pnl_per_year", "n_trades",
                 "trades_per_year", "hit_rate", "avg_trade", "median_trade", "profit_factor", "max_drawdown",
                 "calmar", "time_in_market", "avg_bars", "trade_t", "best_trade_share", "worst_trade"]
PAGED = ("inputs", "setups", "trades")
PAGE_SIZE = {"inputs": 15, "setups": 15, "trades": 10}
DEFAULT_SORT = {"inputs": "score", "setups": "score", "trades": "entry"}
DEFAULT_WINDOW = "All"
SNAP_LEFT_BUFFER_BDAYS = 5  # as the dashboard's Snap chart to view
ET = ZoneInfo("America/New_York")
STATS_ROW = {"display": "flex", "gap": 28, "flexWrap": "wrap", "padding": "8px 2px 8px"}
TWO_UP = {"display": "grid", "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr)", "gap": 12,
          "alignItems": "start"}
# every chart cell the same shape: the PNGs come out at slightly different aspect
# ratios (each chart type has its own figure size, then tight cropping), so each is
# fitted inside an identical frame rather than setting its own row height
IMG = {"width": "100%", "aspectRatio": "1.75", "objectFit": "contain", "background": "#FFFFFF", "display": "block",
       "border": f"1px solid {BORDER}"}


# ---- data --------------------------------------------------------------------


def baked_at(stamp: str) -> str:
    """A run's UTC stamp ('20261016T235922Z') as 'Oct 16, 2026 · 7:59 PM ET'."""
    when = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).astimezone(ET)
    return f"{when:%b} {when.day}, {when.year} · {when.hour % 12 or 12}:{when:%M %p} ET"


def run_options() -> list[dict]:
    out = []
    for path in engine.list_runs():
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        out.append({"label": f"{meta['target']} · {len(meta['inputs'])} inputs · {meta['n_setups']:,} setups · "
                             f"{baked_at(meta['created_at'])}", "value": str(path)})
    return out


def _with_basis(frame: pl.DataFrame, hedge_lb: int) -> pl.DataFrame:
    """Older bakes: before the portfolio basis everything was levels-basis, before
    half-life time stops no setup had one, and before the hedge lookback was swept
    every setup used the bake's single one."""
    if "hedge_lb" not in frame.columns:
        frame = frame.with_columns(pl.lit(hedge_lb, pl.Int64).alias("hedge_lb"))
    if "basis" not in frame.columns:
        frame = frame.with_columns(pl.lit("levels").alias("basis"))
    if "hl_mult" not in frame.columns:
        frame = frame.with_columns(pl.lit(0.0).alias("hl_mult"))
    return frame


@lru_cache(maxsize=4)
def load_run(path: str) -> dict:
    """A run's meta and board: all the boards need. Trades load only when a setup is opened."""
    p = Path(path)
    meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
    return dict(meta=meta, board=_with_basis(pl.read_parquet(p / "board.parquet"), _old_hedge_lb(meta)))


@lru_cache(maxsize=2)
def load_trades(path: str) -> pl.DataFrame:
    return _with_basis(pl.read_parquet(Path(path) / "trades.parquet"), _old_hedge_lb(load_run(path)["meta"]))


def _old_hedge_lb(meta: dict) -> int:
    return int(meta["settings"].get("hedge_lookback", engine.HEDGE_LOOKBACK))


def describe(setup: dict) -> str:
    """A setup in one line: 'lb 252 · hedge 126 · portfolio · raw > 5 · revert 1 · hl×1 · hold 120 · stop 2'."""
    revert = f"revert {setup['revert']:g}" if setup["revert"] else "no revert"
    half_life = f" · hl×{setup['hl_mult']:g}" if setup["hl_mult"] else ""
    return (f"lb {setup['lookback']} · hedge {setup['hedge_lb']} · {setup['basis']} · {setup['scale']} > {setup['entry']:g} · {revert}"
            f"{half_life} · hold {setup['max_hold']} · stop {setup['stop']:g}")


@lru_cache(maxsize=32)
def model_frame(target: str, name: str, lookback: int, hedge_lookback: int) -> pl.DataFrame:
    """ts, the target and the input under their own names, every model array, the
    hedged portfolio's level and the 126d correlation of daily moves."""
    panel = build_panel(CATALOG[target], features=[name])
    aligned = panel.data.select("ts", target, name).drop_nulls()
    y = aligned[target].to_numpy().astype(float)
    x = aligned[name].to_numpy().astype(float)
    arrays = engine.model_arrays(y, x, lookback, hedge_lookback)
    package = engine.held_package(y, x, arrays["beta_changes"], 1.0 if engine.hedgeable(name) else 0.0)
    corr = (pl.DataFrame({"dy": np.diff(y, prepend=np.nan), "dx": np.diff(x, prepend=np.nan)})
            .select(pl.rolling_corr("dy", "dx", window_size=126)).to_series())
    return aligned.with_columns(*[pl.Series(k, v) for k, v in arrays.items()], pl.Series("portfolio", package),
                                corr.alias("corr126"))


def input_board(board: pl.DataFrame, by: str, min_trades: int) -> pl.DataFrame:
    """One row per input: its best setup by ``by`` and how many of its setups pay."""
    b = engine.rank(board, by, min_trades, top=None)
    if b.is_empty():
        return b
    best = b.unique("input", keep="first", maintain_order=True)
    counts = b.group_by("input").agg(
        pl.len().alias("setups"), pl.col("sharpe").median().alias("median_sharpe"),
        (pl.col("sharpe") > 0.5).sum().alias("sharpe_above_0.5"),
        (pl.col("worst_half_sharpe") > 0.5).sum().alias("both_halves_above_0.5"))
    best = best.with_columns(pl.struct(SETUP_COLS).map_elements(describe, return_dtype=pl.Utf8).alias("best_setup"))
    return (best.select("input", "score", "sharpe", "worst_half_sharpe", "total_pnl", "n_trades", "hit_rate",
                        "max_drawdown", "best_setup", "hedged")
            .join(counts, on="input")
            .sort("score", descending=True, nulls_last=True)  # the join doesn't keep the ranking's order
            .with_columns(pl.col("input").replace_strict(FEATURE_LABELS, default="").alias("label")))


def setup_id(row: dict) -> str:
    return "|".join(str(row[c]) for c in SETUP_COLS)


def setup_from_id(name: str, sid: str) -> dict:
    lb, hedge_lb, basis, scale, entry, revert, hold, stop, hl_mult = sid.split("|")
    return dict(input=name, lookback=int(lb), hedge_lb=int(hedge_lb), basis=basis, scale=scale, entry=float(entry), revert=float(revert),
                max_hold=int(hold), stop=float(stop), hl_mult=float(hl_mult))


def matches(frame: pl.DataFrame, setup: dict) -> pl.DataFrame:
    return frame.filter(pl.all_horizontal([pl.col(k) == v for k, v in setup.items()]))


def daily_frame(frame: pl.DataFrame, target: str, name: str, trades: pl.DataFrame, cost: float) -> pl.DataFrame:
    """ts, daily P&L, cumulative P&L and drawdown of one setup."""
    marks = engine.daily_marks(frame["ts"], frame[target].to_numpy().astype(float),
                               frame[name].to_numpy().astype(float), trades, cost)
    cum = np.cumsum(marks)
    return pl.DataFrame({"ts": frame["ts"], "daily_pnl": marks, "cumulative_pnl": cum,
                         "drawdown": cum - np.maximum.accumulate(cum)})


def by_year(daily: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """Per calendar year: P&L, Sharpe of the daily marks, trades and hit rate."""
    year_marks = daily.group_by(pl.col("ts").dt.year().alias("year")).agg(
        pl.col("daily_pnl").sum().alias("pnl"),
        (pl.col("daily_pnl").mean() / pl.col("daily_pnl").std(ddof=0) * np.sqrt(engine.PERIODS)).alias("sharpe"),
        (pl.col("daily_pnl") != 0).mean().alias("time_in_market"),
        (pl.col("daily_pnl") != 0).sum().alias("active_days"))
    year_trades = trades.group_by(pl.col("exit_ts").dt.year().alias("year")).agg(
        pl.len().alias("trades_closed"), (pl.col("pnl") > 0).mean().alias("hit_rate"),
        pl.col("pnl").mean().alias("avg_trade"))
    return (year_marks.join(year_trades, on="year", how="left")
            .with_columns(pl.col("trades_closed").fill_null(0).cast(pl.Int64)).sort("year"))


@lru_cache(maxsize=16)
def bundle(path: str, name: str, sid: str) -> dict:
    """Everything one setup's views draw from, built once and reused by the chart
    window, Snap and trade pages."""
    run = load_run(path)
    meta = run["meta"]
    setup = setup_from_id(name, sid)
    target = meta["target"]
    trades = matches(load_trades(path), setup).sort("entry_ts")
    row = matches(run["board"], setup).to_dicts()
    frame = model_frame(target, name, setup["lookback"], setup["hedge_lb"])
    daily = daily_frame(frame, target, name, trades, float(meta["settings"].get("cost", engine.COST)))
    levels = frame.select("ts", pl.col(target).alias("y"), pl.col(name).alias("x"), "portfolio")
    at = lambda when: levels.rename({"ts": f"{when}_ts", "y": f"{when}_level", "x": f"{when}_input",
                                     "portfolio": f"{when}_portfolio"})
    # the P&L split into its legs: what the target did for the trade, and what the hedge did
    with_levels = (trades.join(at("entry"), on="entry_ts", how="left").join(at("exit"), on="exit_ts", how="left")
                   .with_columns(
                       pl.when(pl.col("side") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("direction"),
                       (pl.col("side") * (pl.col("exit_level") - pl.col("entry_level"))).alias("target_leg"),
                       (-pl.col("side") * pl.col("hedge_ratio") * (pl.col("exit_input") - pl.col("entry_input")))
                       .alias("hedge_leg")))

    def markers(prefix: str) -> pl.DataFrame:
        return with_levels.select(
            "direction", pl.col("entry_ts").cast(pl.Datetime).alias("entry_date"),
            pl.col(f"entry_{prefix}").alias("entry_level"), pl.col("exit_ts").cast(pl.Datetime).alias("exit_date"),
            pl.col(f"exit_{prefix}").alias("exit_level"))
    return dict(meta=meta, setup=setup, target=target, trades=trades, with_levels=with_levels,
                row=row[0] if row else None, frame=frame, daily=daily, markers=markers("level"),
                portfolio_markers=markers("portfolio"), signal=engine.basis_arrays(frame, setup["basis"]),
                latest=frame.drop_nulls("beta_levels").tail(1).to_dicts()[0], years=_active_years(by_year(daily, trades)))


def _active_years(years: pl.DataFrame) -> pl.DataFrame:
    """Years from the first one the setup was ever in the market: the warm-up before
    the signal existed is all zeros and says nothing."""
    active = years.filter(pl.col("active_days") > 0)
    return years.filter(pl.col("year") >= active["year"].min()) if len(active) else years


def monthly_pnl(trades: pl.DataFrame) -> pl.DataFrame:
    return trades.group_by(pl.col("exit_ts").dt.truncate("1mo").alias("month")).agg(pl.col("pnl").sum())


def peer_correlations(path: str, setup: dict, by: str, min_trades: int, peers: int = 12) -> pl.DataFrame:
    """Monthly P&L correlation of this setup with the best setup of each other top input:
    does it add something to a book of signals, or is it the same bet?"""
    trades = load_trades(path)
    ranked = engine.rank(load_run(path)["board"], by, min_trades, top=None)
    best = (ranked.filter(pl.col("input") != setup["input"])
            .unique("input", keep="first", maintain_order=True).head(peers))
    mine = monthly_pnl(matches(trades, setup))
    rows = []
    for r in best.to_dicts():
        theirs = monthly_pnl(matches(trades, {k: r[k] for k in engine.SETUP}))
        both = mine.join(theirs, on="month", how="full", coalesce=True, suffix="_peer").fill_null(0.0)
        rows.append(dict(peer=f"{r['input']} · {describe(r)}",
                         peer_sharpe=r["sharpe"], monthly_corr=both.select(pl.corr("pnl", "pnl_peer")).item(),
                         months_both_active=int(((both["pnl"] != 0) & (both["pnl_peer"] != 0)).sum())))
    return pl.DataFrame(rows)


def feature_correlations(trades: pl.DataFrame) -> pl.DataFrame:
    """Spearman correlation of each entry-state feature with trade P&L and with winning."""
    rows = []
    for f in engine.FEATURES:
        v = trades.select(f, "pnl").drop_nulls().filter(pl.col(f).is_finite())
        if len(v) <= 5:
            continue
        rows.append(dict(feature=f, corr_with_pnl=v.select(pl.corr(f, "pnl", method="spearman")).item(),
                         corr_with_win=v.select(pl.corr(pl.col(f), (pl.col("pnl") > 0).cast(pl.Float64),
                                                        method="spearman")).item(),
                         winners_median=v.filter(pl.col("pnl") > 0)[f].median(),
                         losers_median=v.filter(pl.col("pnl") <= 0)[f].median()))
    return pl.DataFrame(rows)


HORIZON_Z = 1.0  # |z| beyond which another horizon counts as saying something


def multi_resolution(trades: pl.DataFrame, setup: dict) -> pl.DataFrame:
    """This setup's trades split by what each OTHER signal horizon said at entry:
    'agrees' (same sign, |z| > HORIZON_Z: also rich, or also cheap), 'disagrees'
    (opposite sign beyond it) or 'neutral'. A state whose average trade beats the
    setup's own is one where that horizon adds information."""
    horizons = sorted((int(c[4:]), c) for c in trades.columns if c.startswith("z_lb"))
    base = trades["pnl"].mean()
    own = pl.col("entry_z").sign()
    rows = []
    for lb, col in horizons:
        if lb == setup["lookback"]:
            continue
        z = pl.col(col)
        state = (pl.when(z.is_null() | z.is_nan() | (z.abs() <= HORIZON_Z)).then(pl.lit("neutral"))
                 .when(z.sign() == own).then(pl.lit("agrees")).otherwise(pl.lit("disagrees")))
        both = trades.select("entry_z", col).drop_nulls().filter(pl.col(col).is_not_nan())
        corr = both.select(pl.corr("entry_z", col, method="spearman")).item() if len(both) > 5 else None
        split = (trades.with_columns(state.alias("state")).group_by("state").agg(
            pl.len().alias("trades"), pl.col("pnl").mean().alias("avg_trade"), (pl.col("pnl") > 0).mean().alias("hit_rate"),
            pl.col("pnl").sum().alias("total_pnl")))
        for row in split.to_dicts():
            rows.append(dict(other_horizon=f"lb {lb}", z_corr_at_entry=corr, state=row["state"], trades=row["trades"],
                             share=row["trades"] / len(trades), avg_trade=row["avg_trade"],
                             vs_setup_avg=row["avg_trade"] - base, hit_rate=row["hit_rate"],
                             total_pnl=row["total_pnl"]))
    if not rows:
        return pl.DataFrame()
    order = {"agrees": 0, "neutral": 1, "disagrees": 2}
    return (pl.DataFrame(rows).with_columns(pl.col("state").replace_strict(order, default=3).alias("_o"),
                                            pl.col("other_horizon").str.slice(3).cast(pl.Int64).alias("_lb"))
            .sort("_lb", "_o").drop("_o", "_lb"))


def lookback_grid(board: pl.DataFrame, setup: dict, by: str) -> pl.DataFrame:
    """``by`` over lookback x entry for this input, holding the setup's scale, exit and stop."""
    fixed = {k: setup[k] for k in ("input", "hedge_lb", "basis", "scale", "revert", "max_hold", "stop", "hl_mult")}
    return (matches(board, fixed).select("lookback", pl.format("entry {}", pl.col("entry")).alias("entry"), by)
            .pivot(on="entry", index="lookback", values=by).sort("lookback"))


# ---- paged tables ------------------------------------------------------------


def default_view(name: str) -> dict:
    return {"page": 0, "sort": DEFAULT_SORT[name], "desc": True}


def inputs_rows(path: str, by: str, min_trades: int) -> pl.DataFrame:
    rows = input_board(load_run(path)["board"], by, min_trades)
    if rows.is_empty():
        return rows
    return rows.with_row_index("#", offset=1).select(
        "#", "input", "label", "score", "worst_half_sharpe", "total_pnl", "n_trades", "hit_rate", "max_drawdown",
        "best_setup", "setups", "median_sharpe", "sharpe_above_0.5", "both_halves_above_0.5", "hedged")


def setups_rows(path: str, name: str, by: str, min_trades: int) -> pl.DataFrame:
    ranked = engine.rank(load_run(path)["board"].filter(pl.col("input") == name), by, min_trades, top=None)
    if ranked.is_empty():
        return ranked
    return ranked.with_row_index("#", offset=1).with_columns(
        pl.struct(SETUP_COLS).map_elements(setup_id, return_dtype=pl.Utf8).alias("sid")
    ).select("#", "sid", "score", *SETUP_COLS, *STRATEGY_COLS)


def trades_rows(path: str, picked: dict) -> pl.DataFrame:
    """The Dig tab's trade table columns, plus the frozen hedge and entry state."""
    b = bundle(path, picked["input"], picked["sid"])
    return b["with_levels"].select(
        pl.col("entry_ts").alias("entry"), pl.col("exit_ts").alias("exit"), pl.col("direction").alias("dir"),
        "entry_level", "exit_level", pl.col("pnl").alias("pnl_bps"), "target_leg", "hedge_leg",
        pl.col("hedge_ratio").alias("hedge_β"), pl.col("bars").alias("held"), pl.col("reason").alias("exit_reason"),
        "entry_z", "half_life", "r2")


def _sorted_page(rows: pl.DataFrame, name: str, view: dict) -> tuple[pl.DataFrame, dict]:
    """The rows on ``view``'s page, sorted by its column; the page is clamped to what exists."""
    size = PAGE_SIZE[name]
    pages = max(1, math.ceil(len(rows) / size))
    view = {**view, "page": min(max(int(view.get("page", 0)), 0), pages - 1)}
    if view["sort"] not in rows.columns:
        view = {**view, "sort": DEFAULT_SORT[name] if DEFAULT_SORT[name] in rows.columns else rows.columns[0]}
    ordered = rows.sort(view["sort"], descending=view["desc"], nulls_last=True)
    return ordered.slice(view["page"] * size, size), view


def inspect_button(kind: str, key: str) -> html.Button:
    """A row's Inspect button, sized to sit inside a table cell."""
    return html.Button("Inspect", id={"type": kind, "index": key}, n_clicks=0, className="ref-btn",
                       style={**btn_style(), "padding": "0 7px", "fontSize": 10, "lineHeight": "16px"})


def show(frame: pl.DataFrame):
    """A frame for the house table, with missing values blank rather than 'nan'."""
    pdf = frame.to_pandas()
    return pdf.astype(object).where(pdf.notna(), "")


# short display headers for the boards; sorting still keys off the real column
HEADERS = {"worst_half_sharpe": "worst half", "total_pnl": "total pnl", "n_trades": "trades", "hit_rate": "hit",
           "max_drawdown": "max dd", "best_setup": "best setup", "median_sharpe": "median sharpe",
           "sharpe_above_0.5": "sharpe>0.5", "both_halves_above_0.5": "both halves>0.5", "sharpe_h1": "h1",
           "sharpe_h2": "h2", "pnl_per_year": "pnl/yr", "trades_per_year": "trades/yr", "avg_trade": "avg",
           "median_trade": "median", "profit_factor": "pf", "time_in_market": "in mkt", "avg_bars": "avg bars",
           "trade_t": "trade t", "best_trade_share": "best share", "worst_trade": "worst", "max_hold": "hold",
           "hl_mult": "hl×", "exit_reason": "exit", "entry_level": "entry lvl", "exit_level": "exit lvl",
           "target_leg": "target leg", "hedge_leg": "hedge leg", "half_life": "half-life"}
TEXT_COLS = {"input", "label", "best_setup", "basis", "scale", "exit_reason"}


def _cell_style(trade_scale: float | None):
    """Text columns read left-aligned; on trades, direction and P&L are coloured."""
    def style(column, value, _row):
        if column in TEXT_COLS:
            return {"textAlign": "left"}
        if column == "dir":
            return {"color": C1 if value == "long" else C0, "fontWeight": "bold"}
        if column == "pnl_bps" and trade_scale is not None and value != "":
            return _diverging(value, trade_scale)
        return None
    return style


def paged_table(name: str, rows: pl.DataFrame, view: dict, title: str) -> tuple[html.Div, dict]:
    """One page of a board: the house table (every header sorts all rows on the
    server), its Inspect buttons, and where the page sits."""
    if rows.is_empty():
        return note("Nothing to show.", "dim"), view
    # bakes saved before the engine left an undefined ratio blank carry ±inf; show those blank too
    rows = rows.with_columns(pl.when(pl.selectors.float().is_infinite()).then(None)
                             .otherwise(pl.selectors.float()).name.keep())
    shown, view = _sorted_page(rows, name, view)
    hidden = {"inputs": [], "setups": ["sid"], "trades": []}[name]
    cols = [c for c in rows.columns if c not in hidden]
    arrow = " ▼" if view["desc"] else " ▲"
    headers = {c: HEADERS.get(c, c) + (arrow if c == view["sort"] else "") for c in cols}
    store = f"oven-{name}-sort"
    cell_style = _cell_style(float(np.nanpercentile(np.abs(rows["pnl_bps"].to_numpy()), 95))
                             if name == "trades" else None)
    frame = show(shown)
    key = {"inputs": ("input", "oven-input-pick"), "setups": ("sid", "oven-setup-pick")}.get(name)
    if key:  # the row's own Inspect button, as its first cell
        frame.insert(0, "inspect", [inspect_button(key[1], k) for k in shown[key[0]].to_list()])
        cols, headers = ["inspect", *cols], {**headers, "inspect": ""}
    parts = [table(frame, title=title, columns=cols, headers=headers, float_fmt=",.2f",
                   max_rows=PAGE_SIZE[name], sortable=True, cell_style=cell_style,
                   header_props={c: {"data-board-rule": c, "data-board-store": store} for c in cols
                                 if c != "inspect"})]
    first = view["page"] * PAGE_SIZE[name] + 1
    pages = max(1, math.ceil(len(rows) / PAGE_SIZE[name]))
    parts.append(html.Span(f"rows {first:,}–{first + len(shown) - 1:,} of {len(rows):,} · page {view['page'] + 1} "
                           f"of {pages} · sorted by {HEADERS.get(view['sort'], view['sort'])} "
                           f"{'high → low' if view['desc'] else 'low → high'}",
                           className="oven-pager-status"))
    # a new key per view makes React rebuild the table, clearing the " …" the sort click wrote
    return html.Div(parts, key=f"{view['sort']}-{view['desc']}-{view['page']}"), view


def pager(name: str, extra=()) -> html.Div:
    """The Dig tab's trade pager, for any board. Lives outside the replaced page."""
    small = {**btn_style(), "padding": "2px 8px", "fontSize": 11}
    return html.Div([
        html.Button("<<", id=f"oven-{name}-first", n_clicks=0, className="ref-btn", title="First page", style=small),
        html.Button("< prev", id=f"oven-{name}-prev", n_clicks=0, className="ref-btn", style=small),
        html.Button("next >", id=f"oven-{name}-next", n_clicks=0, className="ref-btn", style=small),
        html.Button(">>", id=f"oven-{name}-last", n_clicks=0, className="ref-btn", title="Last page", style=small),
        *extra,
    ], className="oven-pager")


def render_page(name: str, view: dict, path: str, by: str, min_trades, picked_input, picked_setup):
    min_trades = int(min_trades or 1)
    if name == "inputs":
        rows = inputs_rows(path, by, min_trades)
        return paged_table(name, rows, view, f"inputs · each input's best setup by {engine.RANKS[by]}")
    if name == "setups":
        rows = setups_rows(path, picked_input, by, min_trades)
        return paged_table(name, rows, view, f"{len(rows):,} setups with ≥{min_trades} trades")
    rows = trades_rows(path, picked_setup)
    return paged_table(name, rows, view, f"trades · {len(rows)} · P&L green gains, red losses")


# ---- views -------------------------------------------------------------------


def run_view(path: str, by: str, min_trades: int) -> tuple[html.Div, dict]:
    run = load_run(path)
    meta, board = run["meta"], run["board"]
    eligible = board.filter(pl.col("n_trades") >= min_trades)
    sharpe = eligible["sharpe"].drop_nulls()
    if not len(sharpe):
        return note(f"No setup has {min_trades}+ trades. Lower the minimum trades.", "warn"), default_view("inputs")
    # one row: where the setups' Sharpes sit, per signal basis and overall
    qs = {"10%": 0.1, "25%": 0.25, "median": 0.5, "75%": 0.75, "90%": 0.9, "99%": 0.99}
    quantiles = pl.DataFrame([
        {"setups": label, "n": len(s), **{k: s.quantile(q) for k, q in qs.items()}, "max": s.max()}
        for label, s in [("all", sharpe), *[(basis, eligible.filter(pl.col("basis") == basis)["sharpe"].drop_nulls())
                                            for basis in engine.BASES]] if len(s)])
    page, view = render_page("inputs", default_view("inputs"), path, by, min_trades, None, None)
    return html.Div([
        html.Div(style={**STATS_ROW, "borderBottom": f"1px solid {BORDER}", "marginBottom": 10}, children=[
            stat_block("target", meta["target"]), stat_block("inputs", f"{len(meta['inputs'])}"),
            stat_block("setups baked", f"{len(board):,}"),
            stat_block(f"with ≥{min_trades} trades", f"{len(eligible):,}"),
            stat_block("Sharpe > 0.5", f"{(sharpe > 0.5).sum():,}"),
            stat_block("both halves > 0.5", f"{(eligible['worst_half_sharpe'] > 0.5).sum():,}"),
            stat_block("best Sharpe", f"{sharpe.max():.2f}"),
            stat_block("trades", f"{meta['n_trades']:,}"), stat_block("baked", baked_at(meta["created_at"]))]),
        info("How to read these boards",
             "Each setup is one input, signal lookback, signal basis, entry scale and threshold, and exit. Every "
             "trade holds the portfolio target − changes β × input, with β frozen at entry; an input that isn't "
             "tradable trades the target alone. The basis is what the entry signal measures: 'levels' is the "
             "levels-OLS residual of the target on the input (rich/cheap vs the fit); 'portfolio' is that traded "
             "portfolio's own deviation from its mean, so its reversion is the trade's P&L. The model freezes at "
             "entry, so a trade exits only when prices move.",
             "Sharpe is annualised from daily mark-to-market P&L, flat days included. Worst-half Sharpe is the lower "
             "of the setup's Sharpe in the first and second half of its own sample: a setup that only worked in one "
             "era scores low. Best-trade share is the largest trade's share of total P&L.",
             "Click any column header to sort every row by it (click again to flip), and page with the buttons. "
             "Thousands of overlapping setups are searched, so the best of them look good partly by luck: prefer "
             "setups whose neighbours also pay, with both halves positive and no single trade carrying the total."),
        table(show(quantiles), title=f"Sharpe across setups with ≥{min_trades} trades", float_fmt=",.2f"),
        html.Div(page, id="oven-inputs-page"),
        pager("inputs"),
    ]), view


def setups_view(path: str, name: str, by: str, min_trades: int) -> tuple[html.Div, dict]:
    page, view = render_page("setups", default_view("setups"), path, by, min_trades, name, None)
    return html.Div([heading(f"setups · {FEATURE_LABELS.get(name, name)}"), html.Div(page, id="oven-setups-page"),
                     pager("setups")]), view


def _fmt(v, spec):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)


def _slice(frame: pl.DataFrame, date_range) -> pl.DataFrame:
    if date_range is None:
        return frame
    start, end = (pd.Timestamp(d).date() for d in date_range)
    return frame.filter(pl.col("ts").is_between(start, end))


def signal_now(b: dict) -> float:
    """Today's reading of the setup's own signal, on its own scale."""
    sig = b["signal"]["resid" if b["setup"]["scale"] == "raw" else "z"]
    finite = sig.drop_nulls().drop_nans()
    return float(finite[-1]) if len(finite) else float("nan")


def charts_view(b: dict, window: str | None, date_range=None) -> list:
    """The Dig card's chart grid, two to a row: the traded portfolio | the target (both
    with trades), the entry signal | the input, cumulative P&L | the return
    distribution -- then the Oven's own (drawdown, P&L by year, betas, fit quality)."""
    setup, target, name, frame, daily = b["setup"], b["target"], b["setup"]["input"], b["frame"], b["daily"]
    bars = None if date_range is not None else WINDOW_PRESETS[window or DEFAULT_WINDOW]
    raw = setup["scale"] == "raw"
    now = signal_now(b)
    fired = "short" if now > setup["entry"] else "long" if now < -setup["entry"] else "flat"
    sig_frame = pl.DataFrame({"resid": b["signal"]["resid"], "ou_z": b["signal"]["z"]})
    beta = b["latest"]["beta_changes"]
    traded = (f"{target} {'−' if beta >= 0 else '+'} {abs(beta):.2f} × {name}" if b["row"]["hedged"]
              else f"{target} alone")
    shown = frame.with_columns(pl.col("portfolio").alias(f"portfolio · {traded}"),
                               pl.Series("signal_half_life", b["signal"]["half_life"]))
    f, d = _slice(shown, date_range), _slice(daily, date_range)
    years = b["years"]
    pngs = [
        level_chart(shown, f"portfolio · {traded}", trades=b["portfolio_markers"], window_bars=bars,
                    date_range=date_range),
        level_chart(frame, target, trades=b["markers"], window_bars=bars, date_range=date_range),
        # signal_chart draws "resid" for the label "residual" and "ou_z" for any other label
        signal_chart(frame, sig_frame, "residual" if raw else f"{setup['basis']} z-score", setup["entry"],
                     window_bars=bars, date_range=date_range, fired=fired),
        input_chart(frame, name, window_bars=bars, date_range=date_range),
        pnl_chart(daily.select("ts", "cumulative_pnl"), window_bars=bars, date_range=date_range),
        return_distribution_chart(b["trades"].select(pl.col("pnl").alias("pnl_bps")), bins=40, density=True),
        series_chart(d, ["drawdown"], "drawdown", "bps", window_bars=bars, line_colors={"drawdown": C0}),
        residual_bars_chart([str(y) for y in years["year"]], years["pnl"].to_list(), "P&L by year", "bps",
                            rotate_labels=True),
        # series_chart: yaxis_title names the right-hand axis, yaxis_right_title the left one (the `left` columns)
        series_chart(f, ["beta_levels", "beta_changes", "corr126"], "levels beta · changes beta (the hedge) · "
                     "126d corr", "corr", left=["beta_levels", "beta_changes"], yaxis_right_title="beta",
                     window_bars=bars),
        series_chart(f.with_columns(  # a residual that barely reverts has a near-infinite half-life: leave those out
                         pl.when(pl.col("signal_half_life") <= 5 * setup["lookback"]).then(pl.col("signal_half_life"))
                         .otherwise(None).alias("signal_half_life")),
                     ["r2", "signal_half_life"], f"levels fit R² · {setup['basis']} residual half-life (≤ 5× lookback)",
                     "half-life (bars)", left=["r2"], yaxis_right_title="R²", window_bars=bars),
    ]
    return [html.Img(src=f"data:image/png;base64,{png}", style=IMG) for png in pngs]


def window_row(window: str | None) -> list:
    """The Dig card's chart-window buttons and Copy, as styles keyed by window."""
    return [
        html.Span("Chart window", style={"fontSize": 10, "color": DIM, "textTransform": "uppercase",
                                         "marginRight": 2}),
        *[html.Button(k, id=f"oven-window-{k}", n_clicks=0, className="ref-btn", style=btn_style(primary=k == window))
          for k in WINDOW_PRESETS],
        html.Button("Copy", id="copy-charts-oven", n_clicks=0, className="ref-btn copy-charts-btn",
                    title="Copy all visible charts as one image", style={**btn_style(), "marginLeft": "auto"}),
        html.Span(id="copy-status-oven", style={"fontSize": 11, "color": DIM, "minWidth": 42}),
    ]


# the scatter fit windows: the setup's own lookback, then the same presets as the chart window
SCATTER_WINDOWS = ["lookback", *WINDOW_PRESETS]


def scatter_pair(b: dict, window: str) -> list:
    """The levels and changes scatters, fitted over ``window`` (a SCATTER_WINDOWS key)
    with the stretch of the same length before it in grey; All fits every day, with no grey."""
    target, name = b["target"], b["setup"]["input"]
    ts = b["frame"].select("ts", target, name).drop_nulls()["ts"]
    preset = WINDOW_PRESETS.get(window)
    if window == "lookback":
        days = b["setup"]["lookback"]
    elif preset == "YTD":
        days = int((ts.dt.year() == ts.max().year).sum())
    elif preset is None:  # All: regression_scatter wants 3 rows beyond its window (4 once changes drop a day)
        days = len(ts) - 4
    else:
        days = preset
    past = "none" if window == "All" else "previous"
    return [regression_scatter(b["frame"], target, name, basis, days, past) for basis in ("levels", "changes")]


def scatter_window_row(lookback: int, window: str) -> list:
    """The scatters' fit-window buttons."""
    return [
        html.Span("Fit window", style={"fontSize": 10, "color": DIM, "textTransform": "uppercase", "marginRight": 2}),
        *[html.Button(f"{lookback}d · lookback" if k == "lookback" else k, id=f"oven-scatter-{k}", n_clicks=0,
                      className="ref-btn", style=btn_style(primary=k == window)) for k in SCATTER_WINDOWS],
    ]


def detail_view(path: str, picked: dict, by: str, min_trades: int, window: str) -> tuple[html.Div, dict]:
    b = bundle(path, picked["input"], picked["sid"])
    setup, target, name, row, latest = b["setup"], b["target"], picked["input"], b["row"], b["latest"]
    trades, daily = b["trades"], b["daily"]
    if trades.is_empty() or row is None:
        return note("This setup has no trades.", "warn"), default_view("trades")
    now = signal_now(b)
    firing = abs(now) > setup["entry"]
    z_now, resid_now = (float(b["signal"][k].drop_nulls().drop_nans()[-1]) for k in ("z", "resid"))
    beta = latest["beta_changes"]
    hedge_text = (f"P&L: {target} {'−' if beta >= 0 else '+'} {abs(beta):.2f} × {name} (changes β today; frozen "
                  f"per trade)" if row["hedged"] else f"P&L: {target} alone ({name} isn't tradable)")
    page, view = render_page("trades", default_view("trades"), path, by, min_trades, name, picked)
    reasons = (trades.group_by("reason").agg(pl.len().alias("trades"), pl.col("pnl").mean().alias("mean_pnl_bps"),
                                             (pl.col("pnl") > 0).mean().alias("hit_rate"),
                                             pl.col("bars").mean().alias("avg_held"))
               .sort("trades", descending=True))
    periods = b["years"].select("year", pl.col("pnl").alias("pnl_bps"), "active_days")

    card = html.Div(style={"border": f"1px solid {BORDER}", "background": PANEL, "padding": "14px 18px",
                           "marginBottom": 18}, children=[
        html.Div(style={"display": "flex", "alignItems": "baseline", "gap": 12, "marginBottom": 4}, children=[
            html.Span(f"{target} vs {name}", style={"fontSize": 15, "fontWeight": "bold", "color": ORANGE}),
            html.Span(describe(setup), style={"fontSize": 12, "color": DIM}),
            html.Span(hedge_text, style={"fontSize": 11, "color": DIM, "marginLeft": "auto", "fontStyle": "italic"}),
        ]),
        note(f"Signal: {BASIS_TEXT[setup['basis']]}.", "dim"),
        html.Div(style=STATS_ROW, children=[
            stat_block("reading", "short" if now > setup["entry"] else "long" if now < -setup["entry"] else "flat",
                       alert=firing),
            stat_block(f"now · {setup['basis']} z / resid", f"{z_now:+.2f} / {resid_now:+.1f}"),
            stat_block("levels β / changes β", f"{latest['beta_levels']:+.3f} / {beta:+.3f}"),
            stat_block("r²", _fmt(latest["r2"], ".2f")),
            stat_block("half-life", _fmt(float(b["signal"]["half_life"][-1] or np.nan), ".1f") + "d"),
            stat_block("126d corr", _fmt(latest["corr126"], "+.2f")),
            stat_block("halves split", str(row["split_date"]))]),
        html.Div(style={**STATS_ROW, "borderTop": f"1px solid {BORDER}", "marginTop": 4, "paddingBottom": 14},
                 children=[
            stat_block("sharpe", _fmt(row["sharpe"], ".2f")),
            stat_block("halves", f"{_fmt(row['sharpe_h1'], '.2f')} / {_fmt(row['sharpe_h2'], '.2f')}"),
            stat_block("total pnl", f"{row['total_pnl']:+,.0f} bps"),
            stat_block("per year", f"{row['pnl_per_year']:+,.1f} bps"),
            stat_block("n trades", f"{row['n_trades']} · {row['trades_per_year']:.1f}/yr"),
            stat_block("hit rate", f"{row['hit_rate']:.0%}"),
            stat_block("avg / median", f"{row['avg_trade']:+.2f} / {row['median_trade']:+.2f}"),
            stat_block("profit factor", _fmt(row["profit_factor"], ".2f")),
            stat_block("max drawdown", f"{row['max_drawdown']:+,.1f} bps"),
            stat_block("calmar", _fmt(row["calmar"], ".2f")),
            stat_block("time in market", f"{row['time_in_market']:.0%}"),
            stat_block("best trade share", _fmt(row["best_trade_share"], ".0%"))]),
        html.Div(window_row(window), style={"display": "flex", "gap": 8, "marginBottom": 8, "alignItems": "center",
                                            "flexWrap": "wrap"}),
        html.Div(charts_view(b, window), id="card-charts-oven",
                 style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": 10}),
        html.Div(style={"marginTop": 14}, children=[
            html.Div(page, id="oven-trades-page"),
            pager("trades", extra=[html.Button("Snap chart to view", id="oven-snap", n_clicks=0, className="ref-btn",
                                               style={**btn_style(), "padding": "2px 8px", "fontSize": 11})]),
        ]),
    ])
    mechanics = html.Div([
        heading("trade mechanics"),
        note(f"Median closed-trade P&L: {trades['pnl'].median():,.2f} bps · of total P&L, the target leg made "
             f"{b['with_levels']['target_leg'].sum():+,.1f} bps and the hedge leg "
             f"{b['with_levels']['hedge_leg'].sum():+,.1f} bps", "dim"),
        html.Div([
            html.Div(pnl_heatmap(daily.filter(pl.col("ts").dt.year() >= periods["year"].min())
                                 .select("ts", pl.col("daily_pnl").alias("pnl_bps")), periods),
                     style={"flex": "0 1 auto", "minWidth": 0}),
            html.Div(table(show(reasons), title="Exit reasons", float_fmt=",.2f"),
                     style={"flex": "0 1 auto", "minWidth": 0}),
        ], style={"display": "flex", "gap": 28, "flexWrap": "wrap", "alignItems": "flex-start"}),
    ], style={"marginBottom": 18})
    diagnostics = html.Div([
        heading("relationship diagnostics"),
        html.Div(scatter_window_row(setup["lookback"], "lookback"),
                 style={"display": "flex", "gap": 8, "marginBottom": 8, "alignItems": "center", "flexWrap": "wrap"}),
        html.Div(scatter_pair(b, "lookback"), id="oven-scatters", style=TWO_UP),
        html.Div(style={**TWO_UP, "marginTop": 12}, children=[
            table(show(lookback_grid(load_run(path)["board"], setup, by)),
                  title=f"{engine.RANKS[by]} · lookback × entry ({setup['scale']}, same exit)", float_fmt=",.2f"),
            table(show(b["years"]), title="by year", float_fmt=",.2f", sortable=True),
        ]),
        html.Div(style=TWO_UP, children=[
            table(show(feature_correlations(trades)), title="entry state vs trade P&L · spearman",
                  float_fmt=",.3f", sortable=True),
            table(show(peer_correlations(path, setup, by, min_trades)),
                  title="monthly P&L correlation with other inputs' best setups", float_fmt=",.3f", sortable=True),
        ]),
        table(show(engine.characteristics(trades, setup)),
              title="entry-state characteristics · terciles of each feature at entry", float_fmt=",.3f",
              sortable=True),
    ])
    return html.Div([card, mechanics, resolution_view(trades, setup), diagnostics]), view


def resolution_view(trades: pl.DataFrame, setup: dict) -> html.Div:
    """The multi-resolution section: does another horizon change what this one's trades are worth?"""
    table_rows = multi_resolution(trades, setup)
    if table_rows.is_empty():
        body = note("This bake has no other horizons recorded. Re-bake with more than one signal lookback to see "
                    "what the other horizons said at each entry.", "dim")
    else:
        scale = float(np.nanmax(np.abs(table_rows["vs_setup_avg"].fill_null(0).to_numpy()))) or 1.0
        body = table(show(table_rows), title=f"this setup's {len(trades)} trades, split by each other horizon's "
                                             f"{setup['basis']} z at entry", float_fmt=",.2f",
                     headers={"other_horizon": "other horizon", "z_corr_at_entry": "z corr at entry",
                              "vs_setup_avg": "vs setup avg", "avg_trade": "avg trade", "hit_rate": "hit",
                              "total_pnl": "total pnl"},
                     cell_style=lambda col, value, _row: (_diverging(value, scale) if col == "vs_setup_avg" and
                                                          value != "" else {"textAlign": "left"}
                                                          if col in ("other_horizon", "state") else None))
    return html.Div([
        heading("multi-resolution"),
        note(f"Same input, basis and hedge, read at the other signal lookbacks when this setup entered. 'agrees': "
             f"that horizon was also beyond ±{HORIZON_Z:g} z the same way (it saw the same rich/cheap); 'disagrees': "
             f"beyond it the other way. Green 'vs setup avg' means trades in that state beat the setup's average "
             f"trade, so that horizon carries information; watch the trade counts, small states are noise. z corr "
             f"near 1 means the horizons are the same signal measured twice.", "dim"),
        body,
    ], style={"marginBottom": 18})


def snap_range(path: str, picked: dict, view: dict):
    """The dashboard's Snap chart to view: the dates of the trades on the visible page."""
    shown, _ = _sorted_page(trades_rows(path, picked), "trades", view or default_view("trades"))
    dates = [*shown["entry"].to_list(), *shown["exit"].to_list()]
    if not dates:
        return None
    return pd.Timestamp(min(dates)) - pd.offsets.BDay(SNAP_LEFT_BUFFER_BDAYS), pd.Timestamp(max(dates))


# ---- tabs --------------------------------------------------------------------


SMALL_BTN = {**btn_style(), "padding": "1px 8px", "fontSize": 10}


def loader(child) -> dcc.Loading:
    """The dashboard's logo loader, the previous content dimmed underneath."""
    return dcc.Loading(child, custom_spinner=shimmer_loader(image="guy.png", caption="loading"),
                       overlay_style={"visibility": "visible", "opacity": 0.35},
                       parent_className="dashboard-overview-loader")


def _all_none(prefix: str) -> html.Div:
    """The small all / none buttons under one multi-select."""
    return html.Div(style={"display": "flex", "gap": 6, "marginTop": -6, "marginBottom": 12}, children=[
        html.Button("all", id=f"{prefix}-all", n_clicks=0, className="ref-btn", style=SMALL_BTN),
        html.Button("none", id=f"{prefix}-none", n_clicks=0, className="ref-btn", style=SMALL_BTN),
    ])


def _multi(key: str) -> html.Div:
    label, options, default = GRID_CONTROLS[key]
    return html.Div([field(label, dcc.Dropdown(id=f"oven-{key}", value=default, multi=True,
                                               options=[{"label": f"{v:g}", "value": v} for v in options])),
                     _all_none(f"oven-{key}")])


def bake_controls() -> html.Div:
    return html.Div([
        field("target", dcc.Dropdown(id="oven-target", value="10s30s", clearable=False, options=TARGET_OPTIONS)),
        field("inputs · default: everything in the panel", dcc.Dropdown(
            id="oven-inputs", value=ALL_INPUTS, multi=True,
            options=[{"label": f"{FEATURE_LABELS.get(n, n)} · {n}", "value": n} for n in ALL_INPUTS],
            style={"maxHeight": 200, "overflowY": "auto"})),
        _all_none("oven-inputs"),
        html.Div(style={"display": "flex", "gap": 8, "marginBottom": 12}, children=[
            html.Button("Select all", id="oven-select-all", n_clicks=0, className="ref-btn",
                        title="Every input and every parameter value", style={**btn_style(), "flex": 1}),
            html.Button("Reset to defaults", id="oven-defaults", n_clicks=0, className="ref-btn",
                        style={**btn_style(), "flex": 1}),
        ]),
        section("Signal & entries", [
            html.Div([field("signal basis · what the entry signal measures", dcc.Dropdown(
                id="oven-bases", value=list(engine.BASES), multi=True,
                options=[{"label": f"{b} · {BASIS_TEXT[b]}", "value": b} for b in engine.BASES])),
                _all_none("oven-bases")]),
            *[_multi(k) for k in ("lookbacks", "z_entries", "raw_entries")]]),
        section("Exits", [
            note("Combined per setup: a reversion exit, a half-life time stop, the stop and the max-hold cap. No "
                 "revert with a half-life multiple is a pure half-life time exit; neither is a pure max-hold exit.",
                 "dim"),
            *[_multi(k) for k in ("reverts", "hl_mults", "max_holds", "stops")]]),
        section("Hedge & costs", [
            _multi("hedge_lookbacks"),
            field("cost · bps per round trip", dcc.Input(id="oven-cost", type="number", value=engine.COST, min=0,
                                                         step=0.1, style=INPUT)),
        ], open=False),
        html.Div(id="oven-grid-size"),
        html.Button("Bake", id="oven-run", n_clicks=0, className="ref-btn",
                    style={**btn_style(primary=True), "width": "100%"}),
        html.Div(id="oven-run-note"),
    ])


def bake_tab() -> html.Div:
    return html.Div(style={"padding": "18px 24px"}, children=[
        html.Div([heading("oven"), note(
            "Bake one target against a boatload of inputs to find the relationships whose trades pay. For every "
            "input and lookback: the signal is the levels-OLS residual of the target on the input; the trade holds "
            "the target hedged by the input at the changes-OLS beta; and the model freezes at entry, so a trade "
            "exits only when prices move. Every setup is scored as a strategy: Sharpe, P&L, drawdown, hit rate, "
            "and both halves of the sample. Results open on the Results tab.", "dim")],
            className="research-banner", style={"display": "block"}),
        html.Div(style={"display": "grid", "gridTemplateColumns": "300px minmax(0, 1fr)", "gap": 26,
                        "alignItems": "start"}, children=[
            bake_controls(),
            loading_panel("oven-bake-out", "oven", "baking"),
        ]),
    ])


def results_tab() -> html.Div:
    runs = run_options()
    return html.Div(style={"padding": "14px 24px"}, children=[
        html.Div(style={"display": "grid", "gridTemplateColumns": "minmax(0, 520px) 220px 120px", "gap": 14},
                 children=[
            field("saved bake", dcc.Dropdown(id="oven-run-pick", options=runs, clearable=False,
                                              value=runs[0]["value"] if runs else None)),
            field("rank by", dcc.Dropdown(id="oven-rank", value="sharpe", clearable=False,
                                          options=[{"label": v, "value": k} for k, v in engine.RANKS.items()])),
            field("min trades", dcc.Input(id="oven-min-trades", type="number", value=20, min=1, debounce=True,
                                          style=INPUT)),
        ]),
        loader(html.Div(id="oven-results")),
        loader(html.Div(id="oven-setups")),
        loader(html.Div(id="oven-detail")),
        dcc.Store(id="oven-input"),
        dcc.Store(id="oven-setup"),
        dcc.Store(id="oven-window", data=DEFAULT_WINDOW),
        *[dcc.Store(id=f"oven-{name}-{kind}") for name in PAGED for kind in ("view", "sort")],
        *[dcc.Store(id=f"oven-{section_id}-scrolled") for section_id in ("setups", "detail")],
    ])


def tabs() -> html.Div:
    return html.Div([
        dcc.Store(id="oven-session", data=uuid4().hex),
        dcc.Store(id="oven-seen-job"),
        dcc.Interval(id="oven-poll", interval=750),
        dcc.Tabs(id="oven-tabs", value="bake", children=[
            dcc.Tab(label="Bake", value="bake", style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE,
                    children=bake_tab()),
            dcc.Tab(label="Results", value="results", style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE,
                    children=results_tab()),
        ]),
    ], className="research-bench oven-tight")


# ---- callbacks ---------------------------------------------------------------


def _clicked() -> bool:
    """A pattern-matched button really was clicked (re-rendered pickers fire with 0)."""
    return bool(ctx.triggered and ctx.triggered[0]["value"])


def register(app) -> None:
    choices = {"inputs": (ALL_INPUTS, ALL_INPUTS), "bases": (list(engine.BASES), list(engine.BASES)),
               **{k: (opts, default) for k, (_label, opts, default) in GRID_CONTROLS.items()}}
    for key, (options, default) in choices.items():
        # one callback per control: its own all / none, and the master Select all / Reset to defaults
        def _set(*_clicks, options=options, default=default, key=key):
            return {f"oven-{key}-all": list(options), "oven-select-all": list(options),
                    f"oven-{key}-none": [], "oven-defaults": list(default)}[ctx.triggered_id]

        app.callback(Output(f"oven-{key}", "value"),
                     Input(f"oven-{key}-all", "n_clicks"), Input(f"oven-{key}-none", "n_clicks"),
                     Input("oven-select-all", "n_clicks"), Input("oven-defaults", "n_clicks"),
                     prevent_initial_call=True)(_set)

    @app.callback(Output("oven-grid-size", "children"), Input("oven-target", "value"), Input("oven-inputs", "value"),
                  *[Input(f"oven-{k}", "value") for k in GRID_CONTROLS], Input("oven-bases", "value"))
    def _grid_size(target, inputs, lookbacks, z, raw, reverts, hl_mults, holds, stops, hedge_lbs, bases):
        bases = bases or []
        # the portfolio basis only runs for inputs that can be held as a hedge
        models = sum(("levels" in bases) + ("portfolio" in bases and engine.hedgeable(i))
                     for i in dict.fromkeys(inputs or []) if i != target)
        per_model = ((len(z or []) + len(raw or [])) * len(reverts or []) * len(holds or [])
                     * max(len(stops or []), 1) * max(len(hl_mults or []), 1))
        setups = models * len(lookbacks or []) * len(hedge_lbs or []) * per_model
        # measured on a select-everything 10s30s bake: 554,580 setups -> 39.2M trades, 18.8 GB peak, 23s
        trades = setups * 71
        ram_gb = trades / 1e6 * 0.48
        tone = "bad" if ram_gb > 40 else "warn" if ram_gb > 10 else "dim"
        text = f"{setups:,} setups · about {trades / 1e6:.1f}M trades · about {ram_gb:.0f} GB RAM while baking"
        if tone == "bad":
            text += " · too big for this machine; trim inputs or parameters"
        elif tone == "warn":
            text += " · big: the saved run and Results tab will be slower"
        return note(text, tone)

    @app.callback(Output("oven-run-note", "children"), Input("oven-run", "n_clicks"),
                  State("oven-target", "value"), State("oven-inputs", "value"),
                  *[State(f"oven-{k}", "value") for k in GRID_CONTROLS],
                  State("oven-cost", "value"), State("oven-bases", "value"),
                  prevent_initial_call=True)
    def _bake(_clicks, target, inputs, *grid):
        *lists, cost, bases = grid
        settings = dict(zip(GRID_CONTROLS, [sorted(v or []) for v in lists]), cost=float(cost or 0),
                        bases=[b for b in engine.BASES if b in (bases or [])])
        settings["stops"] = settings["stops"] or [0.0]
        settings["hl_mults"] = settings["hl_mults"] or [0.0]
        if not settings["bases"]:
            return note("Choose at least one signal basis.", "warn")
        inputs = [i for i in dict.fromkeys(inputs or []) if i != target]
        if not inputs:
            return note("Choose at least one input other than the target.", "warn")
        if not (settings["lookbacks"] and settings["hedge_lookbacks"] and (settings["z_entries"] or settings["raw_entries"])
                and settings["reverts"] and settings["max_holds"]):
            return note("The grid needs a signal lookback, a hedge lookback, an entry, a revert share and a max "
                        "hold.", "warn")
        try:
            jobs.submit("oven", f"bake {target} vs {len(inputs)} inputs", "oven.engine:oven_job",
                        target, inputs, settings)
        except jobs.JobRunning:
            return note("A bake is already running; its progress is shown on the right.", "warn")
        return note("Baking as a background job. It keeps running if you reload or leave this page; the result "
                    "opens on the Results tab when it finishes.", "dim")

    @app.callback(Output("oven-progress", "children"), Output("oven-run", "disabled"), Output("oven-run", "children"),
                  Input("oven-poll", "n_intervals"))
    def _progress(_tick):
        record = jobs.latest("oven")
        busy = bool(record) and record["status"] == "running"
        return (progress_view(jobs.progress_of("oven"), "baking") if record else "", busy,
                "Baking… (background job)" if busy else "Bake")

    @app.callback(Output("oven-bake-out", "children"), Output("oven-run-pick", "options"),
                  Output("oven-run-pick", "value"), Output("oven-tabs", "value"), Output("oven-seen-job", "data"),
                  Input("oven-poll", "n_intervals"), State("oven-seen-job", "data"), prevent_initial_call=True)
    def _finished(_tick, seen):
        record = jobs.latest("oven")
        if not record or record["status"] not in jobs.FINISHED or record["id"] == seen:
            return (no_update,) * 5
        if seen is None:  # first tick after a page load: remember what had already finished
            return no_update, no_update, no_update, no_update, record["id"]
        if record["status"] != "done":
            return (html.Div([note(record.get("error") or "The bake did not finish.", "bad"),
                              html.Pre(record.get("traceback") or "", style={"fontSize": 10, "whiteSpace": "pre-wrap"})]),
                    no_update, no_update, no_update, record["id"])
        path = record["result"]["run_path"]
        return (note(f"Done · {record['result']['summary']}. Opened on the Results tab.", "good"), run_options(), path,
                "results", record["id"])

    # Each step renders only on its own click, straight into its own section: the boards
    # never wait on a setup's charts, and nothing hangs off a chain of stores.
    @app.callback(Output("oven-results", "children"), Output("oven-setups", "children"),
                  Output("oven-detail", "children"), Output("oven-input", "data"), Output("oven-setup", "data"),
                  Output("oven-inputs-view", "data"),
                  Input("oven-run-pick", "value"), Input("oven-rank", "value"), Input("oven-min-trades", "value"))
    def _results(path, by, min_trades):
        if not path:
            return note("No bake yet: set one up on the Bake tab.", "dim"), "", "", None, None, None
        view, page_view = run_view(path, by, int(min_trades or 1))
        return view, "", "", None, None, page_view

    @app.callback(Output("oven-setups", "children", allow_duplicate=True),
                  Output("oven-detail", "children", allow_duplicate=True),
                  Output("oven-input", "data", allow_duplicate=True),
                  Output("oven-setups-view", "data", allow_duplicate=True),
                  Input({"type": "oven-input-pick", "index": ALL}, "n_clicks"),
                  State("oven-run-pick", "value"), State("oven-rank", "value"), State("oven-min-trades", "value"),
                  prevent_initial_call=True)
    def _pick_input(_clicks, path, by, min_trades):
        if not _clicked() or not path:
            return (no_update,) * 4
        name = ctx.triggered_id["index"]
        view, page_view = setups_view(path, name, by, int(min_trades or 1))
        return view, "", name, page_view

    @app.callback(Output("oven-detail", "children", allow_duplicate=True),
                  Output("oven-setup", "data", allow_duplicate=True),
                  Output("oven-trades-view", "data", allow_duplicate=True),
                  Output("oven-window", "data", allow_duplicate=True),
                  Input({"type": "oven-setup-pick", "index": ALL}, "n_clicks"),
                  State("oven-input", "data"), State("oven-run-pick", "value"), State("oven-rank", "value"),
                  State("oven-min-trades", "value"), State("oven-window", "data"), prevent_initial_call=True)
    def _pick_setup(_clicks, name, path, by, min_trades, window):
        if not _clicked() or not name or not path:
            return (no_update,) * 4
        picked = {"input": name, "sid": ctx.triggered_id["index"]}
        view, page_view = detail_view(path, picked, by, int(min_trades or 1), window or DEFAULT_WINDOW)
        return view, picked, page_view, window or DEFAULT_WINDOW

    # paging and server-side sorting, one callback per board
    for name in PAGED:
        def _page(*args, name=name):
            *_clicks, sort, view, path, by, min_trades, picked_input, picked_setup = args
            trigger = ctx.triggered_id
            if not path or (name != "inputs" and not picked_input) or (name == "trades" and not picked_setup):
                return no_update, no_update
            view = dict(view or default_view(name))
            if trigger == f"oven-{name}-sort":
                rule = (sort or {}).get("rule")
                if not rule:
                    return no_update, no_update
                view.update(desc=not view["desc"] if rule == view["sort"] else True, sort=rule, page=0)
            else:  # render_page clamps the page to what exists, so "last" can just overshoot
                button = trigger.rsplit("-", 1)[1]
                view["page"] = {"first": 0, "prev": view["page"] - 1, "next": view["page"] + 1,
                                "last": 10 ** 9}[button]
            return render_page(name, view, path, by, min_trades, picked_input, picked_setup)

        app.callback(Output(f"oven-{name}-page", "children"), Output(f"oven-{name}-view", "data", allow_duplicate=True),
                     *[Input(f"oven-{name}-{b}", "n_clicks") for b in ("first", "prev", "next", "last")],
                     Input(f"oven-{name}-sort", "data"), State(f"oven-{name}-view", "data"),
                     State("oven-run-pick", "value"), State("oven-rank", "value"), State("oven-min-trades", "value"),
                     State("oven-input", "data"), State("oven-setup", "data"), prevent_initial_call=True)(_page)

    # the scatters' fit window redraws only the two scatters
    @app.callback(Output("oven-scatters", "children"),
                  *[Output(f"oven-scatter-{k}", "style") for k in SCATTER_WINDOWS],
                  *[Input(f"oven-scatter-{k}", "n_clicks") for k in SCATTER_WINDOWS],
                  State("oven-setup", "data"), State("oven-run-pick", "value"), prevent_initial_call=True)
    def _scatter_window(*args):
        *clicks, picked, path = args
        if not any(clicks) or not picked or not path:
            return (no_update,) * (1 + len(SCATTER_WINDOWS))
        chosen = ctx.triggered_id.removeprefix("oven-scatter-")
        return (scatter_pair(bundle(path, picked["input"], picked["sid"]), chosen),
                *[btn_style(primary=(k == chosen)) for k in SCATTER_WINDOWS])

    # the Dig card's chart window and Snap chart to view redraw only the chart grid
    @app.callback(Output("card-charts-oven", "children", allow_duplicate=True), Output("oven-window", "data"),
                  *[Output(f"oven-window-{k}", "style") for k in WINDOW_PRESETS],
                  *[Input(f"oven-window-{k}", "n_clicks") for k in WINDOW_PRESETS],
                  State("oven-setup", "data"), State("oven-run-pick", "value"), prevent_initial_call=True)
    def _window(*args):
        *clicks, picked, path = args
        if not any(clicks) or not picked or not path:
            return (no_update,) * (2 + len(WINDOW_PRESETS))
        chosen = ctx.triggered_id.removeprefix("oven-window-")
        return (charts_view(bundle(path, picked["input"], picked["sid"]), chosen), chosen,
                *[btn_style(primary=(k == chosen)) for k in WINDOW_PRESETS])

    @app.callback(Output("card-charts-oven", "children", allow_duplicate=True),
                  *[Output(f"oven-window-{k}", "style", allow_duplicate=True) for k in WINDOW_PRESETS],
                  Input("oven-snap", "n_clicks"), State("oven-trades-view", "data"), State("oven-setup", "data"),
                  State("oven-run-pick", "value"), prevent_initial_call=True)
    def _snap(clicks, view, picked, path):
        if not clicks or not picked or not path:
            return (no_update,) * (1 + len(WINDOW_PRESETS))
        rng = snap_range(path, picked, view)
        if rng is None:
            return (no_update,) * (1 + len(WINDOW_PRESETS))
        return (charts_view(bundle(path, picked["input"], picked["sid"]), None, rng),
                *[btn_style() for _ in WINDOW_PRESETS])

    # bring a section into view when it fills in: it opens below a long board
    for section_id in ("setups", "detail"):
        app.clientside_callback(
            f"""function(children) {{
                if (children) setTimeout(() => document.getElementById('oven-{section_id}')
                    .scrollIntoView({{behavior: 'smooth', block: 'start'}}), 60);
                return window.dash_clientside.no_update;
            }}""",
            Output(f"oven-{section_id}-scrolled", "data"), Input(f"oven-{section_id}", "children"),
            prevent_initial_call=True)


def build_app():
    app = make_app(title="Oven", sliders=[], body=tabs)
    register(app)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Oven: bake signals and dig through them")
    parser.add_argument("--port", type=int, default=8054)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    run(build_app(), port=args.port, host=args.host)


if __name__ == "__main__":
    main()
