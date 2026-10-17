"""Oven app: bake one target against a boatload of inputs, then hunt for the setups
whose trades pay -- and pay consistently.

    python -m oven                              # http://127.0.0.1:8054
    python -m oven --host raptor --port 8054

Built from the research app's own parts -- its tabs, controls, house tables,
progress log and dashboard charts -- so it looks and reads the same. Two tabs:

  Bake     choose a target, inputs and grid; Bake runs as a background job
           (research.jobs), so a reload doesn't lose it. Every bake is saved
           under oven/data/.
  Results  the inputs board (one row per input: its best setups' strategy
           stats), then the chosen input's setups, then the chosen setup:
           strategy stats, charts (target with trades, the hedged portfolio,
           equity and drawdown, signal, betas and correlation, regression
           scatters, P&L by year, trade distribution) and tables (exits, years,
           lookback x entry Sharpe, entry-state characteristics and their
           correlation with P&L, P&L correlation with other inputs' best
           setups, every trade).

Each trade holds the portfolio y hedged by x (the changes beta, frozen at
entry); an input that isn't tradable trades y alone. The engine is
oven.engine; this module only lays it out.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from dashboard.charts import (WINDOW_PRESETS, level_chart, residual_bars_chart, return_distribution_chart,
                              series_chart)
from oven import engine
from research import jobs
from research.app import (FEATURE_GROUPS, FEATURE_LABELS, INPUT, SELECTED_TAB_STYLE, TAB_STYLE, TARGET_OPTIONS,
                          RESEARCH_CHART_MAX_WIDTH, _png_img, btn_style, field, heading, info, level_window_nav,
                          loading_panel, note, progress_view, regression_scatter, section, table)
from research.panel import CATALOG, build_panel
from utils.research_app import BORDER, C0, C1, TEXT, make_app, run, stat_block

ALL_INPUTS = list(dict.fromkeys(name for names in FEATURE_GROUPS.values() for name in names))
GRID_CONTROLS = {
    "lookbacks": ("signal lookbacks · levels OLS (d)", [21, 63, 126, 252, 504], engine.LOOKBACKS),
    "z_entries": ("entries · normalized z", [1.0, 1.5, 2.0, 2.5, 3.0], engine.Z_ENTRIES),
    "raw_entries": ("entries · raw bps", [3.0, 5.0, 10.0, 15.0, 20.0, 30.0], engine.RAW_ENTRIES),
    "reverts": ("exit · share of frozen residual recovered", [0.25, 0.5, 0.75, 1.0], engine.REVERTS),
    "max_holds": ("exit · max hold (bars)", [5, 10, 20, 40, 60, 120], engine.MAX_HOLDS),
    "stops": ("exit · stop, entry sigmas wider (0 = none)", [0.0, 0.5, 1.0, 2.0], engine.STOPS),
}
SETUP_COLS = ["lookback", "scale", "entry", "revert", "max_hold", "stop"]
STRATEGY_COLS = ["sharpe", "worst_half_sharpe", "sharpe_h1", "sharpe_h2", "total_pnl", "pnl_per_year", "n_trades",
                 "trades_per_year", "hit_rate", "avg_trade", "median_trade", "profit_factor", "max_drawdown",
                 "calmar", "time_in_market", "avg_bars", "trade_t", "best_trade_share", "worst_trade"]
DEFAULT_WINDOW = "All"
PICKER_ROWS = 60
SETUP_TABLE_ROWS = 150
ET = ZoneInfo("America/New_York")
STATS_ROW = {"display": "flex", "gap": 28, "flexWrap": "wrap", "marginBottom": 14, "paddingBottom": 12,
             "borderBottom": f"1px solid {BORDER}"}
TWO_UP = {"display": "grid", "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr)", "gap": 16,
          "alignItems": "start"}


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


@lru_cache(maxsize=4)
def load_run(path: str) -> dict:
    """A run's meta and board: all the boards need. Trades load only when a setup is opened."""
    p = Path(path)
    return dict(meta=json.loads((p / "meta.json").read_text(encoding="utf-8")),
                board=pl.read_parquet(p / "board.parquet"))


@lru_cache(maxsize=2)
def load_trades(path: str) -> pl.DataFrame:
    return pl.read_parquet(Path(path) / "trades.parquet")


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
    return (best.select("input", "score", *SETUP_COLS, "sharpe", "worst_half_sharpe", "total_pnl", "n_trades",
                        "hit_rate", "max_drawdown", "hedged")
            .join(counts, on="input")
            .with_columns(pl.col("input").replace_strict(FEATURE_LABELS, default=pl.col("input")).alias("label")))


def setup_id(row: dict) -> str:
    return "|".join(str(row[c]) for c in SETUP_COLS)


def setup_from_id(name: str, sid: str) -> dict:
    lb, scale, entry, revert, hold, stop = sid.split("|")
    return dict(input=name, lookback=int(lb), scale=scale, entry=float(entry), revert=float(revert),
                max_hold=int(hold), stop=float(stop))


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
        (pl.col("daily_pnl") != 0).mean().alias("time_in_market"))
    year_trades = trades.group_by(pl.col("exit_ts").dt.year().alias("year")).agg(
        pl.len().alias("trades_closed"), (pl.col("pnl") > 0).mean().alias("hit_rate"),
        pl.col("pnl").mean().alias("avg_trade"))
    return year_marks.join(year_trades, on="year", how="left").sort("year")


def monthly_pnl(trades: pl.DataFrame) -> pl.DataFrame:
    return trades.group_by(pl.col("exit_ts").dt.truncate("1mo").alias("month")).agg(pl.col("pnl").sum())


def peer_correlations(run: dict, setup: dict, by: str, min_trades: int, peers: int = 12) -> pl.DataFrame:
    """Monthly P&L correlation of this setup with the best setup of each other top input:
    does it add something to a book of signals, or is it the same bet?"""
    ranked = engine.rank(run["board"], by, min_trades, top=None)
    best = (ranked.filter(pl.col("input") != setup["input"])
            .unique("input", keep="first", maintain_order=True).head(peers))
    mine = monthly_pnl(matches(run["trades"], setup))
    rows = []
    for r in best.to_dicts():
        theirs = monthly_pnl(matches(run["trades"], {k: r[k] for k in engine.SETUP}))
        both = mine.join(theirs, on="month", how="full", coalesce=True, suffix="_peer").fill_null(0.0)
        rows.append(dict(peer=f"{r['input']} · lb {r['lookback']} · {r['scale']} > {r['entry']:g}",
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


def lookback_grid(board: pl.DataFrame, setup: dict, by: str) -> pl.DataFrame:
    """``by`` over lookback x entry for this input, holding the setup's scale, exit and stop."""
    fixed = {k: setup[k] for k in ("input", "scale", "revert", "max_hold", "stop")}
    return (matches(board, fixed).select("lookback", pl.format("entry {}", pl.col("entry")).alias("entry"), by)
            .pivot(on="entry", index="lookback", values=by).sort("lookback"))


# ---- views -------------------------------------------------------------------


def picker(rows: list[tuple[str, str]], kind: str, selected: str | None) -> html.Div:
    """One line per board row with an Inspect button -- the research board's row picker."""
    lines = [html.Div([
        html.Span(("▶ " if key == selected else "") + label,
                  style={"fontSize": 11, "fontFamily": "monospace", "color": TEXT, "whiteSpace": "pre",
                         "fontWeight": "bold" if key == selected else "normal"}),
        html.Button("Inspect", id={"type": kind, "index": key}, n_clicks=0, className="ref-btn",
                    style={**btn_style(primary=key == selected), "padding": "2px 10px", "fontSize": 11,
                           "marginLeft": "auto", "flexShrink": 0}),
    ], style={"display": "flex", "alignItems": "center", "gap": 8, "padding": "3px 0",
              "borderBottom": f"1px solid {BORDER}"}) for key, label in rows]
    return html.Div(lines, style={"maxHeight": 260, "overflowY": "auto", "marginTop": 10, "marginBottom": 18})


def run_view(run: dict, by: str, min_trades: int, selected_input: str | None) -> html.Div:
    meta = run["meta"]
    board = run["board"]
    eligible = board.filter(pl.col("n_trades") >= min_trades)
    inputs = input_board(board, by, min_trades)
    if inputs.is_empty():
        return note(f"No setup has {min_trades}+ trades. Lower the minimum trades.", "warn")
    sharpe = eligible["sharpe"].drop_nulls()
    shown = inputs.with_row_index("#", offset=1).select(
        "#", "input", "label", "score", "sharpe", "worst_half_sharpe", "total_pnl", "n_trades", "hit_rate",
        "max_drawdown", *SETUP_COLS, "setups", "median_sharpe", "sharpe_above_0.5", "both_halves_above_0.5",
        "hedged")
    quantiles = pl.DataFrame({"quantile": ["10%", "25%", "median", "75%", "90%", "99%", "max"],
                              "sharpe": [sharpe.quantile(q) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)] + [sharpe.max()]})
    return html.Div([
        html.Div(style=STATS_ROW, children=[
            stat_block("target", meta["target"]), stat_block("inputs", f"{len(meta['inputs'])}"),
            stat_block("setups baked", f"{len(board):,}"),
            stat_block(f"with ≥{min_trades} trades", f"{len(eligible):,}"),
            stat_block("Sharpe > 0.5", f"{(sharpe > 0.5).sum():,}"),
            stat_block("both halves > 0.5", f"{(eligible['worst_half_sharpe'] > 0.5).sum():,}"),
            stat_block("best Sharpe", f"{sharpe.max():.2f}" if len(sharpe) else "–"),
            stat_block("trades", f"{meta['n_trades']:,}"), stat_block("baked", baked_at(meta["created_at"]))]),
        info("How to read these boards",
             "Each setup is one input, signal lookback, entry scale and threshold, and exit. The signal is the "
             "levels-OLS residual of the target on the input. The trade holds the portfolio target hedged by the "
             "input (the changes-OLS beta, frozen at entry); an input that isn't tradable trades the target alone. "
             "The model freezes at entry, so a trade exits only when prices move.",
             "Sharpe is annualised from daily mark-to-market P&L, flat days included. Worst-half Sharpe is the lower "
             "of the setup's Sharpe in the first and second half of its own sample: a setup that only worked in one "
             "era scores low. Best-trade share is the largest trade's share of total P&L.",
             "Thousands of overlapping setups are searched, so the best of them look good partly by luck. Prefer "
             "setups whose neighbours (other lookbacks and entries for the same input) also pay, with both halves "
             "positive and no single trade carrying the total."),
        html.Div(style=TWO_UP, children=[
            table(quantiles.to_pandas(), title=f"Sharpe across the {len(sharpe):,} setups with ≥{min_trades} trades",
                  float_fmt=",.2f"),
            html.Div(),
        ]),
        table(shown.to_pandas(), title=f"inputs · each input's best setup by {engine.RANKS[by]}", float_fmt=",.2f",
              sortable=True, sorted_by=("score", "desc")),
        note("Inspect an input to see all its setups below.", "dim"),
        picker([(r["input"], f"#{i + 1:<4}{r['input']:<12} {engine.RANKS[by]} {r['score']:+.2f}  "
                             f"Sharpe {r['sharpe']:+.2f}  trades {r['n_trades']:>3}  {r['label']}")
                for i, r in enumerate(inputs.to_dicts())], "oven-input-pick", selected_input),
    ])


def setups_view(run: dict, name: str, by: str, min_trades: int, selected: str | None) -> tuple[html.Div, str | None]:
    ranked = engine.rank(run["board"].filter(pl.col("input") == name), by, min_trades, top=None)
    if ranked.is_empty():
        return note(f"{name} has no setup with {min_trades}+ trades.", "warn"), None
    total = len(ranked)
    # A full-grid bake gives an input thousands of setups; sending them all was 13 MB and
    # froze the page. Show the top rows by the chosen ranking -- re-ranking re-picks them.
    ranked = ranked.head(SETUP_TABLE_ROWS).with_columns(
        pl.struct(SETUP_COLS).map_elements(setup_id, return_dtype=pl.Utf8).alias("sid"))
    selected = selected if selected in set(ranked["sid"]) else None
    shown = ranked.with_row_index("#", offset=1).select("#", *SETUP_COLS, *STRATEGY_COLS)
    return html.Div([
        heading(f"setups · {FEATURE_LABELS.get(name, name)}"),
        table(shown.to_pandas(), title=f"top {len(ranked):,} of {total:,} setups with ≥{min_trades} trades · by "
                                       f"{engine.RANKS[by]}", float_fmt=",.2f", sortable=True,
              sorted_by=(by, "desc")),
        note(f"Inspect a setup for its trades, charts and diagnostics. Listing the top "
             f"{min(PICKER_ROWS, len(ranked))}.", "dim"),
        picker([(r["sid"], f"#{i + 1:<4}Sharpe {r['sharpe']:+.2f}  halves {r['sharpe_h1'] or 0:+.2f} / "
                           f"{r['sharpe_h2'] or 0:+.2f}  trades {r['n_trades']:>3}  lb {r['lookback']:>3}  "
                           f"{r['scale']} > {r['entry']:g}  revert {r['revert']:g}  hold {r['max_hold']}  "
                           f"stop {r['stop']:g}") for i, r in enumerate(ranked.head(PICKER_ROWS).to_dicts())],
               "oven-setup-pick", selected),
    ]), selected


def detail_view(run: dict, setup: dict, by: str, min_trades: int, window: str) -> html.Div:
    meta = run["meta"]
    target, name = meta["target"], setup["input"]
    trades = matches(run["trades"], setup).sort("entry_ts")
    row = matches(run["board"], setup).to_dicts()
    if trades.is_empty() or not row:
        return note("This setup has no trades.", "warn")
    row = row[0]
    cost = float(meta["settings"].get("cost", engine.COST))
    frame = model_frame(target, name, setup["lookback"],
                        int(meta["settings"].get("hedge_lookback", engine.HEDGE_LOOKBACK)))
    daily = daily_frame(frame, target, name, trades, cost)
    years = by_year(daily, trades)
    latest = frame.drop_nulls("beta_levels").tail(1).to_dicts()[0]
    bars = WINDOW_PRESETS[window]
    hedged = bool(row["hedged"])
    b = latest["beta_changes"]
    hedge_text = (f"{target} {'−' if b >= 0 else '+'} {abs(b):.2f} × {name} (changes beta now; frozen per trade)"
                  if hedged else f"{target} alone ({name} isn't tradable)")

    # trade markers in the dashboard's level_chart format
    levels = frame.select("ts", target)
    markers = (trades.join(levels.rename({"ts": "entry_ts", target: "entry_level"}), on="entry_ts", how="left")
               .join(levels.rename({"ts": "exit_ts", target: "exit_level"}), on="exit_ts", how="left")
               .select(pl.when(pl.col("side") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("direction"),
                       pl.col("entry_ts").cast(pl.Datetime).alias("entry_date"), "entry_level",
                       pl.col("exit_ts").cast(pl.Datetime).alias("exit_date"), "exit_level"))
    signal_col = "entry_z" if setup["scale"] == "z" else "resid"
    units = "z" if setup["scale"] == "z" else "bps"
    reasons = (trades.group_by("reason").agg(pl.len().alias("trades"), pl.col("bars").mean().alias("avg_bars"),
                                             pl.col("pnl").mean().alias("avg_trade"),
                                             (pl.col("pnl") > 0).mean().alias("hit_rate"),
                                             pl.col("pnl").sum().alias("total_pnl"))
               .sort("trades", descending=True))
    open_now = abs(latest["entry_z"] if setup["scale"] == "z" else latest["resid"]) > setup["entry"]

    def fmt(v, spec):
        return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)

    return html.Div([
        heading(f"{target} vs {name} · lb {setup['lookback']} · {setup['scale']} > {setup['entry']:g} · "
                f"revert {setup['revert']:g} · hold {setup['max_hold']} · stop {setup['stop']:g}"),
        note(f"Trades {hedge_text}. Sample halves split at {row['split_date']}.", "dim"),
        html.Div(style=STATS_ROW, children=[
            stat_block("Sharpe", fmt(row["sharpe"], ".2f")),
            stat_block("halves", f"{fmt(row['sharpe_h1'], '.2f')} / {fmt(row['sharpe_h2'], '.2f')}"),
            stat_block("total P&L", f"{row['total_pnl']:+,.0f} bps"),
            stat_block("per year", f"{row['pnl_per_year']:+,.1f} bps"),
            stat_block("trades", f"{row['n_trades']} · {row['trades_per_year']:.1f}/yr"),
            stat_block("hit rate", f"{row['hit_rate']:.0%}"),
            stat_block("avg / median trade", f"{row['avg_trade']:+.2f} / {row['median_trade']:+.2f}"),
            stat_block("profit factor", fmt(row["profit_factor"], ".2f")),
            stat_block("max drawdown", f"{row['max_drawdown']:,.1f} bps"),
            stat_block("Calmar", fmt(row["calmar"], ".2f")),
            stat_block("time in market", f"{row['time_in_market']:.0%}"),
            stat_block("best trade share", fmt(row["best_trade_share"], ".0%")),
            stat_block("now", [f"z {latest['entry_z']:+.2f} · resid {latest['resid']:+.1f}", html.Br(),
                               "signal firing" if open_now else "no signal"], alert=open_now)]),
        level_window_nav(window, prefix="oven-window"),
        html.Div(style={"maxWidth": RESEARCH_CHART_MAX_WIDTH}, children=[
            _png_img(level_chart(frame, target, trades=markers, features=[name], window_bars=bars, fig_height=4.2)),
            html.Div(style=TWO_UP, children=[
                _png_img(series_chart(daily, ["cumulative_pnl"], "cumulative P&L · daily marks", "bps",
                                      window_bars=bars, line_colors={"cumulative_pnl": C1})),
                _png_img(series_chart(daily, ["drawdown"], "drawdown", "bps", window_bars=bars,
                                      line_colors={"drawdown": C0})),
            ]),
            html.Div(style=TWO_UP, children=[
                _png_img(series_chart(frame, [signal_col],
                                      f"rolling levels residual ({units}) vs entry · trades freeze it at entry",
                                      units, hlines=[(setup["entry"], f"+{setup['entry']:g}"),
                                                     (-setup["entry"], f"-{setup['entry']:g}")], window_bars=bars)),
                _png_img(series_chart(frame, ["portfolio"], f"the portfolio · {hedge_text}", "bps",
                                      window_bars=bars)),
            ]),
            html.Div(style=TWO_UP, children=[
                _png_img(series_chart(frame, ["beta_levels", "beta_changes", "corr126"],
                                      "levels β (signal) · changes β (hedge) · 126d corr", "beta",
                                      left=["beta_levels", "beta_changes"], yaxis_right_title="corr",
                                      window_bars=bars)),
                _png_img(series_chart(frame, ["r2", "half_life"], "levels fit R² · residual half-life", "R²",
                                      left=["r2"], yaxis_right_title="half-life (bars)", window_bars=bars)),
            ]),
            html.Div(style=TWO_UP, children=[
                regression_scatter(frame, target, name, "levels", setup["lookback"], "previous"),
                regression_scatter(frame, target, name, "changes", setup["lookback"], "previous"),
            ]),
            html.Div(style=TWO_UP, children=[
                _png_img(residual_bars_chart([str(y) for y in years["year"]], years["pnl"].to_list(),
                                             "P&L by year", "bps")),
                _png_img(return_distribution_chart(trades.select(pl.col("pnl").alias("pnl_bps")))),
            ]),
        ]),
        html.Div(style=TWO_UP, children=[
            table(reasons.to_pandas(), title="exits", float_fmt=",.2f"),
            table(lookback_grid(run["board"], setup, by).to_pandas(),
                  title=f"{engine.RANKS[by]} · lookback × entry ({setup['scale']}, same exit)", float_fmt=",.2f"),
        ]),
        table(years.to_pandas(), title="by year", float_fmt=",.2f", sortable=True),
        html.Div(style=TWO_UP, children=[
            table(feature_correlations(trades).to_pandas(), title="entry state vs trade P&L · spearman",
                  float_fmt=",.3f", sortable=True),
            table(peer_correlations(run, setup, by, min_trades).to_pandas(),
                  title="monthly P&L correlation with other inputs' best setups", float_fmt=",.3f", sortable=True),
        ]),
        table(engine.characteristics(run["trades"], setup).to_pandas(),
              title="entry-state characteristics · terciles of each feature at entry", float_fmt=",.3f",
              sortable=True),
        table(trades.select("entry_ts", "exit_ts", "bars", "side", "reason", "pnl", "hedge_ratio", *engine.FEATURES)
              .sort("entry_ts", descending=True).to_pandas(), title=f"trades · {len(trades)}", float_fmt=",.2f",
              sortable=True),
    ])


# ---- tabs --------------------------------------------------------------------


SMALL_BTN = {**btn_style(), "padding": "1px 8px", "fontSize": 10}


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
        section("Signal & entries", [_multi(k) for k in ("lookbacks", "z_entries", "raw_entries")]),
        section("Exits", [_multi(k) for k in ("reverts", "max_holds", "stops")]),
        section("Hedge & costs", [
            field("hedge lookback · changes OLS (d)", dcc.Dropdown(
                id="oven-hedge-lb", value=engine.HEDGE_LOOKBACK, clearable=False,
                options=[{"label": str(v), "value": v} for v in (63, 126, 252)])),
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
    return html.Div(style={"padding": "18px 24px"}, children=[
        html.Div(style={"display": "grid", "gridTemplateColumns": "minmax(0, 2fr) 220px 140px", "gap": 16,
                        "maxWidth": RESEARCH_CHART_MAX_WIDTH}, children=[
            field("saved bake", dcc.Dropdown(id="oven-run-pick", options=runs, clearable=False,
                                              value=runs[0]["value"] if runs else None)),
            field("rank by", dcc.Dropdown(id="oven-rank", value="sharpe", clearable=False,
                                          options=[{"label": v, "value": k} for k, v in engine.RANKS.items()])),
            field("min trades", dcc.Input(id="oven-min-trades", type="number", value=20, min=1, debounce=True,
                                          style=INPUT)),
        ]),
        dcc.Loading(html.Div(id="oven-results"), type="dot"),
        dcc.Loading(html.Div(id="oven-setups"), type="dot"),
        dcc.Loading(html.Div(id="oven-detail"), type="dot"),
        dcc.Store(id="oven-input"),
        dcc.Store(id="oven-setup"),
        dcc.Store(id="oven-setups-scrolled"),
        dcc.Store(id="oven-detail-scrolled"),
        dcc.Store(id="oven-window", data=DEFAULT_WINDOW),
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
    ], className="research-bench")


# ---- callbacks ---------------------------------------------------------------


def _clicked() -> bool:
    """A pattern-matched button really was clicked (re-rendered pickers fire with 0)."""
    return bool(ctx.triggered and ctx.triggered[0]["value"])


def register(app) -> None:
    choices = {"inputs": (ALL_INPUTS, ALL_INPUTS),
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
                  *[Input(f"oven-{k}", "value") for k in GRID_CONTROLS])
    def _grid_size(target, inputs, lookbacks, z, raw, reverts, holds, stops):
        n_inputs = len([i for i in dict.fromkeys(inputs or []) if i != target])
        per_model = (len(z or []) + len(raw or [])) * len(reverts or []) * len(holds or []) * max(len(stops or []), 1)
        setups = n_inputs * len(lookbacks or []) * per_model
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
                  State("oven-hedge-lb", "value"), State("oven-cost", "value"), prevent_initial_call=True)
    def _bake(_clicks, target, inputs, *grid):
        *lists, hedge_lb, cost = grid
        settings = dict(zip(GRID_CONTROLS, [sorted(v or []) for v in lists]),
                        hedge_lookback=int(hedge_lb), cost=float(cost or 0))
        settings["stops"] = settings["stops"] or [0.0]
        inputs = [i for i in dict.fromkeys(inputs or []) if i != target]
        if not inputs:
            return note("Choose at least one input other than the target.", "warn")
        if not (settings["lookbacks"] and (settings["z_entries"] or settings["raw_entries"])
                and settings["reverts"] and settings["max_holds"]):
            return note("The grid needs a lookback, an entry, a revert share and a max hold.", "warn")
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
                  Input("oven-run-pick", "value"), Input("oven-rank", "value"), Input("oven-min-trades", "value"))
    def _results(path, by, min_trades):
        if not path:
            return note("No bake yet: set one up on the Bake tab.", "dim"), "", "", None, None
        return run_view(load_run(path), by, int(min_trades or 1), None), "", "", None, None

    @app.callback(Output("oven-setups", "children", allow_duplicate=True),
                  Output("oven-detail", "children", allow_duplicate=True),
                  Output("oven-input", "data", allow_duplicate=True),
                  Input({"type": "oven-input-pick", "index": ALL}, "n_clicks"),
                  State("oven-run-pick", "value"), State("oven-rank", "value"), State("oven-min-trades", "value"),
                  prevent_initial_call=True)
    def _pick_input(_clicks, path, by, min_trades):
        if not _clicked() or not path:
            return no_update, no_update, no_update
        name = ctx.triggered_id["index"]
        view, _ = setups_view(load_run(path), name, by, int(min_trades or 1), None)
        return view, "", name

    @app.callback(Output("oven-detail", "children", allow_duplicate=True),
                  Output("oven-setup", "data", allow_duplicate=True),
                  Input({"type": "oven-setup-pick", "index": ALL}, "n_clicks"),
                  State("oven-input", "data"), State("oven-run-pick", "value"), State("oven-rank", "value"),
                  State("oven-min-trades", "value"), State("oven-window", "data"), prevent_initial_call=True)
    def _pick_setup(_clicks, name, path, by, min_trades, window):
        if not _clicked() or not name or not path:
            return no_update, no_update
        picked = {"input": name, "sid": ctx.triggered_id["index"]}
        return _render_detail(path, picked, by, min_trades, window), picked

    @app.callback(Output("oven-window", "data"), Output("oven-detail", "children", allow_duplicate=True),
                  *[Input(f"oven-window-{k}", "n_clicks") for k in WINDOW_PRESETS],
                  State("oven-setup", "data"), State("oven-run-pick", "value"), State("oven-rank", "value"),
                  State("oven-min-trades", "value"), prevent_initial_call=True)
    def _window(*args):
        *clicks, picked, path, by, min_trades = args
        if not any(clicks) or not picked or not path:
            return no_update, no_update
        chosen = ctx.triggered_id.removeprefix("oven-window-")
        return chosen, _render_detail(path, picked, by, min_trades, chosen)

    # bring a section into view when it fills in: it opens below a long board
    for section_id in ("oven-setups", "oven-detail"):
        app.clientside_callback(
            f"""function(children) {{
                if (children) setTimeout(() => document.getElementById('{section_id}')
                    .scrollIntoView({{behavior: 'smooth', block: 'start'}}), 60);
                return window.dash_clientside.no_update;
            }}""",
            Output(f"{section_id}-scrolled", "data"), Input(section_id, "children"), prevent_initial_call=True)


def _render_detail(path: str, picked: dict, by: str, min_trades, window: str | None) -> html.Div:
    run = {**load_run(path), "trades": load_trades(path)}
    return detail_view(run, setup_from_id(picked["input"], picked["sid"]), by, int(min_trades or 1),
                       window or DEFAULT_WINDOW)


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
