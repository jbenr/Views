"""Static chart rendering for the live dashboard.

Reuses Viz's actual drawing code (colors, endpoint-value flags, hlines,
residual sign-fill) via a thin subclass that renders straight to a base64
PNG instead of PlotlyViz's own auto-refreshing server -- this dashboard
refreshes on button clicks, not a background loop, so it doesn't need its
own Dash server/registry, just the rendering.
"""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from backtest.lab import parse_gate
from utils.research_app import C0, C1, C2, DIM
from utils.viz import Viz

WINDOW_PRESETS = {"1M": 21, "3M": 63, "6M": 126, "YTD": "YTD", "1Y": 252, "2Y": 504, "5Y": 1260, "All": None}
DEFAULT_WINDOW = "2Y"


class _PngViz(Viz):
    """Viz, but _make_time_nav renders to a base64 PNG instead of a
    notebook widget / PlotlyViz's live-server registry."""

    def __init__(self, *args, fig_height: float | None = None, **kwargs):
        plt.switch_backend("Agg")  # safe off the Dash callback thread
        super().__init__(*args, **kwargs)
        self.fig_height = fig_height

    def _make_time_nav(self, df, render_fn, title=None, nrows=1,
                        height_ratios=None, fig_height=None):
        h = (
            fig_height
            if fig_height is not None
            else self.fig_height
            if self.fig_height is not None
            else (5.4 if nrows == 1 else 4.2 * nrows)
        )
        fig, axes = plt.subplots(
            nrows, 1, figsize=(9, h), sharex=(nrows > 1),
            gridspec_kw={"height_ratios": height_ratios} if height_ratios else {},
        )
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.04, right=0.94, top=0.88, bottom=0.16)
        render_fn(fig, axes, df.index.min(), df.index.max())
        if title:
            fig.suptitle(title.upper(), fontsize=self.TITLE_SIZE, fontweight="bold",
                         color="#333", x=0.02, ha="left", y=0.98)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                    facecolor="white", edgecolor="white")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _pandas_indexed(data: pl.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = data.select(["ts", *cols]).to_pandas()
    return out.set_index("ts")


def _slice_window(
    frame: pd.DataFrame,
    window_bars: int | str | None,
    date_range: tuple | None,
) -> pd.DataFrame:
    """date_range (explicit start/end) wins over the window_bars tail preset.
    window_bars is a bar count, "YTD" (Jan 1 of the latest year to date), or
    None (all history)."""
    if date_range is not None:
        start, end = date_range
        return frame.loc[start:end]
    if window_bars == "YTD":
        end = frame.index.max()
        start = pd.Timestamp(year=end.year, month=1, day=1)
        return frame.loc[start:end]
    if window_bars is not None:
        return frame.tail(window_bars)
    return frame


def _trade_markers(
    trades: pl.DataFrame | None,
    open_entry: dict | None,
    start,
    end,
) -> list[dict]:
    """Entry/exit marker groups for level_chart, scoped to the visible window.
    Entry and exit are filtered independently so a trade whose entry is
    off-screen but exit is on-screen still shows its exit marker."""
    groups: list[dict] = []
    if trades is not None and not trades.is_empty():
        t = trades.to_pandas()
        longs = t[(t["direction"] == "long") & t["entry_date"].between(start, end)]
        shorts = t[(t["direction"] == "short") & t["entry_date"].between(start, end)]
        exits = t[t["exit_date"].between(start, end)]
        if not longs.empty:
            groups.append({"x": longs["entry_date"], "y": longs["entry_level"],
                            "label": "long entry", "color": C1, "marker": "^"})
        if not shorts.empty:
            groups.append({"x": shorts["entry_date"], "y": shorts["entry_level"],
                            "label": "short entry", "color": C0, "marker": "v"})
        if not exits.empty:
            groups.append({"x": exits["exit_date"], "y": exits["exit_level"],
                            "label": "exit", "color": DIM, "marker": "x", "size": 55})
    if open_entry and start <= pd.Timestamp(open_entry["date"]) <= end:
        color, marker = (C1, "^") if open_entry["direction"] == "long" else (C0, "v")
        groups.append({"x": [open_entry["date"]], "y": [open_entry["level"]],
                        "label": "open position", "color": color, "marker": marker,
                        "size": 110})
    return groups


def level_chart(
    data: pl.DataFrame,
    target: str,
    trades: pl.DataFrame | None = None,
    open_entry: dict | None = None,
    window_bars: int | None = WINDOW_PRESETS[DEFAULT_WINDOW],
    date_range: tuple | None = None,
) -> str:
    """Tradable level, recent window, with trade entry/exit markers -- base64 PNG."""
    frame = _pandas_indexed(data, [target])
    frame = _slice_window(frame, window_bars, date_range)
    markers = _trade_markers(trades, open_entry, frame.index.min(), frame.index.max())
    return _PngViz().line(frame, cols=[target], title=target, yaxis_title="bps",
                           markers=markers)


def input_chart(
    data: pl.DataFrame,
    feature: str,
    window_bars: int | None = WINDOW_PRESETS[DEFAULT_WINDOW],
    date_range: tuple | None = None,
) -> str:
    """The model's X/input series, paired beside the traded target chart.

    Inputs may be a market level or a derived feature such as a principal
    component, so the neutral ``level`` axis label is more honest than bps.
    Trade markers intentionally stay on the target chart: their prices are in
    target units and would be misleading on the X series.
    """
    frame = _pandas_indexed(data, [feature])
    frame = _slice_window(frame, window_bars, date_range)
    return _PngViz().line(
        frame,
        cols=[feature],
        title=f"input · {feature}",
        yaxis_title="level",
        line_colors={feature: C2},
    )


def signal_chart(
    data: pl.DataFrame,
    sig_frame: pl.DataFrame,
    entry_signal: str,
    entry_threshold: float,
    window_bars: int | None = WINDOW_PRESETS[DEFAULT_WINDOW],
    date_range: tuple | None = None,
    fired: str = "flat",
) -> str:
    """Residual/OU-z chart with entry threshold bands and the current
    reading flagged -- base64 PNG. Line is colored red while a sell (short)
    signal is firing, green while a buy (long) signal is firing."""
    col = "resid" if entry_signal == "residual" else "ou_z"
    combined = data.select("ts").with_columns(sig_frame[col].alias(col))
    frame = _pandas_indexed(combined, [col])
    frame = _slice_window(frame, window_bars, date_range)
    units = "bps" if entry_signal == "residual" else "z"
    line_colors = None
    if fired == "short":
        line_colors = {col: C0}
    elif fired == "long":
        line_colors = {col: C1}
    return _PngViz().line(
        frame, cols=[col],
        title=f"{entry_signal} vs entry ({entry_threshold:g} {units})",
        yaxis_title=units,
        residual=True,
        hlines=[
            (entry_threshold, f"+{entry_threshold:g}"),
            (-entry_threshold, f"-{entry_threshold:g}"),
        ],
        line_colors=line_colors,
    )


def _gate_bucket_description(kind: str, qs: tuple[float, ...]) -> str:
    pct = [round(q * 100) for q in qs]
    if kind == "below":
        return f"BELOW {pct[0]}TH PCT"
    if kind == "above":
        return f"ABOVE {pct[0]}TH PCT"
    if kind == "between":
        return f"BETWEEN {pct[0]}TH–{pct[1]}TH PCT"
    return f"OUTSIDE {pct[0]}TH–{pct[1]}TH PCT"


def gate_chart(
    data: pl.DataFrame,
    sig_frame: pl.DataFrame,
    gate_spec,
    window_bars: int | None = WINDOW_PRESETS[DEFAULT_WINDOW],
    date_range: tuple | None = None,
    gate_window: int | None = None,
) -> str | None:
    """Causal historical percentile of the promoted gate condition.

    The title reports the current gate state, the bucket rule, and what the
    percentile is measured against -- ``gate_window=None`` means every bar is
    ranked against all history to date, which is not evident from the curve.
    Threshold lines are the same percentile boundaries used by the strategy.
    """
    if gate_spec is None or "gate_percentile" not in sig_frame.columns:
        return None

    name, kind, qs = parse_gate(gate_spec)
    combined = data.select("ts").with_columns(
        (sig_frame["gate_percentile"] * 100.0).alias("historical percentile"),
        sig_frame["gate_allow"].alias("gate_allow"),
    )
    frame = _pandas_indexed(combined, ["historical percentile", "gate_allow"])
    frame = _slice_window(frame, window_bars, date_range)
    finite = frame["historical percentile"].dropna()
    allow = frame["gate_allow"].fillna(False).astype(bool)
    if finite.empty:
        state = "WARMING UP"
        current = ""
    else:
        is_open = bool(allow.loc[finite.index[-1]])
        state = "OPEN" if is_open else "CLOSED"
        current = f" @ {finite.iloc[-1]:.0f}TH PCT"

    basis = "expanding" if gate_window is None else f"roll {gate_window}d"
    title = (
        f"gate: {name} · {_gate_bucket_description(kind, qs)} ({basis}) · "
        f"{state}{current}"
    )
    viz = _PngViz()

    def render(fig, ax, start, end):
        subset = frame.loc[start:end]
        values = subset["historical percentile"].to_numpy(dtype=float)
        states = subset["gate_allow"].fillna(False).to_numpy(dtype=bool)
        x = mdates.date2num(subset.index.to_pydatetime())
        points = np.column_stack([x, values])
        valid = np.isfinite(values[:-1]) & np.isfinite(values[1:])
        segments = np.stack([points[:-1], points[1:]], axis=1)[valid]
        segment_states = states[1:][valid]
        if len(segments):
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=np.where(segment_states, C1, C0),
                    linewidths=1.6,
                    zorder=3,
                )
            )
        # Empty handles give the LineCollection a conventional dashboard legend.
        ax.plot([], [], color=C1, linewidth=1.6, label="gate open")
        ax.plot([], [], color=C0, linewidth=1.6, label="gate closed")
        for q in qs:
            pct = q * 100.0
            ax.axhline(
                pct,
                color=DIM,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"{round(pct)}th pct",
                zorder=2,
            )
        ax.set_ylim(0.0, 100.0)
        viz._style_ax(ax, yaxis_title="percentile")
        viz._format_dates(ax, start, end)
        viz._legend(ax)
        ax.set_xlim(start, end)
        fig.subplots_adjust(bottom=0.18)

    return viz._make_time_nav(frame, render, title=title)


def _window_pnl_frame(
    equity_curve: pl.DataFrame,
    window_bars: int | None = WINDOW_PRESETS[DEFAULT_WINDOW],
    date_range: tuple | None = None,
) -> pd.DataFrame:
    """Slice the exact equity curve and rebase the visible window to zero."""
    frame = _pandas_indexed(equity_curve, ["cumulative_pnl"])
    frame = _slice_window(frame, window_bars, date_range)
    finite = frame["cumulative_pnl"].dropna()
    if not finite.empty:
        frame = frame.copy()
        frame["cumulative_pnl"] -= finite.iloc[0]
    return frame


def pnl_chart(
    equity_curve: pl.DataFrame,
    window_bars: int | None = WINDOW_PRESETS[DEFAULT_WINDOW],
    date_range: tuple | None = None,
) -> str:
    """Exact-engine marked-to-market PnL, rebased at the visible window."""
    frame = _window_pnl_frame(equity_curve, window_bars, date_range)
    latest = frame["cumulative_pnl"].dropna()
    color = C1 if latest.empty or latest.iloc[-1] >= 0 else C0
    return _PngViz().line(
        frame,
        cols=["cumulative_pnl"],
        title="cumulative pnl · window reset",
        yaxis_title="bps",
        hlines=[
            {
                "value": 0.0,
                "style": "solid",
                "color": DIM,
                "alpha": 0.5,
            }
        ],
        line_colors={"cumulative_pnl": color},
    )


def return_distribution_chart(trades: pl.DataFrame | None) -> str:
    """Histogram of realized, net closed-trade returns in basis points.

    This is deliberately strategy-wide rather than windowed: the histogram is
    meant to show the shape of the strategy's realized return distribution,
    while the time-series panels above answer the recent-window question.
    """
    values = np.array([], dtype=float)
    if trades is not None and not trades.is_empty() and "pnl_bps" in trades.columns:
        values = trades["pnl_bps"].drop_nulls().to_numpy().astype(float)
        values = values[np.isfinite(values)]

    viz = _PngViz()
    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.84, bottom=0.20)

    if values.size:
        # Square-root scaling is stable for the modest trade counts typical of
        # these strategies, without the jumpy bin changes of a fully automatic
        # rule as one trade is added or removed.
        bins = max(8, min(20, int(np.ceil(np.sqrt(values.size) * 2))))
        _counts, edges, patches = ax.hist(
            values,
            bins=bins,
            edgecolor="white",
            linewidth=0.7,
        )
        for patch, left, right in zip(patches, edges[:-1], edges[1:]):
            patch.set_facecolor(C1 if (left + right) / 2 >= 0 else C0)
            patch.set_alpha(0.82)
        ax.axvline(0.0, color=DIM, linestyle="--", linewidth=1.0,
                   alpha=0.8, label="flat")
        mean = float(values.mean())
        ax.axvline(
            mean,
            color="#333",
            linestyle="-",
            linewidth=1.2,
            label=f"avg {mean:+.1f} bps",
        )
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        title = f"closed-trade return distribution · n={values.size}"
    else:
        ax.text(
            0.5, 0.5, "No closed trades yet",
            ha="center", va="center", transform=ax.transAxes,
            color=DIM, fontsize=10,
        )
        title = "closed-trade return distribution"

    viz._style_ax(ax, yaxis_title="trades", xaxis_title="net return (bps)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    fig.suptitle(title.upper(), fontsize=viz.TITLE_SIZE, fontweight="bold",
                 color="#333", x=0.02, ha="left", y=0.98)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="white", edgecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
