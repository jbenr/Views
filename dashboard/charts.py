"""Static chart rendering for the live dashboard.

Reuses Viz's actual drawing code (colors, endpoint-value flags, hlines,
residual sign-fill) via a thin subclass that renders straight to a base64
PNG instead of PlotlyViz's own auto-refreshing server -- this dashboard
refreshes on button clicks, not a background loop, so it doesn't need its
own Dash server/registry, just the rendering.
"""

from __future__ import annotations

import base64
import functools
import threading
from io import BytesIO

import matplotlib

# Charts render inside Dash request threads. The interactive default (TkAgg on
# Windows) may only be touched from the thread that created it: drawing from a
# request thread and freeing Tk objects from another crashes the whole server
# ("Tcl_AsyncDelete: async handler deleted by the wrong thread"). Agg never
# opens a window, so choose it once, before pyplot loads.
matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from backtest.lab import parse_gate
from utils.research_app import C0, C1, C2, DIM, ORANGE
from utils.viz import Viz

# pyplot keeps one process-wide figure registry and is not thread-safe: two
# requests drawing at once can corrupt it. One chart renders at a time.
_RENDER_LOCK = threading.RLock()


def _one_at_a_time(render):
    @functools.wraps(render)
    def wrapper(*args, **kwargs):
        with _RENDER_LOCK:
            return render(*args, **kwargs)
    return wrapper


WINDOW_PRESETS = {"1M": 21, "3M": 63, "6M": 126, "YTD": "YTD", "1Y": 252, "2Y": 504, "5Y": 1260, "All": None}
DEFAULT_WINDOW = "2Y"


class _PngViz(Viz):
    """Viz, but _make_time_nav renders to a base64 PNG instead of a
    notebook widget / PlotlyViz's live-server registry."""

    def __init__(self, *args, fig_height: float | None = None, **kwargs):
        # No plt.switch_backend here: it closes every open figure, including
        # ones another request is still drawing. Agg is chosen at import.
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


@_one_at_a_time
def level_chart(
    data: pl.DataFrame,
    target: str,
    trades: pl.DataFrame | None = None,
    open_entry: dict | None = None,
    window_bars: int | None = WINDOW_PRESETS[DEFAULT_WINDOW],
    date_range: tuple | None = None,
    features: list[str] | None = None,
    invert_features: bool = False,
    fig_height: float | None = None,
) -> str:
    """Tradable level with optional research inputs.

    Inputs use the independently scaled left axis, while the tradable target
    stays on the dashboard's standard right-hand BPS axis. This lets a target
    and feature share time/plot space without one level scale flattening the
    other.
    """
    features = [feature for feature in (features or []) if feature != target]
    cols = [target, *features]
    frame = _pandas_indexed(data, cols)
    frame = _slice_window(frame, window_bars, date_range)
    display_features = features
    if invert_features and features:
        display_features = [f"-{feature}" for feature in features]
        frame = frame.rename(columns={
            feature: display for feature, display in zip(features, display_features)
        })
        frame[display_features] = -frame[display_features]
        cols = [target, *display_features]
    markers = _trade_markers(trades, open_entry, frame.index.min(), frame.index.max())
    palette = (C2, C1, C0, DIM)
    colors = {
        target: ORANGE,
        **{
            feature: palette[i % len(palette)]
            for i, feature in enumerate(display_features)
        },
    }
    feature_title = display_features[0] if len(display_features) == 1 else "features"
    return _PngViz(fig_height=fig_height).line(
        frame, cols=cols, title=target, yaxis_title="bps",
        yaxis_right_title=feature_title, left=display_features,
        markers=markers, line_colors=colors,
    )


@_one_at_a_time
def hedge_weights_chart(
    data: pl.DataFrame,
    weight_cols: list[str],
    fixed_priors: dict[str, float],
    window_bars: int | str | None = WINDOW_PRESETS[DEFAULT_WINDOW],
    fig_height: float | None = None,
) -> str:
    """Rolling hedge ratios in the same static treatment as a signal level.

    The fitted ratios share one beta scale, so unlike a target-plus-feature
    chart they deliberately use one right-hand axis. Fixed construction
    weights are quiet dotted reference lines, not additional live series.
    """
    frame = _pandas_indexed(data, weight_cols)
    frame = _slice_window(frame, window_bars, None)
    labels = {col: col.removeprefix("w_") for col in weight_cols}
    frame = frame.rename(columns=labels)
    cols = [labels[col] for col in weight_cols]
    hlines = [
        {
            "value": prior,
            "label": f"{labels[col]} fixed {prior:g}",
            "style": "dotted",
            "color": DIM,
            "alpha": 0.7,
        }
        for col, prior in fixed_priors.items()
    ]
    return _PngViz(fig_height=fig_height).line(
        frame,
        cols=cols,
        title="rolling betas vs fixed weights",
        yaxis_title="beta",
        hlines=hlines,
        line_colors={
            col: (C2, C1, C0, DIM)[i % 4]
            for i, col in enumerate(cols)
        },
    )


@_one_at_a_time
def coverage_chart(coverage: pl.DataFrame) -> str:
    """Series start/end ranges, drawn in the dashboard's static chart style."""
    rows = [
        row for row in coverage.iter_rows(named=True)
        if row["first_valid"] is not None and row["last_valid"] is not None
    ]
    fig, ax = plt.subplots(figsize=(9, max(2.4, 0.52 * len(rows) + 1.2)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")
    fig.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.22)

    for i, row in enumerate(rows):
        is_overlap = row["series"] == "OVERLAP"
        ax.hlines(
            i, pd.Timestamp(row["first_valid"]), pd.Timestamp(row["last_valid"]),
            color=ORANGE if is_overlap else C2,
            linewidth=7 if is_overlap else 5,
            alpha=1.0 if is_overlap else 0.75,
        )
    ax.set_yticks(range(len(rows)), [row["series"] for row in rows])
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax.grid(axis="y", color="#F0F0F0", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=9, colors="#333", length=4, width=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333")
        ax.spines[spine].set_linewidth(1.1)
    fig.suptitle("SERIES COVERAGE", fontsize=11, fontweight="bold",
                 color="#333", x=0.02, ha="left", y=0.98)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="white", edgecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@_one_at_a_time
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


@_one_at_a_time
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


@_one_at_a_time
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


@_one_at_a_time
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


@_one_at_a_time
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
        # Fixed one-basis-point buckets make the distribution directly
        # comparable across refreshes and strategies.
        lower = float(np.floor(values.min()))
        upper = float(np.ceil(values.max()))
        if upper <= lower:
            upper = lower + 1.0
        bins = np.arange(lower, upper + 1.0, 1.0)
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


@_one_at_a_time
def regression_scatter_chart(
    latest: tuple[np.ndarray, np.ndarray],
    past: tuple[np.ndarray, np.ndarray],
    fits: dict,
    now: tuple[float, float],
    *,
    title: str,
    x_title: str,
    y_title: str,
    labels: dict,
    zero_lines: bool = False,
) -> str:
    """Target against feature: the latest window and its fit, an older stretch in grey with its own.

    ``latest`` and ``past`` are (x, y) arrays (``past`` may be empty);
    ``fits`` holds {"latest": fit, "past": fit} from ``stats.fit_lr``;
    ``now`` is today's point; ``labels`` names the legend entries
    ("latest", "latest_fit", "past", "past_fit", "now").
    """
    viz = _PngViz()
    fig, ax = plt.subplots(figsize=(9, 6.2))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.04, right=0.92, top=0.90, bottom=0.16)

    def fit_line(fit, xs, **style):
        if xs.size and np.isfinite(fit["beta"]):
            span = np.array([xs.min(), xs.max()])
            ax.plot(span, fit["alpha"] + fit["beta"] * span, **style)

    if zero_lines:
        ax.axhline(0.0, color=DIM, linewidth=0.8, alpha=0.7)
        ax.axvline(0.0, color=DIM, linewidth=0.8, alpha=0.7)
    if past[0].size:
        ax.scatter(*past, s=10, color="#BDBDBD", alpha=0.55, linewidths=0, label=labels["past"], zorder=2)
        fit_line(fits["past"], past[0], color="#7F7F7F", linestyle="--", linewidth=1.4, label=labels["past_fit"],
                 zorder=3)
    ax.scatter(*latest, s=18, color=C2, alpha=0.8, linewidths=0, label=labels["latest"], zorder=4)
    fit_line(fits["latest"], np.r_[latest[0], past[0]] if labels.get("extend_latest") else latest[0],
             color=C2, linewidth=2.2, label=labels["latest_fit"], zorder=5)
    ax.scatter([now[0]], [now[1]], s=80, marker="D", color=ORANGE, edgecolors="white", linewidths=1.2,
               label=labels["now"], zorder=6)

    viz._style_ax(ax, yaxis_title=y_title, xaxis_title=x_title)
    ax.margins(0.04)  # _style_ax hugs the x edges for time series; points need room
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=5, fontsize=8, frameon=False,
              handlelength=1.6, handletextpad=0.4, columnspacing=1.0)
    fig.suptitle(title.upper(), fontsize=viz.TITLE_SIZE, fontweight="bold",
                 color="#333", x=0.02, ha="left", y=0.98)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="white", edgecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _png(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white", edgecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@_one_at_a_time
def curve_fit_chart(years: list[int], actual: list[float], fair: list[float], title: str) -> str:
    """Today's Treasury curve and the curve the PCA factors imply, by tenor."""
    viz = _PngViz()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.patch.set_facecolor("white")
    ax.plot(years, actual, color=ORANGE, linewidth=2.2, marker="o", markersize=7, label="actual", zorder=3)
    ax.plot(years, fair, color=C2, linewidth=1.8, linestyle="--", marker="o", markersize=5,
            markerfacecolor="white", label="PCA-implied", zorder=4)
    for x, y in zip(years, actual):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8,
                    color="#333")
    viz._style_ax(ax, yaxis_title="yield (bp)", xaxis_title="tenor (years)")
    ax.set_xticks(years)
    ax.margins(x=0.04, y=0.12)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, fontsize=9, frameon=False)
    fig.suptitle(title.upper(), fontsize=viz.TITLE_SIZE, fontweight="bold", color="#333", x=0.02, ha="left", y=0.98)
    return _png(fig)


@_one_at_a_time
def residual_bars_chart(tenors: list[str], values: list[float], title: str, unit: str,
                        bands: tuple[float, ...] = ()) -> str:
    """One bar per tenor, blue above zero and red below, labelled; optional ± reference lines."""
    viz = _PngViz()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.patch.set_facecolor("white")
    clean = [0.0 if v is None or not np.isfinite(v) else float(v) for v in values]
    bars = ax.bar(tenors, clean, color=[C2 if v >= 0 else C0 for v in clean], alpha=0.85, width=0.62, zorder=3)
    for bar, v in zip(bars, clean):
        ax.annotate(f"{v:+.2f}", (bar.get_x() + bar.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 4 if v >= 0 else -11), ha="center", fontsize=8, color="#333")
    ax.axhline(0.0, color="#333", linewidth=0.9)
    for band in bands:
        for sign in (1, -1):
            ax.axhline(sign * band, color=DIM, linestyle="--" if band >= 2 else ":", linewidth=0.9)
    viz._style_ax(ax, yaxis_title=unit)
    ax.margins(x=0.03, y=0.18)
    fig.suptitle(title.upper(), fontsize=viz.TITLE_SIZE, fontweight="bold", color="#333", x=0.02, ha="left", y=0.98)
    return _png(fig)


REGIME_COLOURS = (C2, "#BDBDBD", ORANGE)  # low side, middle, high side


@_one_at_a_time
def regime_timeline_chart(series: pl.DataFrame, states: list[str], title: str, units: str,
                          thresholds: tuple[float, ...] = (), window_bars: int | str | None = None) -> str:
    """A regime's value through time, with the background shaded by its confirmed state.

    ``window_bars`` zooms the chart (a bar count, "YTD" or None for all); the
    regime itself is always computed on its full history.
    """
    viz = _PngViz()
    frame = series.to_pandas().set_index("ts")
    frame.index = pd.to_datetime(frame.index)
    frame = _slice_window(frame, window_bars, None)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    fig.patch.set_facecolor("white")
    for state, colour in zip(states, REGIME_COLOURS):
        ax.fill_between(frame.index, 0, 1, where=(frame["state"] == state).to_numpy(), color=colour, alpha=0.22,
                        linewidth=0, transform=ax.get_xaxis_transform(), label=state, step="mid")
    ax.plot(frame.index, frame["value"], color="#333", linewidth=1.0)
    for level in thresholds:
        ax.axhline(level, color=DIM, linestyle="--", linewidth=0.9)
    viz._style_ax(ax, yaxis_title=units)
    viz._format_dates(ax, frame.index.min(), frame.index.max())
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=len(states), fontsize=9, frameon=False)
    fig.suptitle(title.upper(), fontsize=viz.TITLE_SIZE, fontweight="bold", color="#333", x=0.02, ha="left", y=0.98)
    return _png(fig)


@_one_at_a_time
def regime_gate_chart(series: pl.DataFrame, title: str, window_bars: int | str | None = None,
                      fig_height: float = 2.8) -> str:
    """When a macro regime gate is open: days the regime is in the gate's state, shaded, through time.

    ``series`` holds ``ts`` and a boolean ``gate_allow``. Same window
    handling as ``gate_chart`` (a bar count, "YTD", or None for all).
    """
    frame = _pandas_indexed(series.select("ts", pl.col("gate_allow").cast(pl.Float64)), ["gate_allow"])
    frame = _slice_window(frame, window_bars, None)
    viz = _PngViz(fig_height=fig_height)

    def render(fig, ax, start, end):
        subset = frame.loc[start:end, "gate_allow"].fillna(0.0)
        ax.fill_between(subset.index, 0.0, subset.to_numpy(), step="post", color=C1, alpha=0.55,
                        linewidth=0, label="gate open (regime in this state)")
        ax.fill_between(subset.index, 0.0, 1.0 - subset.to_numpy(), step="post", color=C0, alpha=0.12,
                        linewidth=0, label="gate closed")
        ax.set_ylim(0.0, 1.0)
        viz._style_ax(ax)
        ax.set_yticks([])
        viz._format_dates(ax, start, end)
        viz._legend(ax)
        ax.set_xlim(start, end)
        fig.subplots_adjust(bottom=0.3)

    return viz._make_time_nav(frame, render, title=title)
