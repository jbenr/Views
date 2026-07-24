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

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from utils.research_app import C0, C1, DIM
from utils.viz import Viz

WINDOW_PRESETS = {"3M": 63, "6M": 126, "YTD": "YTD", "1Y": 252, "2Y": 504, "5Y": 1260, "All": None}
DEFAULT_WINDOW = "2Y"


class _PngViz(Viz):
    """Viz, but _make_time_nav renders to a base64 PNG instead of a
    notebook widget / PlotlyViz's live-server registry."""

    def __init__(self, *args, **kwargs):
        plt.switch_backend("Agg")  # safe off the Dash callback thread
        super().__init__(*args, **kwargs)

    def _make_time_nav(self, df, render_fn, title=None, nrows=1,
                        height_ratios=None, fig_height=None):
        h = fig_height if fig_height is not None else (5.4 if nrows == 1 else 4.2 * nrows)
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
