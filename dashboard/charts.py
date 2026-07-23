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

from utils.viz import Viz

WINDOW_BARS = 500  # ~2 years of business days, plenty for a dashboard card


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


def level_chart(data: pl.DataFrame, target: str, window_bars: int = WINDOW_BARS) -> str:
    """Tradable level, recent window -- base64 PNG."""
    frame = _pandas_indexed(data, [target]).tail(window_bars)
    return _PngViz().line(frame, cols=[target], title=target, yaxis_title="bps")


def signal_chart(
    data: pl.DataFrame,
    sig_frame: pl.DataFrame,
    entry_signal: str,
    entry_threshold: float,
    window_bars: int = WINDOW_BARS,
) -> str:
    """Residual/OU-z chart with entry threshold bands and the current
    reading flagged -- base64 PNG."""
    col = "resid" if entry_signal == "residual" else "ou_z"
    combined = data.select("ts").with_columns(sig_frame[col].alias(col))
    frame = _pandas_indexed(combined, [col]).tail(window_bars)
    units = "bps" if entry_signal == "residual" else "z"
    return _PngViz().line(
        frame, cols=[col],
        title=f"{entry_signal} vs entry ({entry_threshold:g} {units})",
        yaxis_title=units,
        residual=True,
        hlines=[
            (entry_threshold, f"+{entry_threshold:g}"),
            (-entry_threshold, f"-{entry_threshold:g}"),
        ],
    )
