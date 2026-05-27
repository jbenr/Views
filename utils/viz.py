#!/usr/bin/env python3
"""
Visualization and time series analysis helpers for rates/fixed income work.
PrismFP/Bloomberg style formatting with interactive time navigation.

Usage:
    from utils.viz import Viz

    v = Viz()
    v.line(df, title='UST Yields', yaxis_title='Yield (%)')
    v.rolling_corr(df, col1='2Y', col2='10Y', window=60)
"""

from __future__ import annotations

import base64
import json
import pickle
import uuid
from io import BytesIO

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Polygon
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, NullLocator
import seaborn as sns

import ipywidgets as widgets
from IPython.display import HTML, display, clear_output


def _endpoint_flag(text: str, *, facecolor: str, textcolor: str, edgecolor: str):
    """Return a tight Bloomberg-style left-pointing endpoint flag."""
    fontprops = FontProperties(size=9, weight="bold")
    fig = plt.gcf()
    renderer = fig.canvas.get_renderer()
    text_w_px, text_h_px, _ = renderer.get_text_width_height_descent(
        text, fontprops, ismath=False,
    )
    dpi_scale = 72.0 / fig.dpi
    text_w = text_w_px * dpi_scale
    text_h = text_h_px * dpi_scale

    pad_x = 5.0
    pad_y = 3.5
    pointer_w = 5.0
    rect_w = text_w + 2 * pad_x
    rect_h = text_h + 2 * pad_y
    da = DrawingArea(pointer_w + rect_w, rect_h, 0, 0)
    da.add_artist(Polygon(
        [
            (0, rect_h / 2),
            (pointer_w, 0),
            (pointer_w + rect_w, 0),
            (pointer_w + rect_w, rect_h),
            (pointer_w, rect_h),
        ],
        closed=True,
        facecolor=facecolor, edgecolor=edgecolor, linewidth=0.8,
    ))
    da.add_artist(Text(
        pointer_w + rect_w / 2,
        rect_h / 2,
        text,
        color=textcolor,
        fontproperties=fontprops,
        ha="center",
        va="center",
    ))
    return da


def figure_copy_html(fig, dpi: int = 170, title: Optional[str] = None, title_size: int = 11) -> str:
    """Return a browser-side Copy chart button for a matplotlib figure."""
    copy_fig = fig
    if title:
        copy_fig = pickle.loads(pickle.dumps(fig))
        copy_fig.suptitle(
            title.upper(),
            fontsize=title_size,
            color="#333",
            x=0.01,
            ha="left",
            y=0.98,
        )
        copy_fig.subplots_adjust(top=0.88)
    buf = BytesIO()
    try:
        copy_fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="white",
            transparent=False,
        )
    finally:
        if copy_fig is not fig:
            plt.close(copy_fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    button_id = f"copy-chart-{uuid.uuid4().hex}"
    status_id = f"copy-chart-status-{uuid.uuid4().hex}"
    return f"""
<button id="{button_id}" style="height:28px;padding:0 10px;border:1px solid #999;border-radius:4px;background:#f7f7f7;cursor:pointer;">
  Copy chart
</button>
<span id="{status_id}" style="margin-left:8px;color:#555;font-size:12px;"></span>
<script>
document.getElementById({json.dumps(button_id)}).onclick = async () => {{
  const status = document.getElementById({json.dumps(status_id)});
  try {{
    const res = await fetch("data:image/png;base64,{encoded}");
    const blob = await res.blob();
    await navigator.clipboard.write([new ClipboardItem({{"image/png": blob}})]);
    status.textContent = "Copied to clipboard";
  }} catch (err) {{
    status.textContent = "Clipboard copy failed in this browser/kernel context";
    console.error(err);
  }}
}};
</script>
"""


class _SquareHandler:
    """Legend handler: square for solid lines, actual line sample for dashed/dotted/etc.

    Keeps the PrismFP-style colored square for the common solid case but renders
    a real linestyle sample for non-solid styles so legends are readable when a
    chart mixes dashed/dotted/solid series.
    """
    _SOLID = ('-', 'solid', None, '')

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D

        x0, y0 = handlebox.xdescent, handlebox.ydescent
        w, h = handlebox.width, handlebox.height

        ls = orig_handle.get_linestyle() if hasattr(orig_handle, 'get_linestyle') else '-'
        color = orig_handle.get_color() if hasattr(orig_handle, 'get_color') else 'black'

        if ls in self._SOLID:
            patch = Rectangle(
                (x0, y0), h, h, facecolor=color, edgecolor='none',
                transform=handlebox.get_transform(),
            )
            handlebox.add_artist(patch)
            return patch

        # Non-solid: render an actual line sample so dashed/dotted are identifiable
        lw = orig_handle.get_linewidth() if hasattr(orig_handle, 'get_linewidth') else 1.5
        line = Line2D(
            [x0, x0 + w], [y0 + h / 2.0, y0 + h / 2.0],
            color=color, linestyle=ls, linewidth=lw, solid_capstyle='butt',
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(line)
        return line


class Viz:
    """Time series visualization toolkit for rates analysis - PrismFP style."""

    # -------------------------------------------------------------------------
    # PrismFP style configuration
    # -------------------------------------------------------------------------

    COLORS = [
        '#D35400',  # Dark Orange (TU)
        '#F1C40F',  # Yellow/Gold (FV)
        '#27AE60',  # Green (TY)
        '#2980B9',  # Blue (US)
        '#8E44AD',  # Purple
        '#C0392B',  # Red
        '#16A085',  # Teal
        '#7F8C8D',  # Gray
    ]

    FONT_SIZE = 10
    TITLE_SIZE = 11
    BG_COLOR = '#F5F5F5'
    GRID_COLOR = '#FFFFFF'

    TIME_NAV_BUTTONS = [
        dict(count=1, label="1M", unit="month"),
        dict(count=3, label="3M", unit="month"),
        dict(count=6, label="6M", unit="month"),
        dict(label="YTD"),
        dict(count=1, label="1Y", unit="year"),
        dict(count=2, label="2Y", unit="year"),
        dict(count=5, label="5Y", unit="year"),
        dict(count=10, label="10Y", unit="year"),
        dict(label="ALL"),
    ]

    # backend aliases — Viz(backend=...) dispatches via __new__
    _BACKEND_MPL    = {None, "mpl", "matplotlib", "jupyter", "notebook"}
    _BACKEND_PLOTLY = {"plotly", "dash", "browser"}

    def __new__(cls, backend=None, **kwargs):
        """Dispatch on `backend`. Default is matplotlib (existing behavior).

        Viz()                     → matplotlib Viz (notebook auto-detected)
        Viz(backend='jupyter')    → same as above (alias)
        Viz(backend='plotly')     → PlotlyViz, serves a Dash app at http://127.0.0.1:8050
        """
        if cls is Viz and backend in cls._BACKEND_PLOTLY:
            return object.__new__(PlotlyViz)
        if cls is Viz and backend not in cls._BACKEND_MPL:
            raise ValueError(
                f"unknown backend {backend!r}; choose from "
                f"{sorted(b for b in cls._BACKEND_MPL if b)} or "
                f"{sorted(cls._BACKEND_PLOTLY)}"
            )
        return object.__new__(cls)

    def __init__(self, backend=None, **kwargs):
        # backend handled in __new__; accepted here so kwargs flow cleanly.
        self.colors = self.COLORS
        plt.ioff()  # prevent auto-display — we control display explicitly
        plt.rcdefaults()
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': self.FONT_SIZE,
            'figure.facecolor': 'white',
            'axes.facecolor': self.BG_COLOR,
            'axes.axisbelow': True,
            'axes.grid': True,
            'grid.color': '#DCDCDC',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.6,
            'toolbar': 'toolbar2',
        })

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    # Named line styles → (linestyle, linewidth)
    _LS_MAP = {
        "solid":  ("-",  1.5),
        "bold":   ("-",  2.5),
        "thin":   ("-",  0.8),
        "dashed": ("--", 1.0),
        "dash":   ("--", 1.0),
        "dotted": (":",  1.4),
        "dot":    (":",  1.4),
    }

    @classmethod
    def _resolve_style(cls, style, default="solid"):
        """Resolve a named line style to (linestyle, linewidth).
        style may be None, a string ('solid', 'dashed', 'dotted', 'bold', 'thin'),
        or a (linestyle, linewidth) tuple to pass through unchanged.
        """
        if style is None:
            return cls._LS_MAP[default]
        if isinstance(style, (tuple, list)) and len(style) == 2:
            return tuple(style)
        return cls._LS_MAP.get(style, cls._LS_MAP[default])

    def _format_legend_name(self, name: str, avg: float = None, unit: str = '') -> str:
        if avg is not None:
            return f"{name} = {avg:.1f} {unit}".strip()
        return name

    def _style_ax(self, ax, title=None, yaxis_title=None, xaxis_title=None):
        """Apply PrismFP styling — bold bottom+right spines, tick marks pointing
        out at every major label, lighter minor ticks between. Top/left spines
        hidden (the left one is re-enabled by callers using a secondary axis).
        """
        if title:
            ax.set_title(title.upper(), fontsize=self.TITLE_SIZE, color='#333',
                         pad=10, loc='left')
        if yaxis_title:
            ax.set_ylabel(yaxis_title.upper(), fontsize=self.FONT_SIZE, color='#333')
        if xaxis_title:
            ax.set_xlabel(xaxis_title.upper(), fontsize=self.FONT_SIZE, color='#333')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=9))

        # Minor ticks between every pair of major ticks — fills out the axis.
        # Date axes will get a date-aware minor locator in _format_dates; the
        # AutoMinorLocator below is right for numeric (y) and harmless for x
        # since _format_dates overrides it.
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(NullLocator())

        # Tick spikes: bold major, lighter minor — both point outward.
        ax.tick_params(axis='both', which='major', labelsize=9, colors='#333',
                       direction='out', length=5, width=1.0)
        ax.tick_params(axis='both', which='minor', colors='#333',
                       direction='out', length=3, width=0.6)

        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        for side in ('bottom', 'right'):
            ax.spines[side].set_color('#333')
            ax.spines[side].set_linewidth(1.2)
        ax.margins(x=0)  # line hugs left and right edges of the plot area

    def _format_dates(self, ax, start=None, end=None):
        """Auto-format date axis with tighter label spacing."""
        if start is not None and end is not None:
            span = (end - start).days
            if span <= 45:
                minticks, maxticks = 12, 42   # daily-ish ticks for 1M
            elif span <= 180:
                minticks, maxticks = 10, 28
            else:
                minticks, maxticks = 14, 30
        else:
            minticks, maxticks = 14, 30
        locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.get_offset_text().set_visible(False)

    def _legend(self, ax):
        """Legend below the chart, left-aligned, with colored squares."""
        leg = ax.legend(
            loc='upper left', bbox_to_anchor=(0, -0.08), ncol=4,
            fontsize=9, frameon=False, handlelength=2.0, handleheight=1,
            handler_map={plt.Line2D: _SquareHandler()},
        )

    # -------------------------------------------------------------------------
    # Interactive time navigation
    # -------------------------------------------------------------------------

    @staticmethod
    def _in_notebook() -> bool:
        try:
            from IPython import get_ipython
            shell = get_ipython()
            return shell is not None and 'zmqshell' in type(shell).__module__
        except Exception:
            return False

    def _make_time_nav(self, df, render_fn, title=None, nrows=1, height_ratios=None):
        """Date pickers + range buttons wired to render_fn(fig, ax_or_axes, start, end).

        When nrows=1 (default): render_fn receives a single Axes as second arg.
        When nrows>1: render_fn receives a numpy array of Axes (sharex=True).
        Falls back to plt.show() outside notebooks.
        """
        kw = dict(sharex=(nrows > 1), gridspec_kw={'height_ratios': height_ratios} if height_ratios else {})

        def _make_fig():
            h = 5 if nrows == 1 else 4 * nrows
            _fig, _axes = plt.subplots(nrows, 1, figsize=(12, h), **kw)
            _fig.patch.set_facecolor('white')
            if nrows == 1:
                return _fig, _axes, [_axes]   # (fig, axes_arg, original_list)
            return _fig, _axes, list(_axes)

        def _static_show():
            _fig, axes_arg, _ = _make_fig()
            render_fn(_fig, axes_arg, df.index.min(), df.index.max())
            plt.show()
            return _fig

        if not self._in_notebook():
            return _static_show()

        _updating = False

        start_picker = widgets.DatePicker(
            description='Start:',
            value=df.index.min().date(),
            layout=widgets.Layout(width='220px'),
        )
        end_picker = widgets.DatePicker(
            description='End:',
            value=df.index.max().date(),
            layout=widgets.Layout(width='220px'),
        )

        btn_layout = widgets.Layout(width='52px', height='28px', padding='0px')
        btn_widgets = []
        for b_def in self.TIME_NAV_BUTTONS:
            b = widgets.Button(description=b_def['label'], layout=btn_layout)
            btn_widgets.append(b)
        btn_widgets[-1].button_style = 'warning'  # ALL starts active

        fig, axes_arg, original_axes = _make_fig()
        plt.close(fig)

        chart_widget = widgets.Output()
        copy_widget = widgets.Output(layout=widgets.Layout(width='260px', height='32px'))
        active_window = {'label': 'ALL Window'}

        def _copy_title() -> Optional[str]:
            if not title:
                return None
            return f"{title} ({active_window['label']})" if active_window.get('label') else title

        def update_range(start, end):
            _refresh_title()
            for extra in [a for a in fig.axes if a not in original_axes]:
                extra.remove()
            for a in original_axes:
                a.clear()
            render_fn(fig, axes_arg, start, end)
            with chart_widget:
                clear_output(wait=True)
                display(fig)
            with copy_widget:
                clear_output(wait=True)
                display(HTML(figure_copy_html(fig, title=_copy_title(), title_size=self.TITLE_SIZE)))

        def on_btn_click(b):
            nonlocal _updating
            if _updating:
                return
            _updating = True

            for btn in btn_widgets:
                btn.button_style = ''
            b.button_style = 'warning'

            end = df.index.max()
            label = b.description
            active_window['label'] = f'{label} Window'
            if label == 'ALL':
                start = df.index.min()
            elif label == 'YTD':
                start = pd.Timestamp(end.year, 1, 1)
            elif label.endswith('M'):
                start = end - pd.DateOffset(months=int(label[:-1]))
            else:
                start = end - pd.DateOffset(years=int(label[:-1]))
            start = max(start, df.index.min())

            start_picker.value = start.date()
            end_picker.value = end.date()
            _updating = False
            update_range(pd.Timestamp(start.date()), pd.Timestamp(end.date()))

        def on_date_change(change):
            nonlocal _updating
            if _updating:
                return
            if not start_picker.value or not end_picker.value:
                return
            for btn in btn_widgets:
                btn.button_style = ''
            active_window['label'] = f"{start_picker.value} to {end_picker.value}"
            update_range(pd.Timestamp(start_picker.value),
                         pd.Timestamp(end_picker.value))

        for b in btn_widgets:
            b.on_click(on_btn_click)
        start_picker.observe(on_date_change, names='value')
        end_picker.observe(on_date_change, names='value')

        btn_row = widgets.HBox(btn_widgets, layout=widgets.Layout(gap='3px'))
        title_widget = widgets.HTML(
            value=f'<b style="font-size:13px; color:#333;">{_copy_title().upper()}</b>' if title else ''
        )
        title_spacer = widgets.Box(layout=widgets.Layout(flex='1 1 auto'))
        title_row = widgets.HBox(
            [title_widget, title_spacer, copy_widget],
            layout=widgets.Layout(width='100%', align_items='center', margin='0 0 4px 0'),
        )
        date_controls = widgets.HBox(
            [start_picker, end_picker],
            layout=widgets.Layout(gap='8px', align_items='center', flex_flow='row wrap'),
        )
        controls = widgets.HBox(
            [date_controls, btn_row],
            layout=widgets.Layout(gap='8px', align_items='center', flex_flow='row wrap', margin='0 0 6px 0'),
        )
        container = widgets.VBox([title_row, controls, chart_widget])

        def _refresh_title():
            if title:
                title_widget.value = f'<b style="font-size:13px; color:#333;">{_copy_title().upper()}</b>'

        # initial render
        update_range(df.index.min(), df.index.max())

        # Display now so multiple v.line() calls in one cell all render.
        # Jupyter only auto-displays the last expression in a cell, so without
        # this only the last chart would show. Return None so Jupyter doesn't
        # also auto-display the container as the last expression (would duplicate).
        display(container)
        return None

    # -------------------------------------------------------------------------
    # Line chart (interactive)
    # -------------------------------------------------------------------------

    def line(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        yaxis_right_title: Optional[str] = None,
        left: Optional[List[str]] = None,
        show_avg: bool = False,
        avg_unit: str = '',
        interval: str = None,
        show_endpoint_marker: bool = True,
        residual: bool = False,
        nas: bool = True,
        hlines: Optional[list] = None,
        linestyles: Optional[dict] = None,
        bar: bool = False,
    ):
        """Line chart with interactive time navigation.

        left : list of column names to plot on secondary (left) y-axis.
               Primary axis stays on the right per PrismFP style.
        linestyles : optional dict {col_name: style} for per-series line style.
               style is one of 'solid' (default), 'bold', 'thin', 'dashed', 'dotted'
               or a (linestyle, linewidth) tuple.
        hlines : optional list of horizontal reference lines. Each item may be:
               - a number (drawn dashed grey, no legend label)
               - a (value, label) tuple (drawn dashed grey, labeled in legend)
               - a (value, label, style) tuple
               - a dict {value, label, style, color, alpha}
               Examples:
                   hlines=[2.0, -2.0]                                # ±2 z-thresholds
                   hlines=[(442, '60d max', 'dashed')]
                   hlines=[{'value': 0, 'style': 'solid', 'color': '#666'}]
        bar : if True, render as bars colored green (positive) / red (negative).
              Best for daily changes, PnL, residuals where sign is the story.
              Multi-series with bar=True bars are stacked side-by-side per date.
        """
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        left = left or []
        ls_map = linestyles or {}
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        def render(fig, ax, start, end):
            subset = df.loc[start:end, cols]
            x_end = end
            if show_endpoint_marker:
                span = end - start
                if span > pd.Timedelta(0):
                    # Keep a modest gap between the final observation and y-axis.
                    x_end = end + span * 0.0225
            ax2 = None
            if left:
                ax2 = ax.twinx()
                ax2.grid(False)

            for i, col in enumerate(cols):
                color = self.colors[i % len(self.colors)]
                target = ax2 if col in left else ax
                series = subset[col].dropna()
                if series.empty:
                    continue
                avg = series.mean() if show_avg else None
                label = self._format_legend_name(col, avg, avg_unit) if show_avg else col
                ls, lw = self._resolve_style(ls_map.get(col), default="solid")

                if not nas:
                    # dotted connectors across weekend/holiday gaps
                    for j in range(len(series) - 1):
                        if (series.index[j + 1] - series.index[j]).days > 1:
                            target.plot(
                                [series.index[j], series.index[j + 1]],
                                [series.iloc[j], series.iloc[j + 1]],
                                color=color, linewidth=1.2, linestyle=':', alpha=0.6, zorder=2,
                            )
                    cal_idx = pd.date_range(series.index[0], series.index[-1], freq='D')
                    plot_series = series.reindex(cal_idx)
                else:
                    plot_series = series

                if bar:
                    # Bars colored by sign: green positive, red negative
                    bar_colors = [
                        '#27AE60' if val >= 0 else '#C0392B'
                        for val in plot_series.values
                    ]
                    target.bar(
                        plot_series.index, plot_series.values,
                        color=bar_colors, width=1.0, linewidth=0,
                        zorder=3, label=label,
                    )
                else:
                    target.plot(plot_series.index, plot_series, color=color,
                                linewidth=lw, linestyle=ls, label=label, zorder=3)

                if residual and not bar and len(cols) == 1:
                    target.axhline(y=0, color='#666', linestyle=':', linewidth=1)
                    target.fill_between(
                        plot_series.index, 0, plot_series,
                        where=plot_series >= 0,
                        interpolate=True, color='#27AE60', alpha=0.15,
                    )
                    target.fill_between(
                        plot_series.index, 0, plot_series,
                        where=plot_series < 0,
                        interpolate=True, color='#C0392B', alpha=0.15,
                    )

                if show_endpoint_marker and not bar and len(series) > 0:
                    last_val = series.iloc[-1]
                    endpoint_face = color
                    endpoint_text = '#FFFFFF'
                    # Custom geometry keeps the pointer flush with the label body.
                    flag = AnnotationBbox(
                        _endpoint_flag(
                            f'{last_val:.2f}',
                            facecolor=endpoint_face,
                            textcolor=endpoint_text,
                            edgecolor=endpoint_face,
                        ),
                        (x_end, last_val),
                        xybox=(0, 0),
                        xycoords='data',
                        boxcoords='offset points',
                        box_alignment=(0, 0.5),
                        frameon=False,
                        pad=0,
                        annotation_clip=False,
                        zorder=20,
                    )
                    target.add_artist(flag)

                if show_avg and avg is not None:
                    target.axhline(y=avg, color=color, linestyle='--', linewidth=1, alpha=0.7)

            # Horizontal reference lines (z-thresholds, level markers, etc.)
            if hlines:
                for h in hlines:
                    if isinstance(h, dict):
                        value = h["value"]
                        h_label = h.get("label")
                        h_style = h.get("style", "dashed")
                        h_color = h.get("color", "#666")
                        h_alpha = h.get("alpha", 0.7)
                    elif isinstance(h, (tuple, list)):
                        value = h[0]
                        h_label = h[1] if len(h) > 1 else None
                        h_style = h[2] if len(h) > 2 else "dashed"
                        h_color = h[3] if len(h) > 3 else "#666"
                        h_alpha = 0.7
                    else:
                        value, h_label, h_style, h_color, h_alpha = float(h), None, "dashed", "#666", 0.7
                    h_ls, h_lw = self._resolve_style(h_style, default="dashed")
                    ax.axhline(
                        y=value, color=h_color, linestyle=h_ls, linewidth=h_lw,
                        alpha=h_alpha, zorder=2,
                        label=h_label if h_label else '_nolegend_',
                    )

            self._style_ax(ax, yaxis_title=yaxis_title)
            if left:
                # primary axis (right side per PrismFP), secondary on left
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position('right')
                ax2.yaxis.tick_left()
                ax2.yaxis.set_label_position('left')
                if yaxis_right_title:
                    ax2.set_ylabel(yaxis_right_title.upper(), fontsize=self.FONT_SIZE, color='#333')
                # Make ax2's left spine the visible bold one; hide its other spines
                # so it doesn't double-draw the bottom/right with ax.
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['bottom'].set_visible(False)
                ax2.spines['left'].set_visible(True)
                ax2.spines['left'].set_color('#333')
                ax2.spines['left'].set_linewidth(1.2)
                ax2.yaxis.set_major_locator(MaxNLocator(nbins=9))
                ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax2.tick_params(axis='y', which='major', labelsize=9, colors='#333',
                                direction='out', length=5, width=1.0)
                ax2.tick_params(axis='y', which='minor', colors='#333',
                                direction='out', length=3, width=0.6)
                ax2.margins(x=0)
                # merge legends from both axes
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(
                    h1 + h2, l1 + l2,
                    loc='upper left', bbox_to_anchor=(0, -0.08), ncol=4,
                    fontsize=9, frameon=False, handlelength=2.0, handleheight=1,
                    handler_map={plt.Line2D: _SquareHandler()},
                )
            else:
                self._legend(ax)
            self._format_dates(ax, start, end)
            ax.set_xlim(start, x_end)
            fig.subplots_adjust(bottom=0.15)

        return self._make_time_nav(df, render, title=title)

    # -------------------------------------------------------------------------
    # Centered Date Chart (static)
    # -------------------------------------------------------------------------

    def centered_chart(
        self,
        df: pd.DataFrame,
        center_date: Union[str, datetime],
        window: int = 30,
        cols: Optional[List[str]] = None,
        normalize: str = 'level',
        title: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        show_avg: bool = False,
    ):
        """Plot time series centered around a specific date."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        center_date = pd.to_datetime(center_date)
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        start = center_date - timedelta(days=window)
        end = center_date + timedelta(days=window)
        data = df.loc[(df.index >= start) & (df.index <= end), cols].copy()

        if data.empty:
            raise ValueError(f"No data in window around {center_date}")

        if center_date not in data.index:
            idx = data.index.get_indexer([center_date], method='nearest')[0]
            center_date = data.index[idx]

        center_values = data.loc[center_date]

        if normalize == 'level':
            data = data - center_values
        elif normalize == 'change':
            data = data.diff().cumsum()
            data = data - data.loc[center_date]
        elif normalize == 'pct':
            data = (data / center_values - 1) * 100

        data['days'] = (data.index - center_date).days
        data = data.set_index('days')

        fig, ax = plt.subplots(figsize=(12, 5))

        for i, col in enumerate(cols):
            color = self.colors[i % len(self.colors)]
            avg = data[col].mean() if show_avg else None
            label = self._format_legend_name(col, avg, 'bps') if show_avg else col
            ax.plot(data.index, data[col], color=color, linewidth=1.5, label=label)

            if show_avg and avg is not None:
                ax.axhline(y=avg, color=color, linestyle='--', linewidth=1, alpha=0.7)

        ax.axvline(x=0, color='#666', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=0, color='#666', linestyle='--', linewidth=1, alpha=0.7)

        ylabel = yaxis_title or {
            'level': 'Change (bps)', 'change': 'Cumulative Change', 'pct': 'Change (%)',
        }.get(normalize, 'Value')

        self._style_ax(ax, title=title or f"Centered on {center_date.strftime('%m/%d/%y')}",
                       yaxis_title=ylabel, xaxis_title='Days from Event')
        self._legend(ax)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Multi-Event Overlay (static)
    # -------------------------------------------------------------------------

    def event_overlay(
        self,
        df: pd.DataFrame,
        events: List[Union[str, datetime]],
        col: str,
        window: int = 30,
        labels: Optional[List[str]] = None,
        normalize: str = 'level',
        title: Optional[str] = None,
    ):
        """Overlay multiple events on the same chart to compare reactions."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        labels = labels or [pd.to_datetime(e).strftime('%m/%d/%y') for e in events]

        fig, ax = plt.subplots(figsize=(12, 5))

        for i, (event, label) in enumerate(zip(events, labels)):
            event = pd.to_datetime(event)
            start = event - timedelta(days=window)
            end = event + timedelta(days=window)

            data = df.loc[(df.index >= start) & (df.index <= end), [col]].copy()
            if data.empty:
                continue

            if event not in data.index:
                idx = data.index.get_indexer([event], method='nearest')[0]
                event = data.index[idx]

            center_val = data.loc[event, col]
            if normalize == 'level':
                data[col] = data[col] - center_val
            elif normalize == 'pct':
                data[col] = (data[col] / center_val - 1) * 100

            data['days'] = (data.index - event).days
            color = self.colors[i % len(self.colors)]
            ax.plot(data['days'], data[col], color=color, linewidth=1.5, label=label)

        ax.axvline(x=0, color='#666', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=0, color='#666', linestyle='--', linewidth=1, alpha=0.7)

        self._style_ax(ax, title=title or f"Event Comparison: {col}",
                       xaxis_title='Days from Event',
                       yaxis_title='Change (bps)' if normalize == 'level' else 'Change (%)')
        self._legend(ax)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Curve Snapshots (static)
    # -------------------------------------------------------------------------

    def curve_snapshot(
        self,
        df: pd.DataFrame,
        dates: List[Union[str, datetime]],
        tenors: Optional[List[str]] = None,
        title: Optional[str] = None,
    ):
        """Plot yield curves for specific dates."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        tenors = tenors or df.columns.tolist()

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, date in enumerate(dates):
            date = pd.to_datetime(date)
            if date not in df.index:
                idx = df.index.get_indexer([date], method='nearest')[0]
                date = df.index[idx]
            values = df.loc[date, tenors]
            color = self.colors[i % len(self.colors)]
            ax.plot(tenors, values, color=color, linewidth=1.5,
                    marker='o', markersize=6, label=date.strftime('%m/%d/%y'))

        self._style_ax(ax, title=title or 'Yield Curve Snapshots',
                       xaxis_title='Tenor', yaxis_title='Yield (%)')
        self._legend(ax)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Rolling Statistics (interactive)
    # -------------------------------------------------------------------------

    def rolling_corr(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str,
        window: int = 60,
        title: Optional[str] = None,
    ):
        """Plot rolling correlation between two series."""
        corr = df[col1].rolling(window).corr(df[col2]).dropna()
        corr_df = corr.to_frame(name='corr')

        def render(fig, ax, start, end):
            subset = corr.loc[start:end]
            ax.plot(subset.index, subset, color=self.colors[0], linewidth=1.5,
                    label=f'{window}d Corr')
            ax.axhline(y=0, color='#666', linestyle='--', linewidth=1, alpha=0.7)

            self._style_ax(ax, yaxis_title='Correlation')
            self._format_dates(ax, start, end)
            self._legend(ax)
            fig.tight_layout()

        return self._make_time_nav(corr_df, render,
                                   title=title or f'Rolling {window}d Correlation: {col1} vs {col2}')

    def rolling_zscore(
        self,
        data: Union[pd.DataFrame, pd.Series],
        col: Optional[str] = None,
        window: int = 60,
        title: Optional[str] = None,
    ):
        """Plot rolling z-score. Accepts DataFrame+col or a bare Series."""
        if isinstance(data, pd.Series):
            col_name = data.name or 'value'
            series = data
        else:
            col_name = col
            series = data[col]

        if not isinstance(series.index, pd.DatetimeIndex):
            series = series.copy()
            series.index = pd.to_datetime(series.index)

        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        zscore = ((series - mean) / std).dropna()
        zscore_df = zscore.to_frame(name='z')

        def render(fig, ax, start, end):
            subset = zscore.loc[start:end]
            ax.plot(subset.index, subset, color=self.colors[0], linewidth=1.5,
                    label=f'{col_name} Z-Score')
            ax.axhline(y=2, color='#C0392B', linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(y=-2, color='#C0392B', linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(y=0, color='#666', linestyle='--', linewidth=1, alpha=0.7)

            self._style_ax(ax, yaxis_title='Z-Score')
            self._format_dates(ax, start, end)
            self._legend(ax)
            fig.tight_layout()

        return self._make_time_nav(zscore_df, render,
                                   title=title or f'Rolling {window}d Z-Score: {col_name}')

    # -------------------------------------------------------------------------
    # Heatmaps (static)
    # -------------------------------------------------------------------------

    def corr_heatmap(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        title: Optional[str] = None,
    ):
        """Correlation heatmap."""
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
        self._style_ax(ax, title=title or 'Correlation Matrix')
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def changes_heatmap(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        freq: str = 'M',
        title: Optional[str] = None,
    ):
        """Heatmap of periodic changes."""
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        resampled = df[cols].resample(freq).last()
        changes = resampled.diff()

        fig, ax = plt.subplots(figsize=(12, max(6, len(changes) * 0.3)))
        sns.heatmap(changes, cmap='RdBu_r', center=0, ax=ax, linewidths=0.5,
                    yticklabels=[d.strftime('%Y-%m') for d in changes.index],
                    cbar_kws={'shrink': 0.8})
        self._style_ax(ax, title=title or f'{freq} Changes Heatmap')
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # -------------------------------------------------------------------------
    # PCA Visualization (static)
    # -------------------------------------------------------------------------

    def pca_loadings(
        self,
        loadings: pd.DataFrame,
        n_components: int = 3,
        title: Optional[str] = None,
    ):
        """Plot PCA loadings (eigenvectors)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        cols = loadings.columns[:n_components]

        for i, col in enumerate(cols):
            ax.plot(loadings.index, loadings[col], color=self.colors[i],
                    linewidth=1.5, marker='o', markersize=6, label=col)

        ax.axhline(y=0, color='#666', linestyle='--', linewidth=1, alpha=0.7)
        self._style_ax(ax, title=title or 'PCA Loadings by Tenor',
                       xaxis_title='Tenor', yaxis_title='Loading')
        self._legend(ax)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def pca_variance(
        self,
        explained_variance: np.ndarray,
        title: Optional[str] = None,
    ):
        """Plot explained variance by component (scree plot)."""
        cumulative = np.cumsum(explained_variance)
        components = [f'PC{i+1}' for i in range(len(explained_variance))]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(components, explained_variance * 100, color=self.colors[0],
                label='Individual', alpha=0.85)
        ax1.set_ylabel('VARIANCE (%)', fontsize=self.FONT_SIZE, color='#333')

        ax2 = ax1.twinx()
        ax2.plot(components, cumulative * 100, color=self.colors[1],
                 linewidth=1.5, marker='o', markersize=6, label='Cumulative')
        ax2.set_ylabel('CUMULATIVE (%)', fontsize=self.FONT_SIZE, color='#333')

        self._style_ax(ax1, title=title or 'PCA Explained Variance')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                   fontsize=9, framealpha=0.85, edgecolor='none')

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Residual Distribution (static)
    # -------------------------------------------------------------------------

    def residual_dist(
        self,
        data: Union[pd.DataFrame, pd.Series],
        cols: Optional[List[str]] = None,
        title: Optional[str] = None,
        bins: int = 50,
        show_stats: bool = True,
    ):
        """Histogram + KDE of residual distributions with normal overlay.

        Accepts a DataFrame (plots each column) or a Series.
        Shows mean, std, skew, kurtosis, and a fitted normal curve.
        """
        if isinstance(data, pd.Series):
            df = data.dropna().to_frame(name=data.name or "resid")
        else:
            df = data.copy()

        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        n_cols = len(cols)

        fig, axes = plt.subplots(
            1, n_cols, figsize=(5 * n_cols, 5), squeeze=False,
        )

        for i, col in enumerate(cols):
            ax = axes[0, i]
            series = df[col].dropna()
            if series.empty:
                continue

            color = self.colors[i % len(self.colors)]

            # Histogram
            ax.hist(
                series, bins=bins, density=True, color=color,
                alpha=0.35, edgecolor=color, linewidth=0.5,
            )

            # KDE
            from scipy.stats import gaussian_kde, norm
            kde = gaussian_kde(series)
            x_grid = np.linspace(series.min(), series.max(), 200)
            ax.plot(x_grid, kde(x_grid), color=color, linewidth=2, label="KDE")

            # Normal overlay
            mu, sigma = series.mean(), series.std()
            ax.plot(
                x_grid, norm.pdf(x_grid, mu, sigma),
                color="#666", linewidth=1.5, linestyle="--", label="Normal",
            )

            # Current value marker
            current = float(series.iloc[-1])
            ax.axvline(
                x=current, color=color, linewidth=2, linestyle="-",
                label=f"Current: {current:.2f}",
            )

            # Stats box
            if show_stats:
                from scipy.stats import skew, kurtosis
                stats_text = (
                    f"μ = {mu:.2f}\n"
                    f"σ = {sigma:.2f}\n"
                    f"skew = {skew(series):.2f}\n"
                    f"kurt = {kurtosis(series):.2f}\n"
                    f"n = {len(series)}"
                )
                ax.text(
                    0.97, 0.97, stats_text,
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="#CCC", alpha=0.9),
                )

            self._style_ax(ax, title=col if n_cols > 1 else None)
            ax.legend(
                loc="upper left", fontsize=8, frameon=False,
                handlelength=1.5,
            )

        if title:
            fig.suptitle(
                title.upper(), fontsize=self.TITLE_SIZE,
                color="#333", x=0.01, ha="left", y=1.02,
            )
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Event Annotation Helper
    # -------------------------------------------------------------------------

    def add_event_annotation(self, ax, x, text: str, y_position: float = 0.85):
        """Add event annotation to a matplotlib axes."""
        ax.axvline(x=x, linestyle=':', color='#333', linewidth=1)
        ax.annotate(text, xy=(x, y_position), xycoords=('data', 'axes fraction'),
                    fontsize=8, color='#333', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#999', alpha=0.9))
        return ax

    # -------------------------------------------------------------------------
    # Scatter plot (static)
    # -------------------------------------------------------------------------

    def scatter(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
    ):
        """Scatter plot with optional color dimension."""
        fig, ax = plt.subplots(figsize=(10, 7))

        if color:
            for i, group in enumerate(df[color].unique()):
                mask = df[color] == group
                ax.scatter(df.loc[mask, x], df.loc[mask, y],
                           color=self.colors[i % len(self.colors)],
                           label=group, s=30, alpha=0.7)
        else:
            ax.scatter(df[x], df[y], color=self.colors[0], s=30, alpha=0.7)

        self._style_ax(ax, title=title, xaxis_title=x, yaxis_title=y)
        if color:
            self._legend(ax)
        plt.tight_layout()
        plt.show()
        plt.close(fig)


# =============================================================================
# Plotly / Dash backend — browser-served charts with native time-range buttons
# =============================================================================
#
# Use PlotlyViz when you want charts in a real browser (PyCharm matplotlib pane
# stinks; ipywidgets time-nav buttons only work in Jupyter). The matplotlib
# `Viz` class above is untouched — existing notebooks keep working. Plotly/dash
# imports are lazy so envs without them only fail when PlotlyViz is used.
#
# Usage:
#     from utils.viz import PlotlyViz
#     v = PlotlyViz()                           # boots Dash on :8050, opens browser
#     v.line(df, title='UST yields')
#     v.line(other_df, title='residual', hlines=[0])
#     # charts stack vertically on the page; native 1M/3M/6M/YTD/1Y/2Y/5Y/10Y/ALL buttons.

import os
import socket
import subprocess
import threading
import time
import webbrowser


RANGE_BUTTON_LABELS = ["1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y", "10Y", "ALL"]


class _PlotlyChartRegistry:
    """Module-level list of charts shown by the Dash app.

    Each entry is a dict {id, title, png_b64, df, render_fn, viz_ref}.
    Charts are rendered matplotlib figures saved as base64 PNGs; `df` +
    `render_fn` are kept so range buttons can re-render server-side. The
    'Plotly' name is legacy.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._charts = []
        self._version = 0

    def add(self, *, title, png_b64, df, render_fn, viz_ref, static=False, html_block=None, nrows=1, height_ratios=None):
        with self._lock:
            self._charts.append({
                "id": uuid.uuid4().hex,
                "title": title,
                "png_b64": png_b64,
                "df": df,
                "render_fn": render_fn,
                "viz_ref": viz_ref,
                "static": static,
                "html_block": html_block,
                "nrows": nrows,
                "height_ratios": height_ratios,
            })
            self._version += 1

    def get(self, chart_id):
        with self._lock:
            for c in self._charts:
                if c["id"] == chart_id:
                    return c
            return None

    def clear(self):
        with self._lock:
            self._charts = []
            self._version += 1

    def snapshot(self):
        with self._lock:
            return list(self._charts), self._version


_REGISTRY = _PlotlyChartRegistry()
_SERVER_STATE = {"started": False, "port": None, "url": None}


def _pick_port(start=8050, tries=20):
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"no free TCP port in range {start}..{start + tries}")


def _open_browser(url):
    # WSL: standard webbrowser often fails. Try Windows host fallbacks.
    try:
        if webbrowser.open(url):
            return
    except Exception:
        pass
    for cmd in (["wslview", url],
                ["cmd.exe", "/c", "start", url],
                ["xdg-open", url]):
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (FileNotFoundError, OSError):
            continue


def _build_dash_app():
    import dash
    from dash import Dash, dcc, html, Output, Input, State, MATCH, ALL, ctx

    app = Dash(__name__, title="Viz", update_title=None)
    app.layout = html.Div(
        style={"fontFamily": "Arial, Helvetica, sans-serif",
               "background": "#fff", "padding": "8px 4px", "color": "#333"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "borderBottom": "1px solid #ddd", "paddingBottom": "6px",
                       "marginBottom": "10px", "paddingLeft": "4px"},
                children=[
                    html.H3("VIZ", style={"margin": 0, "letterSpacing": "1px"}),
                    html.Span(id="_count",
                              style={"marginLeft": "12px", "color": "#777", "fontSize": "12px"}),
                ],
            ),
            dcc.Interval(id="_poll", interval=1000, n_intervals=0),
            dcc.Store(id="_version", data=-1),
            html.Div(id="_charts"),
        ],
    )

    _btn_base_style = {
        "minWidth": "44px", "height": "26px", "padding": "0 8px",
        "marginRight": "4px", "fontSize": "11px", "border": "1px solid #999",
        "borderRadius": "3px", "background": "#F7F7F7", "color": "#333",
        "cursor": "pointer",
    }
    _btn_active_style = {**_btn_base_style, "background": "#F1C40F",
                          "borderColor": "#B7950B"}

    def _chart_block(c):
        chart_id = c["id"]
        title = (c["title"] or "").upper()
        buttons = [] if c.get("static") else [
            html.Button(
                lbl,
                id={"type": "range-btn", "chart": chart_id, "label": lbl},
                n_clicks=0,
                style=(_btn_active_style if lbl == "ALL" else _btn_base_style),
            )
            for lbl in RANGE_BUTTON_LABELS
        ]
        if c.get("html_block") is not None:
            return html.Div(
                style={"marginBottom": "24px"},
                children=[c["html_block"]],
            )
        return html.Div(
            style={"marginBottom": "24px"},
            children=[
                # title is baked into the image so copy-chart includes it; the
                # row here only carries the range buttons (right-aligned).
                html.Div(buttons,
                         style={"display": "flex", "justifyContent": "flex-end",
                                "marginBottom": "6px"}),
                html.Img(
                    id={"type": "chart-img", "chart": chart_id},
                    src=f"data:image/png;base64,{c['png_b64']}",
                    style={"maxWidth": "100%", "height": "auto",
                           "display": "block", "margin": 0,
                           "border": "1px solid #eee"},
                ),
            ],
        )

    @app.callback(
        Output("_charts", "children"),
        Output("_version", "data"),
        Output("_count", "children"),
        Input("_poll", "n_intervals"),
        State("_version", "data"),
    )
    def _refresh(_n, last_version):
        charts, version = _REGISTRY.snapshot()
        if version == last_version:
            raise dash.exceptions.PreventUpdate
        return ([_chart_block(c) for c in charts],
                version,
                f"{len(charts)} chart{'s' if len(charts) != 1 else ''}")

    @app.callback(
        Output({"type": "chart-img", "chart": MATCH}, "src"),
        Output({"type": "range-btn", "chart": MATCH, "label": ALL}, "style"),
        Input({"type": "range-btn", "chart": MATCH, "label": ALL}, "n_clicks"),
        State({"type": "range-btn", "chart": MATCH, "label": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _on_range_click(_clicks, ids):
        trig = ctx.triggered_id
        if not isinstance(trig, dict):
            raise dash.exceptions.PreventUpdate
        chart_id = trig["chart"]
        label = trig["label"]
        chart = _REGISTRY.get(chart_id)
        if chart is None:
            raise dash.exceptions.PreventUpdate
        start, end = _range_window(chart["df"], label)
        png = chart["viz_ref"]._render_png(
            chart["df"], chart["render_fn"], start, end, title=chart["title"],
            nrows=chart.get("nrows", 1), height_ratios=chart.get("height_ratios"),
        )
        styles = [
            (_btn_active_style if d["label"] == label else _btn_base_style)
            for d in ids
        ]
        return f"data:image/png;base64,{png}", styles

    return app


def _ensure_server():
    if _SERVER_STATE["started"]:
        return _SERVER_STATE["url"]

    # Silence per-request access logs — Dash polls every second so the
    # default Werkzeug logger buries any print() output from the caller.
    import logging as _logging
    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

    port = _pick_port(int(os.getenv("VIZ_PORT", "8050")))
    app = _build_dash_app()

    def _run():
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

    # Non-daemon so the server keeps the process alive after main() returns —
    # the user can keep poking the page until they Ctrl-C.
    threading.Thread(target=_run, daemon=False, name=f"viz-dash-{port}").start()

    for _ in range(60):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.05)

    url = f"http://127.0.0.1:{port}"
    _SERVER_STATE.update({"started": True, "port": port, "url": url})
    print(f"[viz] dash app running at {url}  (Ctrl-C to stop)")
    _open_browser(url)
    return url


class PlotlyViz(Viz):
    """Browser-served counterpart to Viz. Renders the SAME matplotlib charts
    you'd see in Jupyter — identical colors, legend, axis style — and serves
    them as PNGs on a Dash page. (No plotly anymore; class name is legacy.)

    First instantiation boots a Dash server in a background thread (port 8050
    or next free) and opens the browser. Each `.line()` call appends a chart;
    the page polls and refreshes every second.
    """

    def __init__(self, backend=None, port=None, auto_clear=True, **kwargs):
        # Force the non-interactive Agg backend before anything draws. Dash
        # range-button callbacks fire on worker threads, and Windows' default
        # interactive backend (TkAgg/QtAgg) isn't safe off the main thread.
        # Agg is in-memory only — perfect since we just want PNG bytes.
        plt.switch_backend("Agg")

        Viz.__init__(self)
        if port:
            os.environ["VIZ_PORT"] = str(port)
        if auto_clear:
            _REGISTRY.clear()
        _ensure_server()

    def _render_png(self, df, render_fn, start, end, title=None, nrows=1, height_ratios=None):
        """Render render_fn into a fresh matplotlib Figure and return base64 PNG.

        Title gets baked into the image so right-click-copy includes it. The
        plot area is pushed flush to the figure's left edge so there's no
        internal whitespace between the title and the chart.
        """
        h = 5.4 if nrows == 1 else 4.2 * nrows
        kw = dict(sharex=(nrows > 1), gridspec_kw={'height_ratios': height_ratios} if height_ratios else {})
        fig, _axes = plt.subplots(nrows, 1, figsize=(13, h), **kw)
        axes_arg = _axes if nrows > 1 else _axes
        fig.patch.set_facecolor("white")
        # Push plot area to span (almost) the full figure width — y-labels and
        # the endpoint ribbons live in the right margin (~5%).
        fig.subplots_adjust(left=0.02, right=0.95, top=0.91)
        render_fn(fig, axes_arg, start, end)
        if title:
            fig.suptitle(title.upper(), fontsize=self.TITLE_SIZE,
                          fontweight="bold", color="#333",
                          x=0.02, ha="left", y=0.97)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                    facecolor="white", edgecolor="white")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _make_time_nav(self, df, render_fn, title=None, nrows=1, height_ratios=None):
        """Render the matplotlib figure over the full date range and stash the
        closure so range buttons can re-render server-side later.

        Overrides Viz._make_time_nav so all chart helpers (line, residual_dist,
        etc.) flow into the browser instead of triggering plt.show() / Jupyter
        widgets.
        """
        png = self._render_png(df, render_fn, df.index.min(), df.index.max(), title=title, nrows=nrows, height_ratios=height_ratios)
        _REGISTRY.add(title=title, png_b64=png, df=df, render_fn=render_fn, viz_ref=self, nrows=nrows, height_ratios=height_ratios)
        return None

    def table(self, df: pd.DataFrame, title: Optional[str] = None, max_rows: int = 20):
        """Render a pandas DataFrame as a plain HTML table in the browser app."""
        from dash import html as dhtml

        def _fmt(v):
            if isinstance(v, (float, np.floating)):
                return f"{v:,.3f}"
            if isinstance(v, pd.Timestamp):
                return v.strftime("%Y-%m-%d")
            return str(v)

        show = df.head(max_rows).copy()
        _th_style = {
            "padding": "5px 10px", "textAlign": "right",
            "borderBottom": "2px solid #bbb", "fontWeight": "bold",
            "fontSize": "12px", "color": "#333", "whiteSpace": "nowrap",
        }
        _td_style = {
            "padding": "4px 10px", "textAlign": "right",
            "borderBottom": "1px solid #e8e8e8", "fontSize": "12px",
            "whiteSpace": "nowrap",
        }
        header = dhtml.Tr([dhtml.Th(str(col), style=_th_style) for col in show.columns])
        rows = [
            dhtml.Tr([dhtml.Td(_fmt(v), style=_td_style) for v in row])
            for row in show.to_numpy()
        ]
        children = []
        if title:
            children.append(dhtml.P(
                title.upper(),
                style={"fontWeight": "bold", "fontSize": "11px",
                       "color": "#333", "marginBottom": "6px", "letterSpacing": "0.04em"},
            ))
        children.append(dhtml.Table(
            [dhtml.Thead(header), dhtml.Tbody(rows)],
            style={"borderCollapse": "collapse", "minWidth": "100%", "width": "max-content"},
        ))
        block = dhtml.Div(children, style={"overflowX": "auto"})
        dummy = pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp.today().normalize()]))
        _REGISTRY.add(
            title=title,
            png_b64="",
            df=dummy,
            render_fn=lambda *_args: None,
            viz_ref=self,
            static=True,
            html_block=block,
        )
        return None


def _range_window(df, label):
    """Translate a button label into (start, end) timestamps within df.index."""
    end = df.index.max()
    if label == "ALL":
        start = df.index.min()
    elif label == "YTD":
        start = pd.Timestamp(end.year, 1, 1)
    elif label.endswith("M"):
        start = end - pd.DateOffset(months=int(label[:-1]))
    elif label.endswith("Y"):
        start = end - pd.DateOffset(years=int(label[:-1]))
    else:
        start = df.index.min()
    return max(start, df.index.min()), end


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------

_viz = Viz()

def line(*args, **kwargs):
    return _viz.line(*args, **kwargs)

def centered_chart(*args, **kwargs):
    return _viz.centered_chart(*args, **kwargs)

def event_overlay(*args, **kwargs):
    return _viz.event_overlay(*args, **kwargs)

def curve_snapshot(*args, **kwargs):
    return _viz.curve_snapshot(*args, **kwargs)

def rolling_corr(*args, **kwargs):
    return _viz.rolling_corr(*args, **kwargs)

def rolling_zscore(*args, **kwargs):
    return _viz.rolling_zscore(*args, **kwargs)

def corr_heatmap(*args, **kwargs):
    return _viz.corr_heatmap(*args, **kwargs)

def pca_loadings(*args, **kwargs):
    return _viz.pca_loadings(*args, **kwargs)

def pca_variance(*args, **kwargs):
    return _viz.pca_variance(*args, **kwargs)

def residual_dist(*args, **kwargs):
    return _viz.residual_dist(*args, **kwargs)
