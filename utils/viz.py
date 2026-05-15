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
import seaborn as sns

import ipywidgets as widgets
from IPython.display import HTML, display, clear_output


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
        """Apply PrismFP styling."""
        if title:
            ax.set_title(title.upper(), fontsize=self.TITLE_SIZE, color='#333',
                         pad=10, loc='left')
        if yaxis_title:
            ax.set_ylabel(yaxis_title.upper(), fontsize=self.FONT_SIZE, color='#333')
        if xaxis_title:
            ax.set_xlabel(xaxis_title.upper(), fontsize=self.FONT_SIZE, color='#333')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.tick_params(labelsize=9, colors='#333')
        for spine in ax.spines.values():
            spine.set_color('#333')
            spine.set_linewidth(0.5)

    def _format_dates(self, ax, start=None, end=None):
        """Auto-format date axis with tighter label spacing."""
        if start is not None and end is not None:
            span = (end - start).days
            if span <= 45:
                minticks, maxticks = 10, 35   # daily ticks for 1M
            elif span <= 180:
                minticks, maxticks = 8, 20
            else:
                minticks, maxticks = 12, 24
        else:
            minticks, maxticks = 12, 24
        locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
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

    def _make_time_nav(self, df, render_fn, title=None):
        """Date pickers + range buttons wired to render_fn(fig, ax, start, end).
        Falls back to plt.show() outside notebooks.
        """
        def _static_show():
            fig, ax = plt.subplots(figsize=(12, 5))
            render_fn(fig, ax, df.index.min(), df.index.max())
            plt.show()
            return fig

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

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('white')
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
            ax.clear()
            for extra in fig.axes[1:]:
                extra.remove()
            render_fn(fig, ax, start, end)
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
                    target.annotate(f'{last_val:.2f}',
                                xy=(series.index[-1], last_val),
                                fontsize=8, color='white', fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                                          edgecolor='none', alpha=0.9),
                                ha='left', va='center',
                                xytext=(5, 0), textcoords='offset points',
                                zorder=20)

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
                ax2.tick_params(labelsize=9, colors='#333')
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


class _PlotlyChartRegistry:
    """Module-level list of figures shown by the Dash app."""

    def __init__(self):
        self._lock = threading.Lock()
        self._charts = []
        self._version = 0

    def add(self, figure, title=None):
        with self._lock:
            self._charts.append({"id": uuid.uuid4().hex, "title": title, "figure": figure})
            self._version += 1

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
    from dash import Dash, dcc, html, Output, Input, State

    app = Dash(__name__, title="Viz", update_title=None)
    app.layout = html.Div(
        style={"fontFamily": "Arial, Helvetica, sans-serif",
               "background": "#fff", "padding": "16px", "color": "#333"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "borderBottom": "1px solid #ddd", "paddingBottom": "8px",
                       "marginBottom": "12px"},
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
        items = []
        for c in charts:
            items.append(html.Div(
                style={"marginBottom": "20px"},
                children=[
                    dcc.Graph(
                        figure=c["figure"],
                        config={
                            "displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                            "toImageButtonOptions": {"format": "png", "scale": 2,
                                                     "filename": (c["title"] or "chart").lower().replace(" ", "_")},
                        },
                    ),
                ],
            ))
        return items, version, f"{len(charts)} chart{'s' if len(charts) != 1 else ''}"

    return app


def _ensure_server():
    if _SERVER_STATE["started"]:
        return _SERVER_STATE["url"]

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


_PLOTLY_LS_MAP = {
    "solid":  ("solid", 1.5),
    "bold":   ("solid", 2.5),
    "thin":   ("solid", 0.8),
    "dashed": ("dash",  1.2),
    "dash":   ("dash",  1.2),
    "dotted": ("dot",   1.4),
    "dot":    ("dot",   1.4),
}
_MPL_TO_PLOTLY_DASH = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}


def _resolve_plotly_style(style, default="solid"):
    if style is None:
        return _PLOTLY_LS_MAP[default]
    if isinstance(style, (tuple, list)) and len(style) == 2:
        ls, lw = style
        return _MPL_TO_PLOTLY_DASH.get(ls, ls), lw
    return _PLOTLY_LS_MAP.get(style, _PLOTLY_LS_MAP[default])


_PLOTLY_RANGE_BUTTONS = [
    {"count": 1,  "step": "month", "stepmode": "backward", "label": "1M"},
    {"count": 3,  "step": "month", "stepmode": "backward", "label": "3M"},
    {"count": 6,  "step": "month", "stepmode": "backward", "label": "6M"},
    {"step": "year", "stepmode": "todate", "label": "YTD"},
    {"count": 1,  "step": "year",  "stepmode": "backward", "label": "1Y"},
    {"count": 2,  "step": "year",  "stepmode": "backward", "label": "2Y"},
    {"count": 5,  "step": "year",  "stepmode": "backward", "label": "5Y"},
    {"count": 10, "step": "year",  "stepmode": "backward", "label": "10Y"},
    {"step": "all", "label": "ALL"},
]


class PlotlyViz(Viz):
    """Browser-served counterpart to Viz. Same .line() signature; charts stack on a Dash page.

    Prefer `Viz(backend='plotly')` over instantiating this directly. Inherits
    from Viz so `isinstance(v, Viz)` and existing type hints still hold —
    centered_chart/heatmaps/etc. still emit matplotlib (only .line() is plotly).

    First instantiation boots a Dash server in a background thread (port 8050
    or next free) and opens the browser. Each .line() call appends a chart;
    the page polls and refreshes every second.
    """

    def __init__(self, backend=None, port=None, auto_clear=True, **kwargs):
        # cheap matplotlib setup runs too so the non-plotly methods still work
        Viz.__init__(self)
        if port:
            os.environ["VIZ_PORT"] = str(port)
        if auto_clear:
            _REGISTRY.clear()
        _ensure_server()

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
        interval: str = None,             # accepted for signature compat
        show_endpoint_marker: bool = True,
        residual: bool = False,
        nas: bool = True,                 # accepted for signature compat
        hlines: Optional[list] = None,
        linestyles: Optional[dict] = None,
        bar: bool = False,
    ):
        """Plotly line chart with native rangeselector buttons. Drop-in for Viz.line."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        left_set = set(left or [])
        ls_map = linestyles or {}

        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        use_secondary = bool(left_set)
        fig = make_subplots(specs=[[{"secondary_y": True}]]) if use_secondary else go.Figure()
        annotations = []

        # residual fills go in FIRST so the line plots on top
        if residual and not bar and len(cols) == 1:
            r = df[cols[0]].dropna()
            if not r.empty:
                pos = np.where(r.values >= 0, r.values, 0)
                neg = np.where(r.values < 0, r.values, 0)
                fill_args = dict(mode="none", showlegend=False, hoverinfo="skip")
                for y_vals, color in ((pos, "rgba(39,174,96,0.18)"),
                                      (neg, "rgba(192,57,43,0.18)")):
                    tr = go.Scatter(x=r.index, y=y_vals, fill="tozeroy",
                                    fillcolor=color, **fill_args)
                    fig.add_trace(tr, secondary_y=False) if use_secondary else fig.add_trace(tr)
                fig.add_hline(y=0, line=dict(color="#666", width=1, dash="dot"))

        for i, col in enumerate(cols):
            color = self.colors[i % len(self.colors)]
            series = df[col].dropna()
            if series.empty:
                continue
            on_secondary = col in left_set
            avg = series.mean() if show_avg else None
            label = (f"{col} = {avg:.1f} {avg_unit}".strip()
                     if show_avg and avg is not None else col)
            dash, width = _resolve_plotly_style(ls_map.get(col), default="solid")

            if bar:
                bar_colors = ["#27AE60" if v >= 0 else "#C0392B" for v in series.values]
                trace = go.Bar(x=series.index, y=series.values,
                               marker_color=bar_colors, name=label)
            else:
                trace = go.Scatter(
                    x=series.index, y=series.values,
                    mode="lines", name=label,
                    line=dict(color=color, width=width, dash=dash),
                    hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.4f}}<extra></extra>",
                    connectgaps=nas,
                )

            if use_secondary:
                fig.add_trace(trace, secondary_y=on_secondary)
            else:
                fig.add_trace(trace)

            if show_endpoint_marker and not bar and len(series):
                yref = ("y2" if (on_secondary and use_secondary) else "y")
                annotations.append(dict(
                    x=series.index[-1], y=float(series.iloc[-1]),
                    xref="x", yref=yref,
                    text=f"{float(series.iloc[-1]):.2f}",
                    showarrow=False,
                    bgcolor=color, font=dict(color="white", size=10),
                    xanchor="left", yanchor="middle",
                    xshift=6, borderpad=3,
                ))

            if show_avg and avg is not None and not bar:
                if use_secondary:
                    fig.add_hline(y=avg, line=dict(color=color, width=1, dash="dash"),
                                  opacity=0.6, secondary_y=on_secondary)
                else:
                    fig.add_hline(y=avg, line=dict(color=color, width=1, dash="dash"),
                                  opacity=0.6)

        if hlines:
            for h in hlines:
                if isinstance(h, dict):
                    value   = h["value"]
                    h_label = h.get("label")
                    h_style = h.get("style", "dashed")
                    h_color = h.get("color", "#666")
                    h_alpha = h.get("alpha", 0.7)
                elif isinstance(h, (tuple, list)):
                    value   = h[0]
                    h_label = h[1] if len(h) > 1 else None
                    h_style = h[2] if len(h) > 2 else "dashed"
                    h_color = h[3] if len(h) > 3 else "#666"
                    h_alpha = 0.7
                else:
                    value, h_label, h_style, h_color, h_alpha = float(h), None, "dashed", "#666", 0.7
                dash, width = _resolve_plotly_style(h_style, default="dashed")
                fig.add_hline(
                    y=value, line=dict(color=h_color, width=width, dash=dash),
                    opacity=h_alpha,
                    annotation_text=h_label,
                    annotation_position="top right",
                    annotation_font=dict(size=9, color=h_color),
                )

        title_text = title.upper() if title else None
        fig.update_layout(
            title=(dict(text=title_text, x=0.01, xanchor="left", y=0.97,
                        font=dict(size=13, color="#333")) if title_text else None),
            plot_bgcolor="#F5F5F5",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Arial, Helvetica, sans-serif", size=10, color="#333"),
            hovermode="x unified",
            legend=dict(orientation="h", x=0, y=-0.16, font=dict(size=9)),
            margin=dict(l=50, r=70, t=70, b=70),
            annotations=annotations,
            height=460,
            xaxis=dict(
                showgrid=True, gridcolor="#DCDCDC", gridwidth=0.5,
                rangeselector=dict(
                    buttons=_PLOTLY_RANGE_BUTTONS,
                    bgcolor="#F7F7F7", activecolor="#F1C40F",
                    borderwidth=1, bordercolor="#999",
                    x=1, xanchor="right", y=1.06, yanchor="bottom",
                    font=dict(size=10),
                ),
                rangeslider=dict(visible=False),
                type="date",
            ),
        )
        # axis titles — primary on right (Prism style), secondary (if any) on left
        if use_secondary:
            fig.update_yaxes(title_text=(yaxis_title.upper() if yaxis_title else None),
                             secondary_y=False, side="right",
                             showgrid=True, gridcolor="#DCDCDC", gridwidth=0.5)
            fig.update_yaxes(title_text=(yaxis_right_title.upper() if yaxis_right_title else None),
                             secondary_y=True, side="left", showgrid=False)
        else:
            fig.update_yaxes(title_text=(yaxis_title.upper() if yaxis_title else None),
                             side="right",
                             showgrid=True, gridcolor="#DCDCDC", gridwidth=0.5)

        _REGISTRY.add(fig, title=title)
        return fig


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
