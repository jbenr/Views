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

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import ipywidgets as widgets
from IPython.display import display, clear_output


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

    def __init__(self):
        self.colors = self.COLORS
        sns.set_theme(style='whitegrid', rc={
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': self.FONT_SIZE,
            'axes.facecolor': self.BG_COLOR,
            'figure.facecolor': 'white',
        })

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _format_legend_name(self, name: str, avg: float = None, unit: str = '') -> str:
        if avg is not None:
            return f"{name} = {avg:.1f} {unit}".strip()
        return name

    def _style_ax(self, ax, title=None, yaxis_title=None, xaxis_title=None):
        """Apply PrismFP / 3fiftyseven styling to a matplotlib axes."""
        ax.set_facecolor(self.BG_COLOR)
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
        ax.grid(True, color=self.GRID_COLOR, linewidth=1)
        for spine in ax.spines.values():
            spine.set_color('#333')
            spine.set_linewidth(1)

    def _format_dates(self, ax):
        """Auto-format date axis."""
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.xaxis.get_offset_text().set_visible(False)

    def _legend(self, ax):
        """Standard legend in lower left."""
        ax.legend(loc='lower left', fontsize=9, framealpha=0.85,
                  edgecolor='none', fancybox=True)

    def _add_hover(self, fig, ax):
        """Add crosshair + value tooltip on hover. Requires %matplotlib widget."""
        # Disconnect any previous hover handler
        if hasattr(fig, '_hover_cid') and fig._hover_cid is not None:
            fig.canvas.mpl_disconnect(fig._hover_cid)
            fig._hover_cid = None

        # Save limits before adding hover elements (xy=0 would blow out the axis)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        vline = ax.axvline(x=0, color='#ccc', linewidth=0.8)
        vline.set_visible(False)
        annot = ax.annotate('', xy=(0, 0), xytext=(15, 15),
                            textcoords='offset points', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc'),
                            zorder=10)
        annot.set_visible(False)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # data lines only (skip dashed avg/ref lines)
        plot_lines = [l for l in ax.get_lines()
                      if l.get_linestyle() == '-' and l.get_visible()
                      and len(l.get_xdata()) > 2]

        def on_move(event):
            if event.inaxes != ax or not plot_lines:
                if annot.get_visible():
                    annot.set_visible(False)
                    vline.set_visible(False)
                    fig.canvas.draw_idle()
                return

            xdata = plot_lines[0].get_xdata()
            if len(xdata) == 0:
                return
            idx = np.argmin(np.abs(xdata - event.xdata))
            snap_x = xdata[idx]

            date_str = mdates.num2date(snap_x).strftime('%b %d, %Y')
            parts = [date_str]
            for line in plot_lines:
                yd = line.get_ydata()
                if idx < len(yd):
                    parts.append(f'{line.get_label()}: {yd[idx]:.2f}')

            annot.set_text('\n'.join(parts))
            annot.xy = (snap_x, event.ydata)
            annot.set_visible(True)
            vline.set_xdata([snap_x, snap_x])
            vline.set_visible(True)
            fig.canvas.draw_idle()

        fig._hover_cid = fig.canvas.mpl_connect('motion_notify_event', on_move)

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
        """Build date pickers + range buttons, wire them to render_fn(fig, ax, start, end).

        Uses fig.canvas directly as the chart widget for true interactivity (hover, etc).
        Falls back to Output widget if ipympl backend isn't available.
        Plain script mode: just plt.show() with no widgets.
        """
        # plain script — skip widgets, just show the chart
        if not self._in_notebook():
            fig, ax = plt.subplots(figsize=(12, 5))
            if title:
                fig.suptitle(title.upper(), fontsize=self.TITLE_SIZE, color='#333',
                             x=0.01, ha='left')
            render_fn(fig, ax, df.index.min(), df.index.max())
            plt.show()
            return fig

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

        btn_layout = widgets.Layout(width='42px', height='26px', padding='0px')
        btn_widgets = []
        for b_def in self.TIME_NAV_BUTTONS:
            b = widgets.Button(description=b_def['label'], layout=btn_layout)
            btn_widgets.append(b)
        btn_widgets[-1].button_style = 'warning'  # ALL starts active

        # Create figure once - use canvas directly as widget for interactivity
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.close(fig)  # Remove from pyplot manager (prevents double display)

        # Check if canvas is a widget (ipympl backend) or fallback to Output
        canvas_is_widget = isinstance(fig.canvas, widgets.Widget)

        if canvas_is_widget:
            chart_widget = fig.canvas

            def update_range(start, end):
                ax.clear()
                render_fn(fig, ax, start, end)
                fig.canvas.draw_idle()
        else:
            # Fallback: use Output widget (no hover, but still works)
            chart_widget = widgets.Output()

            def update_range(start, end):
                ax.clear()
                render_fn(fig, ax, start, end)
                with chart_widget:
                    clear_output(wait=True)
                    display(fig)

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
            update_range(pd.Timestamp(start_picker.value),
                         pd.Timestamp(end_picker.value))

        for b in btn_widgets:
            b.on_click(on_btn_click)
        start_picker.observe(on_date_change, names='value')
        end_picker.observe(on_date_change, names='value')

        btn_row = widgets.HBox(btn_widgets, layout=widgets.Layout(gap='2px'))
        controls = widgets.HBox(
            [start_picker, end_picker, btn_row],
            layout=widgets.Layout(gap='8px', align_items='center'),
        )
        title_widget = widgets.HTML(
            value=f'<b style="font-size:13px; color:#333;">{title.upper()}</b>' if title else ''
        )
        container = widgets.VBox([title_widget, controls, chart_widget])

        # initial render
        update_range(df.index.min(), df.index.max())
        return container

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
        show_avg: bool = False,
        avg_unit: str = '',
        interval: str = None,
        show_endpoint_marker: bool = True,
        residual: bool = False,
    ):
        """Line chart with interactive time navigation."""
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        def render(fig, ax, start, end):
            subset = df.loc[start:end, cols]

            for i, col in enumerate(cols):
                color = self.colors[i % len(self.colors)]
                series = subset[col].dropna()
                if series.empty:
                    continue
                avg = series.mean() if show_avg else None
                label = self._format_legend_name(col, avg, avg_unit) if show_avg else col
                ax.plot(series.index, series, color=color, linewidth=1.5, label=label)

                if residual and len(cols) == 1:
                    ax.axhline(y=0, color='#666', linestyle=':', linewidth=1)
                    ax.fill_between(
                        series.index, 0, series,
                        where=series >= 0,
                        interpolate=True, color='#27AE60', alpha=0.15,
                    )
                    ax.fill_between(
                        series.index, 0, series,
                        where=series < 0,
                        interpolate=True, color='#C0392B', alpha=0.15,
                    )

                if show_endpoint_marker and len(series) > 0:
                    last_val = series.iloc[-1]
                    ax.annotate(f'{last_val:.2f}',
                                xy=(series.index[-1], last_val),
                                fontsize=8, color='white', fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                                          edgecolor='none', alpha=0.9),
                                ha='left', va='center',
                                xytext=(5, 0), textcoords='offset points',
                                zorder=5)

                if show_avg and avg is not None:
                    ax.axhline(y=avg, color=color, linestyle='--', linewidth=1, alpha=0.7)

            self._style_ax(ax, yaxis_title=yaxis_title)
            self._format_dates(ax)
            self._legend(ax)
            self._add_hover(fig, ax)
            fig.tight_layout()

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
            self._format_dates(ax)
            self._legend(ax)
            self._add_hover(fig, ax)
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
            self._format_dates(ax)
            self._legend(ax)
            self._add_hover(fig, ax)
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
