#!/usr/bin/env python3
"""
Visualization and time series analysis helpers for rates/fixed income work.
PrismFP/Bloomberg style formatting.

Usage:
    from viz import Viz
    
    v = Viz()
    v.centered_chart(df, center_date='2025-06-15', window=30, cols=['2Y', '10Y'])
    v.curve_snapshot(df, dates=['2025-01-01', '2025-06-01'])
    v.rolling_corr(df, col1='2Y', col2='10Y', window=60)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict
from datetime import datetime, timedelta

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


class Viz:
    """Time series visualization toolkit for rates analysis - PrismFP style."""

    # -------------------------------------------------------------------------
    # PrismFP style configuration
    # -------------------------------------------------------------------------
    
    # Color palette matching PrismFP exactly
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
    
    # Font settings
    FONT_FAMILY = 'Arial, Helvetica, sans-serif'
    FONT_SIZE = 10
    TITLE_SIZE = 11
    
    # Background colors
    BG_COLOR = '#F5F5F5'
    GRID_COLOR = '#FFFFFF'
    
    def __init__(self, style: str = 'plotly'):
        """
        Initialize visualizer.
        
        Args:
            style: 'plotly' or 'seaborn' (default plotting backend)
        """
        self.style = style
        self.colors = self.COLORS
        
        # Seaborn defaults
        sns.set_theme(style='whitegrid')
    
    def _format_title(
        self, 
        main_title: str, 
        start_date: datetime = None, 
        end_date: datetime = None,
        interval: str = None,
    ) -> str:
        """Format multi-line title like PrismFP."""
        lines = [main_title.upper()]
        
        if start_date and end_date:
            date_line = f"{start_date.strftime('%m/%d/%y %H:%M')} ET TO {end_date.strftime('%m/%d/%y %H:%M')} ET"
            if interval:
                date_line += f", {interval.upper()}"
            lines.append(date_line)
        
        return '<br>'.join(lines)
    
    def _format_legend_name(self, name: str, avg: float = None, unit: str = '') -> str:
        """Format legend entry like 'TU 1mth = 58.9 nv'."""
        if avg is not None:
            return f"{name} = {avg:.1f} {unit}".strip()
        return name
        
    def _base_layout(
        self, 
        title: str = None, 
        xaxis_title: str = None, 
        yaxis_title: str = None,
        show_source: bool = True,
        source_text: str = "SOURCE: Custom Analysis",
    ):
        """Return base layout matching PrismFP style."""
        layout = dict(
            title=dict(
                text=title,
                font=dict(family=self.FONT_FAMILY, size=self.TITLE_SIZE, color='#333'),
                x=0.5,
                xanchor='center',
            ),
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE, color='#333'),
            plot_bgcolor=self.BG_COLOR,
            paper_bgcolor='white',
            xaxis=dict(
                title=dict(
                    text=xaxis_title.upper() if xaxis_title else None,
                    font=dict(size=self.FONT_SIZE),
                ),
                showgrid=True,
                gridcolor=self.GRID_COLOR,
                gridwidth=1,
                showline=True,
                linewidth=1,
                linecolor='#333',
                tickfont=dict(size=9),
                ticks='outside',
                ticklen=4,
            ),
            yaxis=dict(
                title=dict(
                    text=yaxis_title.upper() if yaxis_title else None,
                    font=dict(size=self.FONT_SIZE),
                ),
                showgrid=True,
                gridcolor=self.GRID_COLOR,
                gridwidth=1,
                showline=True,
                linewidth=1,
                linecolor='#333',
                tickfont=dict(size=9),
                ticks='outside',
                ticklen=4,
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='left',
                x=0,
                font=dict(size=9),
                bgcolor='rgba(255,255,255,0)',
                borderwidth=0,
                itemsizing='constant',
            ),
            margin=dict(l=60, r=120, t=80, b=60),
            hovermode='x unified',
        )
        
        # Add source annotation
        if show_source:
            layout['annotations'] = [
                dict(
                    text=source_text,
                    xref='paper',
                    yref='paper',
                    x=1.0,
                    y=0.02,
                    xanchor='right',
                    yanchor='bottom',
                    font=dict(size=8, color='#666', family=self.FONT_FAMILY),
                    showarrow=False,
                )
            ]
        
        return layout
        
    # -------------------------------------------------------------------------
    # Main line chart (PrismFP style)
    # -------------------------------------------------------------------------
    
    def line(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        show_avg: bool = True,
        avg_unit: str = '',
        interval: str = None,
        source: str = "SOURCE: Custom Analysis",
        show_endpoint_marker: bool = True,
    ):
        """
        Line chart matching PrismFP style.
        
        Args:
            df: DataFrame with DatetimeIndex
            cols: Columns to plot
            title: Main chart title
            subtitle: Optional subtitle (date range auto-generated if None)
            yaxis_title: Y-axis label
            show_avg: If True, add horizontal dashed lines for averages with legend labels
            avg_unit: Unit to show after average value in legend (e.g., 'nv', 'bps')
            interval: Interval description (e.g., '10MIN INTERVALS')
            source: Source text for bottom right
            show_endpoint_marker: Show 'x' marker at end of each line
        """
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        
        fig = go.Figure()
        
        for i, col in enumerate(cols):
            color = self.colors[i % len(self.colors)]
            avg = df[col].mean() if show_avg else None
            
            # Main line
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=self._format_legend_name(col, avg, avg_unit) if show_avg else col,
                line=dict(color=color, width=1.5),
                legendgroup=col,
            ))
            
            # Endpoint marker (x)
            if show_endpoint_marker and not df[col].empty:
                last_idx = df[col].last_valid_index()
                if last_idx is not None:
                    fig.add_trace(go.Scatter(
                        x=[last_idx],
                        y=[df.loc[last_idx, col]],
                        mode='markers',
                        marker=dict(symbol='x', size=8, color=color, line=dict(width=2)),
                        showlegend=False,
                        legendgroup=col,
                        hoverinfo='skip',
                    ))
            
            # Average line
            if show_avg and avg is not None:
                fig.add_trace(go.Scatter(
                    x=[df.index.min(), df.index.max()],
                    y=[avg, avg],
                    mode='lines',
                    line=dict(color=color, width=1, dash='dash'),
                    showlegend=False,
                    legendgroup=col,
                    hoverinfo='skip',
                ))
        
        # Format title
        full_title = self._format_title(
            title or 'Time Series',
            start_date=df.index.min() if len(df) > 0 else None,
            end_date=df.index.max() if len(df) > 0 else None,
            interval=interval,
        )
        
        layout = self._base_layout(
            title=full_title,
            xaxis_title='Date',
            yaxis_title=yaxis_title or 'Value',
            source_text=source,
        )
        fig.update_layout(**layout)
        
        # X-axis date format
        fig.update_xaxes(
            tickformat='%m/%d/%y\n%H:%M',
            dtick=86400000 * 2,  # Every 2 days in milliseconds
        )
        
        return fig

    # -------------------------------------------------------------------------
    # Centered Date Chart
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
        source: str = "SOURCE: Custom Analysis",
    ):
        """
        Plot time series centered around a specific date.
        
        X-axis shows days relative to center_date (e.g., -30 to +30).
        Y-axis shows change from the center_date value.
        """
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        
        center_date = pd.to_datetime(center_date)
        
        # Select columns
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to window
        start = center_date - timedelta(days=window)
        end = center_date + timedelta(days=window)
        mask = (df.index >= start) & (df.index <= end)
        data = df.loc[mask, cols].copy()
        
        if data.empty:
            raise ValueError(f"No data in window around {center_date}")
        
        # Get center values
        if center_date not in data.index:
            idx = data.index.get_indexer([center_date], method='nearest')[0]
            center_date = data.index[idx]
        
        center_values = data.loc[center_date]
        
        # Normalize
        if normalize == 'level':
            data = data - center_values
        elif normalize == 'change':
            data = data.diff().cumsum()
            data = data - data.loc[center_date]
        elif normalize == 'pct':
            data = (data / center_values - 1) * 100
        
        # Convert index to days from center
        data['days'] = (data.index - center_date).days
        data = data.set_index('days')
        
        fig = go.Figure()
        
        for i, col in enumerate(cols):
            color = self.colors[i % len(self.colors)]
            avg = data[col].mean() if show_avg else None
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                mode='lines',
                name=self._format_legend_name(col, avg, 'bps') if show_avg else col,
                line=dict(color=color, width=1.5),
            ))
            
            if show_avg and avg is not None:
                fig.add_trace(go.Scatter(
                    x=[data.index.min(), data.index.max()],
                    y=[avg, avg],
                    mode='lines',
                    line=dict(color=color, width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip',
                ))
        
        # Reference lines
        fig.add_vline(x=0, line_dash='dash', line_color='#666', line_width=1, opacity=0.7)
        fig.add_hline(y=0, line_dash='dash', line_color='#666', line_width=1, opacity=0.7)
        
        ylabel = yaxis_title or {
            'level': 'Change (bps)',
            'change': 'Cumulative Change',
            'pct': 'Change (%)',
        }.get(normalize, 'Value')
        
        full_title = self._format_title(
            title or f"Centered on {center_date.strftime('%m/%d/%y')}",
        )
        
        layout = self._base_layout(
            title=full_title,
            xaxis_title='Days from Event',
            yaxis_title=ylabel,
            source_text=source,
        )
        fig.update_layout(**layout)
        
        return fig

    # -------------------------------------------------------------------------
    # Multi-Event Overlay
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
        source: str = "SOURCE: Custom Analysis",
    ):
        """Overlay multiple events on the same chart to compare reactions."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        
        labels = labels or [pd.to_datetime(e).strftime('%m/%d/%y') for e in events]
        
        fig = go.Figure()
        
        for i, (event, label) in enumerate(zip(events, labels)):
            event = pd.to_datetime(event)
            start = event - timedelta(days=window)
            end = event + timedelta(days=window)
            
            mask = (df.index >= start) & (df.index <= end)
            data = df.loc[mask, [col]].copy()
            
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
            
            fig.add_trace(go.Scatter(
                x=data['days'],
                y=data[col],
                mode='lines',
                name=label,
                line=dict(color=color, width=1.5),
            ))
        
        fig.add_vline(x=0, line_dash='dash', line_color='#666', line_width=1, opacity=0.7)
        fig.add_hline(y=0, line_dash='dash', line_color='#666', line_width=1, opacity=0.7)
        
        layout = self._base_layout(
            title=self._format_title(title or f"Event Comparison: {col}"),
            xaxis_title='Days from Event',
            yaxis_title='Change (bps)' if normalize == 'level' else 'Change (%)',
            source_text=source,
        )
        fig.update_layout(**layout)
        
        return fig

    # -------------------------------------------------------------------------
    # Curve Snapshots
    # -------------------------------------------------------------------------
    
    def curve_snapshot(
        self,
        df: pd.DataFrame,
        dates: List[Union[str, datetime]],
        tenors: Optional[List[str]] = None,
        title: Optional[str] = None,
        source: str = "SOURCE: Custom Analysis",
    ):
        """Plot yield curves for specific dates."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        
        tenors = tenors or df.columns.tolist()
        
        fig = go.Figure()
        
        for i, date in enumerate(dates):
            date = pd.to_datetime(date)
            
            if date not in df.index:
                idx = df.index.get_indexer([date], method='nearest')[0]
                date = df.index[idx]
            
            values = df.loc[date, tenors]
            color = self.colors[i % len(self.colors)]
            
            fig.add_trace(go.Scatter(
                x=tenors,
                y=values,
                mode='lines+markers',
                name=date.strftime('%m/%d/%y'),
                line=dict(color=color, width=1.5),
                marker=dict(size=6),
            ))
        
        layout = self._base_layout(
            title=self._format_title(title or 'Yield Curve Snapshots'),
            xaxis_title='Tenor',
            yaxis_title='Yield (%)',
            source_text=source,
        )
        fig.update_layout(**layout)
        
        return fig

    # -------------------------------------------------------------------------
    # Rolling Statistics
    # -------------------------------------------------------------------------
    
    def rolling_corr(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str,
        window: int = 60,
        title: Optional[str] = None,
        source: str = "SOURCE: Custom Analysis",
    ):
        """Plot rolling correlation between two series."""
        corr = df[col1].rolling(window).corr(df[col2])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=corr.index,
            y=corr,
            mode='lines',
            name=f'{window}d Corr',
            line=dict(color=self.colors[0], width=1.5),
        ))
        
        fig.add_hline(y=0, line_dash='dash', line_color='#666', line_width=1, opacity=0.7)
        
        layout = self._base_layout(
            title=self._format_title(title or f'Rolling {window}d Correlation: {col1} vs {col2}'),
            xaxis_title='Date',
            yaxis_title='Correlation',
            source_text=source,
        )
        fig.update_layout(**layout)
        
        return fig
    
    def rolling_zscore(
        self,
        df: pd.DataFrame,
        col: str,
        window: int = 60,
        title: Optional[str] = None,
        source: str = "SOURCE: Custom Analysis",
    ):
        """Plot rolling z-score of a series."""
        mean = df[col].rolling(window).mean()
        std = df[col].rolling(window).std()
        zscore = (df[col] - mean) / std
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=zscore.index,
            y=zscore,
            mode='lines',
            name=f'{col} Z-Score',
            line=dict(color=self.colors[0], width=1.5),
        ))
        
        fig.add_hline(y=2, line_dash='dash', line_color='#C0392B', line_width=1, opacity=0.7)
        fig.add_hline(y=-2, line_dash='dash', line_color='#C0392B', line_width=1, opacity=0.7)
        fig.add_hline(y=0, line_dash='dash', line_color='#666', line_width=1, opacity=0.7)
        
        layout = self._base_layout(
            title=self._format_title(title or f'Rolling {window}d Z-Score: {col}'),
            xaxis_title='Date',
            yaxis_title='Z-Score',
            source_text=source,
        )
        fig.update_layout(**layout)
        
        return fig

    # -------------------------------------------------------------------------
    # Heatmaps
    # -------------------------------------------------------------------------
    
    def corr_heatmap(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        title: Optional[str] = None,
        source: str = "SOURCE: Custom Analysis",
    ):
        """Correlation heatmap."""
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu_r',
            zmid=0,
            text=corr.round(2).values,
            texttemplate='%{text}',
            textfont=dict(size=10, family=self.FONT_FAMILY),
        ))
        
        layout = self._base_layout(
            title=self._format_title(title or 'Correlation Matrix'),
            source_text=source,
        )
        fig.update_layout(**layout)
        
        return fig

    def changes_heatmap(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        freq: str = 'M',
        title: Optional[str] = None,
        source: str = "SOURCE: Custom Analysis",
    ):
        """Heatmap of periodic changes."""
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        
        resampled = df[cols].resample(freq).last()
        changes = resampled.diff()
        
        fig = go.Figure(data=go.Heatmap(
            z=changes.values,
            x=changes.columns,
            y=changes.index.strftime('%Y-%m'),
            colorscale='RdBu_r',
            zmid=0,
            textfont=dict(family=self.FONT_FAMILY),
        ))
        
        layout = self._base_layout(
            title=self._format_title(title or f'{freq} Changes Heatmap'),
            xaxis_title='Tenor',
            yaxis_title='Period',
            source_text=source,
        )
        fig.update_layout(**layout)
        
        return fig

    # -------------------------------------------------------------------------
    # PCA Visualization
    # -------------------------------------------------------------------------
    
    def pca_loadings(
        self,
        loadings: pd.DataFrame,
        n_components: int = 3,
        title: Optional[str] = None,
        source: str = "SOURCE: Custom Analysis",
    ):
        """Plot PCA loadings (eigenvectors)."""
        fig = go.Figure()
        
        cols = loadings.columns[:n_components]
        
        for i, col in enumerate(cols):
            fig.add_trace(go.Scatter(
                x=loadings.index,
                y=loadings[col],
                mode='lines+markers',
                name=col,
                line=dict(color=self.colors[i], width=1.5),
                marker=dict(size=6),
            ))
        
        fig.add_hline(y=0, line_dash='dash', line_color='#666', line_width=1, opacity=0.7)
        
        layout = self._base_layout(
            title=self._format_title(title or 'PCA Loadings by Tenor'),
            xaxis_title='Tenor',
            yaxis_title='Loading',
            source_text=source,
        )
        fig.update_layout(**layout)
        
        return fig
    
    def pca_variance(
        self,
        explained_variance: np.ndarray,
        title: Optional[str] = None,
        source: str = "SOURCE: Custom Analysis",
    ):
        """Plot explained variance by component (scree plot)."""
        cumulative = np.cumsum(explained_variance)
        components = [f'PC{i+1}' for i in range(len(explained_variance))]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=components,
                y=explained_variance * 100,
                name='Individual',
                marker_color=self.colors[0],
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=components,
                y=cumulative * 100,
                mode='lines+markers',
                name='Cumulative',
                line=dict(color=self.colors[1], width=1.5),
                marker=dict(size=6),
            ),
            secondary_y=True,
        )
        
        layout = self._base_layout(
            title=self._format_title(title or 'PCA Explained Variance'),
            source_text=source,
        )
        fig.update_layout(**layout)
        
        fig.update_yaxes(title_text='VARIANCE (%)', secondary_y=False)
        fig.update_yaxes(title_text='CUMULATIVE (%)', secondary_y=True)
        
        return fig
    
    # -------------------------------------------------------------------------
    # Event Annotation Helper
    # -------------------------------------------------------------------------
    
    def add_event_annotation(
        self,
        fig,
        x,
        text: str,
        y_position: float = 0.85,
    ):
        """
        Add event annotation like 'NFP RELEASE @ 09JAN 08:30 ET'.
        
        Args:
            fig: Plotly figure
            x: X position (date/datetime)
            text: Annotation text
            y_position: Y position as fraction of plot (0-1)
        """
        fig.add_vline(
            x=x, 
            line_dash='dot', 
            line_color='#333', 
            line_width=1,
        )
        
        fig.add_annotation(
            x=x,
            y=y_position,
            yref='paper',
            text=text,
            showarrow=False,
            font=dict(size=8, family=self.FONT_FAMILY, color='#333'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#999',
            borderwidth=1,
            borderpad=2,
        )
        
        return fig
    
    # -------------------------------------------------------------------------
    # Scatter plot
    # -------------------------------------------------------------------------
    
    def scatter(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        source: str = "SOURCE: Custom Analysis",
    ):
        """Scatter plot with optional color dimension."""
        fig = px.scatter(
            df.reset_index(),
            x=x,
            y=y,
            color=color,
        )
        
        # Update trace colors
        for i, trace in enumerate(fig.data):
            trace.marker.color = self.colors[i % len(self.colors)]
        
        layout = self._base_layout(
            title=self._format_title(title) if title else None,
            xaxis_title=x,
            yaxis_title=y,
            source_text=source,
        )
        fig.update_layout(**layout)
        
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