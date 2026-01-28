#!/usr/bin/env python3
"""
Visualization and time series analysis helpers for rates/fixed income work.

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
from typing import List, Optional, Union
from datetime import datetime, timedelta

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


class Viz:
    """Time series visualization toolkit for rates analysis."""

    def __init__(self, style: str = 'plotly'):
        """
        Initialize visualizer.
        
        Args:
            style: 'plotly' or 'seaborn' (default plotting backend)
        """
        self.style = style
        self.colors = px.colors.qualitative.Set2
        
        # Seaborn defaults
        sns.set_theme(style='whitegrid')
        
    # -------------------------------------------------------------------------
    # Centered Date Chart
    # -------------------------------------------------------------------------
    
    def centered_chart(
        self,
        df: pd.DataFrame,
        center_date: Union[str, datetime],
        window: int = 30,
        cols: Optional[List[str]] = None,
        normalize: str = 'level',  # 'level', 'change', 'pct'
        title: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        """
        Plot time series centered around a specific date.
        
        X-axis shows days relative to center_date (e.g., -30 to +30).
        Y-axis shows change from the center_date value.
        
        Args:
            df: DataFrame with DatetimeIndex and numeric columns
            center_date: The date to center on (becomes x=0)
            window: Days before and after center_date to show
            cols: Columns to plot (default: all numeric)
            normalize: 
                'level' - show absolute change from center_date value
                'change' - show cumulative change (diff) from center
                'pct' - show percent change from center_date value
            title: Chart title
            backend: Override default backend ('plotly' or 'seaborn')
        """
        backend = backend or self.style
        
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
            # Find nearest date
            idx = data.index.get_indexer([center_date], method='nearest')[0]
            center_date = data.index[idx]
        
        center_values = data.loc[center_date]
        
        # Normalize
        if normalize == 'level':
            # Change from center value (in original units, e.g., bps)
            data = data - center_values
        elif normalize == 'change':
            # Cumulative sum of daily changes, zeroed at center
            data = data.diff().cumsum()
            data = data - data.loc[center_date]
        elif normalize == 'pct':
            # Percent change from center
            data = (data / center_values - 1) * 100
        
        # Convert index to days from center
        data['days'] = (data.index - center_date).days
        data = data.set_index('days')
        
        # Plot
        if backend == 'plotly':
            return self._centered_plotly(data, cols, center_date, normalize, title)
        else:
            return self._centered_seaborn(data, cols, center_date, normalize, title)
    
    def _centered_plotly(self, data, cols, center_date, normalize, title):
        fig = go.Figure()
        
        for i, col in enumerate(cols):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                mode='lines',
                name=col,
                line=dict(color=self.colors[i % len(self.colors)], width=2),
            ))
        
        # Add vertical line at center
        fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.5)
        
        # Add horizontal line at 0
        fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
        
        ylabel = {
            'level': 'Change (bps)',
            'change': 'Cumulative Change',
            'pct': 'Change (%)',
        }.get(normalize, 'Value')
        
        fig.update_layout(
            title=title or f"Centered on {center_date.strftime('%Y-%m-%d')}",
            xaxis_title='Days from Event',
            yaxis_title=ylabel,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        
        return fig
    
    def _centered_seaborn(self, data, cols, center_date, normalize, title):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in cols:
            ax.plot(data.index, data[col], label=col, linewidth=2)
        
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        ylabel = {
            'level': 'Change (bps)',
            'change': 'Cumulative Change',
            'pct': 'Change (%)',
        }.get(normalize, 'Value')
        
        ax.set_xlabel('Days from Event')
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"Centered on {center_date.strftime('%Y-%m-%d')}")
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1.15), ncol=len(cols))
        
        plt.tight_layout()
        return fig, ax

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
    ):
        """
        Overlay multiple events on the same chart to compare reactions.
        
        Args:
            df: DataFrame with DatetimeIndex
            events: List of dates to overlay
            col: Single column to plot
            window: Days before/after each event
            labels: Optional labels for each event
            normalize: 'level', 'change', or 'pct'
            title: Chart title
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        
        labels = labels or [str(e) for e in events]
        
        fig = go.Figure()
        
        for i, (event, label) in enumerate(zip(events, labels)):
            event = pd.to_datetime(event)
            start = event - timedelta(days=window)
            end = event + timedelta(days=window)
            
            mask = (df.index >= start) & (df.index <= end)
            data = df.loc[mask, [col]].copy()
            
            if data.empty:
                continue
            
            # Find nearest center
            if event not in data.index:
                idx = data.index.get_indexer([event], method='nearest')[0]
                event = data.index[idx]
            
            center_val = data.loc[event, col]
            
            if normalize == 'level':
                data[col] = data[col] - center_val
            elif normalize == 'pct':
                data[col] = (data[col] / center_val - 1) * 100
            
            data['days'] = (data.index - event).days
            
            fig.add_trace(go.Scatter(
                x=data['days'],
                y=data[col],
                mode='lines',
                name=label,
                line=dict(color=self.colors[i % len(self.colors)], width=2),
            ))
        
        fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.5)
        fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
        
        fig.update_layout(
            title=title or f"Event Comparison: {col}",
            xaxis_title='Days from Event',
            yaxis_title='Change (bps)' if normalize == 'level' else 'Change (%)',
            hovermode='x unified',
            template='plotly_white',
        )
        
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
    ):
        """
        Plot yield curves for specific dates.
        
        Args:
            df: DataFrame with DatetimeIndex, columns are tenors
            dates: Dates to snapshot
            tenors: Column order (for x-axis). If None, uses all columns.
            title: Chart title
        """
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
            
            fig.add_trace(go.Scatter(
                x=tenors,
                y=values,
                mode='lines+markers',
                name=date.strftime('%Y-%m-%d'),
                line=dict(color=self.colors[i % len(self.colors)], width=2),
            ))
        
        fig.update_layout(
            title=title or 'Yield Curve Snapshots',
            xaxis_title='Tenor',
            yaxis_title='Yield (%)',
            hovermode='x unified',
            template='plotly_white',
        )
        
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
    ):
        """Plot rolling correlation between two series."""
        corr = df[col1].rolling(window).corr(df[col2])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=corr.index,
            y=corr,
            mode='lines',
            name=f'{window}d Correlation',
            line=dict(color=self.colors[0], width=2),
        ))
        
        fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
        
        fig.update_layout(
            title=title or f'Rolling {window}d Correlation: {col1} vs {col2}',
            xaxis_title='Date',
            yaxis_title='Correlation',
            template='plotly_white',
        )
        
        return fig
    
    def rolling_zscore(
        self,
        df: pd.DataFrame,
        col: str,
        window: int = 60,
        title: Optional[str] = None,
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
            line=dict(color=self.colors[0], width=2),
        ))
        
        # Add bands
        fig.add_hline(y=2, line_dash='dash', line_color='red', opacity=0.5)
        fig.add_hline(y=-2, line_dash='dash', line_color='red', opacity=0.5)
        fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
        
        fig.update_layout(
            title=title or f'Rolling {window}d Z-Score: {col}',
            xaxis_title='Date',
            yaxis_title='Z-Score',
            template='plotly_white',
        )
        
        return fig

    # -------------------------------------------------------------------------
    # Heatmaps
    # -------------------------------------------------------------------------
    
    def corr_heatmap(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        title: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        """Correlation heatmap."""
        backend = backend or self.style
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr = df[cols].corr()
        
        if backend == 'plotly':
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale='RdBu_r',
                zmid=0,
                text=corr.round(2).values,
                texttemplate='%{text}',
                textfont=dict(size=10),
            ))
            
            fig.update_layout(
                title=title or 'Correlation Matrix',
                template='plotly_white',
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
            ax.set_title(title or 'Correlation Matrix')
            plt.tight_layout()
            return fig, ax

    def changes_heatmap(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        freq: str = 'M',
        title: Optional[str] = None,
    ):
        """
        Heatmap of periodic changes (e.g., monthly changes by tenor).
        
        Args:
            df: DataFrame with DatetimeIndex
            cols: Columns to include
            freq: Resampling frequency ('M', 'W', 'Q')
            title: Chart title
        """
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Resample and get period-over-period change
        resampled = df[cols].resample(freq).last()
        changes = resampled.diff()
        
        fig = go.Figure(data=go.Heatmap(
            z=changes.values,
            x=changes.columns,
            y=changes.index.strftime('%Y-%m'),
            colorscale='RdBu_r',
            zmid=0,
        ))
        
        fig.update_layout(
            title=title or f'{freq} Changes Heatmap',
            xaxis_title='Tenor',
            yaxis_title='Period',
            template='plotly_white',
        )
        
        return fig

    # -------------------------------------------------------------------------
    # PCA Visualization
    # -------------------------------------------------------------------------
    
    def pca_loadings(
        self,
        loadings: pd.DataFrame,
        n_components: int = 3,
        title: Optional[str] = None,
    ):
        """
        Plot PCA loadings (eigenvectors) for interpretation.
        
        Args:
            loadings: DataFrame where rows are tenors, columns are PC1, PC2, etc.
            n_components: Number of components to plot
            title: Chart title
        """
        fig = go.Figure()
        
        cols = loadings.columns[:n_components]
        
        for i, col in enumerate(cols):
            fig.add_trace(go.Scatter(
                x=loadings.index,
                y=loadings[col],
                mode='lines+markers',
                name=col,
                line=dict(color=self.colors[i], width=2),
            ))
        
        fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
        
        fig.update_layout(
            title=title or 'PCA Loadings by Tenor',
            xaxis_title='Tenor',
            yaxis_title='Loading',
            template='plotly_white',
        )
        
        return fig
    
    def pca_variance(
        self,
        explained_variance: np.ndarray,
        title: Optional[str] = None,
    ):
        """
        Plot explained variance by component (scree plot).
        
        Args:
            explained_variance: Array of explained variance ratios
            title: Chart title
        """
        cumulative = np.cumsum(explained_variance)
        components = [f'PC{i+1}' for i in range(len(explained_variance))]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart for individual variance
        fig.add_trace(
            go.Bar(
                x=components,
                y=explained_variance * 100,
                name='Individual',
                marker_color=self.colors[0],
            ),
            secondary_y=False,
        )
        
        # Line for cumulative
        fig.add_trace(
            go.Scatter(
                x=components,
                y=cumulative * 100,
                mode='lines+markers',
                name='Cumulative',
                line=dict(color=self.colors[1], width=2),
            ),
            secondary_y=True,
        )
        
        fig.update_layout(
            title=title or 'PCA Explained Variance',
            template='plotly_white',
        )
        
        fig.update_yaxes(title_text='Variance (%)', secondary_y=False)
        fig.update_yaxes(title_text='Cumulative (%)', secondary_y=True)
        
        return fig

    # -------------------------------------------------------------------------
    # Quick Plots
    # -------------------------------------------------------------------------
    
    def line(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        title: Optional[str] = None,
    ):
        """Simple line chart."""
        cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
        
        fig = go.Figure()
        
        for i, col in enumerate(cols):
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=self.colors[i % len(self.colors)], width=2),
            ))
        
        fig.update_layout(
            title=title,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        
        return fig
    
    def scatter(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
    ):
        """Scatter plot with optional color dimension."""
        fig = px.scatter(
            df.reset_index(),
            x=x,
            y=y,
            color=color,
            title=title,
            template='plotly_white',
        )
        
        return fig


# -----------------------------------------------------------------------------
# Convenience functions (no class instantiation needed)
# -----------------------------------------------------------------------------

_viz = Viz()

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