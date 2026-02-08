"""PCA analysis — fit, roll, reconstruct, residuals."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


def fit_pca(df: pd.DataFrame, n_components: int = 3, use_changes: bool = False) -> dict:
    """fit PCA, returns dict: loadings, scores, eigenvalues, explained_variance, cumulative_variance, mean"""
    data = df.dropna()
    if use_changes:
        data = data.diff().dropna()

    mean = data.mean()
    centered = data - mean
    cov = centered.T @ centered / (len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov.values)

    order = np.argsort(eigenvalues)[::-1]  # descending
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    n_components = min(n_components, len(eigenvalues))
    total_var = eigenvalues.sum()

    eig = eigenvalues[:n_components]
    evec = eigenvectors[:, :n_components]
    pc_names = [f'PC{i+1}' for i in range(n_components)]

    return {
        'loadings': pd.DataFrame(evec, index=data.columns, columns=pc_names),
        'scores': pd.DataFrame(centered.values @ evec, index=data.index, columns=pc_names),
        'eigenvalues': eig,
        'explained_variance': eig / total_var,
        'cumulative_variance': np.cumsum(eig / total_var),
        'mean': mean,
    }


def roll_pca(df: pd.DataFrame, lookback: int = 252, n_components: int = 3,
             use_changes: bool = False, min_periods: int = None) -> dict:
    """rolling PCA — returns rolling explained_variance (df) and pc1_loadings (df)"""
    if min_periods is None:
        min_periods = lookback

    data = df.dropna()
    if use_changes:
        data = data.diff().dropna()

    n, n_cols = len(data), len(data.columns)
    n_components = min(n_components, n_cols)
    pc_names = [f'PC{i+1}' for i in range(n_components)]
    ev_out = np.full((n, n_components), np.nan)
    load_out = np.full((n, n_cols), np.nan)

    for i in range(min_periods - 1, n):
        s = max(0, i - lookback + 1)
        w = data.iloc[s:i+1]
        if len(w) < min_periods:
            continue

        centered = w - w.mean()
        cov = centered.T @ centered / (len(centered) - 1)
        try:
            evals, evecs = np.linalg.eigh(cov.values)
        except np.linalg.LinAlgError:
            continue

        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        total = evals.sum()
        if total > 0:
            ev_out[i, :] = evals[:n_components] / total
        load_out[i, :] = evecs[:, 0]

    return {
        'explained_variance': pd.DataFrame(ev_out, index=data.index, columns=pc_names),
        'pc1_loadings': pd.DataFrame(load_out, index=data.index, columns=data.columns),
    }


def reconstruct(result: dict, n_components: int = None) -> pd.DataFrame:
    """reconstruct data from first n components"""
    loadings, scores, mean = result['loadings'], result['scores'], result['mean']
    if n_components is not None:
        pcs = [f'PC{i+1}' for i in range(n_components)]
        loadings, scores = loadings[pcs], scores[pcs]

    return pd.DataFrame(
        scores.values @ loadings.values.T + mean.values,
        index=scores.index, columns=loadings.index,
    )


def residual_from_pca(result: dict, n_components: int = 1) -> pd.DataFrame:
    """what the first n components don't explain"""
    full = reconstruct(result)
    partial = reconstruct(result, n_components=n_components)
    return full - partial


def explain(result: dict) -> pd.DataFrame:
    """summary table: eigenvalue, variance %, cumulative %"""
    pc_names = [f'PC{i+1}' for i in range(len(result['eigenvalues']))]
    return pd.DataFrame({
        'component': pc_names,
        'eigenvalue': result['eigenvalues'],
        'variance_pct': result['explained_variance'] * 100,
        'cumulative_pct': result['cumulative_variance'] * 100,
    }).set_index('component')
