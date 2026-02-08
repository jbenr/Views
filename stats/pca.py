"""PCA analysis — fit, roll, reconstruct, residuals.

Polars-native I/O. NumPy for eigendecomposition (cannot be replaced).
"""

from __future__ import annotations
import numpy as np
import polars as pl
import pandas as pd
from typing import Optional, Union
from utils.helpers import to_pl_df


def fit_pca(
    df: Union[pl.DataFrame, pd.DataFrame],
    n_components: int = 3,
    use_changes: bool = False,
) -> dict:
    """Fit PCA. Returns dict: loadings, scores, eigenvalues,
    explained_variance, cumulative_variance, mean.
    """
    data = to_pl_df(df).drop_nulls()
    if use_changes:
        data = data.select(pl.all().diff()).slice(1)

    cols = data.columns
    mat = data.to_numpy().astype(float)
    mean = mat.mean(axis=0)
    centered = mat - mean

    cov = centered.T @ centered / (len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    n_components = min(n_components, len(eigenvalues))
    total_var = eigenvalues.sum()

    eig = eigenvalues[:n_components]
    evec = eigenvectors[:, :n_components]
    pc_names = [f"PC{i+1}" for i in range(n_components)]

    return {
        "loadings": pl.DataFrame(
            {pc: evec[:, i] for i, pc in enumerate(pc_names)},
        ).with_columns(pl.Series("column", cols)),
        "scores": pl.DataFrame(
            {pc: (centered @ evec)[:, i] for i, pc in enumerate(pc_names)},
        ),
        "eigenvalues": eig,
        "explained_variance": eig / total_var,
        "cumulative_variance": np.cumsum(eig / total_var),
        "mean": mean,
        "_columns": cols,
    }


def roll_pca(
    df: Union[pl.DataFrame, pd.DataFrame],
    lookback: int = 252,
    n_components: int = 3,
    use_changes: bool = False,
    min_periods: int = None,
) -> dict:
    """Rolling PCA — returns rolling explained_variance and pc1_loadings.

    Loop stays (eigendecomposition is inherently sequential) but operates
    on pre-extracted numpy array for speed.
    """
    if min_periods is None:
        min_periods = lookback

    data = to_pl_df(df).drop_nulls()
    if use_changes:
        data = data.select(pl.all().diff()).slice(1)

    cols = data.columns
    mat = data.to_numpy().astype(float)
    n, n_cols = mat.shape
    n_components = min(n_components, n_cols)
    pc_names = [f"PC{i+1}" for i in range(n_components)]
    ev_out = np.full((n, n_components), np.nan)
    load_out = np.full((n, n_cols), np.nan)

    for i in range(min_periods - 1, n):
        s = max(0, i - lookback + 1)
        w = mat[s : i + 1]
        if len(w) < min_periods:
            continue

        centered = w - w.mean(axis=0)
        cov = centered.T @ centered / (len(centered) - 1)
        try:
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue

        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        total = evals.sum()
        if total > 0:
            ev_out[i, :] = evals[:n_components] / total
        load_out[i, :] = evecs[:, 0]

    return {
        "explained_variance": pl.DataFrame(
            {pc: ev_out[:, i] for i, pc in enumerate(pc_names)}
        ),
        "pc1_loadings": pl.DataFrame(
            {col: load_out[:, i] for i, col in enumerate(cols)}
        ),
    }


def reconstruct(result: dict, n_components: int = None) -> pl.DataFrame:
    """Reconstruct data from first n components."""
    loadings = result["loadings"].drop("column").to_numpy()
    scores = result["scores"].to_numpy()
    mean = result["mean"]
    cols = result["_columns"]

    if n_components is not None:
        loadings = loadings[:, :n_components]
        scores = scores[:, :n_components]

    reconstructed = scores @ loadings.T + mean
    return pl.DataFrame({col: reconstructed[:, i] for i, col in enumerate(cols)})


def residual_from_pca(result: dict, n_components: int = 1) -> pl.DataFrame:
    """What the first n components don't explain."""
    full = reconstruct(result)
    partial = reconstruct(result, n_components=n_components)

    full_np = full.to_numpy()
    partial_np = partial.to_numpy()
    residual = full_np - partial_np

    cols = result["_columns"]
    return pl.DataFrame({col: residual[:, i] for i, col in enumerate(cols)})


def explain(result: dict) -> pl.DataFrame:
    """Summary table: eigenvalue, variance %, cumulative %."""
    pc_names = [f"PC{i+1}" for i in range(len(result["eigenvalues"]))]
    return pl.DataFrame(
        {
            "component": pc_names,
            "eigenvalue": result["eigenvalues"],
            "variance_pct": result["explained_variance"] * 100,
            "cumulative_pct": result["cumulative_variance"] * 100,
        }
    )
