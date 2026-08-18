"""PCA analysis — fit, roll, reconstruct, residuals.

Polars-native I/O. NumPy for eigendecomposition (cannot be replaced).
"""

from __future__ import annotations
import numpy as np
import polars as pl
import pandas as pd
from dataclasses import dataclass
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


def _roll_pc1(
    mat: np.ndarray, lookback: int, min_periods: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shared rolling PC1 pass: returns (scores, loadings) for one matrix.

    Loadings are returned alongside the score because they are what tells you
    what the factor is made of -- see pc1_self_weight, which needs them to
    measure how much of a target spread the "hedge" already contains.
    """
    n, k = mat.shape
    scores = np.full(n, np.nan)
    loadings = np.full((n, k), np.nan)

    for i in range(min_periods - 1, n):
        w = mat[max(0, i - lookback + 1) : i + 1]
        if len(w) < min_periods or np.isnan(w).any():
            continue
        mean = w.mean(axis=0)
        centered = w - mean
        cov = centered.T @ centered / (len(centered) - 1)
        try:
            _, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        pc1 = evecs[:, -1]  # eigh sorts ascending: last = largest eigenvalue
        if pc1.mean() < 0:
            pc1 = -pc1
        scores[i] = (mat[i] - mean) @ pc1
        loadings[i] = pc1

    return scores, loadings


def roll_pc1_score(
    df: Union[pl.DataFrame, pd.DataFrame],
    lookback: int = 252,
    min_periods: int = None,
) -> pl.Series:
    """Point-in-time PC1 score: at each bar, fit PCA on the trailing window
    and project the CURRENT observation onto that window's first component.

    No lookahead (each bar only sees its own window), and loadings are
    sign-fixed (mean loading forced positive) so on a yield panel PC1 is the
    "level" factor with a stable sign through time. Output has the same
    length as the input; warmup rows and rows in windows containing nulls
    are null.
    """
    if min_periods is None:
        min_periods = lookback
    scores, _ = _roll_pc1(to_pl_df(df).to_numpy().astype(float), lookback, min_periods)
    return pl.Series("pc1", scores).fill_nan(None)


def roll_pc1_loadings(
    df: Union[pl.DataFrame, pd.DataFrame],
    lookback: int = 252,
    min_periods: int = None,
) -> pl.DataFrame:
    """Per-bar PC1 loadings, one column per input series, same rolling window
    and sign convention as roll_pc1_score."""
    if min_periods is None:
        min_periods = lookback
    frame = to_pl_df(df)
    _, loadings = _roll_pc1(frame.to_numpy().astype(float), lookback, min_periods)
    return pl.DataFrame(
        {c: pl.Series(loadings[:, i]).fill_nan(None)
         for i, c in enumerate(frame.columns)}
    )


def pc1_self_weight(
    loadings: pl.DataFrame, short: str, long: str
) -> pl.Series:
    """How much of the spread (long - short) the PC1 score itself contains.

    Writing the two legs in terms of their mean m and spread s = long - short,
    the PC1 score's contribution from those legs is

        v_short*short + v_long*long = (v_short + v_long)*m + (v_long - v_short)/2 * s

    so the factor carries the spread with coefficient (v_long - v_short)/2.
    When a panel includes both legs of the curve being modelled, that term is
    the target leaking into its own explanatory variable: the regression then
    lets part of the target explain itself, which shrinks the residual and
    makes it look more mean-reverting than it is.

    Exactly zero when the two legs load equally -- which is why an
    equal-weighted level factor is clean by construction and a fitted PC1 is
    not. Returns the per-bar coefficient; take abs().mean() for a summary.
    """
    for leg in (short, long):
        if leg not in loadings.columns:
            raise ValueError(f"leg {leg!r} not in loadings columns {loadings.columns}")
    return ((loadings[long] - loadings[short]) / 2.0).alias("self_weight")


@dataclass(frozen=True)
class PCSpec:
    """A named recipe for a rolling level-factor feature: which tenors, what
    window, fitted or equal-weighted.

    Exists so the input panel is a variable you can test rather than a
    constant baked into each strategy module. The choice matters: a factor
    built from a panel that contains the target curve's own legs carries that
    curve inside it (see pc1_self_weight), so the model is partly explaining
    the target with itself.

    method:
        "pca"    fitted first principal component, sign-fixed. Data-driven
                 weights, but they are unequal, so a panel containing both
                 target legs leaks the target into the feature.
        "equal"  equal-weighted mean of the panel, demeaned on the same
                 trailing window. Leaks exactly zero when both legs are in
                 the panel, at the cost of not letting the data pick weights.

    Naming stays backward compatible: an unlabelled 252d spec is "pc1_252",
    matching what the curve book already writes by hand.
    """

    cols: tuple[str, ...]
    lookback: int = 252
    method: str = "pca"
    label: str = ""

    def __post_init__(self):
        if self.method not in {"pca", "equal"}:
            raise ValueError(f"unknown method={self.method!r}; expected 'pca' or 'equal'")
        if not self.cols:
            raise ValueError("PCSpec needs at least one column")

    @property
    def name(self) -> str:
        """Feature column name, e.g. 'pc1_252' or 'pc1_front_252'."""
        stem = "pc1" if self.method == "pca" else "lvl"
        return f"{stem}_{self.label}_{self.lookback}" if self.label else f"{stem}_{self.lookback}"

    def _panel(self, data: pl.DataFrame) -> pl.DataFrame:
        missing = [c for c in self.cols if c not in data.columns]
        if missing:
            raise ValueError(f"{self.name}: columns missing from data: {missing}")
        return data.select(["ts", *self.cols]).drop_nulls()

    def score(self, data: pl.DataFrame) -> pl.DataFrame:
        """ts + this spec's factor score, over rows where the panel is complete."""
        panel = self._panel(data)
        values = panel.select(self.cols)
        if self.method == "pca":
            series = roll_pc1_score(values, lookback=self.lookback)
        else:
            mean = values.select(pl.mean_horizontal(pl.all()).alias("m"))["m"]
            series = (mean - mean.rolling_mean(self.lookback)).fill_nan(None)
        return panel.select("ts").with_columns(series.alias(self.name))

    def loadings(self, data: pl.DataFrame) -> pl.DataFrame:
        """Per-bar weights on each input. Constant 1/k for method='equal'."""
        panel = self._panel(data)
        values = panel.select(self.cols)
        if self.method == "pca":
            frame = roll_pc1_loadings(values, lookback=self.lookback)
        else:
            w = 1.0 / len(self.cols)
            frame = pl.DataFrame({c: [w] * len(panel) for c in self.cols})
        return panel.select("ts").hstack(frame)

    def self_weight(self, data: pl.DataFrame, short: str, long: str) -> pl.Series:
        """Per-bar coefficient on (long - short) inside this factor.

        Zero when a leg is absent from the panel: a factor that never saw the
        leg cannot carry it directly.
        """
        if short not in self.cols or long not in self.cols:
            return pl.Series("self_weight", [0.0] * len(self._panel(data)))
        return pc1_self_weight(self.loadings(data), short, long)

    def attach(self, data: pl.DataFrame) -> pl.DataFrame:
        """Feature hook for Strategy(feature_fn=...): data + this spec's column."""
        return data.join(self.score(data), on="ts", how="left")


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
