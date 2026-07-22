"""Selection-aware diagnostics for strategy research.

Implements the Deflated Sharpe Ratio (DSR) of Bailey and López de Prado and
the combinatorially symmetric cross-validation (CSCV) estimator of the
Probability of Backtest Overfitting (PBO) of Bailey et al.

Both operate on synchronous daily PnL/return paths. They assess the strategy
*selection process*, not whether the underlying economic model is causal.
Every attempted candidate from the selection stage must be represented;
running these only on reported winners understates overfitting risk.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations
from statistics import NormalDist

import numpy as np

_NORMAL = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329


def _return_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError(f"returns must be a non-empty 1D/2D array, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("returns must be finite and synchronous across trials")
    return matrix


def _column_sharpes(matrix: np.ndarray, periods_per_year: int = 252) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    return np.divide(
        mean,
        std,
        out=np.zeros_like(mean, dtype=float),
        where=std > 0,
    ) * math.sqrt(periods_per_year)


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized population-moment Sharpe, matching the backtest engine."""
    return float(_column_sharpes(_return_matrix(returns), periods_per_year)[0])


def return_moments(returns: np.ndarray) -> dict[str, float]:
    """Sample length plus population skewness and Pearson kurtosis."""
    values = _return_matrix(returns)[:, 0]
    centered = values - values.mean()
    variance = float(np.mean(centered**2))
    if variance <= 0:
        return {"n_obs": len(values), "skewness": 0.0, "kurtosis": 3.0}
    sigma = math.sqrt(variance)
    return {
        "n_obs": len(values),
        "skewness": float(np.mean(centered**3) / sigma**3),
        "kurtosis": float(np.mean(centered**4) / sigma**4),
    }


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Probability the true Sharpe exceeds an annualized benchmark."""
    values = _return_matrix(returns)[:, 0]
    moments = return_moments(values)
    sr = annualized_sharpe(values, periods_per_year) / math.sqrt(periods_per_year)
    benchmark = benchmark_sharpe / math.sqrt(periods_per_year)
    denominator = 1.0 - moments["skewness"] * sr
    denominator += ((moments["kurtosis"] - 1.0) / 4.0) * sr**2
    if len(values) < 2 or denominator <= 0:
        return float("nan")
    z = (sr - benchmark) * math.sqrt(len(values) - 1) / math.sqrt(denominator)
    return float(_NORMAL.cdf(z))


def expected_maximum_sharpe(
    trial_sharpes: np.ndarray,
    independent_trials: float | None = None,
) -> float:
    """Expected maximum annualized Sharpe across the attempted trials."""
    sharpes = np.asarray(trial_sharpes, dtype=float)
    sharpes = sharpes[np.isfinite(sharpes)]
    if len(sharpes) == 0:
        raise ValueError("trial_sharpes contains no finite values")
    n_trials = float(len(sharpes) if independent_trials is None else independent_trials)
    if not 1.0 <= n_trials <= len(sharpes):
        raise ValueError(
            f"independent_trials must be in [1, {len(sharpes)}], got {n_trials}"
        )
    mean = float(sharpes.mean())
    if len(sharpes) == 1 or n_trials == 1:
        return mean
    sigma = float(sharpes.std(ddof=1))
    if sigma == 0:
        return mean
    max_z = (
        (1.0 - _EULER_MASCHERONI) * _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
        + _EULER_MASCHERONI
        * _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )
    return mean + sigma * max_z


def effective_number_of_trials(return_paths: np.ndarray) -> tuple[float, float]:
    """Implied independent trials and mean pairwise path correlation.

    Uses the paper's linear interpolation: ``N_eff = rho + (1-rho) * M``.
    Retain the raw trial count as a more conservative alternative.
    """
    matrix = _return_matrix(return_paths)
    n_trials = matrix.shape[1]
    if n_trials == 1:
        return 1.0, 1.0
    corr = np.corrcoef(matrix, rowvar=False)
    off_diagonal = corr[np.triu_indices(n_trials, k=1)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    mean_corr = float(finite.mean()) if len(finite) else 0.0
    mean_corr = min(1.0, max(-1.0 / (n_trials - 1), mean_corr))
    implied = mean_corr + (1.0 - mean_corr) * n_trials
    return float(min(n_trials, max(1.0, implied))), mean_corr


def deflated_sharpe_ratio(
    selected_returns: np.ndarray,
    trial_sharpes: np.ndarray,
    independent_trials: float | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """DSR and its inputs for a strategy selected from ``trial_sharpes``."""
    sharpes = np.asarray(trial_sharpes, dtype=float)
    sharpes = sharpes[np.isfinite(sharpes)]
    n_independent = float(len(sharpes) if independent_trials is None else independent_trials)
    benchmark = expected_maximum_sharpe(sharpes, n_independent)
    selected = _return_matrix(selected_returns)[:, 0]
    moments = return_moments(selected)
    return {
        "dsr": probabilistic_sharpe_ratio(
            selected, benchmark_sharpe=benchmark, periods_per_year=periods_per_year
        ),
        "selected_sharpe": annualized_sharpe(selected, periods_per_year),
        "expected_max_sharpe": benchmark,
        "n_trials": float(len(sharpes)),
        "independent_trials": n_independent,
        **moments,
    }


@dataclass(frozen=True)
class PBOResult:
    """Summary of a combinatorially symmetric cross-validation run."""

    pbo: float
    probability_of_loss: float
    n_combinations: int
    n_trials: int
    n_slices: int
    mean_is_sharpe: float
    mean_oos_sharpe: float
    mean_degradation: float
    median_oos_rank: float
    is_oos_correlation: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _relative_rank(scores: np.ndarray, selected: int) -> float:
    value = scores[selected]
    less = int(np.sum(scores < value))
    equal = int(np.sum(scores == value))
    average_rank = less + (equal + 1.0) / 2.0
    return float(average_rank / (len(scores) + 1.0))


def probability_of_backtest_overfitting(
    return_paths: np.ndarray,
    n_slices: int = 8,
    periods_per_year: int = 252,
) -> PBOResult:
    """Estimate PBO with combinatorially symmetric cross-validation.

    Rows are synchronous daily PnLs/returns and columns are all candidates in
    one selection stage. Contiguous row slices are combined symmetrically;
    the IS Sharpe winner is ranked among the same candidates OOS.
    """
    matrix = _return_matrix(return_paths)
    n_obs, n_trials = matrix.shape
    if n_trials < 2:
        raise ValueError("PBO requires at least two trial paths")
    if n_slices < 2 or n_slices % 2:
        raise ValueError("n_slices must be an even integer >= 2")
    if n_slices > n_obs:
        raise ValueError("n_slices cannot exceed the number of observations")

    slices = np.array_split(np.arange(n_obs), n_slices)
    chosen_is: list[float] = []
    chosen_oos: list[float] = []
    oos_ranks: list[float] = []
    logits: list[float] = []

    for selected_slices in combinations(range(n_slices), n_slices // 2):
        selected_set = set(selected_slices)
        is_rows = np.concatenate([slices[i] for i in selected_slices])
        oos_rows = np.concatenate(
            [slices[i] for i in range(n_slices) if i not in selected_set]
        )
        is_scores = _column_sharpes(matrix[is_rows], periods_per_year)
        oos_scores = _column_sharpes(matrix[oos_rows], periods_per_year)
        winner = int(np.argmax(is_scores))
        rank = _relative_rank(oos_scores, winner)
        rank = min(1.0 - 1e-12, max(1e-12, rank))

        chosen_is.append(float(is_scores[winner]))
        chosen_oos.append(float(oos_scores[winner]))
        oos_ranks.append(rank)
        logits.append(math.log(rank / (1.0 - rank)))

    is_array = np.asarray(chosen_is)
    oos_array = np.asarray(chosen_oos)
    corr = (
        float(np.corrcoef(is_array, oos_array)[0, 1])
        if len(is_array) > 1 and is_array.std() > 0 and oos_array.std() > 0
        else float("nan")
    )
    return PBOResult(
        pbo=float(np.mean(np.asarray(logits) < 0.0)),
        probability_of_loss=float(np.mean(oos_array < 0.0)),
        n_combinations=len(logits),
        n_trials=n_trials,
        n_slices=n_slices,
        mean_is_sharpe=float(is_array.mean()),
        mean_oos_sharpe=float(oos_array.mean()),
        mean_degradation=float((oos_array - is_array).mean()),
        median_oos_rank=float(np.median(oos_ranks)),
        is_oos_correlation=corr,
    )


def event_overlap_diagnostics(
    event_indices: np.ndarray, horizon: int
) -> dict[str, float | int]:
    """Count how many threshold events have non-overlapping label windows."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    events = np.unique(np.asarray(event_indices, dtype=int))
    if len(events) == 0:
        return {
            "n_events": 0,
            "n_non_overlapping": 0,
            "overlap_fraction": 0.0,
            "median_spacing": float("nan"),
        }
    selected = []
    for event in events:
        if not selected or event >= selected[-1] + horizon:
            selected.append(int(event))
    spacing = np.diff(events)
    return {
        "n_events": len(events),
        "n_non_overlapping": len(selected),
        "overlap_fraction": float(np.mean(spacing < horizon)) if len(spacing) else 0.0,
        "median_spacing": float(np.median(spacing)) if len(spacing) else float("nan"),
    }
