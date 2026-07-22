"""Selection-aware validation diagnostics."""

import numpy as np
import pytest

from backtest.validation import (
    deflated_sharpe_ratio,
    effective_number_of_trials,
    event_overlap_diagnostics,
    probability_of_backtest_overfitting,
    probabilistic_sharpe_ratio,
)


def test_probabilistic_and_deflated_sharpe_penalize_multiple_trials():
    rng = np.random.default_rng(3)
    selected = rng.normal(0.12, 1.0, 1500)
    trials = np.linspace(-0.5, 1.2, 40)

    psr = probabilistic_sharpe_ratio(selected)
    few = deflated_sharpe_ratio(selected, trials, independent_trials=2)["dsr"]
    many = deflated_sharpe_ratio(selected, trials)["dsr"]

    assert psr > few > many
    assert 0.0 <= many <= 1.0


def test_effective_trials_reflect_path_correlation():
    rng = np.random.default_rng(7)
    independent = rng.normal(size=(2000, 8))
    common = rng.normal(size=(2000, 1))
    correlated = common + 0.05 * rng.normal(size=(2000, 8))

    n_independent, rho_independent = effective_number_of_trials(independent)
    n_correlated, rho_correlated = effective_number_of_trials(correlated)

    assert n_independent > n_correlated
    assert rho_independent < rho_correlated
    assert n_correlated == pytest.approx(1.0, abs=0.1)


def test_cscv_detects_stable_edge_and_deliberate_slice_overfit():
    rng = np.random.default_rng(11)
    stable = rng.normal(size=(800, 12))
    stable[:, 0] += 0.35
    stable_result = probability_of_backtest_overfitting(stable, n_slices=8)

    overfit = np.full((800, 8), -0.1)
    for trial, rows in enumerate(np.array_split(np.arange(800), 8)):
        overfit[rows, trial] = 1.0
    overfit_result = probability_of_backtest_overfitting(overfit, n_slices=8)

    assert stable_result.pbo < 0.2
    assert overfit_result.pbo > 0.8
    assert overfit_result.mean_degradation < 0


def test_event_overlap_diagnostics_counts_independent_windows():
    stats = event_overlap_diagnostics(np.array([0, 2, 5, 12, 14, 30]), horizon=10)
    assert stats["n_events"] == 6
    assert stats["n_non_overlapping"] == 3
    assert stats["overlap_fraction"] == pytest.approx(4 / 5)
    assert stats["median_spacing"] == 3.0
