from datetime import date, timedelta

import numpy as np
import polars as pl

from backtest.engine import TradeDef
from research.dislocation import dislocation_scan
from research.dislocation_backtest import signal_frame, run_grid
from research.saved_runs import grid_spec, compare_run


def test_raw_uses_unscaled_residual_and_separate_entry_grid():
    rng = np.random.default_rng(9)
    x = rng.normal(size=420).cumsum()
    data = pl.DataFrame(dict(ts=[date(2020, 1, 1)+timedelta(days=i) for i in range(420)],
                             x=x, y=.6*x+3*rng.normal(size=420)))
    args = dict(target='y', feature='x', fit_on=['levels'], beta_lookbacks=[30],
                residual_lookbacks=[10], normalization_lookbacks=[40], thresholds=[.5],
                raw_thresholds=[2., 5.], horizons=[5], gate_names=[], device='cpu')
    _, results = dislocation_scan(data, signal_kind=['raw', 'normalized', 'ou_z'], **args)
    assert set(results.filter(pl.col('signal_kind') == 'raw')['entry_z']) == {2., 5.}
    assert set(results.filter(pl.col('signal_kind') != 'raw')['entry_z']) == {.5}
    candidate = dict(target='y', feature='x', fit_on='levels', beta_lb=30,
                     residual_lb=None, norm_lb=40, signal_kind='raw', gate='(none)')
    state = signal_frame(data, **{k: v for k, v in candidate.items() if k != 'gate'})
    np.testing.assert_allclose(state['signal'].to_numpy(), state['resid'].to_numpy(), equal_nan=True)
    other_window = signal_frame(data, **{k: v for k, v in candidate.items() if k not in ('gate', 'norm_lb')}, norm_lb=80)
    np.testing.assert_allclose(state['signal'].to_numpy(), other_window['signal'].to_numpy(), equal_nan=True)
    grid, runs = run_grid(data, TradeDef.outright('y', 'y'), candidate, entry_zs=[2.],
        exit_params={'time': [5], 'band': [0.5], 'revert_frac': [.5], 'half_life_frac': [2.]},
        execution_lag=1)
    assert len(grid) == 4
    assert set(grid['signal_units']) == {'target units'}
    assert all(run['trades'] for run in runs['runs'].values())


def test_saved_raw_threshold_coverage_is_independent():
    args = (['levels'], [30], [10], [40], [.5], [5], [], [], ['raw'], .7)
    saved = grid_spec(*args, raw_entries=[2.])
    requested = grid_spec(*args, raw_entries=[2., 5.])
    assert 'raw_entry (target units): [5.0]' in compare_run(dict(grid=saved), requested, 'hash')
