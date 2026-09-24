import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from research.dislocation import dislocation_scan
from research.saved_runs import grid_spec, compare_run


def test_combined_signals_equal_separate_scans_and_rank_independently():
    from research.app import dislocation_view
    from datetime import date, timedelta
    rng = np.random.default_rng(14)
    x = rng.normal(size=420).cumsum()
    data = pl.DataFrame(dict(ts=[date(2020, 1, 1)+timedelta(days=i) for i in range(420)],
                             x=x, y=.7*x+rng.normal(size=420)))
    args = dict(target='y', feature='x', beta_lookbacks=[30], residual_lookbacks=[10],
                normalization_lookbacks=[40], thresholds=[.5], horizons=[5],
                gate_names=['resid_phi'], gate_windows=[126], fit_on=['levels', 'changes'],
                train_fraction=.7, device='cpu')
    _, combined = dislocation_scan(data, signal_kind=['normalized', 'ou_z'], **args)
    sort = ['signal_kind', 'fit_on', 'gate', 'gate_bucket', 'sample']
    independent = []
    for signal in ['normalized', 'ou_z']:
        _, single = dislocation_scan(data, signal_kind=signal, **args)
        assert_frame_equal(combined.filter(pl.col('signal_kind') == signal).sort(sort), single.sort(sort))
        independent.append(single)
    info = dict(target='y', train_fraction=.7, backend='cpu', elapsed_s=1, cells_per_second=1)
    _, rows = dislocation_view(combined, 'x', info, min_events=1)
    assert len(rows) <= 40
    for row in rows:
        _, single_rows = dislocation_view(combined.filter(pl.col('signal_kind') == row['signal_kind']), 'x', info, min_events=1)
        match = next(r for r in single_rows if all(r[k] == row[k] for k in sort if k != 'sample'))
        assert row['model_family_median_ic'] == match['model_family_median_ic']
        assert row['later_ic'] == match['later_ic']


def test_select_all_uses_every_live_option_without_changing_defaults():
    from research.app import build_app, dislocation_tab
    app = build_app()
    controls = {}
    def visit(component):
        if getattr(component, 'id', None) and isinstance(component.id, str):
            controls[component.id] = component
        children = getattr(component, 'children', [])
        for child in children if isinstance(children, (list, tuple)) else [children]:
            if hasattr(child, 'to_plotly_json'):
                visit(child)
    visit(dislocation_tab())
    callback = next(v for v in app.callback_map.values()
                    if v['callback'].__wrapped__.__name__ == '_select_all_sweeps')
    options = [controls[s['id']].options for s in callback['state']]
    actual = callback['callback'].__wrapped__(1, *options)
    assert actual == [[o['value'] for o in group if not o.get('disabled', False)] for group in options]
    assert controls['dis-signal'].value == ['normalized']
    assert actual[0] == ['normalized', 'ou_z', 'raw']
    assert controls['dis-train'].value == .7
    assert app.server.test_client().get('/_dash-dependencies').status_code == 200


def test_legacy_scalar_signal_archive_can_cover_single_signal_request():
    request = grid_spec(['levels'], [63], [10], [126], [1], [5], [], [], ['normalized'], .7)
    legacy = dict(request, signal_kind='normalized')
    assert 'covers all requested' in compare_run(dict(grid=legacy), request, 'hash')
    both = dict(request, signal_kind=['normalized', 'ou_z'])
    assert "signal_kind: ['ou_z']" in compare_run(dict(grid=legacy), both, 'hash')
