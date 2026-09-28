import json
from pathlib import Path

import polars as pl
import pytest

from research import artifacts
from research.saved_runs import grid_spec, input_hash, list_runs, compare_run, load_run, run_path


def spec():
    return grid_spec(['levels'], [63], [10], [126], [1.0], [5], [], [], 'ou_z', .7)


def test_archive_coverage_and_legacy_status(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, 'RUNS', tmp_path)
    data = pl.DataFrame({'ts': ['2024-01-01', '2024-01-02'], 'y': [1., 2.], 'x': [2., 3.]})
    stored = dict(target='y', legs={'y': 1}, rows=data.to_dicts())
    fingerprint = input_hash(stored, 'x')
    path = Path(artifacts.save_run('discovery', data, pl.DataFrame({'ic': [.2]}),
                dict(target='y', feature='x', grid=spec(), input_sha256=fingerprint)))
    meta = list_runs('y', 'x')[0]
    assert meta['data_end'] == '2024-01-02'
    report = compare_run(meta, spec(), fingerprint)
    assert 'covers all requested settings' in report
    assert 'snapshot matches' in report
    assert 'code matches' in report
    expanded = dict(spec(), beta_lb=[63, 126])
    assert 'beta_lb: [126]' in compare_run(meta, expanded, fingerprint)
    revised = dict(stored, rows=[dict(row, y=10.) for row in stored['rows']])
    assert input_hash(revised, 'x') != fingerprint
    assert 'snapshot differs' in compare_run(meta, spec(), input_hash(revised, 'x'))
    legacy = dict(meta)
    legacy.pop('grid')
    legacy.pop('input_sha256')
    assert 'coverage cannot be certified' in compare_run(legacy, spec(), fingerprint)
    assert load_run(path.name)[1].equals(data)
    assert load_run(path.name, include_results=False)[2] is None
    assert list_runs('other', 'x') == []
    assert 'train_fraction' in compare_run(meta, dict(spec(), train_fraction=.8), fingerprint)


def test_incomplete_runs_and_path_escape_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, 'RUNS', tmp_path)
    incomplete = tmp_path/'20240101_discovery_incomplete'
    incomplete.mkdir()
    (incomplete/'metadata.json').write_text(json.dumps({'target': 'y'}))
    assert list_runs() == []
    with pytest.raises(ValueError):
        run_path('..')


def test_target_definition_separates_fixed_beta_and_missing_history():
    from research.saved_runs import definition_match, target_label
    current = dict(target='10s30s', weighting='beta', legs={'10y': 1., '30y': -1.},
                   beta_lookback=126, beta_dependent='30y', weight_columns={'10y': 'w10', '30y': 'w30'})
    assert definition_match(dict(current), current) == 'match'
    assert definition_match(dict(current, weighting='fixed'), current) == 'different'
    assert definition_match(dict(current, beta_lookback=252), current) == 'different'
    assert definition_match(dict(current, beta_dependent='10y'), current) == 'different'
    assert definition_match(dict(current, beta_lookback=None), current) == 'unverified'
    assert definition_match(dict(target='10s30s'), current) == 'unverified'
    assert 'hedge beta lookback 126' in target_label(current)
    assert 'fixed-weight' in target_label(dict(target='10s30s', weighting='fixed'))


def test_archive_lookup_hides_other_target_definitions(tmp_path, monkeypatch):
    from research import app as ui
    monkeypatch.setattr(artifacts, 'RUNS', tmp_path)
    data = pl.DataFrame(dict(ts=['2024-01-01', '2024-01-02'], y=[1., 2.], x=[2., 3.], wy=[1., 1.], wx=[-1., -1.]))
    current = dict(target='y', feature='x', features=['x'], weighting='beta', legs={'y': 1., 'x': -1.},
                   beta_lookback=126, beta_dependent='x', weight_columns={'y': 'wy', 'x': 'wx'}, rows=data.to_dicts())
    paths = []
    for extra in [{}, dict(weighting='fixed'), dict(beta_lookback=252), dict(beta_lookback=None)]:
        meta = {k: v for k, v in dict(current, **extra).items() if k not in ('rows', 'features')}
        paths.append(Path(artifacts.save_run('discovery', data, pl.DataFrame({'ic': [.2]}), meta)).name)
    app = ui.build_app()
    callback = next(v['callback'].__wrapped__ for v in app.callback_map.values()
                    if v['callback'].__wrapped__.__name__ == '_find_saved')
    args = (current, 'x', ['levels'], [63], [10], [126], [1.], [5], [], [126], ['ou_z'], .7, None, None)
    options, chosen, query = callback(*args)
    assert [o['value'] for o in options] == [paths[0]]
    assert chosen == paths[0]
    assert query['hidden_definitions'] == 3
    assert 'beta-weighted' in options[0]['label']
    options, _, _ = callback(*args, None, ['show'])
    assert len(options) == 4
    assert app.server.test_client().get('/_dash-dependencies').status_code == 200
