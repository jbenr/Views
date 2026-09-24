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
