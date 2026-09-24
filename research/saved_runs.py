"""Discovery archive matching. Historical snapshots are never merged implicitly."""
import hashlib
import json
from pathlib import Path

import polars as pl

from research import artifacts


def grid_spec(bases, beta, residual, norm, entries, horizons, gates, windows, signal, train):
    return dict(fit_on=sorted(bases), beta_lb=sorted(beta), residual_lb=sorted(residual),
                norm_lb=sorted(norm), entry_z=sorted(entries), horizon=sorted(horizons),
                gates=sorted(gates), gate_windows=sorted(windows) if gates else [],
                signal_kind=sorted([signal] if isinstance(signal, str) else signal), train_fraction=float(train))


def input_hash(stored, feature):
    columns = sorted(set([stored['target'], feature, *stored['legs'],
                          *stored.get('weight_columns', {}).values()]))
    payload = dict(legs=stored['legs'], weight_columns=stored.get('weight_columns', {}),
                   rows=[{k: row[k] for k in ['ts', *columns]} for row in stored['rows']])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run_path(run_id):
    root = artifacts.RUNS.resolve()
    path = (root / run_id).resolve()
    if path.parent != root or not path.is_dir():
        raise ValueError('Invalid saved run identifier')
    return path


def list_runs(target=None, feature=None):
    found = []
    for path in sorted(artifacts.RUNS.glob('*_discovery_*'), reverse=True):
        try:
            meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
            if not all((path / f'{name}.parquet').is_file() for name in ('data', 'results')):
                continue
            if target and meta.get('target') != target:
                continue
            if feature and meta.get('feature') != feature:
                continue
            if 'data_start' not in meta:
                dates = pl.read_parquet(path / 'data.parquet', columns=['ts'])['ts']
                meta.update(data_start=str(dates.min()), data_end=str(dates.max()), data_rows=len(dates))
            found.append(dict(meta, run_id=path.name))
        except (OSError, ValueError, pl.exceptions.PolarsError):
            continue
    return found


def compare_run(meta, requested, current_hash):
    saved = meta.get('grid')
    messages = []
    if not saved:
        messages.append('Legacy run: full requested grid was not recorded; coverage cannot be certified.')
    else:
        missing = []
        for key in ('fit_on', 'beta_lb', 'norm_lb', 'entry_z', 'horizon', 'gates', 'gate_windows'):
            extra = sorted(set(requested[key]) - set(saved[key]))
            if extra:
                missing.append(f'{key}: {extra}')
        if 'changes' in requested['fit_on']:
            extra = sorted(set(requested['residual_lb']) - set(saved['residual_lb']))
            if extra:
                missing.append(f'residual_lb: {extra}')
        saved_signals = saved['signal_kind']
        saved_signals = [saved_signals] if isinstance(saved_signals, str) else saved_signals
        requested_signals = requested['signal_kind']
        requested_signals = [requested_signals] if isinstance(requested_signals, str) else requested_signals
        missing_signals = sorted(set(requested_signals) - set(saved_signals))
        if missing_signals:
            missing.append(f'signal_kind: {missing_signals}')
        for key in ('train_fraction',):
            if saved[key] != requested[key]:
                missing.append(f'{key}: requested {requested[key]}, saved {saved[key]} (rerun required)')
        messages.append('Missing requested settings: ' + '; '.join(missing) if missing
                        else 'Saved grid covers all requested settings. Opening shows the entire saved grid.')
    if meta.get('input_sha256') == current_hash:
        messages.append('Input snapshot matches the loaded panel.')
    elif meta.get('input_sha256'):
        messages.append('Input snapshot differs: newer dates, revisions, range or weights may differ. A fresh run is required for current-data results.')
    else:
        messages.append('Legacy input fingerprint unavailable; current-data equivalence is unverified.')
    root = Path(__file__).resolve().parent.parent
    files = ['research/dislocation.py', 'backtest/lab.py', 'stats/ols.py', 'stats/ou.py']
    hashes = meta.get('source_sha256', {})
    changed = any(hashes.get(name) != hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files)
    messages.append('Calculation code differs or is unverified; rerun to use current code.' if changed
                    else 'Calculation code matches.')
    return ' '.join(messages)


def load_run(run_id, include_results=True):
    path = run_path(run_id)
    meta = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    return meta, pl.read_parquet(path/'data.parquet'), (pl.read_parquet(path/'results.parquet') if include_results else None)
