import json

from research import preferences


def test_preferences_round_trip_and_empty_features(tmp_path, monkeypatch):
    monkeypatch.setattr(preferences, 'PREFERENCES', tmp_path/'preferences.json')
    args = (['a', 'b'], ['x', 'y'], 'a', [])
    assert preferences.load_preferences(*args)['target'] == 'a'
    preferences.save_preferences('b', ['y', 'x'])
    assert preferences.load_preferences(*args) == dict(target='b', features=['y', 'x'], custom=None)
    preferences.save_preferences('b', [])
    assert preferences.load_preferences(*args)['features'] == []
    assert not list(tmp_path.glob('*.tmp'))


def test_corrupt_and_stale_preferences_fall_back(tmp_path, monkeypatch):
    path = tmp_path/'preferences.json'
    monkeypatch.setattr(preferences, 'PREFERENCES', path)
    args = (['a'], ['x'], 'a', [])
    path.write_text('{broken')
    assert preferences.load_preferences(*args) == dict(target='a', features=[], custom=None)
    path.write_text(json.dumps(dict(target='removed', features=['gone', 'x', 'x', {}])))
    assert preferences.load_preferences(*args) == dict(target='a', features=['x'], custom=None)
    preferences.save_preferences('custom', ['x'], '10y:1, 30y:-1')
    assert preferences.load_preferences(*args)['custom'] == '10y:1, 30y:-1'


def test_setup_uses_latest_file_and_failed_load_does_not_overwrite(tmp_path, monkeypatch):
    from research import app as ui
    monkeypatch.setattr(preferences, 'PREFERENCES', tmp_path/'preferences.json')
    def get_values():
        result = {}
        def visit(node):
            if getattr(node, 'id', None) in ('target', 'features', 'custom'):
                result[node.id] = node.value
            children = getattr(node, 'children', [])
            for child in children if isinstance(children, list) else [children]:
                if hasattr(child, 'to_plotly_json'):
                    visit(child)
        visit(ui.controls())
        return result
    preferences.save_preferences('10s30s', ['10y'])
    assert get_values()['target'] == '10s30s'
    assert get_values()['features'] == ['10y']
    preferences.save_preferences('custom', [], '10y:1, 30y:-1')
    assert get_values()['custom'] == '10y:1, 30y:-1'
    before = preferences.PREFERENCES.read_bytes()
    app = ui.build_app()
    callback = next(v['callback'].__wrapped__ for v in app.callback_map.values()
                    if v['callback'].__wrapped__.__name__ == '_load')
    def fail(*args, **kwargs):
        raise ValueError('test load failure')
    monkeypatch.setattr(ui, 'build_panel', fail)
    callback(1, 0, '10s30s', None, 'fixed', 126, None, ['10y'], '2000-01-01', [], '6M', 'test')
    assert preferences.PREFERENCES.read_bytes() == before
