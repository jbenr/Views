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
            if getattr(node, 'id', None) in ('target', 'features', 'custom', 'derived-features'):
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
    preferences.save_preferences('10s30s', ['10y', '10y~2y@126', 'bad~spec@1'])
    assert get_values()['features'] == ['10y']
    assert get_values()['derived-features'] == '10y~2y@126'
    preferences.save_preferences('custom', [], '10y:1, 30y:-1')
    assert get_values()['custom'] == '10y:1, 30y:-1'
    before = preferences.PREFERENCES.read_bytes()
    app = ui.build_app()
    callback = next(v['callback'].__wrapped__ for v in app.callback_map.values()
                    if v['callback'].__wrapped__.__name__ == '_load')
    def fail(*args, **kwargs):
        raise ValueError('test load failure')
    monkeypatch.setattr(ui, 'build_panel', fail)
    callback(1, 0, '10s30s', None, 'fixed', 126, None, ['10y'], '2000-01-01', [], None, '6M', 'test')
    assert preferences.PREFERENCES.read_bytes() == before


def test_controls_and_setup_selections_share_the_file_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr(preferences, 'PREFERENCES', tmp_path/'preferences.json')
    assert preferences.load_controls() == {}
    preferences.save_preferences('b', ['x'])
    preferences.save_controls({'dis-cv': 5})
    preferences.save_preferences('a', ['y'])
    assert preferences.load_controls() == {'dis-cv': 5}
    assert preferences.load_preferences(['a', 'b'], ['x', 'y'], 'a', [])['features'] == ['y']
    (tmp_path/'preferences.json').write_text('{broken')
    assert preferences.load_controls() == {}


def test_research_controls_come_back_on_the_next_page_and_stale_values_are_dropped(tmp_path, monkeypatch):
    from research import app as ui
    monkeypatch.setattr(preferences, 'PREFERENCES', tmp_path/'preferences.json')
    app = ui.build_app()
    remember = next(v['callback'].__wrapped__ for v in app.callback_map.values()
                    if v['callback'].__wrapped__.__name__ == '_remember_controls')
    values = {key: None for key in ui.REMEMBERED}
    values.update({'dis-cv': 5, 'dis-thresholds': [1.5, 99.0], 'bt-stop': [0.0, 25.0], 'dis-min-events': 45,
                   'dis-exit-styles': ['band', 'time'], 'dis-rank': 'not-a-rule', 'bt-signal-stops': [7.7]})
    assert remember(*[values[key] for key in ui.REMEMBERED]) is True

    def control(tree, key):
        if getattr(tree, 'id', None) == key:
            return tree
        children = getattr(tree, 'children', None)
        for child in children if isinstance(children, (list, tuple)) else [children]:
            if hasattr(child, 'to_plotly_json') and (found := control(child, key)) is not None:
                return found

    page = ui.dislocation_tab()
    assert control(page, 'dis-cv').value == 5
    assert control(page, 'dis-thresholds').value == [1.5]  # 99 is not offered any more
    assert control(page, 'bt-stop').value == [0.0, 25.0]
    assert control(page, 'dis-min-events').value == 45
    assert control(page, 'dis-exit-styles').value == ['band', 'time']
    assert control(page, 'dis-rank').value == 'family'  # unknown rule -> default
    assert control(page, 'bt-signal-stops').value == [0.0]  # nothing valid left -> default
    preferences.save_preferences('10s30s', ['10y'])
    assert ui.dislocation_tab() and control(ui.dislocation_tab(), 'dis-cv').value == 5
