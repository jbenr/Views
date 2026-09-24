"""Local last-successful Setup selection, separate from research run artifacts."""
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

PREFERENCES = Path(__file__).parent / 'data' / 'preferences.json'
_lock = Lock()


def load_preferences(targets, features, default_target, default_features):
    defaults = dict(target=default_target, features=list(default_features), custom=None)
    try:
        saved = json.loads(PREFERENCES.read_text(encoding='utf-8'))
        if not isinstance(saved, dict):
            return defaults
        target = saved.get('target')
        custom = saved.get('custom')
        if not isinstance(target, str) or target not in {*targets, 'custom'}:
            target = default_target
        if target == 'custom' and (not isinstance(custom, str) or not custom.strip()):
            target = default_target
        selected = saved.get('features', default_features)
        if not isinstance(selected, list):
            selected = default_features
        selected = list(dict.fromkeys(f for f in selected if isinstance(f, str) and f in features))
        return dict(target=target, features=selected, custom=custom if isinstance(custom, str) else None)
    except (OSError, ValueError):
        return defaults


def save_preferences(target, features, custom=None):
    payload = dict(target=target, features=list(features), custom=custom if target == 'custom' else None)
    # A reader sees either the old complete file or the new complete file.
    with _lock:
        PREFERENCES.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with NamedTemporaryFile(mode='w', encoding='utf-8', dir=PREFERENCES.parent,
                                    prefix='preferences-', suffix='.tmp', delete=False) as stream:
                temp_path = Path(stream.name)
                json.dump(payload, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, PREFERENCES)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
