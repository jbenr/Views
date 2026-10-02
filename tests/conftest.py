"""Keep tests away from the user's real research settings and job records."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_research_state(tmp_path_factory, monkeypatch):
    # The app reads saved control values when building its layout and writes
    # them on every change; jobs record themselves on disk. Tests must neither
    # see the user's choices nor overwrite them.
    from research import jobs, preferences
    root = tmp_path_factory.mktemp("research_state")
    monkeypatch.setattr(preferences, "PREFERENCES", root / "preferences.json")
    monkeypatch.setattr(jobs, "JOBS", root / "jobs")
