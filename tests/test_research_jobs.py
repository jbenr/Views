"""Background research jobs: they outlive the page, report failures, and reattach."""

import json
import threading

import pytest
from dash import no_update

from research import jobs


@pytest.fixture(autouse=True)
def job_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    return tmp_path / "jobs"


def test_a_finished_job_records_its_result_on_disk():
    job_id = jobs.submit("dis", "demo", lambda: {"run_path": "somewhere", "summary": "ok"})
    record = jobs.wait(job_id)
    assert record["status"] == "done" and record["result"] == {"run_path": "somewhere", "summary": "ok"}
    assert jobs.latest("dis")["id"] == job_id and jobs.latest("bt") is None
    assert not jobs.running("dis")
    on_disk = json.loads((jobs.JOBS / f"{job_id}.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "done" and on_disk["finished_at"]


def test_a_failed_job_keeps_its_error_and_traceback():
    def boom():
        raise ValueError("bad grid")
    record = jobs.wait(jobs.submit("bt", "demo", boom))
    assert record["status"] == "failed" and record["error"] == "ValueError: bad grid"
    assert "Traceback" in record["traceback"] and not jobs.running("bt")


def test_one_job_per_kind_at_a_time():
    release = threading.Event()
    first = jobs.submit("dis", "slow", lambda: release.wait(10) and {})
    try:
        assert jobs.running("dis") and jobs.latest("dis")["status"] == "running"
        with pytest.raises(jobs.JobRunning):
            jobs.submit("dis", "second", lambda: {})
        other = jobs.submit("bt", "other kind", lambda: {})
        assert jobs.wait(other)["status"] == "done"
    finally:
        release.set()
    assert jobs.wait(first)["status"] == "done"


def test_a_job_orphaned_by_a_server_restart_reads_as_interrupted():
    jobs.JOBS.mkdir(parents=True)
    job_id = "20260101T000000000000Z_dis_deadbeef"
    (jobs.JOBS / f"{job_id}.json").write_text(json.dumps(dict(id=job_id, kind="dis", status="running")))
    record = jobs.latest("dis")
    assert record["status"] == "interrupted" and "stopped" in record["error"]
    assert json.loads((jobs.JOBS / f"{job_id}.json").read_text())["status"] == "interrupted"


def test_a_new_page_reattaches_to_running_and_finished_jobs():
    from research import app as ui
    app = ui.build_app()
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in app.callback_map.values()}
    poll = callbacks["_job_results"]

    release = threading.Event()
    running = jobs.submit("bt", "slow grid", lambda: release.wait(10) and {})
    try:
        out = poll(1, None, None, 30)
        assert out[10] is True and "Running" in out[11]  # Run backtest grid disabled while it runs
        assert out[4] is no_update  # nothing to show yet
    finally:
        release.set()
    jobs.wait(running)

    failed = jobs.wait(jobs.submit("dis", "broken", lambda: (_ for _ in ()).throw(RuntimeError("no data"))))
    out = poll(1, None, running, 30)  # a fresh page: nothing rendered yet
    assert out[3] == failed["id"] and "no data" in json.dumps(out[0].to_plotly_json(), default=str)
    assert out[8] is False and out[9] == "Run discovery"
    again = poll(2, failed["id"], running, 30)  # already shown: not re-rendered every tick
    assert again[0] is no_update and again[3] is no_update


def test_results_from_saved_runs_survive_loading_a_panel():
    from research import app as ui
    app = ui.build_app()
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in app.callback_map.values()}
    archived = {"archive_id": "run", "run_path": "/runs/run"}
    assert callbacks["_invalidate_panel"]({"rows": []}, archived) == (no_update,) * 6
    assert callbacks["_invalidate_panel"]({"rows": []}, {"run_path": "x"})[0] == ""
    assert callbacks["_invalidate_candidate_for_board"](archived, {"discovery_run": "/runs/run"}) == (no_update,) * 4
    assert callbacks["_invalidate_candidate_for_board"](archived, {"discovery_run": "/runs/other"})[0] is None


def test_the_research_server_starts_without_the_file_watching_reloader(monkeypatch):
    # With the reloader on, saving any source file restarts the server and kills running
    # jobs, and a failed restart (e.g. numba threads torn down mid-kernel) exits it entirely.
    from research import app as ui
    from utils import research_app
    seen = {}
    monkeypatch.setattr(type(ui.build_app()), "run", lambda self, **kwargs: seen.update(kwargs))
    research_app.run(ui.build_app(), port=1)
    assert seen["use_reloader"] is False and seen["port"] == 1


def test_charts_never_use_tk_and_survive_concurrent_requests():
    import matplotlib
    import polars as pl
    from dashboard import charts
    assert matplotlib.get_backend().lower() == "agg"
    coverage = pl.DataFrame({"series": ["a", "b"], "first_valid": ["2020-01-01", "2021-01-01"],
                             "last_valid": ["2024-01-01", "2024-06-01"], "n_valid": [100, 80], "pct_missing": [0.0, 0.1]})
    coverage = coverage.with_columns(pl.col("first_valid", "last_valid").str.to_date())
    errors = []

    def draw():
        try:
            for _ in range(3):
                assert charts.coverage_chart(coverage)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(repr(exc))
    threads = [threading.Thread(target=draw) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []


def test_native_crashes_are_logged_with_every_threads_stack(tmp_path):
    import faulthandler
    from research.app import record_native_crashes
    log = record_native_crashes(tmp_path / "logs" / "crashes.log")
    try:
        assert faulthandler.is_enabled()
        faulthandler.dump_traceback(file=log, all_threads=True)
        log.flush()
        text = (tmp_path / "logs" / "crashes.log").read_text(encoding="utf-8")
        assert "research server started" in text and "test_native_crashes_are_logged" in text
    finally:
        faulthandler.disable()
        log.close()
