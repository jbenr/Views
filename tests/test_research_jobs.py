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
    status = callbacks["_job_status"]

    release = threading.Event()
    running = jobs.submit("bt", "slow grid", lambda: release.wait(10) and {})
    try:
        out = status(1, None, None, None, None, False, False)
        assert out[4] is True and "Running" in out[5]  # Run backtest grid disabled while it runs
        assert out[1] is no_update  # nothing to show yet
        assert status(2, None, None, None, None, False, True)[4] is no_update  # unchanged: no update
    finally:
        release.set()
    jobs.wait(running)

    failed = jobs.wait(jobs.submit("dis", "broken", lambda: (_ for _ in ()).throw(RuntimeError("no data"))))
    out = status(1, None, running, None, None, False, False)  # a fresh page: nothing rendered yet
    assert out[0] == failed["id"]
    shown = callbacks["_show_discovery_job"](out[0], 30)
    assert shown[3] == failed["id"] and "no data" in json.dumps(shown[0].to_plotly_json(), default=str)
    again = status(2, failed["id"], running, failed["id"], None, False, False)  # shown: nothing new
    assert again[0] is no_update and again[1] is no_update


def test_results_from_saved_runs_survive_loading_a_panel():
    from research import app as ui
    app = ui.build_app()
    callbacks = {v["callback"].__wrapped__.__name__: v["callback"].__wrapped__ for v in app.callback_map.values()}
    stored = {"target": "10s30s", "weighting": "fixed", "legs": {"10y": -1.0, "30y": 1.0}, "features": ["10y"]}
    same = {"target_definition": {"target": "10s30s", "weighting": "fixed", "legs": {"10y": -1.0, "30y": 1.0},
                                  "weight_columns": {}}, "feature": "10y"}
    other = dict(same, feature="swsp10")
    assert callbacks["_invalidate_panel"](stored, same, same) == (no_update,) * 7  # same setup: results stay
    cleared = json.dumps(callbacks["_invalidate_panel"](stored, other, other)[:7], default=str)
    assert "The board shown was for" in cleared and "The grid shown was for" in cleared
    archived = {"archive_id": "run", "run_path": "/runs/run"}
    assert callbacks["_invalidate_candidate_for_board"](archived, {"discovery_run": "/runs/run"}) == (no_update,) * 2
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



def test_the_per_tick_job_check_never_writes_to_the_result_areas():
    # Writing there every 0.75s made Dash dim them as 'loading' each tick: the page flashed.
    from research import app as ui
    app = ui.build_app()
    tick = next(k for k, v in app.callback_map.items() if v["callback"].__wrapped__.__name__ == "_job_status")
    assert "dis-out" not in tick and "bt-out" not in tick and "dis-board" not in tick and "bt-grid" not in tick
    for name, kind in (("_show_discovery_job", "dis"), ("_show_grid_job", "bt")):
        spec = next(v for v in app.callback_map.values() if v["callback"].__wrapped__.__name__ == name)
        assert [i["id"] for i in spec["inputs"]] == [f"{kind}-ready"]


def test_concurrent_renders_of_one_result_share_a_single_render():
    from research import app as ui
    calls, started = [], threading.Event()

    def slow():
        calls.append(1)
        started.set()
        threading.Event().wait(0.2)
        return "board"
    results = []
    first = threading.Thread(target=lambda: results.append(ui.render_once(("t", "job"), slow)))
    first.start()
    started.wait(5)
    results.append(ui.render_once(("t", "job"), slow))
    first.join()
    assert results == ["board", "board"] and len(calls) == 1


def test_a_damaged_record_is_repaired_on_read_not_raised():
    # Exactly the record that broke the page: a stub with no status.
    jobs.JOBS.mkdir(parents=True)
    job_id = "20261002T055445432827Z_dis_a3ba3beb"
    (jobs.JOBS / f"{job_id}.json").write_text(json.dumps({"id": job_id, "kind": "dis", "pid": 42340, "attempt": 2}))
    record = jobs.latest("dis")
    assert record["status"] == "failed" and "incomplete" in record["error"]
    assert jobs.progress_of("dis") == {} and not jobs.running("dis")
    assert json.loads((jobs.JOBS / f"{job_id}.json").read_text())["status"] == "failed"


def test_worker_processes_finish_back_to_back_and_report_progress():
    first = jobs.wait(jobs.submit("dis", "first", "json:loads", '{"n": 1}'), timeout=120)
    # Started the instant the first record says done: must not be refused as still running.
    second = jobs.wait(jobs.submit("dis", "second", "json:loads", '{"n": 2}'), timeout=120)
    assert (first["status"], first["result"], second["status"], second["result"]) == ("done", {"n": 1}, "done", {"n": 2})
    reported = jobs.wait(jobs.submit("bt", "progress", "research.progress:update", "jobs", "bt", "hello from a worker", 1, 2),
                         timeout=120)
    assert reported["status"] == "done"
    progress = jobs.progress_of("bt")
    assert progress["message"] == "hello from a worker" and (progress["done"], progress["total"]) == (1, 2)


def test_a_crashing_worker_is_relaunched_then_marked_failed(monkeypatch):
    monkeypatch.setattr(jobs, "MAX_ATTEMPTS", 2)
    record = jobs.wait(jobs.submit("dis", "crashes", "os:_exit", 7), timeout=120)
    assert record["status"] == "failed" and record["attempt"] == 2
    assert "exit code 7" in record["error"] and "relaunched" in record["note"]
    assert not jobs.running("dis")


def test_a_locked_progress_file_never_breaks_the_job(tmp_path, monkeypatch):
    # Windows refuses to replace a file the app is reading; progress must skip, not raise.
    from research import progress
    sink = tmp_path / "job.progress.json"
    monkeypatch.setattr(progress, "_sink", sink)
    real = progress.os.replace
    monkeypatch.setattr(progress.os, "replace", lambda *a: (_ for _ in ()).throw(PermissionError(5, "Access is denied")))
    progress.update("s", "dis", "Backtesting model 1/10", 1, 10)  # does not raise
    assert not sink.exists()
    monkeypatch.setattr(progress.os, "replace", real)
    progress.update("s", "dis", "Completed · done", 1, 1)  # the next write lands
    assert progress.read(sink)["message"] == "Completed · done"
