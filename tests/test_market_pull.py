"""Run harmless child processes to verify the live pull's output and lifecycle."""
import sys
import threading
import time

import pytest

from dashboard import market_pull


def wait_until(predicate):
    deadline = time.monotonic() + 10
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("Background pull did not reach the expected state")
        time.sleep(0.02)


def test_live_output_duplicate_click_and_refresh(monkeypatch, tmp_path):
    monkeypatch.setenv("VIEWS_STORE_DIR", str(tmp_path))
    monkeypatch.setattr(market_pull, "COMMAND", [sys.executable, "-u", "-c",
        "import sys,time; print('first quote',flush=True); time.sleep(0.3); print('last quote',file=sys.stderr)"])
    entered, release = threading.Event(), threading.Event()
    def refresh():
        entered.set()
        assert release.wait(10)
    monkeypatch.setattr(market_pull.markets, "pull", refresh)
    job = market_pull.LivePull()
    job.start()
    try:
        wait_until(lambda: "first quote" in job.snapshot()["output"])
        assert job.snapshot()["status"] == "running"
        original_id = job.snapshot()["id"]
        job.start()
        assert job.snapshot()["id"] == original_id
        assert entered.wait(10)
    finally:
        release.set()
    wait_until(lambda: job.snapshot()["status"] != "running")
    state = job.snapshot()
    assert state["status"] == "succeeded"
    assert state["exit_code"] == 0
    assert "last quote" in state["output"]
    assert "first quote" in next((tmp_path / "market_pulls").glob("*.log")).read_text(encoding="utf-8")


@pytest.mark.parametrize("cache_fails", [False, True])
def test_failures_are_reported_without_retries(monkeypatch, tmp_path, cache_fails):
    monkeypatch.setenv("VIEWS_STORE_DIR", str(tmp_path))
    code = 0 if cache_fails else 7
    monkeypatch.setattr(market_pull, "COMMAND", [sys.executable, "-u", "-c",
        f"import sys; print('real failure output'); sys.exit({code})"])
    calls = []
    def refresh():
        calls.append(True)
        raise RuntimeError("database unavailable")
    monkeypatch.setattr(market_pull.markets, "pull", refresh)
    job = market_pull.LivePull()
    job.start()
    wait_until(lambda: job.snapshot()["status"] != "running")
    state = job.snapshot()
    assert state["status"] == "failed"
    assert state["exit_code"] == code
    assert "real failure output" in state["output"]
    assert len(calls) == int(cache_fails)
    assert len(list((tmp_path / "market_pulls").glob("*.log"))) == 1


def test_output_is_bounded_and_credentials_redacted():
    import io
    job = market_pull.LivePull()
    log = io.StringIO()
    job._append("\x1b[31merror\x1b[0m postgresql://name:secret@host/db password='private'\n", log)
    assert "secret" not in log.getvalue()
    assert "private" not in log.getvalue()
    assert "\x1b" not in log.getvalue()
    job._append("x" * (market_pull.MAX_OUTPUT + 20), log)
    assert len(job.snapshot()["output"]) == market_pull.MAX_OUTPUT
