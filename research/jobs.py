"""Background research jobs that outlive the browser page that started them.

A discovery run or trade-mechanics grid executes in a server thread and
records its state in ``research/data/jobs/<id>.json``: kind, status, times, a
short description, the saved run it produced, or its error. Pages poll that
record, so reloading, closing the tab or a sleeping laptop no longer loses a
run: the next page reattaches to a running job or shows its finished result.
While a job runs on Windows, the machine is asked not to sleep.

Jobs live in this server process. Stopping the server stops a running job,
which readers then see as ``interrupted``; finished results are the saved
runs under ``research/data/runs`` and are kept regardless.
"""

from __future__ import annotations

import ctypes
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Callable
from uuid import uuid4

JOBS = Path(__file__).parent / "data" / "jobs"
FINISHED = ("done", "failed", "interrupted")
_lock = Lock()
_live: dict[str, str] = {}  # kind -> running job id, this process only


class JobRunning(RuntimeError):
    """A job of the same kind is already running."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(record: dict) -> None:
    JOBS.mkdir(parents=True, exist_ok=True)
    temp = JOBS / f"{record['id']}.tmp"
    temp.write_text(json.dumps(record, default=str, indent=2), encoding="utf-8")
    temp.replace(JOBS / f"{record['id']}.json")


def _read(job_id: str) -> dict | None:
    try:
        return json.loads((JOBS / f"{job_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while this thread works; release when done."""
    if sys.platform != "win32":
        return
    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def submit(kind: str, description: str, work: Callable[[], dict]) -> str:
    """Start ``work`` in a background thread; its returned dict becomes the job's result."""
    with _lock:
        if kind in _live:
            raise JobRunning(f"a {kind} job is already running")
        job_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{kind}_{uuid4().hex[:8]}"
        _live[kind] = job_id
        _write(dict(id=job_id, kind=kind, status="running", description=description,
                    started_at=_now(), finished_at=None, result=None, error=None, traceback=None))
    Thread(target=_run, args=(job_id, kind, work), name=f"research-job-{job_id}", daemon=True).start()
    return job_id


def _run(job_id: str, kind: str, work: Callable[[], dict]) -> None:
    _keep_awake(True)
    try:
        outcome = dict(status="done", result=work())
    except Exception as exc:
        outcome = dict(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        _keep_awake(False)
    with _lock:
        record = _read(job_id) or dict(id=job_id, kind=kind)
        record.update(outcome, finished_at=_now())
        _write(record)
        _live.pop(kind, None)


def get(job_id: str) -> dict | None:
    """The job's record; a 'running' job no longer alive in this process reads as interrupted."""
    record = _read(job_id)
    if record and record["status"] == "running" and job_id not in _live.values():
        record.update(status="interrupted", finished_at=record.get("finished_at") or _now(),
                      error="The research server stopped while this job was running.")
        with _lock:
            _write(record)
    return record


def latest(kind: str) -> dict | None:
    """The most recently started job of ``kind``."""
    if not JOBS.is_dir():
        return None
    names = sorted(JOBS.glob(f"*_{kind}_*.json"), reverse=True)
    return get(names[0].stem) if names else None


def running(kind: str) -> bool:
    return kind in _live


def wait(job_id: str, timeout: float = 600.0) -> dict:
    """Block until a job finishes (tests and scripts); returns its record."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = get(job_id)
        if record and record["status"] in FINISHED:
            return record
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} still running after {timeout}s")
