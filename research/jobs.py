"""Background research jobs that outlive the browser page that started them.

A discovery run or trade-mechanics grid executes in its own worker process
and records its state in ``research/data/jobs/<id>.json``: kind, status,
times, a short description, the saved run it produced, or its error. Pages
poll that record, so reloading, closing the tab or a sleeping laptop no longer
loses a run: the next page reattaches to a running job or shows its result.

A separate process means a native crash in the backtest code kills only the
worker, never the app, and all of a run's memory goes back to the system when
it ends. If a worker dies without finishing, the job is relaunched (up to
``MAX_ATTEMPTS`` in all); discovery checkpoints each model, so a relaunch
resumes rather than restarts. Workers keep the machine awake on Windows, and
keep running if the app restarts: the record holds the worker's process id,
so a restarted app reattaches instead of calling the job interrupted.

Jobs given a plain callable (tests, quick tasks) run in a thread instead.

Records are only ever changed by merging into what is on disk, read with
retries: on Windows a file being replaced can be briefly unreadable, and
treating that as "no record" once overwrote a real record with a stub.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import multiprocessing
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock, Thread
from typing import Callable
from uuid import uuid4

JOBS = Path(__file__).parent / "data" / "jobs"
FINISHED = ("done", "failed", "interrupted")
MAX_ATTEMPTS = 3
_lock = RLock()  # re-entrant: submit checks running() while holding it
_live: dict[str, str] = {}  # kind -> job id watched by this process


class JobRunning(RuntimeError):
    """A job of the same kind is already running."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(record: dict) -> None:
    JOBS.mkdir(parents=True, exist_ok=True)
    temp = JOBS / f"{record['id']}.{uuid4().hex[:8]}.tmp"  # one temp file per write: no writer races
    temp.write_text(json.dumps(record, default=str, indent=2), encoding="utf-8")
    for pause in (0.0, 0.05, 0.1, 0.2, 0.4):
        time.sleep(pause)
        try:
            temp.replace(JOBS / f"{record['id']}.json")
            return
        except PermissionError:  # a reader has it open this instant (Windows)
            continue
    temp.replace(JOBS / f"{record['id']}.json")


def _read(job_id: str) -> dict | None:
    """The record as stored, or None if there is no such job. Retries a briefly locked file."""
    path = JOBS / f"{job_id}.json"
    for pause in (0.0, 0.05, 0.1, 0.2, 0.4):
        time.sleep(pause)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def _update(job_id: str, kind: str, **changes) -> dict:
    """Merge ``changes`` into a record on disk; a record that cannot be read is rebuilt, never blanked."""
    with _lock:
        record = _read(job_id) or dict(id=job_id, kind=kind, status="running", description="(record was rebuilt)")
        record.update(changes)
        _write(record)
        return record


def _keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while this thread works; release when done."""
    if sys.platform != "win32":
        return
    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).is_running()
    except Exception:
        return False


def submit(kind: str, description: str, work: str | Callable[..., dict], *args) -> str:
    """Start a job; its returned dict becomes the job's result.

    ``work`` as ``"module:function"`` runs ``function(*args)`` in a worker
    process (args must pickle); a callable runs in a thread.
    """
    with _lock:
        if running(kind):
            raise JobRunning(f"a {kind} job is already running")
        job_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{kind}_{uuid4().hex[:8]}"
        _live[kind] = job_id
        _write(dict(id=job_id, kind=kind, status="running", description=description, started_at=_now(),
                    finished_at=None, result=None, error=None, traceback=None, attempt=1, pid=None))
    if isinstance(work, str):
        _launch(job_id, kind, work, args, attempt=1)
    else:
        Thread(target=_run_thread, args=(job_id, kind, work, args), name=f"research-job-{job_id}", daemon=True).start()
    return job_id


def _run_thread(job_id: str, kind: str, work: Callable, args) -> None:
    _keep_awake(True)
    try:
        outcome = dict(status="done", result=work(*args))
    except Exception as exc:
        outcome = dict(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        _keep_awake(False)
    with _lock:
        _update(job_id, kind, **outcome, finished_at=_now())
        if _live.get(kind) == job_id:
            _live.pop(kind)


def _launch(job_id: str, kind: str, target: str, args, attempt: int) -> None:
    process = multiprocessing.get_context("spawn").Process(
        target=_worker, args=(str(JOBS), job_id, kind, target, args), name=f"research-job-{job_id}")
    process.start()
    _update(job_id, kind, pid=process.pid, attempt=attempt)
    Thread(target=_watch, args=(job_id, kind, target, args, process, attempt), daemon=True).start()


def _watch(job_id: str, kind: str, target: str, args, process, attempt: int) -> None:
    """Wait for a worker; if it died without finishing, relaunch or record the failure."""
    process.join()
    record = _read(job_id)
    if record is None or record.get("status") in FINISHED:  # finished (or deleted): never relaunch
        with _lock:
            if _live.get(kind) == job_id:
                _live.pop(kind)
        return
    code = process.exitcode
    shown = f"0x{code & 0xFFFFFFFF:08X}" if code and (code < 0 or code > 255) else str(code)
    if attempt < MAX_ATTEMPTS:
        _update(job_id, kind, note=f"Worker stopped unexpectedly (exit code {shown}); relaunched, attempt "
                                   f"{attempt + 1} of {MAX_ATTEMPTS}, resuming from its checkpoint.")
        _launch(job_id, kind, target, args, attempt + 1)
        return
    with _lock:
        _update(job_id, kind, status="failed", finished_at=_now(),
                error=f"Worker process stopped unexpectedly (exit code {shown}) on all {MAX_ATTEMPTS} attempts.")
        if _live.get(kind) == job_id:
            _live.pop(kind)


def _worker(jobs_dir: str, job_id: str, kind: str, target: str, args) -> None:
    """Worker process entry: run the job, record its outcome, report progress to a file."""
    global JOBS
    JOBS = Path(jobs_dir)
    from research import progress
    progress.persist_to(JOBS / f"{job_id}.progress.json")
    _keep_awake(True)
    try:
        module, name = target.split(":")
        outcome = dict(status="done", result=getattr(importlib.import_module(module), name)(*args))
    except Exception as exc:
        outcome = dict(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        _keep_awake(False)
    _update(job_id, kind, **outcome, finished_at=_now())


def get(job_id: str) -> dict | None:
    """The job's record, repaired where needed.

    A record without a status (damaged) reads as failed; a 'running' job with
    no live worker or thread anywhere reads as interrupted. Both are saved, so
    every reader sees the same thing.
    """
    record = _read(job_id)
    if record is None:
        return None
    if record.get("status") not in ("running", *FINISHED):
        return _update(job_id, record.get("kind", ""), status="failed", finished_at=record.get("finished_at") or _now(),
                       error=record.get("error") or "This job's record was incomplete; the job's outcome is unknown.")
    if (record["status"] == "running" and job_id not in _live.values()
            and not _pid_alive(record.get("pid"))):
        return _update(job_id, record.get("kind", ""), status="interrupted",
                       finished_at=record.get("finished_at") or _now(),
                       error="The research server stopped while this job was running.")
    return record


def latest(kind: str) -> dict | None:
    """The most recently started job of ``kind``."""
    if not JOBS.is_dir():
        return None
    names = sorted(p for p in JOBS.glob(f"*_{kind}_*.json") if not p.name.endswith(".progress.json"))
    return get(names[-1].stem) if names else None


def running(kind: str) -> bool:
    """The latest ``kind`` job's record says running (here, or in a worker that outlived an app restart)."""
    record = latest(kind)
    return bool(record) and record["status"] == "running"


def progress_of(kind: str) -> dict:
    """The latest ``kind`` job's progress, as written by its worker; {} if none."""
    from research import progress
    record = latest(kind)
    return progress.read(JOBS / f"{record['id']}.progress.json") if record else {}


def wait(job_id: str, timeout: float = 600.0) -> dict:
    """Block until a job finishes (tests and scripts); returns its record."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = get(job_id)
        if record and record["status"] in FINISHED:
            return record
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} still running after {timeout}s")
