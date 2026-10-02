"""Session-scoped progress for the local, threaded research server."""

import json
import os
from pathlib import Path
from threading import Lock
from time import monotonic, sleep, time

_lock = Lock()
_tasks: dict[tuple[str, str], dict] = {}
MAX_LOG_LINES = 200
# Set in a worker process: every update is also written here for the app to read.
_sink: Path | None = None
_last_write = 0.0


def update(session: str, task: str, message: str, done=0, total=0) -> None:
    now = monotonic()
    with _lock:
        for key in list(_tasks):
            if now - _tasks[key]["updated"] > 7200:
                del _tasks[key]
        key = (session, task)
        old = _tasks.get(key, {})
        history = old.get("history", [])
        message_count = old.get("message_count", 0)
        if not history or history[-1] != message:
            history = [*history, message][-MAX_LOG_LINES:]
            message_count += 1
        _tasks[key] = dict(message=message, done=done, total=total,
                           message_count=message_count,
                           phase_started=(old.get("phase_started", now)
                                          if old.get("total") == total and done >= old.get("done", 0)
                                          else now),
                           started=old.get("started", now), updated=now,
                           history=history,
                           wall_started=old.get("wall_started", time()),
                           wall_phase_started=(old.get("wall_phase_started", time())
                                               if old.get("total") == total and done >= old.get("done", 0)
                                               else time()))
        if _sink is not None:
            _persist(_tasks[key])


def persist_to(path) -> None:
    """In a worker process: also write progress to ``path`` (read back with ``read``)."""
    global _sink
    _sink = Path(path)


def _persist(state: dict) -> None:
    """Write progress for the app to read. Never raises: progress is a display, not the job.

    On Windows a file cannot be replaced while another process has it open,
    and the app reads this file every poll. So the swap is retried briefly;
    if the app still holds it, this update is skipped and the next one
    rewrites it (a final message retries for longer).
    """
    global _last_write
    finished = state["message"].startswith(("Completed ·", "Failed:"))
    if not finished and time() - _last_write < 0.25:
        return
    payload = {k: state[k] for k in ("message", "done", "total", "message_count", "history",
                                     "wall_started", "wall_phase_started")}
    payload["wall_updated"] = time()
    partial = _sink.with_name(_sink.name + ".partial")
    pauses = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8) if finished else (0.0, 0.02, 0.05)
    for pause in pauses:
        sleep(pause)
        try:
            partial.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(partial, _sink)
        except OSError:
            continue
        _last_write = time()
        return


def read(path) -> dict:
    """A worker's persisted progress, in ``snapshot``'s shape; {} if none yet."""
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    finished = state["message"].startswith(("Completed ·", "Failed:"))
    end = state["wall_updated"] if finished else time()
    return dict(state, started=state["wall_started"], elapsed=end - state["wall_started"],
                phase_elapsed=end - state["wall_phase_started"])


def start(session: str, task: str, message: str) -> None:
    with _lock:
        _tasks.pop((session, task), None)
    update(session, task, message)


def snapshot(session: str, task: str) -> dict:
    with _lock:
        state = dict(_tasks.get((session, task), {}))
    if state:
        finished = state["message"].startswith(("Completed ·", "Failed:"))
        state["elapsed"] = (state["updated"] if finished else monotonic()) - state["started"]
        state["phase_elapsed"] = (state["updated"] if finished else monotonic()) - state["phase_started"]
    return state
