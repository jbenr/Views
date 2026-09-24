"""Session-scoped progress for the local, threaded research server."""

from threading import Lock
from time import monotonic

_lock = Lock()
_tasks: dict[tuple[str, str], dict] = {}


def update(session: str, task: str, message: str, done=0, total=0) -> None:
    now = monotonic()
    with _lock:
        for key in list(_tasks):
            if now - _tasks[key]["updated"] > 7200:
                del _tasks[key]
        key = (session, task)
        old = _tasks.get(key, {})
        history = old.get("history", [])
        if not history or history[-1] != message:
            history = [*history, message][-6:]
        _tasks[key] = dict(message=message, done=done, total=total,
                           phase_started=(old.get("phase_started", now)
                                          if old.get("total") == total and done >= old.get("done", 0)
                                          else now),
                           started=old.get("started", now), updated=now,
                           history=history)


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
