"""Durable local review queue and broker snapshot history, separate from research."""
import datetime as dt
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path


class TradeJournal:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path(os.getenv(
            "VIEWS_STORE_DIR", str(Path(__file__).resolve().parents[1] / "store"))) / "execution.sqlite3"

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT, created TEXT, payload TEXT)")
                yield db
        finally:
            db.close()

    def _save(self, kind, payload, key=None):
        key = key or str(uuid.uuid4())
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO records VALUES (?, ?, ?, ?)",
                       (key, kind, dt.datetime.now(dt.timezone.utc).isoformat(), json.dumps(payload, allow_nan=False)))
        return key

    def records(self, kind):
        with self._connect() as db:
            return [dict(id=r[0], created=r[1], **json.loads(r[2])) for r in db.execute(
                "SELECT id, created, payload FROM records WHERE kind=? ORDER BY created DESC", (kind,))]

    def flag(self, signal_id, module, signal_run=None):
        if not signal_id or not module:
            raise ValueError("A promoted signal and strategy module are required.")
        # One pending review per signal/run, even across duplicate browser clicks.
        key = str(uuid.uuid5(uuid.NAMESPACE_URL, f"views:{signal_id}:{signal_run}"))
        return self._save("intent", dict(signal_id=signal_id, module=module,
                          signal_run=signal_run, status="pending_review", order_ref=f"views:{key}"), key)

    def save_snapshot(self, snapshot):
        # Retain executions when they disappear from TWS's recent-history window.
        for fill in snapshot.get("fills", []):
            if fill.get("exec_id"):
                self._save("fill", dict(fill, account=snapshot["account"]),
                           f"fill:{snapshot['account']}:{fill['exec_id']}")
        return self._save("snapshot", snapshot)

    def saved_fills(self, account):
        return [r for r in self.records("fill") if r["account"] == account]

    def latest_snapshot(self):
        with self._connect() as db:
            row = db.execute("SELECT payload FROM records WHERE kind='snapshot' ORDER BY created DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None
