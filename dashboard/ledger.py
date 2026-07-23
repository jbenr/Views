"""Append-only audit log of live signal analysis runs.

Every dashboard.runner.run_analysis() call appends one row here: what the
signal read and what params produced that reading, timestamped before any
outcome is known -- the auditable record the Macro Signal Ledger is built
around. Same atomic-write convention as backtest.lab.MetricStore.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import polars as pl


def _store_dir() -> Path:
    return Path(
        os.getenv("VIEWS_STORE_DIR", str(Path(__file__).resolve().parents[1] / "store"))
    )


class SignalLedger:
    def __init__(self, path: Path | None = None):
        self.path = path or (_store_dir() / "signal_ledger.parquet")

    def load(self) -> pl.DataFrame:
        if not self.path.exists():
            return pl.DataFrame()
        return pl.read_parquet(self.path, memory_map=False)

    def log(self, entry: dict) -> None:
        row = pl.DataFrame([entry])
        existing = self.load()
        combined = (
            pl.concat([existing, row], how="diagonal_relaxed")
            if not existing.is_empty()
            else row
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.stem}.{uuid.uuid4().hex}.tmp.parquet")
        combined.write_parquet(tmp_path)
        os.replace(tmp_path, self.path)

    def latest(self, module: str) -> dict | None:
        df = self.load()
        if df.is_empty():
            return None
        match = df.filter(pl.col("module") == module).sort("run_ts")
        if match.is_empty():
            return None
        return match.row(-1, named=True)

    def history(self, module: str) -> pl.DataFrame:
        df = self.load()
        if df.is_empty():
            return df
        return df.filter(pl.col("module") == module).sort("run_ts")
