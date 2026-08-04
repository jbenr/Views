"""Append-only audit log of live signal analysis runs.

Every dashboard.runner.run_analysis() call appends one row here: what the
signal read and what params produced that reading, timestamped before any
outcome is known -- the auditable record the Macro Signal Ledger is built
around. Same atomic-write convention as backtest.lab.MetricStore.

Every row is also hash-chained (see hashchain.py): each entry's row_hash
covers its content plus the previous row's hash, so editing, deleting, or
reordering history breaks the chain from that point forward. verify()
replays it.
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import polars as pl

from .hashchain import GENESIS_HASH, chain, verify as verify_chain


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
        frame = pl.read_parquet(self.path, memory_map=False)
        if "signal_id" not in frame.columns:
            frame = frame.with_columns(pl.col("module").alias("signal_id"))
        return frame

    @staticmethod
    def _chain_tip(existing: pl.DataFrame) -> str:
        """Most recent row_hash to build on, or GENESIS_HASH if the ledger
        is empty or predates chaining (no row_hash column yet)."""
        if existing.is_empty() or "row_hash" not in existing.columns:
            return GENESIS_HASH
        tail = existing["row_hash"][-1]
        return tail if tail is not None else GENESIS_HASH

    def log(self, entry: dict) -> None:
        existing = self.load()
        prev_hash = self._chain_tip(existing)
        row = pl.DataFrame([chain(entry, prev_hash)])
        combined = (
            pl.concat([existing, row], how="diagonal_relaxed")
            if not existing.is_empty()
            else row
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.stem}.{uuid.uuid4().hex}.tmp.parquet")
        combined.write_parquet(tmp_path)
        os.replace(tmp_path, self.path)

    def verify(self) -> list[int]:
        """Replay the hash chain. Returns indices of rows where it breaks
        (empty list = the whole ledger is intact, in original append order)."""
        df = self.load()
        if df.is_empty():
            return []
        return verify_chain(df.iter_rows(named=True))

    def latest(self, signal_id: str) -> dict | None:
        df = self.load()
        if df.is_empty():
            return None
        match = df.filter(pl.col("signal_id") == signal_id).sort("run_ts")
        if match.is_empty():
            return None
        return match.row(-1, named=True)

    def history(self, signal_id: str) -> pl.DataFrame:
        df = self.load()
        if df.is_empty():
            return df
        return df.filter(pl.col("signal_id") == signal_id).sort("run_ts")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="replay the hash chain and report any breaks"
    )
    args = parser.parse_args(argv)

    ledger = SignalLedger()
    if args.verify:
        df = ledger.load()
        hashed = df.filter(pl.col("row_hash").is_not_null()) if "row_hash" in df.columns else df.head(0)
        broken = ledger.verify()
        if not broken:
            print(f"{len(hashed)}/{len(df)} rows hash-chained, chain intact")
        else:
            print(f"chain BROKEN at {len(broken)} row(s) (0-indexed): {broken}")
        raise SystemExit(1 if broken else 0)
    parser.print_help()


if __name__ == "__main__":
    main()
