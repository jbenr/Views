"""Registry of promoted ("live") signals tracked on the dashboard.

Promoting a signal snapshots one row from its sweep_results_file (the
frozen entry/exit/gate config) into store/live_signals.parquet. Any book
module that exposes a `STRATEGY = backtest.strategy.Strategy(...)` object
can be promoted -- this isn't curve-specific. Same atomic-write convention
as backtest.lab.MetricStore.

    python -m dashboard.registry --promote book.curve.tens_10s30s
    python -m dashboard.registry --promote book.curve.tens_10s30s --rank 2
    python -m dashboard.registry --list
    python -m dashboard.registry --remove book.curve.tens_10s30s
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import os
import uuid
from pathlib import Path

import polars as pl

import utils

from .params import PARAM_COLS


def _store_dir() -> Path:
    return Path(
        os.getenv("VIEWS_STORE_DIR", str(Path(__file__).resolve().parents[1] / "store"))
    )


def load_strategy(module: str):
    """Import `module` and return its STRATEGY object."""
    mod = importlib.import_module(module)
    if not hasattr(mod, "STRATEGY"):
        raise SystemExit(
            f"{module} has no STRATEGY object -- expected a backtest.strategy.Strategy instance"
        )
    return mod.STRATEGY


class LiveRegistry:
    """Parquet-backed list of promoted signals, one row per module."""

    def __init__(self, path: Path | None = None):
        self.path = path or (_store_dir() / "live_signals.parquet")

    def load(self) -> pl.DataFrame:
        if not self.path.exists():
            return pl.DataFrame()
        return pl.read_parquet(self.path, memory_map=False)

    def _write(self, df: pl.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.stem}.{uuid.uuid4().hex}.tmp.parquet")
        df.write_parquet(tmp_path)
        os.replace(tmp_path, self.path)

    def promote(self, module: str, rank: int = 0) -> dict:
        """Snapshot row `rank` of <module>'s sweep_results_file (0 = best by
        sharpe, the file's own sort order) as the live config for `module`.
        Replaces any prior promotion of the same module."""
        strategy = load_strategy(module)
        if not strategy.sweep_results_file.exists():
            raise SystemExit(
                f"{module}: no sweep results at {strategy.sweep_results_file} -- run --sweep first"
            )
        results = pl.read_parquet(strategy.sweep_results_file)
        if not (0 <= rank < len(results)):
            raise SystemExit(f"{module}: rank {rank} out of range (0..{len(results) - 1})")
        row = results.row(rank, named=True)

        entry = {
            "module": module,
            "name": strategy.name,
            "family": strategy.family,
            "target": strategy.target,
            "feature": strategy.feature,
            "rank": rank,
            "promoted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            **{c: row.get(c) for c in PARAM_COLS},
            "sharpe": row.get("sharpe"),
            "n_trades": row.get("n_trades"),
            "hit_rate": row.get("hit_rate"),
            "max_drawdown_bps": row.get("max_drawdown_bps"),
        }
        existing = self.load()
        if not existing.is_empty():
            existing = existing.filter(pl.col("module") != module)
        new_row = pl.DataFrame([entry])
        combined = (
            pl.concat([existing, new_row], how="diagonal_relaxed")
            if not existing.is_empty()
            else new_row
        )
        self._write(combined)
        return entry

    def remove(self, module: str) -> bool:
        existing = self.load()
        if existing.is_empty() or module not in existing["module"].to_list():
            return False
        self._write(existing.filter(pl.col("module") != module))
        return True

    def list(self) -> pl.DataFrame:
        return self.load()

    def get(self, module: str) -> dict | None:
        df = self.load()
        if df.is_empty():
            return None
        match = df.filter(pl.col("module") == module)
        if match.is_empty():
            return None
        return match.row(0, named=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", metavar="MODULE")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--remove", metavar="MODULE")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    reg = LiveRegistry()
    if args.promote:
        entry = reg.promote(args.promote, rank=args.rank)
        print(
            f"promoted {args.promote} (rank {args.rank}): "
            f"sharpe={entry['sharpe']:.2f}  n_trades={entry['n_trades']:.0f}  "
            f"hit_rate={entry['hit_rate']:.0%}"
        )
    elif args.remove:
        ok = reg.remove(args.remove)
        print(f"{'removed' if ok else 'not found:'} {args.remove}")
    elif args.list:
        df = reg.list()
        if df.is_empty():
            print("no live signals promoted yet")
        else:
            utils.pdf(
                df.select(
                    "module", "name", "family", "sharpe", "n_trades",
                    "hit_rate", "promoted_at",
                )
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
