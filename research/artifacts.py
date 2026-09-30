"""Reproducible, isolated research runs; never overwrite promoted results."""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import polars as pl

RUNS = Path(__file__).parent / "data" / "runs"


def save_run(kind: str, data: pl.DataFrame, results: pl.DataFrame,
             metadata: dict, details: dict | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS / f"{stamp}_{kind}_{uuid4().hex[:8]}"
    path.mkdir(parents=True)
    data.write_parquet(path / "data.parquet")
    results.write_parquet(path / "results.parquet")
    root = Path(__file__).parent.parent
    files = ["research/app.py", "research/dislocation.py", "research/dislocation_backtest.py",
             "research/panel.py", "backtest/engine.py", "backtest/lab.py", "backtest/vector.py",
             "backtest/validation.py", "stats/ols.py", "stats/ou.py"]
    metadata = dict(metadata, data_start=str(data['ts'].min()), data_end=str(data['ts'].max()),
                    data_rows=len(data), created_at=datetime.now(timezone.utc).isoformat(),
                    source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files})
    (path / "metadata.json").write_text(
        json.dumps(metadata, default=str, indent=2), encoding="utf-8")
    if details:
        append_details(path, details)
    return str(path.resolve())


def append_details(path: str | Path, details: dict) -> None:
    """Add per-config trades/equity/periods to a run, replacing any for the same config."""
    path = Path(path)
    for name in ("trades", "equity", "periods"):
        rows = [dict(config_id=key, **row)
                for key, run in details.items() for row in run.get(name, [])]
        if not rows:
            continue
        new = pl.DataFrame(rows, infer_schema_length=None)
        target = path / f"{name}.parquet"
        if target.is_file():
            old = pl.read_parquet(target).filter(~pl.col("config_id").is_in(list(details)))
            new = pl.concat([old, new], how="diagonal_relaxed")
        new.write_parquet(target)
