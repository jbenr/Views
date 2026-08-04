"""Tamper-evident hash chaining for append-only ledgers.

Each entry's hash covers its own canonical content plus the previous
entry's hash, so editing, deleting, or reordering any past row breaks
every hash from that point forward -- the same construction git commits
and blockchains use, minus the network. Verification just replays the
chain from GENESIS_HASH; no external state needed beyond the ledger file
itself.
"""

from __future__ import annotations

import hashlib
import json

GENESIS_HASH = "0" * 64
_HASH_COLS = ("prev_hash", "row_hash")


def _canonical(entry: dict) -> str:
    # Drop Nones as well as the hash columns: parquet schema unions
    # (diagonal_relaxed) backfill older rows with null whenever a later
    # entry adds a new field, and that backfill must not change an old
    # row's hash. Consistently excluding null-valued keys, both here and
    # when the row was first chained, keeps the digest stable as the
    # ledger's schema grows.
    payload = {
        k: v for k, v in entry.items() if k not in _HASH_COLS and v is not None
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _digest(entry: dict, prev_hash: str) -> str:
    return hashlib.sha256((prev_hash + _canonical(entry)).encode("utf-8")).hexdigest()


def chain(entry: dict, prev_hash: str) -> dict:
    """Return entry with prev_hash/row_hash added; entry itself is untouched."""
    return {**entry, "prev_hash": prev_hash, "row_hash": _digest(entry, prev_hash)}


def verify(rows: list[dict]) -> list[int]:
    """Replay the chain against stored hashes. Returns indices where it breaks
    (empty list = every hashed row's hash matches its content and its
    neighbor). Rows with no row_hash predate chaining being added and are
    skipped -- they were never covered by a hash commitment, so they can
    neither pass nor fail verification."""
    broken = []
    prev_hash = GENESIS_HASH
    for i, row in enumerate(rows):
        if row.get("row_hash") is None:
            continue
        expected = _digest(row, prev_hash)
        if row.get("prev_hash") != prev_hash or row.get("row_hash") != expected:
            broken.append(i)
        prev_hash = row["row_hash"]
    return broken
