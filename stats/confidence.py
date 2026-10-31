"""Walk-forward confidence in a signal's next trade, from how past trades did in
the same state.

For each entry-state feature (half-life, |z|, fit quality, ...) the trades that
had already CLOSED before an entry are split into terciles of that feature; the
entry's own value falls in one. That tercile's average P&L, against the past
average and in units of the past trades' spread, is the feature's edge, shrunk
toward zero by n / (n + shrink) so a thin tercile barely counts. The score is
the mean edge across features. Its tier is relative to the scores of earlier
entries: HIGH above two-thirds of them, LOW in the bottom third, MED between --
so the label means "a better or worse state than this signal usually enters
in". Until CONF_MIN_SCORED earlier entries have a score, a fixed band stands
in (HIGH above +BAND, LOW below -BAND). "—" until MIN_PAST trades have closed.
Nothing a trade's confidence uses was unknown at its entry, so the same number
can be logged live.

    per_trade = walk_forward_confidence(trades, {"half_life": "half-life", "abs_z": "|z|"})
    now = confidence_now(trades, {"half_life": 6.1, "abs_z": 1.9}, labels)
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["walk_forward_confidence", "confidence_now", "CONF_BAND", "CONF_MIN_PAST", "CONF_SHRINK"]

CONF_BAND = 0.15  # fallback tiers: mean edge, in past trades' std, beyond which HIGH or LOW
CONF_MIN_PAST = 15  # closed trades needed before any confidence is given
CONF_MIN_SCORED = 10  # earlier scored entries needed to tier against them rather than the band
CONF_SHRINK = 20  # a tercile of n trades counts n / (n + CONF_SHRINK) of its edge


def _tier(score: float, earlier: np.ndarray) -> str:
    """HIGH / MED / LOW: against the terciles of earlier entries' scores, or the fixed
    band while there are too few of them."""
    earlier = earlier[np.isfinite(earlier)]
    if len(earlier) >= CONF_MIN_SCORED:
        lo, hi = np.quantile(earlier, [1 / 3, 2 / 3])
        return "HIGH" if score > hi else "LOW" if score <= lo else "MED"
    return "HIGH" if score >= CONF_BAND else "LOW" if score <= -CONF_BAND else "MED"


def _feature_edge(values: np.ndarray, pnl: np.ndarray, value: float, shrink: float) -> dict | None:
    """One feature's edge for ``value`` against past (values, pnl)."""
    ok = np.isfinite(values) & np.isfinite(pnl)
    values, pnl = values[ok], pnl[ok]
    if len(values) < 3 or not np.isfinite(value):
        return None
    lo, hi = np.quantile(values, [1 / 3, 2 / 3])
    if value <= lo:
        bucket, mask = "low", values <= lo
    elif value > hi:
        bucket, mask = "high", values > hi
    else:
        bucket, mask = "mid", (values > lo) & (values <= hi)
    n, spread = int(mask.sum()), pnl.std()
    if n == 0 or spread <= 0:
        return None
    avg = float(pnl[mask].mean())
    edge = n / (n + shrink) * (avg - pnl.mean()) / spread
    return dict(edge=float(edge), bucket=bucket, avg=avg, n=n)


def _score(past: pl.DataFrame, values: dict, labels: dict, min_past: int, shrink: float, pnl_col: str,
           earlier: np.ndarray) -> dict:
    """Confidence for one entry with feature ``values``, given the closed trades ``past``
    and the ``earlier`` entries' scores."""
    if len(past) < min_past:
        return dict(score=None, label="—", drivers="", n_past=len(past))
    pnl = past[pnl_col].to_numpy().astype(float)
    edges = []
    for col, label in labels.items():
        e = _feature_edge(past[col].to_numpy().astype(float), pnl, float(values.get(col, np.nan)), shrink)
        if e is not None:
            edges.append((label, float(values[col]), e))
    if not edges:
        return dict(score=None, label="—", drivers="", n_past=len(past))
    score = float(np.mean([e["edge"] for _, _, e in edges]))
    tier = _tier(score, earlier)
    top = sorted(edges, key=lambda item: -abs(item[2]["edge"]))[:2]
    drivers = " · ".join(f"{label} {value:.3g} ({e['bucket']}): past {e['avg']:+.1f}bp ×{e['n']}"
                         for label, value, e in top)
    return dict(score=score, label=tier, drivers=drivers, n_past=len(past))


def walk_forward_confidence(trades: pl.DataFrame, labels: dict[str, str], entry: str = "entry_ts",
                            exit: str = "exit_ts", pnl: str = "pnl", min_past: int = CONF_MIN_PAST,
                            shrink: float = CONF_SHRINK) -> pl.DataFrame:
    """Each trade's confidence at its entry, from the trades closed before it.

    ``trades`` needs ``entry``, ``exit``, ``pnl`` and a column per key of ``labels``
    (feature column -> display name). Returns the trades in their order with
    conf_score, conf (HIGH/MED/LOW/—), conf_drivers and conf_n_past."""
    rows = [None] * len(trades)
    scored: list[tuple] = []  # (entry, score) of the entries walked so far, in entry order
    for i in trades.with_row_index("_i").sort(entry)["_i"].to_list():
        t = trades.row(i, named=True)
        past = trades.filter(pl.col(exit) < t[entry])
        earlier = np.array([s for when, s in scored if when < t[entry]], dtype=float)
        r = _score(past, t, labels, min_past, shrink, pnl, earlier)
        if r["score"] is not None:
            scored.append((t[entry], r["score"]))
        rows[i] = dict(conf_score=r["score"], conf=r["label"], conf_drivers=r["drivers"], conf_n_past=r["n_past"])
    out = pl.DataFrame(rows, schema={"conf_score": pl.Float64, "conf": pl.String, "conf_drivers": pl.String,
                                     "conf_n_past": pl.Int64})
    return pl.concat([trades, out], how="horizontal")


def confidence_now(trades: pl.DataFrame, values: dict[str, float], labels: dict[str, str], pnl: str = "pnl",
                   min_past: int = CONF_MIN_PAST, shrink: float = CONF_SHRINK) -> dict:
    """Confidence in an entry today with feature ``values``, from every closed trade:
    score, label, drivers and n_past. Tiered against every trade's own score when
    ``trades`` come from walk_forward_confidence (a conf_score column)."""
    earlier = trades["conf_score"].to_numpy().astype(float) if "conf_score" in trades.columns else np.array([])
    return _score(trades, values, labels, min_past, shrink, pnl, earlier)
