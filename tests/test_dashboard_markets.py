"""Market source freshness must not advance just because the cache is refreshed."""
import datetime as dt
import os

import pandas as pd
import polars as pl

from dashboard import markets


def test_source_stamp_survives_db_reread_and_cache_mtime(monkeypatch, tmp_path):
    monkeypatch.setenv("VIEWS_STORE_DIR", str(tmp_path))
    source_time = dt.datetime(2026, 11, 11, 15, 30, tzinfo=dt.timezone.utc)
    data = pd.DataFrame([
        dict(ts=dt.date(2026, 11, 11), ticker="USGG10YR Index", px=4.2,
             updated_at=source_time, is_live=True),
    ])
    monkeypatch.setattr(markets, "query_db", lambda *args, **kwargs: data)
    first = markets.source_stamp(markets.pull())
    assert "Nov 11, 2026 10:30:00 AM EST" in first
    assert "live snapshot" in first
    os.utime(markets._cache_path(), (2_000_000_000, 2_000_000_000))
    assert markets.source_stamp(markets.cached()[0]) == first
    assert markets.source_stamp(markets.pull()) == first


def test_latest_valid_price_controls_freshness(monkeypatch, tmp_path):
    monkeypatch.setenv("VIEWS_STORE_DIR", str(tmp_path))
    data = pd.DataFrame([
        dict(ts=dt.date(2026, 11, 10), ticker="USGG10YR Index", px=4.2,
             updated_at=dt.datetime(2026, 11, 10, 20, tzinfo=dt.timezone.utc), is_live=False),
        dict(ts=dt.date(2026, 11, 11), ticker="USGG10YR Index", px=float("nan"),
             updated_at=dt.datetime(2026, 11, 11, 20, tzinfo=dt.timezone.utc), is_live=True),
        dict(ts=dt.date(2026, 11, 11), ticker="USGG5YR Index", px=4.1,
             updated_at=None, is_live=True),
    ])
    monkeypatch.setattr(markets, "query_db", lambda *args, **kwargs: data)
    stamp = markets.source_stamp(markets.pull())
    assert "mixed dates" in stamp
    assert "EOD / mixed snapshot" in stamp
    assert "Nov 10, 2026 03:00:00 PM EST" in stamp
    assert "1 source timestamps unavailable" in stamp


def test_old_cache_does_not_invent_source_time():
    old = pl.DataFrame({"ts": [dt.date(2026, 11, 10)], "10y": [420.0]})
    assert "source update time unavailable" in markets.source_stamp(old)
