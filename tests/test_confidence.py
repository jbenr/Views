"""Walk-forward confidence: past trades only, shrunk, and HIGH where the state paid."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stats import CONF_MIN_PAST, confidence_now, walk_forward_confidence


def _trades(n=60, seed=1):
    # one trade per 10 days, each held 5; fast half-life trades win, slow ones lose
    rng = np.random.default_rng(seed)
    half_life = rng.uniform(2, 30, n)
    pnl = np.where(half_life < 10, 4.0, -1.0) + rng.normal(0, 1, n)
    start = date(2010, 1, 4)
    return pl.DataFrame({"entry_ts": [start + timedelta(days=10 * i) for i in range(n)],
                         "exit_ts": [start + timedelta(days=10 * i + 5) for i in range(n)],
                         "pnl": pnl, "half_life": half_life})


LABELS = {"half_life": "half-life"}


def test_no_confidence_before_enough_trades_have_closed():
    out = walk_forward_confidence(_trades(), LABELS)
    assert (out["conf"][:CONF_MIN_PAST] == "—").all()
    assert out["conf_n_past"][CONF_MIN_PAST] == CONF_MIN_PAST


def test_the_state_that_paid_reads_high_and_the_one_that_lost_reads_low():
    out = walk_forward_confidence(_trades(), LABELS).filter(pl.col("conf") != "—")
    fast, slow = out.filter(pl.col("half_life") < 8), out.filter(pl.col("half_life") > 22)
    assert (fast["conf"] == "HIGH").mean() > 0.8
    assert (slow["conf"] == "LOW").mean() > 0.8
    now = confidence_now(_trades(), {"half_life": 3.0}, LABELS)
    assert now["label"] == "HIGH" and "half-life 3" in now["drivers"]


def test_a_trade_never_sees_trades_that_closed_after_it_entered():
    trades = _trades()
    k = 40
    changed = trades.with_columns(pl.when(pl.int_range(pl.len()) >= k).then(pl.col("pnl") * -10)
                                  .otherwise(pl.col("pnl")).alias("pnl"))
    before = walk_forward_confidence(trades, LABELS)["conf_score"][: k + 1]
    after = walk_forward_confidence(changed, LABELS)["conf_score"][: k + 1]
    assert before.to_list() == after.to_list()  # trade k entered before trade k closed


def test_today_carries_every_number_its_score_is_built_from():
    trades = walk_forward_confidence(_trades(), LABELS)
    now = confidence_now(trades, {"half_life": 3.0}, LABELS)
    (c,) = now["components"]
    assert (c["feature"], c["value"], c["bucket"]) == ("half-life", 3.0, "low") and c["cut_lo"] < c["cut_hi"]
    assert c["edge"] == pytest.approx(c["weight"] * (c["avg"] - c["past_avg"]) / c["spread"])
    assert now["score"] == pytest.approx(c["edge"])  # one feature: the score is its edge
    assert now["score"] > now["tier_hi"] and now["label"] == "HIGH"


def test_each_trade_records_the_tercile_and_edge_it_entered_with():
    out = walk_forward_confidence(_trades(), LABELS)
    rated, unrated = out.filter(pl.col("conf") != "—"), out.filter(pl.col("conf") == "—")
    assert rated["conf_bucket_half_life"].is_in(["low", "mid", "high"]).all()
    assert rated["conf_score"].to_list() == pytest.approx(rated["conf_edge_half_life"].to_list())
    assert unrated["conf_bucket_half_life"].is_null().all()
