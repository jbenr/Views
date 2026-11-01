"""backtest.engine: the fast walk and the exact bar-by-bar walk agree trade for trade
on every rule (fills, re-arms, gates, exits, sizing), and the rules do what they say."""

import itertools

import numpy as np
import pytest

from backtest.engine import (EXIT_REASONS, STATS, TRADE_COLS, configs_frame, exact_stats, gate_edges, gate_mask,
                             mask_from_edges, walk, walk_exact)


def _model(n=900, seed=0):
    """A target that mean-reverts around a fit on a random-walk input, with a
    rolling-looking fit: slightly wandering alpha/beta, a scale and a half-life."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 3, n)) + 400
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = 0.93 * resid[i - 1] + rng.normal(0, 2.0)
    beta = 0.4 + 0.05 * np.sin(np.arange(n) / 90)
    alpha = 10 + 0.01 * np.arange(n)
    y = alpha + beta * x + resid
    sigma = np.full(n, resid.std())
    warm = 30
    for a in (alpha, beta, sigma):
        a[:warm] = np.nan
    signal_raw = y - alpha - beta * x
    half_life = np.where(np.arange(n) < warm, np.nan, 9.0 + 3 * np.cos(np.arange(n) / 50))
    half_life[(np.arange(n) // 40) % 5 == 3] = np.nan  # stretches with no measured reversion
    return dict(y=y, x=x, alpha=alpha, beta=beta, center=np.zeros(n), sigma=sigma, half_life=half_life,
                hedge_beta=beta * 0.9, port_vol=np.full(n, 2.5),
                signal_z=signal_raw / sigma, signal_raw=signal_raw)


def _start(model, scale):
    s = model["signal_z" if scale == "z" else "signal_raw"]
    t = 1
    while not np.isfinite(s[t]):
        t += 1
    return t


GRID = [dict(scale=sc, entry=e, revert=r, band=b, max_hold=h, stop=st, hl_mult=hm, pnl_stop=ps, sizing=sz,
             fill=fl, rearm=ra, gate=g, need_half_life=nh)
        for (sc, e), r, b, h, st, hm, ps, sz, fl, ra, g, nh in itertools.product(
            [("z", 1.0), ("z", 1.8), ("raw", 4.0)], [0.0, 0.5, 1.0], [np.nan, 0.3], [5, 40], [0.0, 1.0], [0.0, 1.5],
            [0.0, 4.0], ["flat", "bps_per_day"], ["same", "next"], ["threshold", "half", "mean", "by_exit"], [-1, 0],
            [True, False])]


@pytest.mark.parametrize("seed", [0, 1])
def test_fast_and_exact_walks_agree_trade_for_trade(seed):
    model = _model(seed=seed)
    gates = np.vstack([np.sin(np.arange(len(model["y"])) / 37) > -0.3])
    rng = np.random.default_rng(seed)
    configs = configs_frame([GRID[i] for i in rng.choice(len(GRID), 160, replace=False)])
    split = 500
    trades, stats = walk(model, configs, hedge=1.0, cost=0.5, split=split, gates=gates)
    for c, cfg in enumerate(configs.iter_rows(named=True)):
        exact, marks = walk_exact(model, cfg, hedge=1.0, cost=0.5, gate=gates[0] if cfg["gate"] == 0 else None)
        fast = trades[trades[:, 0] == c]
        assert len(fast) == len(exact), cfg
        for row, tr in zip(fast, exact):
            for i, col in enumerate(TRADE_COLS[1:], start=1):
                want = EXIT_REASONS.index(tr[col]) if col == "reason" else float(tr[col])
                assert row[i] == pytest.approx(want, rel=1e-9, abs=1e-9, nan_ok=True), (cfg, col)
        want = exact_stats(exact, marks, _start(model, cfg["scale"]), split)
        for i, key in enumerate(STATS):
            assert stats[c, i] == pytest.approx(want[key], rel=1e-9, abs=1e-9, nan_ok=True), (cfg, key)


def test_entries_need_a_crossing_and_rearming_follows_the_rule():
    """A signal that leaves the band and stays out trades once until it comes back."""
    n = 200
    s = np.zeros(n)
    s[20:120] = 3.0      # out at 20 and stays out
    s[120:125] = 0.5     # back inside the entry band (but not through the mean)
    s[125:140] = 3.0     # out again
    s[140:145] = -0.2    # through the mean
    s[145:160] = 3.0
    model = dict(y=s.copy(), x=np.zeros(n), alpha=np.zeros(n), beta=np.zeros(n), center=np.zeros(n),
                 sigma=np.ones(n), half_life=np.full(n, 5.0), hedge_beta=np.zeros(n), port_vol=np.ones(n),
                 signal_z=s, signal_raw=s)
    base = dict(scale="z", entry=2.0, max_hold=5, fill="same")
    entries = {}
    for rule in ("threshold", "mean", "by_exit"):
        trades, _ = walk(model, configs_frame(dict(base, rearm=rule)))
        entries[rule] = trades[:, 1].astype(int).tolist()
    # every exit here is a 5-bar time stop with the signal still out: never re-entered inside a run
    assert entries["threshold"] == [20, 125, 145]
    assert entries["mean"] == [20, 145]       # 120 is back inside but not through the mean
    assert entries["by_exit"] == [20, 145]    # time exits wait for the mean


def test_fill_moves_both_ends_and_episodes_merge_one_dislocation():
    n = 120
    s = np.zeros(n)
    s[10:30] = 3.0
    s[30:33] = 1.0   # back inside, same side
    s[33:50] = 3.0
    model = dict(y=s.copy(), x=np.zeros(n), alpha=np.zeros(n), beta=np.zeros(n), center=np.zeros(n),
                 sigma=np.ones(n), half_life=np.full(n, 5.0), hedge_beta=np.zeros(n), port_vol=np.ones(n),
                 signal_z=s, signal_raw=s)
    for fill, lag in (("same", 0), ("next", 1)):
        trades, stats = walk(model, configs_frame(dict(scale="z", entry=2.0, max_hold=4, fill=fill)))
        assert trades[:, 2].tolist() == [t + lag for t in trades[:, 1]]
        assert trades[:, 3].tolist() == [10 + 4 + lag, 33 + 4 + lag]
        assert stats[0, STATS.index("n_trades")] == 2
        assert stats[0, STATS.index("episodes")] == 1  # never crossed the mean: one dislocation


def test_gate_buckets_are_raw_cuts_or_point_in_time():
    v = np.arange(1000, dtype=float)
    kind, edges = gate_edges(v, ("r2", "high_75"))
    assert kind == "above" and edges[0] == pytest.approx(749.25)
    full = gate_mask(v, ("r2", "high_75"))
    assert full.sum() == 250 and np.array_equal(full, mask_from_edges(v, kind, edges))
    # point in time: a rising series is always at the top of its own history
    causal = gate_mask(v, ("r2", "high_75"), mode="causal", min_history=50)
    assert not causal[:49].any() and causal[60:].all()


def test_no_entry_without_a_measured_half_life():
    """A crossing where the residual shows no reversion (no half-life) is not traded, unless
    the config is an older one that ran without the rule."""
    model = _model(seed=3)
    hl = model["half_life"]
    base = dict(scale="z", entry=1.0, revert=0.5, max_hold=20, sizing="flat")
    needed, _ = walk(model, configs_frame(dict(base, need_half_life=True)))
    loose, _ = walk(model, configs_frame(dict(base, need_half_life=False)))
    decided = lambda t: t[:, TRADE_COLS.index("decided")].astype(int)  # noqa: E731
    assert np.isfinite(hl[decided(needed)]).all()
    assert (~np.isfinite(hl[decided(loose)])).any()  # the old rule did enter on them


def test_a_big_grid_walks_in_parallel_to_the_same_trades_and_stats(monkeypatch):
    # the bake's grids run their configs across threads (WALK_THREADS); each config writes
    # only its own rows, so the result is exactly the one-by-one walk's, in the same order
    import backtest.engine as engine

    model = _model(seed=3)
    grid = [dict(scale=sc, entry=e, revert=r, max_hold=h, rearm=ra, sizing=sz, pnl_stop=p)
            for sc, e in [("z", 1.0), ("z", 1.5), ("raw", 2.0), ("raw", 3.0)]
            for r, h, ra, sz, p in itertools.product([0.0, 0.5, 1.0], [10, 40], ["threshold", "mean", "by_exit"],
                                                     ["flat", "exp_return"], [0.0, 3.0])]
    configs = configs_frame(grid)
    assert len(configs) >= engine.PARALLEL_CONFIGS
    parallel = walk(model, configs, cost=0.2, split=450)
    monkeypatch.setattr(engine, "PARALLEL_CONFIGS", 10 ** 9)  # force the serial path
    serial = walk(model, configs, cost=0.2, split=450)
    assert len(parallel[0]) > 100
    assert np.array_equal(parallel[0], serial[0], equal_nan=True)
    assert np.array_equal(parallel[1], serial[1], equal_nan=True)
