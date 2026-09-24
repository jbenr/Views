"""Optimized discovery must preserve the original vectorized scan semantics."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from backtest.lab import predict_scan


@pytest.mark.parametrize("buckets", [3, "regime"])
@pytest.mark.parametrize("custom_forward", [False, True])
def test_batched_single_model_matches_multi_column_reference(buckets, custom_forward):
    rng = np.random.default_rng(86)
    z = rng.normal(size=400)
    z[:12] = np.nan
    z[88:96] = np.nan
    level = rng.normal(size=400).cumsum()
    level[215] = np.nan
    gate = rng.integers(0, 5, size=400).astype(float)
    gate[:30] = np.nan
    forwards = {h: np.r_[np.full(190, np.nan), rng.normal(size=400-190-h),
                         np.full(h, np.nan)] for h in [5, 20]} if custom_forward else None
    kwargs = dict(entries=[0.5, 1.5, 20], horizons=[5, 20], gate_buckets=buckets,
                  gate_min_history=40, gate_windows=[None, 63, 126], forward_moves=forwards)
    fast = predict_scan(z, level, gates={"condition": gate}, device="auto", **kwargs)
    reference = predict_scan(np.column_stack([z, z]), level,
                             gates={"condition": gate}, device="cpu", **kwargs)
    keys = ["entry_threshold", "horizon", "gate", "gate_bucket", "gate_window"]
    assert_frame_equal(fast.sort(keys), reference.filter(pl.col("col") == 0).sort(keys),
                       rel_tol=1e-10, abs_tol=1e-12)


def test_progress_eta_uses_scoring_time_not_preparation(monkeypatch):
    from research import progress
    now = [0.0]
    monkeypatch.setattr(progress, "monotonic", lambda: now[0])
    progress.start("eta-test", "dis", "Preparing")
    now[0] = 600
    progress.update("eta-test", "dis", "Scoring", 0, 100)
    now[0] = 610
    progress.update("eta-test", "dis", "Scored two", 2, 100)
    state = progress.snapshot("eta-test", "dis")
    assert state["elapsed"] == 610
    assert state["phase_elapsed"] == 10


def test_single_model_auto_does_not_probe_cuda(monkeypatch):
    import backtest.lab as lab

    def unexpected_backend(*args):
        pytest.fail("Single-model auto scoring must not initialize CUDA")

    monkeypatch.setattr(lab, "_get_xp", unexpected_backend)
    result = lab.predict_scan(np.arange(40.0), np.arange(40.0), device="auto")
    assert len(result) == 9
