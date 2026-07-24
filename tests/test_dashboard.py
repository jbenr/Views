"""Live-dashboard presentation behavior."""

import base64
import importlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from matplotlib.collections import LineCollection

from dashboard import charts
from dashboard.charts import _window_pnl_frame, gate_chart, pnl_chart


def _load_app_module(monkeypatch, tmp_path):
    monkeypatch.setenv("VIEWS_STORE_DIR", str(tmp_path))
    sys.modules.pop("dashboard.app", None)
    return importlib.import_module("dashboard.app")


def test_snap_range_adds_five_business_days_before_first_entry(
    monkeypatch, tmp_path
):
    app = _load_app_module(monkeypatch, tmp_path)
    trades = pl.DataFrame(
        {
            "entry_date": [pd.Timestamp("2026-07-13")],
            "exit_date": [pd.Timestamp("2026-07-20")],
        }
    )

    start, end = app._visible_trade_range(trades, None, 0, None)

    assert start == pd.Timestamp("2026-07-06")
    assert end == pd.Timestamp("2026-07-20")


def test_param_summary_explains_exit_and_hard_stop(monkeypatch, tmp_path):
    app = _load_app_module(monkeypatch, tmp_path)
    row = {
        "entry_signal": "residual",
        "beta_lb": 90,
        "ou_lb": None,
        "entry_threshold": 19.0,
        "exit_style": "revert_frac",
        "exit_param": 1.0,
        "stop_loss_bps": 25.0,
        "gate": "('beta_cv', 'tails_25_75')",
    }

    summary = app._param_summary(row)

    assert (
        "exit=revert_frac=1 "
        "(after 100% of entry signal reverts toward zero)"
    ) in summary
    assert "hard stop=25bps" in summary


def test_gate_chart_renders_png():
    dates = pl.date_range(
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2026-03-01"),
        interval="1d",
        eager=True,
    )
    n = len(dates)
    data = pl.DataFrame({"ts": dates})
    sig = pl.DataFrame(
        {
            "gate_percentile": [i / (n - 1) for i in range(n)],
            "gate_allow": [i / (n - 1) >= 0.75 for i in range(n)],
        }
    )

    encoded = gate_chart(
        data,
        sig,
        ("beta_cv", "tails_25_75"),
        window_bars=252,
    )

    assert encoded is not None
    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")


def test_gate_chart_colors_only_adjacent_segments(monkeypatch):
    dates = pl.date_range(
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-04"),
        interval="1d",
        eager=True,
    )
    data = pl.DataFrame({"ts": dates})
    sig = pl.DataFrame(
        {
            "gate_percentile": [0.4, 0.6, 0.4, 0.6],
            "gate_allow": [True, False, True, False],
        }
    )
    captured = {}

    def capture_render(self, frame, render_fn, title=None, **_kwargs):
        fig, ax = plt.subplots()
        render_fn(fig, ax, frame.index.min(), frame.index.max())
        collections = [
            collection
            for collection in ax.collections
            if isinstance(collection, LineCollection)
        ]
        captured["segments"] = collections[0].get_segments()
        captured["colors"] = collections[0].get_colors()
        plt.close(fig)
        return "captured"

    monkeypatch.setattr(charts._PngViz, "_make_time_nav", capture_render)
    assert gate_chart(data, sig, ("x", "below_50"), window_bars=None) == "captured"

    assert len(captured["segments"]) == 3
    assert [segment[:, 1].tolist() for segment in captured["segments"]] == [
        [40.0, 60.0],
        [60.0, 40.0],
        [40.0, 60.0],
    ]
    assert len(captured["colors"]) == 3
    assert captured["colors"][0].tolist() == captured["colors"][2].tolist()
    assert captured["colors"][0].tolist() != captured["colors"][1].tolist()


def test_pnl_chart_renders_exact_cumulative_curve():
    dates = pl.date_range(
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-03-01"),
        interval="1d",
        eager=True,
    )
    pnl = [1.0 if i % 3 else -0.5 for i in range(len(dates))]
    equity = pl.DataFrame({"ts": dates, "pnl_bps": pnl}).with_columns(
        pl.col("pnl_bps").cum_sum().alias("cumulative_pnl")
    )

    encoded = pnl_chart(equity, window_bars=30)
    visible = _window_pnl_frame(equity, window_bars=30)

    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")
    assert visible["cumulative_pnl"].iloc[0] == 0.0
    assert visible["cumulative_pnl"].iloc[-1] == (
        equity["cumulative_pnl"][-1] - equity["cumulative_pnl"][-30]
    )
