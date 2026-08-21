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
from dashboard.params import params_from_row
from dashboard.registry import LiveRegistry


def _load_app_module(monkeypatch, tmp_path):
    monkeypatch.setenv("VIEWS_STORE_DIR", str(tmp_path))
    sys.modules.pop("dashboard.app", None)
    return importlib.import_module("dashboard.app")


def _component_ids(component):
    component_id = getattr(component, "id", None)
    if component_id is not None:
        yield component_id
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or getattr(child, "id", None) is not None:
            yield from _component_ids(child)


def test_dashboard_has_live_overview_and_selectable_deep_dive(monkeypatch, tmp_path):
    pl.DataFrame(
        [
            {
                "module": "book.curve.twos_10s30s",
                "name": "twos_10s30s",
                "family": "curve",
                "target": "10s30s",
                "feature": "2y",
                "entry_signal": "ou_z",
                "beta_lb": 90,
                "ou_lb": 470,
                "entry_threshold": 0.9,
                "exit_style": "revert_frac",
                "exit_param": 1.0,
                "gate": None,
                "stop_loss_bps": 25.0,
                "sharpe": 0.71,
                "n_trades": 58,
                "hit_rate": 0.707,
                "max_drawdown_bps": -52.4,
            }
        ]
    ).write_parquet(tmp_path / "live_signals.parquet")

    app_module = _load_app_module(monkeypatch, tmp_path)

    # The layout is deliberately only a shell: building the tabs costs an
    # Engine.run() per signal, and doing that before the response is sent
    # leaves the browser blank with no window to show a loading state in.
    layout = app_module.app.layout
    shell = set(_component_ids(layout() if callable(layout) else layout))
    assert {"page-boot", "page-body"} <= shell
    assert "dashboard-tabs" not in shell

    # ...so the page proper arrives via the boot callback, the way a browser
    # gets it. Exercise that rather than the builder, so the wiring is covered.
    response = app_module.app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": "page-body.children",
            "outputs": {"id": "page-body", "property": "children"},
            "inputs": [
                {"id": "page-boot", "property": "n_intervals", "value": 1}
            ],
            "changedPropIds": ["page-boot.n_intervals"],
        },
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    page = response.data.decode()
    for component_id in (
        "dashboard-tabs",
        "overview-content",
        "refresh-overview",
        "deep-dive-signal",
        "deep-card-book-curve-twos-10s30s",
    ):
        assert component_id in page


def test_overview_snapshot_uses_exact_open_position(monkeypatch, tmp_path):
    app = _load_app_module(monkeypatch, tmp_path)
    state = {
        "params": {
            "entry_signal": "ou_z",
            "entry_threshold": 0.9,
            "gate": None,
        },
        "last": {"signal": -1.2},
        "fired": "long",
        "data_asof": pd.Timestamp("2026-07-24").date(),
    }
    trades = pl.DataFrame({"pnl_bps": [4.0, -1.0]})
    open_entry = {"direction": "long", "pnl_bps": 2.5}
    monkeypatch.setattr(app.runner, "compute_signal", lambda _module: state)
    monkeypatch.setattr(
        app.runner,
        "trade_history",
        lambda _module, _state: (trades, open_entry),
    )

    snapshot = app._overview_snapshot(
        {
            "module": "book.curve.twos_10s30s",
            "target": "10s30s",
            "name": "twos_10s30s",
            "feature": "2y",
            "entry_signal": "ou_z",
            "beta_lb": 90,
            "ou_lb": 470,
            "entry_threshold": 0.9,
            "sharpe": 0.71,
            "n_trades": 58,
            "hit_rate": 0.707,
            "max_drawdown_bps": -52.4,
        }
    )

    assert snapshot["position"] == "LONG OPEN"
    # the live value and the threshold it is judged against share one cell
    assert snapshot["reading"] == "-1.20z / ±0.9z"
    assert snapshot["gate_rule"] == "none"
    assert snapshot["live_pnl_bps"] == 5.5
    # Feature is gone: the Signal name already leads with it
    assert "feature" not in {key for key, _ in app.OVERVIEW_COLUMNS}
    assert snapshot["name"].startswith("2y_10s30s")


def test_dashboard_signal_names_match_the_module_input_target_order(
    monkeypatch, tmp_path
):
    """Display names read `<input>_<target>`, the same order the strategy
    modules are named in, so one signal has one name project-wide."""
    app = _load_app_module(monkeypatch, tmp_path)

    assert app._display_name(
        {
            "target": "10s30s",
            "feature": "2y",
            "variant_label": "OU430 · 0.9z",
        }
    ) == "2y_10s30s · OU430 · 0.9z"
    assert app._display_name(
        {
            "target": "2s10s",
            "feature": "real10y",
            "variant_label": "OU390 · 1.7z",
        }
    ) == "real10y_2s10s · OU390 · 1.7z"


def test_unlabeled_promotions_are_named_from_their_frozen_params(
    monkeypatch, tmp_path
):
    """A promotion with no curated variant name must still say what it is --
    "default" told the desk nothing about how the signal was configured."""
    app = _load_app_module(monkeypatch, tmp_path)

    assert app._display_name(
        {
            "target": "10s30s",
            "feature": "2y",
            "entry_signal": "ou_z",
            "beta_lb": 90,
            "ou_lb": 470,
            "entry_threshold": 0.9,
        }
    ) == "2y_10s30s · OU470 · 0.9z"
    assert app._display_name(
        {
            "target": "10s30s",
            "feature": "10y",
            "entry_signal": "residual",
            "beta_lb": 90,
            "ou_lb": None,
            "entry_threshold": 19.0,
        }
    ) == "10y_10s30s · RES90 · 19bps"


def _state(signal, threshold=1.7, fired="flat", gate=None, gate_window=None):
    return {
        "params": {
            "entry_signal": "ou_z",
            "entry_threshold": threshold,
            "gate": gate,
            "gate_window": gate_window,
        },
        "last": {"signal": signal},
        "fired": fired,
    }


def test_position_column_folds_in_what_the_signal_is_saying(monkeypatch, tmp_path):
    """One column has to carry both what is held and what today calls for --
    a fired-but-gated signal used to render as a bare FLAT."""
    app = _load_app_module(monkeypatch, tmp_path)

    held = {"direction": "long", "pnl_bps": 2.5}
    assert app._position_label(_state(-2.0, fired="long"), held) == "LONG OPEN"
    # an open trade wins even when today's reading has gone quiet
    assert app._position_label(_state(0.1), held) == "LONG OPEN"

    assert app._position_label(_state(0.1), None) == "FLAT"
    assert app._position_label(_state(-2.0, fired="long"), None) == "LONG SIGNAL"
    assert app._position_label(_state(2.0, fired="short"), None) == "SHORT SIGNAL"
    assert (
        app._position_label(_state(-2.0, fired="flat (gated)"), None)
        == "LONG GATED"
    )
    assert app._position_label(_state(float("nan")), None) == "WARMING UP"


def test_gate_column_states_the_rule_and_its_percentile_basis(
    monkeypatch, tmp_path
):
    """The gate cell has to say what is being tested, and against what -- the
    same condition and bucket mean different gates on different lookbacks."""
    app = _load_app_module(monkeypatch, tmp_path)

    assert app._gate_rule({"gate": None}) == "none"
    assert app._gate_rule(
        {"gate": ("r2", "tails_25_75"), "gate_window": 1260}
    ) == "r2 · tails_25_75 · roll 1260d"
    assert app._gate_rule(
        {"gate": ("resid_half_life", "below_50"), "gate_window": None}
    ) == "resid_half_life · below_50 · expanding"


def test_registry_list_states_the_whole_frozen_configuration():
    """--list is the record of what is live, so it has to spell out every
    parameter -- including the gate basis and the filters that are off."""
    from dashboard.registry import describe

    frame = pl.DataFrame([{
        "signal_id": "book.curve.tens_10s30s",
        "module": "book.curve.tens_10s30s",
        "name": "tens_10s30s", "family": "curve",
        "target": "10s30s", "feature": "10y",
        "variant": None, "variant_label": None, "rationale": None,
        "rank": 0, "selection_source": None,
        "promoted_at": "2026-07-29T04:33:59+00:00",
        "entry_signal": "ou_z", "beta_lb": 80, "ou_lb": 400,
        "entry_threshold": 1.7, "exit_style": "revert_frac", "exit_param": 1.0,
        "gate": "('r2', 'tails_25_75')", "gate_window": 1260,
        "stop_loss_bps": 25.0,
        "sharpe": 0.70089, "n_trades": 17.0, "hit_rate": 0.882353,
        "max_drawdown_bps": -22.63,
    }])

    out = describe(frame)

    assert "CURVE" in out                              # grouped by family
    assert "10s30s" in out                             # target leads the row
    assert "ou_z b80/ou400" in out                     # both lookbacks
    assert "1.7z" in out                               # entry threshold + units
    assert "revert 100%" in out                        # exit rule and its size
    assert "25bps" in out
    assert "r2 tails_25_75 /1260d" in out              # gate basis, not just bucket
    assert "0.70" in out and "17" in out               # frozen backtest metrics
    assert "book.curve.tens_10s30s" in out             # full id, pasteable into --remove
    assert "Jul 29" in out                             # when it went live


def test_registry_list_groups_by_family_and_keeps_gates_off_visible():
    """Rows are only comparable within a family, so that is the grouping;
    target leads and orders each row. A signal with no gate still says so."""
    from dashboard.registry import describe

    def row(signal_id, target, feature, gate, window, family="curve"):
        return {
            "signal_id": signal_id, "module": signal_id,
            "name": signal_id, "family": family,
            "target": target, "feature": feature,
            "variant": None, "variant_label": None, "rationale": None,
            "rank": 0, "selection_source": None,
            "promoted_at": "2026-07-29T04:33:59+00:00",
            "entry_signal": "ou_z", "beta_lb": 80, "ou_lb": 400,
            "entry_threshold": 1.7, "exit_style": "band", "exit_param": 1.5,
            "gate": gate, "gate_window": window, "stop_loss_bps": 25.0,
            "sharpe": 0.70, "n_trades": 17.0, "hit_rate": 0.88,
            "max_drawdown_bps": -22.6,
        }

    out = describe(pl.DataFrame([
        row("a", "10s30s", "10y", "('r2', 'tails_25_75')", 1260),
        row("b", "10s30s", "2y", None, None),
        row("c", "2s10s", "real10y", "('r2', 'tails_25_75')", None),
        row("d", "1m10y", "move", None, None, family="rate_vol"),
    ]))

    assert "4 live signals · 3 targets · 2 families" in out
    assert "CURVE · 3 signals · 2 targets" in out
    assert "RATE_VOL · 1 signal · 1 target" in out
    assert "none" in out        # a disabled gate is stated, not blank
    assert "/exp" in out        # expanding percentile basis, distinct from a roll

    # inside a family, rows are ordered by target
    curve = out.split("CURVE")[1].split("RATE_VOL")[0]
    assert curve.index("10s30s") < curve.index("2s10s")


def test_registry_can_promote_curated_module_defaults(monkeypatch, tmp_path):
    mod = importlib.import_module("book.curve.twos_10s30s")
    monkeypatch.setattr(mod.STRATEGY, "load_data", lambda: mod.synthetic_data(n=1500))
    monkeypatch.setattr(
        "dashboard.registry.load_strategy",
        lambda _module: mod.STRATEGY,
    )
    registry = LiveRegistry(tmp_path / "live.parquet")

    entry = registry.promote_defaults("book.curve.twos_10s30s")
    stored = registry.get("book.curve.twos_10s30s")

    assert entry["selection_source"] == "curated module defaults"
    assert stored["rank"] is None
    assert params_from_row(stored)["beta_lb"] == 90
    assert params_from_row(stored)["ou_lb"] == 470
    assert params_from_row(stored).get("gate") is None
    assert stored["n_trades"] > 0


def test_params_from_row_preserves_explicitly_disabled_filters():
    params = params_from_row(
        {
            "entry_signal": "residual",
            "beta_lb": 50.0,
            "ou_lb": 252.0,
            "gate": None,
        }
    )

    assert params["gate"] is None
    assert params["beta_lb"] == 50
    assert params["ou_lb"] == 252


def test_params_from_row_drops_null_required_values_from_union_schema():
    params = params_from_row(
        {
            "entry_signal": "residual",
            "beta_lb": 90.0,
            "ou_lb": None,
            "entry_threshold": 19.0,
            "exit_style": "revert_frac",
            "exit_param": 1.0,
            "stop_loss_bps": 25.0,
            "gate": None,
        }
    )

    assert "ou_lb" not in params
    assert params["gate"] is None


def test_registry_keeps_named_variants_beside_base(monkeypatch, tmp_path):
    mod = importlib.import_module("book.curve.twos_10s30s")
    monkeypatch.setattr(mod.STRATEGY, "load_data", lambda: mod.synthetic_data(n=1500))
    monkeypatch.setattr(
        "dashboard.registry.load_strategy",
        lambda _module: mod.STRATEGY,
    )
    registry = LiveRegistry(tmp_path / "live.parquet")

    base = registry.promote_defaults("book.curve.twos_10s30s")
    challenger = registry.promote_variant(
        "book.curve.twos_10s30s",
        "ou430_e09",
    )
    rows = registry.list().sort("signal_id")

    assert rows.height == 2
    assert set(rows["signal_id"]) == {
        "book.curve.twos_10s30s",
        "book.curve.twos_10s30s::ou430_e09",
    }
    assert base["ou_lb"] == 470
    assert challenger["ou_lb"] == 430
    assert challenger["variant_label"] == "OU430 · 0.9z"
    assert registry.get(challenger["signal_id"])["ou_lb"] == 430
    assert registry.get("book.curve.twos_10s30s")["ou_lb"] == 470
    assert registry.remove(challenger["signal_id"])
    assert registry.list().height == 1
    assert registry.get("book.curve.twos_10s30s")["ou_lb"] == 470


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
