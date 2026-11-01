"""Oven setups as live dashboard signals.

A setup picked in the Oven app (one input, lookback, hedge, basis, entry and
exits from a saved bake) is promoted into the dashboard's registry as an
"oven" signal: its row carries the frozen setup, so the dashboard runner can
re-walk it with oven's own engine on fresh data -- the cards, trades, open
position and ledger rows are exactly what the Oven showed, not a translation
into a different engine. OvenStrategy is the small object the runner talks to
in place of a book module's backtest.strategy.Strategy.

    entry = registry_entry(run_path, "10y", sid)          # what promotion stores
    python -m dashboard.registry --oven RUN --input 10y --setup "63|126|..."
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

from oven import engine
from research.panel import CATALOG, build_panel, panel_tickers


def signal_id(target: str, name: str, sid: str) -> str:
    return f"oven::{target}::{name}::{sid}"


def registry_entry(run_path: str | Path, name: str, sid: str, rationale: str | None = None) -> dict:
    """The registry row that promotes one setup of a saved bake: the frozen setup, the
    dashboard's shared param fields (filled so its labels and charts read it), and the
    bake's strategy stats for the setup."""
    from oven.app import describe, load_run, matches, setup_from_id, setup_gate  # the bake's own readers

    run_path = str(Path(run_path))
    run = load_run(run_path)
    target, cost = run["meta"]["target"], float(run["meta"]["settings"].get("cost", engine.COST))
    setup = setup_from_id(name, sid)
    rows = matches(run["board"], setup).to_dicts()
    if not rows:
        raise ValueError(f"{name} · {sid}: no such setup on the board of {run_path}")
    row = rows[0]
    sig = signal_id(target, name, sid)
    return {
        "signal_id": sig,
        "module": sig,  # the dashboard keys data refreshes by module; each oven signal pulls its own
        "kind": "oven",
        "name": re.sub(r"\W+", "_", f"oven_{name}_{target}_lb{setup['lookback']}_h{setup['hedge_lb']}"),
        "family": "curve",
        "target": target,
        "feature": name,
        "variant": None,
        "variant_label": None,  # named from the setup: dashboard.params.auto_label
        "rationale": rationale,
        "rank": None,
        "selection_source": f"oven bake {Path(run_path).name}",
        "promoted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        # the dashboard's shared param fields
        "entry_signal": "residual" if setup["scale"] == "raw" else "z",
        "beta_lb": int(setup["lookback"]),
        "ou_lb": None,
        "entry_threshold": float(setup["entry"]),
        "exit_style": "oven",
        "exit_param": float(setup["revert"]),
        "exit_text": describe(setup),
        "gate": None,
        "gate_window": None,
        "stop_loss_bps": float(setup["pnl_stop"]) or None,
        # the bake's stats for this setup
        "sharpe": row["sharpe"],
        "n_trades": row["n_trades"],
        "hit_rate": row["hit_rate"],
        "max_drawdown_bps": row["max_drawdown"],
        # what the runner re-walks
        "oven_run": run_path,
        "oven_setup": json.dumps(setup),
        # a full-sample gate's raw cut, frozen from the bake's window: live trading applies the same cut
        "oven_gate_edges": json.dumps(setup_gate(run_path, setup)),
        "oven_cost": cost,
    }


class OvenStrategy:
    """The interface dashboard.runner needs from a signal, for one promoted oven setup."""

    def __init__(self, row: dict):
        self.setup = json.loads(row["oven_setup"])
        self.cost = float(row["oven_cost"])
        self.gate_edges = json.loads(row.get("oven_gate_edges") or "null")
        self.transaction_cost_bps = self.cost
        self.name = row["name"]
        self.target = row["target"]
        self.feature = row["feature"]
        self.family = row["family"]
        self.tickers = panel_tickers(CATALOG[self.target], [self.feature])

    def load_data(self) -> pl.DataFrame:
        return build_panel(CATALOG[self.target], features=[self.feature]).data

    def model_frame(self, data: pl.DataFrame) -> pl.DataFrame:
        """The setup's model on the pulled data: what the cache holds and every card reads."""
        if "beta_changes" in data.columns:  # already a model frame (the cached pull)
            return data
        return engine.model_frame(data, self.target, self.feature, self.setup["lookback"], self.setup["hedge_lb"])

    def compute(self, data: pl.DataFrame, params: dict | None = None) -> pl.DataFrame:
        """Per bar: the setup's own entry signal (its residual in bps, or its z) and the
        fit it comes from, in the columns the dashboard's cards and ledger read."""
        m = engine.basis_arrays({c: data[c].to_numpy() for c in data.columns if c != "ts"}, self.setup["basis"])
        signal = m["resid"] if self.setup["scale"] == "raw" else m["z"]
        # beta is the signal fit's own slope: levels beta on the levels basis, the hedge on the portfolio one
        return pl.DataFrame({"ts": data["ts"], "signal": signal, "resid": m["resid"], "ou_z": m["z"],
                             "beta": m["beta"], "r2": data["r2"], "half_life": m["half_life"]})

    def history(self, data: pl.DataFrame) -> tuple[pl.DataFrame, dict | None, pl.DataFrame]:
        """(closed trade log, open position or None, daily equity curve) in the shapes
        dashboard.runner.trade_history returns for a book strategy."""
        trades = engine.setup_trades(data, self.target, self.setup, self.cost, gate_edges=self.gate_edges)
        y = data[self.target].to_numpy().astype(float)
        x = data[self.feature].to_numpy().astype(float)
        if trades.is_empty():
            marks = np.zeros(len(data))
            return pl.DataFrame(), None, self._equity(data, marks)
        level = data.select(pl.col("ts"), pl.col(self.target).alias("level"))
        trades = (trades.join(level.rename({"ts": "entry_ts", "level": "entry_level"}), on="entry_ts", how="left")
                  .join(level.rename({"ts": "exit_ts", "level": "exit_level"}), on="exit_ts", how="left"))
        closed, held = trades.filter(pl.col("reason") != "open"), trades.filter(pl.col("reason") == "open")
        marks = (engine.daily_marks(data["ts"], y, x, closed, self.cost)
                 + engine.daily_marks(data["ts"], y, x, held, 0.0))  # the open trade has no exit cost yet
        log = closed.select(
            pl.col("entry_ts").alias("entry_date"), pl.col("exit_ts").alias("exit_date"),
            pl.when(pl.col("side") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("direction"),
            "entry_level", "exit_level", pl.lit(None, pl.Float64).alias("target_lvl"),
            pl.col("exp_return").cast(pl.Float64).alias("expected_return_bps"),
            pl.col("exp_bars").cast(pl.Float64).alias("expected_bars"),
            pl.col("pnl").alias("pnl_bps"), pl.col("half_life").cast(pl.Float64).alias("entry_half_life"),
            pl.col("bars").alias("bars_held"), pl.col("reason").alias("exit_reason"),
            # entry state the dashboard's confidence reads (dashboard.runner.live_read)
            pl.col("entry_z").abs().cast(pl.Float64).alias("abs_z"), pl.col("r2").cast(pl.Float64),
            pl.col("beta_drift20").abs().cast(pl.Float64).alias("abs_drift"),
        ).sort("exit_date", descending=True)
        open_entry = None
        if not held.is_empty():
            p = held.row(0, named=True)
            open_entry = {
                "date": p["entry_ts"], "level": p["entry_level"],
                "direction": "long" if p["side"] > 0 else "short",
                "entry_date": p["entry_ts"], "exit_date": None,
                "entry_level": p["entry_level"], "exit_level": float(y[-1]),
                "target_lvl": None, "expected_return_bps": float(p["exp_return"]),
                "expected_bars": float(p["exp_bars"]), "pnl_bps": float(p["pnl"]),
                "entry_half_life": float(p["half_life"]), "bars_held": int(p["bars"]),
                "exit_reason": "active",
            }
        return log, open_entry, self._equity(data, marks)

    @staticmethod
    def _equity(data: pl.DataFrame, marks: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({"ts": data["ts"], "pnl_bps": marks, "cumulative_pnl": np.cumsum(marks)})
