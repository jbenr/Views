"""Parameter sweep and walk-forward optimization.

Grid search over strategy parameters, with optional walk-forward
validation to avoid overfitting.

Usage:
    from backtest.sweep import ParameterSweep, SweepConfig

    def build_and_run(params, data):
        # Build engine + pipeline with params, run backtest, return result
        ...

    sweep = ParameterSweep(
        build_fn=build_and_run,
        data=data,
        config=SweepConfig(
            params={"lookback": [30, 50, 100], "z_entry": [1.5, 2.0, 2.5]},
        ),
    )
    results = sweep.run()          # full-sample grid search
    oos = sweep.walk_forward()     # rolling train/test
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Optional
import polars as pl
import numpy as np
from datetime import date, timedelta

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from .engine import BacktestResult


@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""

    params: dict[str, list] = field(default_factory=dict)
    optimize_metric: str = "sharpe"    # metric to optimize in walk-forward
    train_years: int = 3
    test_years: int = 1
    step_years: int = 1                # slide forward by this many years


class ParameterSweep:
    """Grid search over strategy parameters."""

    def __init__(
        self,
        build_fn: Callable[[dict, pl.DataFrame], BacktestResult],
        data: pl.DataFrame,
        config: SweepConfig,
    ):
        """
        Args:
            build_fn: Callable(params_dict, data) -> BacktestResult
                      Builds engine + pipeline with given params, runs backtest.
            data: Full dataset (wide-format with 'ts' column).
            config: Sweep configuration.
        """
        self.build_fn = build_fn
        self.data = data
        self.config = config

    def _param_combos(self) -> list[dict]:
        """Generate all parameter combinations."""
        keys = list(self.config.params.keys())
        vals = list(self.config.params.values())
        return [dict(zip(keys, combo)) for combo in product(*vals)]

    def run(self) -> pl.DataFrame:
        """Full-sample grid search. Returns one row per param combo."""
        combos = self._param_combos()
        records = []

        for params in tqdm(combos, desc="Grid search"):
            try:
                result = self.build_fn(params, self.data)
                metrics = result.summary()
                row = {**params, **metrics}
            except Exception as e:
                row = {**params, "error": str(e)}
            records.append(row)

        return pl.DataFrame(records).sort(self.config.optimize_metric, descending=True)

    def walk_forward(self) -> pl.DataFrame:
        """Rolling walk-forward optimization.

        For each window:
        1. Train: find best params on [t, t + train_years)
        2. Test: evaluate best params on [t + train_years, t + train_years + test_years)
        3. Slide forward by step_years
        """
        dates = self.data["ts"].sort()
        min_date = dates[0]
        max_date = dates[-1]

        combos = self._param_combos()
        oos_records = []  # out-of-sample results

        # Convert to date objects for arithmetic
        if isinstance(min_date, date):
            start = min_date
        else:
            start = min_date.date() if hasattr(min_date, 'date') else min_date

        if isinstance(max_date, date):
            end = max_date
        else:
            end = max_date.date() if hasattr(max_date, 'date') else max_date

        train_delta = timedelta(days=self.config.train_years * 365)
        test_delta = timedelta(days=self.config.test_years * 365)
        step_delta = timedelta(days=self.config.step_years * 365)

        # Precompute window starts for tqdm
        window_starts = []
        t = start
        while t + train_delta + test_delta <= end:
            window_starts.append(t)
            t += step_delta

        for window_num, current in enumerate(tqdm(window_starts, desc="Walk-forward")):
            train_end = current + train_delta
            test_end = train_end + test_delta

            train_data = self.data.filter(
                (pl.col("ts") >= current) & (pl.col("ts") < train_end)
            )
            test_data = self.data.filter(
                (pl.col("ts") >= train_end) & (pl.col("ts") < test_end)
            )

            if train_data.is_empty() or test_data.is_empty():
                continue

            # Find best params on training set
            best_params = None
            best_metric = -np.inf

            for params in tqdm(combos, desc=f"  Window {window_num} train", leave=False):
                try:
                    result = self.build_fn(params, train_data)
                    m = result.summary().get(self.config.optimize_metric, -np.inf)
                    if m > best_metric:
                        best_metric = m
                        best_params = params
                except Exception:
                    continue

            if best_params is None:
                continue

            # Evaluate best params on test set (out-of-sample)
            try:
                oos_result = self.build_fn(best_params, test_data)
                oos_metrics = oos_result.summary()
                oos_records.append({
                    "window": window_num,
                    "train_start": str(current),
                    "train_end": str(train_end),
                    "test_start": str(train_end),
                    "test_end": str(test_end),
                    **{f"best_{k}": v for k, v in best_params.items()},
                    f"is_train_{self.config.optimize_metric}": best_metric,
                    **{f"oos_{k}": v for k, v in oos_metrics.items()},
                })
            except Exception as e:
                oos_records.append({
                    "window": window_num,
                    "train_start": str(current),
                    "error": str(e),
                })

        if not oos_records:
            return pl.DataFrame()

        return pl.DataFrame(oos_records)
