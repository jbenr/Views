"""Parameter lab — scalable sweeps, metric stores, fast scans, and gates.

The research funnel, cheapest to most exact:

1. `signal_matrix` + `fast_scan` — vectorized approximate backtest of an
   entire parameter grid at once. Pure array math (numpy semantics), so it
   runs unchanged on an NVIDIA GPU via cupy (`device="gpu"`). Use it to carve
   a huge grid down to a shortlist. Approximations: hysteresis-band exits
   (no stops/time-stops), next-bar execution, one position at a time.
2. `sweep_strategy` — the exact row-by-row Engine, one combo per task,
   parallel across CPU cores (ProcessPoolExecutor; a 32-core box runs 32
   combos at a time). Use it to verify the shortlist with real trade
   mechanics: stops, time-stops, costs, trade logs.
3. `MetricStore` — parquet-backed history of every run: leaderboards across
   strategies, metric matrices across any two parameter dimensions.
4. `gate_scan` — conditional edge analysis: bucket candidate state variables
   (beta_cv, r2, vol, half-life, ...) and measure how the strategy's forward
   PnL differs by bucket. This is the foundation for entry gates that keep
   only the highest-confidence setups of an already-good strategy.

Strategy contract for sweeps: the strategy module exposes
`make_pipeline(params: dict) -> SignalPipeline`. See
book/curve/tens_10s30s.py and book/rate_vol/template.py.

GPU setup (on the tower): `pip install -e ".[gpu]"`, then pass
`device="gpu"` to fast_scan. The extra includes the CUDA runtime components;
only a compatible NVIDIA GPU and driver are required.
"""

from __future__ import annotations

import datetime as dt
import importlib
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl


# ── parameter grids ──────────────────────────────────────────────────────────

@dataclass
class ParamGrid:
    """Cartesian product of parameter lists.

    ParamGrid({"beta_lb": [63, 252], "entry_z": [1.5, 2.0]}) -> 4 combos.
    """

    params: dict[str, list] = field(default_factory=dict)

    def combos(self) -> list[dict]:
        keys = list(self.params)
        return [dict(zip(keys, c)) for c in product(*self.params.values())]

    def __len__(self) -> int:
        n = 1
        for v in self.params.values():
            n *= len(v)
        return n


# ── exact sweep: Engine per combo, parallel across cores ────────────────────

_WORKER_STATE: dict = {}


def _import_strategy(module_name: str):
    """Import a strategy, with a source-tree fallback for spawned workers.

    On Windows, launching a strategy file directly from a nested directory
    leaves the repository root off the spawned process's import path. Normal
    package imports remain preferred; the fallback loads the matching source
    file without mutating ``sys.path``.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        top_package = module_name.partition(".")[0]
        if exc.name != top_package:
            raise

        module_path = (
            Path(__file__).resolve().parents[1]
            .joinpath(*module_name.split("."))
            .with_suffix(".py")
        )
        if not module_path.is_file():
            raise

        spec = spec_from_file_location(f"_sweep_{module_path.stem}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load sweep strategy from {module_path}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _init_worker(module_name: str, data: pl.DataFrame, cost: float, slip: float):
    _WORKER_STATE["module"] = _import_strategy(module_name)
    _WORKER_STATE["data"] = data
    _WORKER_STATE["cost"] = cost
    _WORKER_STATE["slip"] = slip


def _run_combo(params: dict) -> dict:
    from .engine import BacktestConfig, Engine

    try:
        pipeline = _WORKER_STATE["module"].make_pipeline(params)
        config = BacktestConfig(
            transaction_cost_bps=_WORKER_STATE["cost"],
            slippage_bps=_WORKER_STATE["slip"],
        )
        result = Engine(config).add_signal(pipeline).run(_WORKER_STATE["data"])
        return {**params, **{k: float(v) for k, v in result.summary().items()}}
    except Exception as e:  # keep the sweep alive; surface the failure in the row
        return {**params, "error": f"{type(e).__name__}: {e}"}


def sweep_strategy(
    module_name: str,
    data: pl.DataFrame,
    grid: Union[ParamGrid, dict],
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    n_jobs: Optional[int] = None,
    sort_by: str = "sharpe",
) -> pl.DataFrame:
    """Exact-engine backtest of every param combo, parallel across CPU cores.

    Args:
        module_name: importable strategy module exposing make_pipeline(params),
                     e.g. "book.curve.tens_10s30s".
        data: wide market-data frame (shipped once to each worker process).
        grid: ParamGrid or plain {param: [values]} dict.
        n_jobs: worker processes; default = all cores. 1 = serial (debugging).

    Returns one row per combo: params + full Engine metrics, best first.
    """
    if not isinstance(grid, ParamGrid):
        grid = ParamGrid(grid)
    combos = grid.combos()

    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    n_jobs = min(n_jobs, len(combos))

    if n_jobs <= 1:
        _init_worker(module_name, data, transaction_cost_bps, slippage_bps)
        rows = [_run_combo(p) for p in combos]
    else:
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_worker,
            initargs=(module_name, data, transaction_cost_bps, slippage_bps),
        ) as pool:
            rows = list(pool.map(_run_combo, combos))

    out = pl.DataFrame(rows, infer_schema_length=None)
    if sort_by in out.columns:
        out = out.sort(sort_by, descending=True, nulls_last=True)
    return out


# ── fast scan: vectorized approximate backtest (GPU-ready) ──────────────────

def _get_xp(device: str = "cpu"):
    """Return numpy, or cupy when a working CUDA device is available."""
    if device == "gpu":
        try:
            import cupy
        except ImportError:
            warnings.warn(
                'CuPy is not installed — falling back to CPU. Install the '
                'prebuilt CUDA extra with: pip install -e ".[gpu]" '
                "(do not use `pip install cupy`, which builds from source)."
            )
            return np

        try:
            device_count = cupy.cuda.runtime.getDeviceCount()
            if device_count > 0:
                # Device discovery alone does not prove kernels can compile.
                # This also catches missing CUDA runtime headers on CUDA 12.2+.
                cupy.add(cupy.asarray([0.0]), 1.0)
                cupy.cuda.Stream.null.synchronize()
        except Exception as exc:
            warnings.warn(
                f"CuPy is installed but CUDA is unavailable ({exc}) — "
                "fast_scan is falling back to CPU. Install the CUDA components "
                'with: pip install "cupy-cuda12x[ctk]"'
            )
            return np
        if device_count < 1:
            warnings.warn(
                "CuPy found no CUDA devices — fast_scan is falling back to CPU."
            )
            return np
        return cupy
    return np


def _to_numpy(arr) -> np.ndarray:
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


_CUPY_FFILL_KERNEL = None
_CUPY_MAX_DRAWDOWN_KERNEL = None


def _ffill_positions_cupy(events, xp):
    """Forward-fill positions on CUDA without unsupported ufunc.accumulate."""
    global _CUPY_FFILL_KERNEL

    if _CUPY_FFILL_KERNEL is None:
        _CUPY_FFILL_KERNEL = xp.RawKernel(
            r"""
            extern "C" __global__
            void ffill_positions(
                const double* events,
                double* positions,
                const int rows,
                const int cols
            ) {
                const int col = blockDim.x * blockIdx.x + threadIdx.x;
                if (col >= cols) return;

                double last = 0.0;
                for (int row = 0; row < rows; ++row) {
                    const int idx = row * cols + col;
                    const double value = events[idx];
                    if (value == value) last = value;  // NaN is not equal to itself
                    positions[idx] = last;
                }
            }
            """,
            "ffill_positions",
        )

    events = xp.ascontiguousarray(events, dtype=xp.float64)
    positions = xp.empty_like(events)
    rows, cols = events.shape
    threads = 128
    blocks = (cols + threads - 1) // threads
    _CUPY_FFILL_KERNEL(
        (blocks,),
        (threads,),
        (events, positions, np.int32(rows), np.int32(cols)),
    )
    return positions


def _ffill_positions(events, xp):
    """Forward-fill a (T, K) event matrix (+1/-1/0 at decision bars, NaN
    elsewhere) into a position matrix. Vectorized — no scan loop."""
    if xp.__name__ == "cupy":
        return _ffill_positions_cupy(events, xp)

    t_len = events.shape[0]
    valid = ~xp.isnan(events)
    idx = xp.where(valid, xp.arange(t_len)[:, None], 0)
    idx = xp.maximum.accumulate(idx, axis=0)
    pos = xp.take_along_axis(events, idx, axis=0)
    return xp.where(xp.isnan(pos), 0.0, pos)


def _max_drawdown(cumulative_pnl, xp):
    """Maximum drawdown by column, using a CUDA scan when needed."""
    if xp.__name__ != "cupy":
        drawdown = cumulative_pnl - xp.maximum.accumulate(cumulative_pnl, axis=0)
        return drawdown.min(axis=0)

    global _CUPY_MAX_DRAWDOWN_KERNEL
    if _CUPY_MAX_DRAWDOWN_KERNEL is None:
        _CUPY_MAX_DRAWDOWN_KERNEL = xp.RawKernel(
            r"""
            extern "C" __global__
            void max_drawdown(
                const double* cumulative,
                double* result,
                const int rows,
                const int cols
            ) {
                const int col = blockDim.x * blockIdx.x + threadIdx.x;
                if (col >= cols) return;

                double peak = cumulative[col];
                double worst = 0.0;
                for (int row = 0; row < rows; ++row) {
                    const double value = cumulative[row * cols + col];
                    if (value > peak) peak = value;
                    const double drawdown = value - peak;
                    if (drawdown < worst) worst = drawdown;
                }
                result[col] = worst;
            }
            """,
            "max_drawdown",
        )

    cumulative_pnl = xp.ascontiguousarray(cumulative_pnl, dtype=xp.float64)
    rows, cols = cumulative_pnl.shape
    result = xp.empty(cols, dtype=xp.float64)
    threads = 128
    blocks = (cols + threads - 1) // threads
    _CUPY_MAX_DRAWDOWN_KERNEL(
        (blocks,),
        (threads,),
        (cumulative_pnl, result, np.int32(rows), np.int32(cols)),
    )
    return result


def fast_scan(
    z: np.ndarray,
    level: np.ndarray,
    entries: Union[list[float], tuple[float, ...]] = (1.5, 2.0, 2.5),
    exit_band: float = 0.25,
    cost_bps: float = 0.0,
    periods_per_year: int = 252,
    combos: Optional[list[dict]] = None,
    device: str = "cpu",
) -> pl.DataFrame:
    """Approximate threshold backtest of a whole signal matrix, vectorized.

    Rules per column of z (a fade signal): go long when z <= -entry, short
    when z >= +entry, flat when |z| <= exit_band; positions carry between
    events and execute on the NEXT bar; costs charged per unit of turnover.

    Args:
        z: (T,) or (T, K) signal matrix — e.g. from signal_matrix().
        level: (T,) trade level in bps (composite the strategy trades).
        entries: entry thresholds to scan — the grid becomes K × len(entries).
        combos: optional K param dicts annotating z's columns.
        device: "cpu" (numpy) or "gpu" (cupy — install the ``gpu`` extra).

    Returns one row per (combo, entry_z): total_pnl_bps, sharpe, hit_rate,
    max_drawdown_bps, n_trades, n_bars_active. Best sharpe first.
    """
    xp = _get_xp(device)

    z = xp.asarray(z, dtype=xp.float64)
    if z.ndim == 1:
        z = z[:, None]
    t_len, k = z.shape
    if combos is not None and len(combos) != k:
        raise ValueError(f"combos has {len(combos)} entries but z has {k} columns")

    lv = xp.asarray(level, dtype=xp.float64)
    dlv = xp.concatenate([xp.asarray([xp.nan]), xp.diff(lv)])[:, None]
    dlv = xp.where(xp.isnan(dlv), 0.0, dlv)

    rows = []
    for entry in entries:
        events = xp.where(
            z <= -entry, 1.0,
            xp.where(z >= entry, -1.0,
                     xp.where(xp.abs(z) <= exit_band, 0.0, xp.nan)),
        )
        pos = _ffill_positions(events, xp)

        pos_lag = xp.concatenate([xp.zeros((1, k)), pos[:-1]], axis=0)
        turnover = xp.abs(pos - pos_lag)
        pnl = pos_lag * dlv - cost_bps * turnover

        total = pnl.sum(axis=0)
        mean, std = pnl.mean(axis=0), pnl.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sharpe = xp.where(std > 0, mean / std * np.sqrt(periods_per_year), xp.nan)

        active = pos_lag != 0
        n_active = active.sum(axis=0)
        wins = ((pnl > 0) & active).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            hit = xp.where(n_active > 0, wins / n_active, xp.nan)

        cum = xp.cumsum(pnl, axis=0)
        max_dd = _max_drawdown(cum, xp)

        n_trades = ((pos != pos_lag) & (pos != 0)).sum(axis=0)

        metrics = {
            "total_pnl_bps": _to_numpy(total),
            "sharpe": _to_numpy(sharpe),
            "hit_rate": _to_numpy(hit),
            "max_drawdown_bps": _to_numpy(max_dd),
            "n_trades": _to_numpy(n_trades).astype(int),
            "n_bars_active": _to_numpy(n_active).astype(int),
        }
        for col in range(k):
            row = dict(combos[col]) if combos is not None else {"col": col}
            row["entry_z"] = float(entry)
            row.update({m: v[col].item() for m, v in metrics.items()})
            rows.append(row)

    return pl.DataFrame(rows).sort("sharpe", descending=True, nulls_last=True)


def signal_matrix(
    x: pl.Series,
    y: pl.Series,
    beta_lbs: list[int],
    z_lbs: list[int],
) -> tuple[np.ndarray, list[dict]]:
    """Build the (T, K) OU z-score matrix for the standard model family:
    changes-based rolling OLS of y on x, residual rolled into level space,
    z-scored — one column per (beta_lb, z_lb) combo, aligned to len(y).

    Feed the result to fast_scan() to evaluate the whole grid at once.
    """
    from stats.ols import roll_lr_diff
    from stats.ou import roll_ou_zscore

    t_len = len(y)
    cols, combos = [], []
    null1 = pl.Series([None], dtype=pl.Float64)
    for beta_lb in beta_lbs:
        reg = roll_lr_diff(x, y, lookback=beta_lb)
        resid_roll = pl.concat([
            null1,
            reg["resid"].rolling_sum(beta_lb, min_samples=beta_lb),
        ])
        for z_lb in z_lbs:
            z = roll_ou_zscore(resid_roll, lookback=z_lb)
            cols.append(z.to_numpy().astype(float)[:t_len])
            combos.append({"beta_lb": beta_lb, "z_lb": z_lb})

    return np.column_stack(cols), combos


# ── metric store ─────────────────────────────────────────────────────────────

class MetricStore:
    """Parquet-backed history of backtest runs across strategies.

    Every log() appends rows (params + metrics + metadata) so grids stay
    comparable across strategies, data spans, and time. Query with
    leaderboard() and matrix().

    Default location is <repo root>/store/backtests.parquet regardless of
    cwd; override with the VIEWS_STORE_DIR env var or an explicit path.
    """

    def __init__(self, path: Union[str, Path, None] = None):
        if path is None:
            store_dir = os.getenv(
                "VIEWS_STORE_DIR", str(Path(__file__).resolve().parents[1] / "store")
            )
            path = Path(store_dir) / "backtests.parquet"
        self.path = Path(path)

    def log(
        self,
        strategy: str,
        results: pl.DataFrame,
        meta: Optional[dict] = None,
    ) -> pl.DataFrame:
        """Append a results frame (e.g. from sweep_strategy / fast_scan)."""
        run = results.with_columns(
            pl.lit(strategy).alias("strategy"),
            pl.lit(dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")).alias("run_ts"),
            *[pl.lit(v).alias(k) for k, v in (meta or {}).items()],
        )
        existing = self.load()
        combined = (
            pl.concat([existing, run], how="diagonal_relaxed")
            if not existing.is_empty()
            else run
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(self.path)
        return run

    def load(self) -> pl.DataFrame:
        if not self.path.exists():
            return pl.DataFrame()
        return pl.read_parquet(self.path)

    def leaderboard(
        self,
        metric: str = "sharpe",
        strategy: Optional[str] = None,
        top: int = 20,
    ) -> pl.DataFrame:
        """Best runs across the store (or one strategy), by any metric."""
        df = self.load()
        if df.is_empty() or metric not in df.columns:
            return pl.DataFrame()
        if strategy is not None:
            df = df.filter(pl.col("strategy") == strategy)
        return df.sort(metric, descending=True, nulls_last=True).head(top)

    def matrix(
        self,
        x: str,
        y: str,
        metric: str = "sharpe",
        strategy: Optional[str] = None,
        agg: str = "mean",
    ) -> pl.DataFrame:
        """Pivot a metric across two parameter dimensions (rows=y, cols=x).

        agg ("mean"/"max"/"min"/"median") collapses the other dimensions —
        "max" answers "best achievable sharpe at this (x, y)".
        """
        df = self.load()
        if strategy is not None and not df.is_empty():
            df = df.filter(pl.col("strategy") == strategy)
        if df.is_empty() or not {x, y, metric} <= set(df.columns):
            return pl.DataFrame()
        return (
            df.pivot(index=y, on=x, values=metric, aggregate_function=agg)
            .sort(y)
        )


# ── gates: conditional edge analysis ─────────────────────────────────────────

def gate_scan(
    signal: Union[pl.Series, np.ndarray],
    level: Union[pl.Series, np.ndarray],
    conditions: pl.DataFrame,
    entry_z: float = 1.5,
    horizon: int = 20,
    n_buckets: int = 3,
    min_n: int = 20,
) -> pl.DataFrame:
    """Which state variables separate good entries from bad ones?

    Takes every bar where |signal| >= entry_z as a hypothetical fade entry,
    measures forward PnL = sign(-z) × (level[t+h] − level[t]), then buckets
    each condition column into quantiles (computed on the entry sample) and
    reports per-bucket n / hit / avg_pnl / per-trade sharpe plus the lift vs
    the unconditional baseline (condition="(all)").

    A gate is promising when one bucket concentrates the hit rate and PnL
    with enough n — that bucket becomes an entry_filter_fn in SignalConfig.
    """
    z = np.asarray(signal.to_numpy() if isinstance(signal, pl.Series) else signal, dtype=float)
    lv = np.asarray(level.to_numpy() if isinstance(level, pl.Series) else level, dtype=float)

    fwd = np.full_like(lv, np.nan)
    fwd[:-horizon] = lv[horizon:] - lv[:-horizon]

    entry = np.isfinite(z) & np.isfinite(fwd) & (np.abs(z) >= entry_z)
    pnl = np.sign(-z[entry]) * fwd[entry]

    def _stats(p: np.ndarray) -> dict:
        if len(p) == 0:
            return {"n": 0, "hit": None, "avg_pnl": None, "trade_sharpe": None}
        std = p.std()
        return {
            "n": int(len(p)),
            "hit": float((p > 0).mean()),
            "avg_pnl": float(p.mean()),
            "trade_sharpe": float(p.mean() / std) if std > 0 else None,
        }

    base = _stats(pnl)
    rows = [{"condition": "(all)", "bucket": "all", **base, "hit_lift": 0.0, "pnl_lift": 0.0}]

    for cond in conditions.columns:
        c = np.asarray(conditions[cond].to_numpy(), dtype=float)[entry]
        ok = np.isfinite(c)
        if ok.sum() < min_n:
            continue

        # discrete conditions (few unique values) bucket by value;
        # continuous ones by quantile, with duplicate edges collapsed
        uniq = np.unique(c[ok])
        if len(uniq) <= n_buckets:
            buckets = [(f"={u:g}", ok & (c == u)) for u in uniq]
        else:
            edges = np.unique(np.nanquantile(c[ok], np.linspace(0, 1, n_buckets + 1)))
            n_b = len(edges) - 1
            buckets = []
            for b in range(n_b):
                lo, hi = edges[b], edges[b + 1]
                mask = ok & (c >= lo) & (c <= hi if b == n_b - 1 else c < hi)
                buckets.append((f"q{b + 1}/{n_b} [{lo:.3g}, {hi:.3g}]", mask))

        for label, in_b in buckets:
            if in_b.sum() < min_n:
                continue
            s = _stats(pnl[in_b])
            rows.append({
                "condition": cond,
                "bucket": label,
                **s,
                "hit_lift": round(s["hit"] - base["hit"], 4) if base["hit"] is not None else None,
                "pnl_lift": round(s["avg_pnl"] - base["avg_pnl"], 4) if base["avg_pnl"] is not None else None,
            })

    return pl.DataFrame(rows).sort("hit_lift", descending=True, nulls_last=True)
