"""Parameter lab â€” scalable sweeps, metric stores, fast scans, and gates.

The research funnel, cheapest to most exact:

1. `signal_matrix` + `fast_scan` â€” vectorized approximate backtest of an
   entire parameter grid at once. Pure array math (numpy semantics), so it
   runs unchanged on an NVIDIA GPU via cupy (`device="gpu"`). Use it to carve
   a huge grid down to a shortlist. Approximations: hysteresis-band exits
   (no stops/time-stops), next-bar execution, one position at a time.
2. `sweep_strategy` â€” the exact row-by-row Engine, one combo per task,
   parallel across CPU cores (ProcessPoolExecutor; a 32-core box runs 32
   combos at a time). Use it to verify the shortlist with real trade
   mechanics: stops, time-stops, costs, trade logs.
3. `MetricStore` â€” parquet-backed history of every run: leaderboards across
   strategies, metric matrices across any two parameter dimensions.
4. `gate_scan` â€” conditional edge analysis: bucket candidate state variables
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
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl

REGIME_GATE_BUCKETS = (
    ("low_10", "below", 0.10),
    ("high_90", "above", 0.90),
    ("mid_10_90", "between", 0.10, 0.90),
    ("tails_10_90", "outside", 0.10, 0.90),
    ("low_25", "below", 0.25),
    ("high_75", "above", 0.75),
    ("mid_25_75", "between", 0.25, 0.75),
    ("tails_25_75", "outside", 0.25, 0.75),
    ("mid_40_60", "between", 0.40, 0.60),
    ("tails_40_60", "outside", 0.40, 0.60),
    ("below_50", "below", 0.50),
    ("above_50", "above", 0.50),
)


# â”€â”€ parameter grids â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ exact sweep: Engine per combo, parallel across cores â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


def _param_row(params: dict) -> dict:
    """Result-row copy of params: structured values (e.g. gate tuples/dicts)
    become strings so combo rows stack into one DataFrame."""
    scalars = (bool, int, float, str, np.integer, np.floating)
    return {
        k: v if v is None or isinstance(v, scalars) else repr(v)
        for k, v in params.items()
    }


def _run_combo(params: dict) -> dict:
    from .engine import BacktestConfig, Engine

    try:
        pipeline = _WORKER_STATE["module"].make_pipeline(params)
        config = BacktestConfig(
            transaction_cost_bps=_WORKER_STATE["cost"],
            slippage_bps=_WORKER_STATE["slip"],
        )
        result = Engine(config).add_signal(pipeline).run(_WORKER_STATE["data"])
        return {**_param_row(params), **{k: float(v) for k, v in result.summary().items()}}
    except Exception as e:  # keep the sweep alive; surface the failure in the row
        return {**_param_row(params), "error": f"{type(e).__name__}: {e}"}


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


# â”€â”€ fast scan: vectorized approximate backtest (GPU-ready) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _get_xp(device: str = "cpu"):
    """Return numpy, or cupy when a working CUDA device is available."""
    device = (device or "cpu").lower()
    if device not in {"cpu", "gpu", "auto"}:
        raise ValueError(f"unknown device={device!r}; expected 'cpu', 'gpu', or 'auto'")
    if device in {"gpu", "auto"}:
        try:
            import cupy
        except ImportError:
            if device == "auto":
                return np
            warnings.warn(
                'CuPy is not installed â€” falling back to CPU. Install the '
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
            if device == "auto":
                return np
            warnings.warn(
                f"CuPy is installed but CUDA is unavailable ({exc}) â€” "
                "fast_scan is falling back to CPU. Install the CUDA components "
                'with: pip install "cupy-cuda12x[ctk]"'
            )
            return np
        if device_count < 1:
            if device == "auto":
                return np
            warnings.warn(
                "CuPy found no CUDA devices â€” fast_scan is falling back to CPU."
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
    elsewhere) into a position matrix. Vectorized â€” no scan loop."""
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


def _evaluate_events(events, dlv, cost_bps, periods_per_year, xp) -> dict[str, np.ndarray]:
    """Metrics for one (T, K) event matrix. All array math â€” CPU or GPU."""
    k = events.shape[1]
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

    return {
        "total_pnl_bps": _to_numpy(total),
        "sharpe": _to_numpy(sharpe),
        "hit_rate": _to_numpy(hit),
        "max_drawdown_bps": _to_numpy(max_dd),
        "n_trades": _to_numpy(n_trades).astype(int),
        "n_bars_active": _to_numpy(n_active).astype(int),
    }


def gate_variant_count(gate_buckets: Union[int, str, tuple, list]) -> int:
    """Number of gate variants generated per condition."""
    if isinstance(gate_buckets, (int, np.integer)):
        return int(gate_buckets)
    if isinstance(gate_buckets, str):
        if gate_buckets.lower() in {"regime", "regimes", "tails"}:
            return len(REGIME_GATE_BUCKETS)
        raise ValueError(f"unknown gate_buckets={gate_buckets!r}")
    return len(gate_buckets)


_GATE_KIND_ARITY = {"below": 1, "above": 1, "between": 2, "outside": 2}
_NAMED_GATE_BUCKETS = {
    spec[0]: (spec[1], tuple(float(q) for q in spec[2:]))
    for spec in REGIME_GATE_BUCKETS
}


def _lookup_gate_bucket(label: str) -> tuple[str, tuple[float, ...]]:
    if label not in _NAMED_GATE_BUCKETS:
        raise ValueError(
            f"unknown gate bucket {label!r}; known: {sorted(_NAMED_GATE_BUCKETS)}"
        )
    return _NAMED_GATE_BUCKETS[label]


def parse_gate(spec: Union[tuple, list, dict]) -> tuple[str, str, tuple[float, ...]]:
    """Normalize a gate spec into (condition, kind, quantiles).

    A gate names a condition column (see CONDITION_NAMES) and a quantile
    bucket — either a REGIME_GATE_BUCKETS label or an explicit kind
    ("below"/"above"/"between"/"outside") with its quantile(s). Accepted:

        ("r2", "high_75")                                # named regime bucket
        ("r2", "above", 0.75)                            # explicit kind + q
        ("beta_cv", "between", 0.25, 0.75)
        {"condition": "r2", "bucket": "high_75"}
        {"condition": "r2", "kind": "above", "q": 0.75}
        {"condition": "beta_cv", "kind": "between", "q": (0.25, 0.75)}
    """
    if isinstance(spec, dict):
        if "condition" not in spec:
            raise ValueError(f"gate dict needs a 'condition' key: {spec!r}")
        condition = spec["condition"]
        if "bucket" in spec:
            kind, qs = _lookup_gate_bucket(spec["bucket"])
        elif "kind" in spec and "q" in spec:
            kind = spec["kind"]
            q = spec["q"]
            qs = (
                tuple(float(x) for x in q)
                if isinstance(q, (list, tuple))
                else (float(q),)
            )
        else:
            raise ValueError(f"gate dict needs 'bucket' or 'kind'+'q': {spec!r}")
    elif isinstance(spec, (tuple, list)) and len(spec) >= 2:
        condition = spec[0]
        if len(spec) == 2:
            kind, qs = _lookup_gate_bucket(spec[1])
        else:
            kind = spec[1]
            qs = tuple(float(x) for x in spec[2:])
    else:
        raise ValueError(f"cannot parse gate spec: {spec!r}")

    if kind not in _GATE_KIND_ARITY:
        raise ValueError(
            f"unknown gate kind={kind!r}; expected one of {sorted(_GATE_KIND_ARITY)}"
        )
    if len(qs) != _GATE_KIND_ARITY[kind]:
        raise ValueError(
            f"gate kind {kind!r} takes {_GATE_KIND_ARITY[kind]} quantile(s), got {qs}"
        )
    return condition, kind, qs


def gate_allow_mask(
    values: Union[pl.Series, np.ndarray], spec: Union[tuple, list, dict]
) -> np.ndarray:
    """Boolean entry-allow mask for one condition series under a gate spec.

    Same bucket semantics as the fast_scan/predict_scan gates: quantile cuts
    are computed on the full sample of finite condition values, and
    non-finite bars are never allowed. Attach the mask to a strategy's
    signal frame and check it in entry_filter_fn so the exact Engine
    reproduces a gate shortlisted by the discovery scans.
    """
    _, kind, qs = parse_gate(spec)
    c = np.asarray(
        values.to_numpy() if isinstance(values, pl.Series) else values, dtype=float
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cuts = np.nanquantile(c, qs)
    finite = np.isfinite(c)
    if kind == "below":
        return finite & (c <= cuts[0])
    if kind == "above":
        return finite & (c >= cuts[0])
    if kind == "between":
        return finite & (c > cuts[0]) & (c < cuts[1])
    return finite & ((c <= cuts[0]) | (c >= cuts[1]))


def _gate_masks(
    gates: Optional[dict[str, np.ndarray]],
    t_len: int,
    k: int,
    gate_buckets: Union[int, str, tuple, list],
    xp,
) -> list[tuple[str, str, object]]:
    """Build entry-allow masks per gate condition and named bucket."""
    gate_masks: list[tuple[str, str, object]] = []
    if not gates:
        return gate_masks

    for name, cond in gates.items():
        c_np = _to_numpy(cond).astype(float)
        if c_np.ndim == 1:
            c_np = np.broadcast_to(c_np[:, None], (t_len, k))
        if c_np.shape != (t_len, k):
            raise ValueError(f"gate '{name}' has shape {c_np.shape}, expected {(t_len, k)}")
        c = xp.asarray(c_np)

        if isinstance(gate_buckets, (int, np.integer)):
            n_buckets = int(gate_buckets)
            qs = np.linspace(0.0, 1.0, n_buckets + 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                edges_np = np.nanquantile(c_np, qs, axis=0)
            edges = xp.asarray(edges_np)
            for b in range(n_buckets):
                lo, hi = edges[b][None, :], edges[b + 1][None, :]
                allow = (c >= lo) & ((c <= hi) if b == n_buckets - 1 else (c < hi))
                gate_masks.append((name, f"q{b + 1}/{n_buckets}", allow))
            continue

        if isinstance(gate_buckets, str):
            if gate_buckets.lower() not in {"regime", "regimes", "tails"}:
                raise ValueError(f"unknown gate_buckets={gate_buckets!r}")
            specs = REGIME_GATE_BUCKETS
        else:
            specs = gate_buckets

        quantiles = sorted({
            q
            for spec in specs
            for q in spec[2:]
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cuts_np = np.nanquantile(c_np, quantiles, axis=0)
        cuts = {
            q: xp.asarray(cuts_np[i])[None, :]
            for i, q in enumerate(quantiles)
        }

        finite = xp.isfinite(c)
        for spec in specs:
            label, kind, *qs = spec
            if kind == "below":
                allow = finite & (c <= cuts[qs[0]])
            elif kind == "above":
                allow = finite & (c >= cuts[qs[0]])
            elif kind == "between":
                allow = finite & (c > cuts[qs[0]]) & (c < cuts[qs[1]])
            elif kind == "outside":
                allow = finite & ((c <= cuts[qs[0]]) | (c >= cuts[qs[1]]))
            else:
                raise ValueError(f"unknown gate bucket kind={kind!r}")
            gate_masks.append((name, label, allow))

    return gate_masks


def fast_scan(
    z: np.ndarray,
    level: np.ndarray,
    entries: Union[list[float], tuple[float, ...]] = (1.5, 2.0, 2.5),
    exit_band: Union[float, list[float], tuple[float, ...]] = 0.25,
    cost_bps: float = 0.0,
    periods_per_year: int = 252,
    combos: Optional[list[dict]] = None,
    gates: Optional[dict[str, np.ndarray]] = None,
    gate_buckets: Union[int, str, tuple, list] = 3,
    device: str = "cpu",
    entry_col: str = "entry_z",
    exit_col: str = "exit_band_bps",
) -> pl.DataFrame:
    """Approximate threshold backtest of a whole signal matrix, vectorized.

    Rules per column of z (a fade signal): go long when z <= -entry, short
    when z >= +entry, flat when |z| <= exit_band; positions carry between
    events and execute on the NEXT bar; costs charged per unit of turnover.
    Pass a list of exit bands to scan exit thresholds too.

    Gating: every (combo, entry) is additionally evaluated once per
    (gate condition Ã— quantile bucket) â€” ENTRIES are allowed only on bars
    where the condition falls inside the bucket (exits always fire; NaN
    condition bars never open trades). Bucket edges are per-column quantiles
    of the condition, so labels are relative ("q2/3"); recover the actual
    cutoff levels for a shortlisted gate with gate_scan(). The baseline rows
    carry gate="(none)".

    Args:
        z: (T,) or (T, K) signal matrix â€” e.g. from signal_matrix().
        level: (T,) trade level in bps (composite the strategy trades).
        entries: entry thresholds to scan.
        exit_band: exit threshold(s) to scan.
        combos: optional K param dicts annotating z's columns.
        gates: optional {name: (T,) or (T, K) condition array} â€” e.g. the
               conditions from signal_matrix(return_conditions=True).
        gate_buckets: int for equal quantile slices, or "regime" for named
                      tails/middles/median regimes.
        device: "cpu" (numpy) or "gpu" (cupy â€” install the ``gpu`` extra).
        entry_col: output column name for the entry threshold.
        exit_col: output column name for the exit threshold.
    Returns one row per (combo, entry_z, exit_band, gate, gate_bucket):
    total_pnl_bps, sharpe, hit_rate, max_drawdown_bps, n_trades,
    n_bars_active. Best sharpe first. Total rows =
    K Ã— len(entries) Ã— len(exit_bands) Ã— (1 + n_gates Ã— gate_buckets).
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

    # Pre-compute entry-allow masks per (gate, bucket): per-column quantile
    # buckets of the condition. Edges computed on CPU once per gate (cupy's
    # nanquantile support varies by version); masks are bool (T, K) â€” cheap.
    gate_masks = _gate_masks(gates, t_len, k, gate_buckets, xp)

    combo_frame = (
        pl.DataFrame(combos) if combos is not None
        else pl.DataFrame({"col": list(range(k))})
    )
    if isinstance(exit_band, (int, float, np.integer, np.floating)):
        exit_bands = [float(exit_band)]
    else:
        exit_bands = [float(x) for x in exit_band]

    def _emit(
        entry: float,
        exit_band_value: float,
        gate: str,
        bucket: str,
        metrics: dict,
    ) -> pl.DataFrame:
        return combo_frame.with_columns(
            pl.lit(float(entry)).alias(entry_col),
            pl.lit(float(exit_band_value)).alias(exit_col),
            pl.lit(gate).alias("gate"),
            pl.lit(bucket).alias("gate_bucket"),
            *[pl.Series(m, v) for m, v in metrics.items()],
        )

    blocks: list[pl.DataFrame] = []
    for exit_band_value in exit_bands:
        for entry in entries:
            events = xp.where(
                z <= -entry, 1.0,
                xp.where(z >= entry, -1.0,
                         xp.where(xp.abs(z) <= exit_band_value, 0.0, xp.nan)),
            )
            blocks.append(_emit(
                entry, exit_band_value, "(none)", "all",
                _evaluate_events(events, dlv, cost_bps, periods_per_year, xp),
            ))

            is_entry = xp.abs(events) == 1.0
            for name, bucket, allow in gate_masks:
                gated = xp.where(is_entry & ~allow, xp.nan, events)
                blocks.append(_emit(
                    entry, exit_band_value, name, bucket,
                    _evaluate_events(gated, dlv, cost_bps, periods_per_year, xp),
                ))

    return (
        pl.concat(blocks)
        # NaN (zero-trade combos) would sort above real values â€” make them null
        .with_columns(pl.col("sharpe", "hit_rate").fill_nan(None))
        .sort("sharpe", descending=True, nulls_last=True)
    )


def predict_scan(
    z: np.ndarray,
    level: np.ndarray,
    entries: Union[list[float], tuple[float, ...]] = (1.5, 2.0, 2.5),
    horizons: Union[list[int], tuple[int, ...]] = (5, 20, 60),
    periods_per_year: int = 252,
    combos: Optional[list[dict]] = None,
    gates: Optional[dict[str, np.ndarray]] = None,
    gate_buckets: Union[int, str, tuple, list] = 3,
    device: str = "cpu",
    entry_col: str = "entry_threshold",
) -> pl.DataFrame:
    """Forward-horizon predictability scan for a signal matrix.

    For every signal column, entry threshold, horizon, and optional gate
    bucket, evaluate whether larger positive signals predict lower future
    levels, and larger negative signals predict higher future levels.

    This is not a trade backtest: no exits, stops, holding state, or costs.
    It is the GPU-friendly discovery layer for signal/threshold/horizon/regime
    predictability. Metrics are event-bar diagnostics, not trade metrics:
    IC, directional hit rate, observation count, and fire rate.
    """
    xp = _get_xp(device)

    z = xp.asarray(z, dtype=xp.float64)
    if z.ndim == 1:
        z = z[:, None]
    t_len, k = z.shape
    if combos is not None and len(combos) != k:
        raise ValueError(f"combos has {len(combos)} entries but z has {k} columns")

    lv = xp.asarray(level, dtype=xp.float64)

    gate_masks = _gate_masks(gates, t_len, k, gate_buckets, xp)
    combo_frame = (
        pl.DataFrame(combos) if combos is not None
        else pl.DataFrame({"col": list(range(k))})
    )

    def _metrics(mask, eligible, fwd) -> dict[str, np.ndarray]:
        directional_move = -xp.sign(z) * fwd
        n = mask.sum(axis=0)
        eligible_n = eligible.sum(axis=0)
        n_float = n.astype(xp.float64)

        with np.errstate(invalid="ignore", divide="ignore"):
            hit = xp.where(n > 0, ((directional_move > 0) & mask).sum(axis=0) / n_float, xp.nan)
            fire_rate = xp.where(eligible_n > 0, n / eligible_n.astype(xp.float64), xp.nan)

        y = -fwd
        x_masked = xp.where(mask, z, 0.0)
        y_masked = xp.where(mask, y, 0.0)
        sx = x_masked.sum(axis=0)
        sy = y_masked.sum(axis=0)
        sxy = xp.where(mask, z * y, 0.0).sum(axis=0)
        sx2 = xp.where(mask, z * z, 0.0).sum(axis=0)
        sy2 = xp.where(mask, y * y, 0.0).sum(axis=0)
        cov_num = n_float * sxy - sx * sy
        den = xp.sqrt((n_float * sx2 - sx * sx) * (n_float * sy2 - sy * sy))
        with np.errstate(invalid="ignore", divide="ignore"):
            ic = xp.where((n > 1) & (den > 0), cov_num / den, xp.nan)

        return {
            "n_obs": _to_numpy(n).astype(int),
            "ic": _to_numpy(ic),
            "hit_rate": _to_numpy(hit),
            "fire_rate": _to_numpy(fire_rate),
        }

    def _emit(entry: float, horizon: int, gate: str, bucket: str, metrics: dict) -> pl.DataFrame:
        return combo_frame.with_columns(
            pl.lit(float(entry)).alias(entry_col),
            pl.lit(int(horizon)).alias("horizon"),
            pl.lit(gate).alias("gate"),
            pl.lit(bucket).alias("gate_bucket"),
            *[pl.Series(m, v) for m, v in metrics.items()],
        )

    blocks: list[pl.DataFrame] = []
    for horizon in horizons:
        fwd = xp.full((t_len, 1), xp.nan, dtype=xp.float64)
        if horizon < t_len:
            fwd[:-horizon, 0] = lv[horizon:] - lv[:-horizon]
        valid_fwd = xp.isfinite(fwd)

        for entry in entries:
            eligible = xp.isfinite(z) & valid_fwd
            base = eligible & (xp.abs(z) >= entry)
            blocks.append(_emit(entry, horizon, "(none)", "all", _metrics(base, eligible, fwd)))
            for name, bucket, allow in gate_masks:
                gated_eligible = eligible & allow
                gated = gated_eligible & (xp.abs(z) >= entry)
                blocks.append(_emit(entry, horizon, name, bucket, _metrics(gated, gated_eligible, fwd)))

    return (
        pl.concat(blocks)
        .with_columns(pl.col("ic", "hit_rate", "fire_rate").fill_nan(None))
        .sort("ic", descending=True, nulls_last=True)
    )


def add_predict_lift(
    results: pl.DataFrame,
    keys: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Join gated predict rows to their ungated baseline and compute lifts."""
    metric_cols = {
        "ic", "hit_rate", "fire_rate", "n_obs", "gate", "gate_bucket",
    }
    if keys is None:
        keys = [c for c in results.columns if c not in metric_cols]

    base = (
        results.filter(pl.col("gate") == "(none)")
        .select(
            *keys,
            pl.col("ic").alias("base_ic"),
            pl.col("hit_rate").alias("base_hit_rate"),
            pl.col("fire_rate").alias("base_fire_rate"),
            pl.col("n_obs").alias("base_n_obs"),
        )
    )
    return (
        results.join(base, on=keys, how="left", nulls_equal=True)
        .with_columns(
            (pl.col("ic") - pl.col("base_ic")).alias("ic_lift"),
            (pl.col("hit_rate") - pl.col("base_hit_rate")).alias("hit_lift"),
            (pl.col("fire_rate") - pl.col("base_fire_rate")).alias("fire_rate_lift"),
        )
    )


def add_gate_lift(
    results: pl.DataFrame,
    keys: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Join each gated row to its ungated baseline and compute the lift.

    keys defaults to every param column (everything except gate/gate_bucket
    and the metric columns). Adds base_sharpe / base_hit_rate / base_n_trades
    plus sharpe_lift and hit_lift â€” the gate's value-add over the same combo
    with no gate. Sort by sharpe_lift (with an n_trades floor) to find gates
    that improve an already-good strategy rather than just riding it.
    """
    metric_cols = {
        "total_pnl_bps", "sharpe", "hit_rate", "max_drawdown_bps",
        "n_trades", "n_bars_active", "gate", "gate_bucket",
    }
    if keys is None:
        keys = [c for c in results.columns if c not in metric_cols]

    base = (
        results.filter(pl.col("gate") == "(none)")
        .select(
            *keys,
            pl.col("sharpe").alias("base_sharpe"),
            pl.col("hit_rate").alias("base_hit_rate"),
            pl.col("n_trades").alias("base_n_trades"),
        )
    )
    return (
        results.join(base, on=keys, how="left", nulls_equal=True)
        .with_columns(
            (pl.col("sharpe") - pl.col("base_sharpe")).alias("sharpe_lift"),
            (pl.col("hit_rate") - pl.col("base_hit_rate")).alias("hit_lift"),
        )
    )


# Condition matrices built alongside the signal â€” the gate candidates.
CONDITION_NAMES = (
    "r2",
    "beta_cv",
    "beta",
    "beta_vol20",
    "beta_mom10",
    "r2_vol20",
    "r2_mom10",
    "resid_phi",
    "resid_half_life",
    "resid_vol20",
    "resid_mom10",
)


def signal_matrix(
    x: pl.Series,
    y: pl.Series,
    beta_lbs: list[int],
    z_lbs: list[int],
    return_conditions: bool = False,
    signal_kind: str = "ou_zscore",
    lookback_name: str = "z_lb",
):
    """Build the (T, K) OU z-score matrix for the standard model family:
    changes-based rolling OLS of y on x, residual rolled into level space,
    z-scored â€” one column per (beta_lb, z_lb) combo, aligned to len(y).

    With return_conditions=True also returns {name: (T, K) array} of gate
    candidates aligned column-for-column with z (they vary by beta_lb):
    model quality/stability (r2, beta_cv, beta), beta/r2 volatility and
    momentum, residual Dickey-Fuller-style AR(1) state, and residual
    volatility/momentum. Volatility is rolling 20-bar standard deviation of
    daily changes; momentum is 10-bar change. Feed z + conditions to
    fast_scan()/predict_scan() to evaluate gated grids.

    Returns (z, combos) or (z, combos, conditions).
    """
    from stats.diagnostics import beta_cv as _beta_cv
    from stats.ols import roll_lr_diff
    from stats.ou import roll_ou_features, roll_ou_zscore

    valid_signal_kinds = {"ou_zscore", "zscore", "z", "residual", "resid"}
    if signal_kind not in valid_signal_kinds:
        raise ValueError(f"unknown signal_kind={signal_kind!r}")

    t_len = len(y)
    cols, combos = [], []
    cond_cols: dict[str, list[np.ndarray]] = {name: [] for name in CONDITION_NAMES}
    null1 = pl.Series([None], dtype=pl.Float64)

    for beta_lb in beta_lbs:
        reg = roll_lr_diff(x, y, lookback=beta_lb)
        resid_roll = pl.concat([
            null1,
            reg["resid"].rolling_sum(beta_lb, min_samples=beta_lb),
        ])
        if return_conditions:
            beta = pl.concat([null1, reg["beta"]])
            r2 = pl.concat([null1, reg["r2"]])
            ou_state = roll_ou_features(resid_roll, lookback=beta_lb)
            conds = {
                "r2": r2,
                "beta_cv": _beta_cv(beta, lookback=beta_lb),
                "beta": beta,
                "beta_vol20": beta.diff().rolling_std(20),
                "beta_mom10": beta.diff(10),
                "r2_vol20": r2.diff().rolling_std(20),
                "r2_mom10": r2.diff(10),
                "resid_phi": ou_state["ou_rho"] - 1.0,
                "resid_half_life": ou_state["half_life"],
                "resid_vol20": resid_roll.diff().rolling_std(20),
                "resid_mom10": resid_roll.diff(10),
            }
        for z_lb in z_lbs:
            if signal_kind in {"ou_zscore", "zscore", "z"}:
                signal = roll_ou_zscore(resid_roll, lookback=z_lb)
            else:
                signal = resid_roll
            cols.append(signal.to_numpy().astype(float)[:t_len])
            combos.append({"beta_lb": beta_lb, lookback_name: z_lb})
            if return_conditions:
                for name in CONDITION_NAMES:
                    cond_cols[name].append(
                        conds[name].to_numpy().astype(float)[:t_len]
                    )

    z_mat = np.column_stack(cols)
    if not return_conditions:
        return z_mat, combos
    conditions = {name: np.column_stack(v) for name, v in cond_cols.items()}
    return z_mat, combos, conditions


# â”€â”€ metric store â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        tmp_path = self.path.with_name(f".{self.path.stem}.{uuid.uuid4().hex}.tmp.parquet")
        try:
            combined.write_parquet(tmp_path)
            os.replace(tmp_path, self.path)
        except OSError as exc:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            fallback = self.path.with_name(
                f"{self.path.stem}."
                f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}."
                f"{uuid.uuid4().hex[:8]}.parquet"
            )
            run.write_parquet(fallback)
            print(
                f"warning: could not update {self.path} ({exc}); "
                f"wrote this run to {fallback}"
            )
        return run

    def load(self) -> pl.DataFrame:
        if not self.path.exists():
            return pl.DataFrame()
        return pl.read_parquet(self.path, memory_map=False)

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

        agg ("mean"/"max"/"min"/"median") collapses the other dimensions â€”
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


# â”€â”€ gates: conditional edge analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    measures forward PnL = sign(-z) Ã— (level[t+h] âˆ’ level[t]), then buckets
    each condition column into quantiles (computed on the entry sample) and
    reports per-bucket n / hit / avg_pnl / per-trade sharpe plus the lift vs
    the unconditional baseline (condition="(all)").

    A gate is promising when one bucket concentrates the hit rate and PnL
    with enough n â€” that bucket becomes an entry_filter_fn in SignalConfig.
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
