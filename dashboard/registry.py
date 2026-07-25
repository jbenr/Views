"""Registry of promoted ("live") signals tracked on the dashboard.

Promoting a signal snapshots one row from its sweep_results_file (the
frozen entry/exit/gate config) into store/live_signals.parquet. Any book
module that exposes a `STRATEGY = backtest.strategy.Strategy(...)` object
can be promoted -- this isn't curve-specific. Same atomic-write convention
as backtest.lab.MetricStore.

    python -m dashboard.registry --promote book.curve.tens_10s30s
    python -m dashboard.registry --promote book.curve.tens_10s30s --rank 2
    python -m dashboard.registry --promote book.curve.twos_10s30s --defaults
    python -m dashboard.registry --promote book.curve.twos_10s30s --variant ou430_e09
    python -m dashboard.registry --list
    python -m dashboard.registry --remove book.curve.twos_10s30s::ou430_e09
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import os
import uuid
from pathlib import Path

import polars as pl

import utils

from .params import PARAM_COLS


def _store_dir() -> Path:
    return Path(
        os.getenv("VIEWS_STORE_DIR", str(Path(__file__).resolve().parents[1] / "store"))
    )


def load_strategy(module: str):
    """Import `module` and return its STRATEGY object."""
    mod = importlib.import_module(module)
    if not hasattr(mod, "STRATEGY"):
        raise SystemExit(
            f"{module} has no STRATEGY object -- expected a backtest.strategy.Strategy instance"
        )
    return mod.STRATEGY


class LiveRegistry:
    """Parquet-backed promoted signals, including named module variants."""

    def __init__(self, path: Path | None = None):
        self.path = path or (_store_dir() / "live_signals.parquet")

    def load(self) -> pl.DataFrame:
        if not self.path.exists():
            return pl.DataFrame()
        frame = pl.read_parquet(self.path, memory_map=False)
        # Backward-compatible migration: legacy rows were uniquely identified
        # by module. Keep that as the base signal_id.
        if "signal_id" not in frame.columns:
            frame = frame.with_columns(pl.col("module").alias("signal_id"))
        if "variant" not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("variant"))
        if "variant_label" not in frame.columns:
            frame = frame.with_columns(
                pl.lit(None, dtype=pl.String).alias("variant_label")
            )
        if "rationale" not in frame.columns:
            frame = frame.with_columns(
                pl.lit(None, dtype=pl.String).alias("rationale")
            )
        return frame

    def _write(self, df: pl.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.stem}.{uuid.uuid4().hex}.tmp.parquet")
        df.write_parquet(tmp_path)
        os.replace(tmp_path, self.path)

    def _replace(self, entry: dict) -> dict:
        existing = self.load()
        if not existing.is_empty():
            existing = existing.filter(pl.col("signal_id") != entry["signal_id"])
        new_row = pl.DataFrame([entry])
        combined = (
            pl.concat([existing, new_row], how="diagonal_relaxed")
            if not existing.is_empty()
            else new_row
        )
        self._write(combined)
        return entry

    @staticmethod
    def _signal_id(module: str, variant: str | None = None) -> str:
        return module if variant is None else f"{module}::{variant}"

    def promote(self, module: str, rank: int = 0) -> dict:
        """Snapshot row `rank` of <module>'s sweep_results_file (0 = best by
        sharpe, the file's own sort order) as the live config for `module`.
        Replaces any prior promotion of the same module."""
        strategy = load_strategy(module)
        if not strategy.sweep_results_file.exists():
            raise SystemExit(
                f"{module}: no sweep results at {strategy.sweep_results_file} -- run --sweep first"
            )
        results = pl.read_parquet(strategy.sweep_results_file)
        if not (0 <= rank < len(results)):
            raise SystemExit(f"{module}: rank {rank} out of range (0..{len(results) - 1})")
        row = results.row(rank, named=True)

        entry = {
            "signal_id": self._signal_id(module),
            "module": module,
            "name": strategy.name,
            "family": strategy.family,
            "target": strategy.target,
            "feature": strategy.feature,
            "variant": None,
            "variant_label": None,
            "rationale": None,
            "rank": rank,
            "selection_source": f"sweep rank {rank}",
            "promoted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            **{c: row.get(c) for c in PARAM_COLS},
            "sharpe": row.get("sharpe"),
            "n_trades": row.get("n_trades"),
            "hit_rate": row.get("hit_rate"),
            "max_drawdown_bps": row.get("max_drawdown_bps"),
        }
        return self._replace(entry)

    def _promote_params(
        self,
        module: str,
        params: dict,
        *,
        variant: str | None,
        variant_label: str | None,
        rationale: str | None,
        selection_source: str,
    ) -> dict:
        """Exact-engine promotion for an explicit frozen parameter set."""
        from backtest.engine import BacktestConfig, Engine, compute_metrics

        strategy = load_strategy(module)
        params = strategy._params(params)
        data = strategy.model_frame(strategy.load_data())
        result = (
            Engine(BacktestConfig(transaction_cost_bps=strategy.transaction_cost_bps))
            .add_signal(strategy.make_pipeline(params))
            .run(data)
        )
        metrics = compute_metrics(result)
        frozen_params = {
            c: (
                repr(params[c])
                if c == "gate" and params.get(c) is not None
                else params.get(c)
            )
            for c in PARAM_COLS
        }
        entry = {
            "signal_id": self._signal_id(module, variant),
            "module": module,
            "name": strategy.name,
            "family": strategy.family,
            "target": strategy.target,
            "feature": strategy.feature,
            "variant": variant,
            "variant_label": variant_label,
            "rationale": rationale,
            "rank": None,
            "selection_source": selection_source,
            "promoted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            **frozen_params,
            "sharpe": metrics["sharpe"],
            "n_trades": metrics["n_trades"],
            "hit_rate": metrics["hit_rate"],
            "max_drawdown_bps": metrics["max_drawdown_bps"],
        }
        return self._replace(entry)

    def promote_defaults(self, module: str) -> dict:
        """Promote a strategy's explicitly curated ``default_params``.

        Unlike :meth:`promote`, this does not imply that the top raw sweep row
        is trusted. It runs the exact engine on current data and snapshots the
        resulting metrics alongside the module's frozen defaults.
        """
        strategy = load_strategy(module)
        return self._promote_params(
            module,
            strategy.default_params,
            variant=None,
            variant_label=None,
            rationale=None,
            selection_source="curated module defaults",
        )

    def promote_variant(self, module: str, variant: str) -> dict:
        """Promote one named config from ``module.DASHBOARD_VARIANTS``."""
        mod = importlib.import_module(module)
        variants = getattr(mod, "DASHBOARD_VARIANTS", {})
        if variant not in variants:
            known = ", ".join(sorted(variants)) or "(none)"
            raise SystemExit(
                f"{module}: unknown dashboard variant {variant!r}; known: {known}"
            )
        spec = variants[variant]
        return self._promote_params(
            module,
            spec["params"],
            variant=variant,
            variant_label=spec.get("label", variant),
            rationale=spec.get("rationale"),
            selection_source=f"curated variant {variant}",
        )

    def remove(self, signal_id: str) -> bool:
        existing = self.load()
        if existing.is_empty() or signal_id not in existing["signal_id"].to_list():
            return False
        self._write(existing.filter(pl.col("signal_id") != signal_id))
        return True

    def list(self) -> pl.DataFrame:
        return self.load()

    def get(self, signal_id: str) -> dict | None:
        df = self.load()
        if df.is_empty():
            return None
        match = df.filter(pl.col("signal_id") == signal_id)
        if match.is_empty():
            # Legacy convenience: a module name resolves its base promotion.
            module_rows = df.filter(pl.col("module") == signal_id)
            base = module_rows.filter(pl.col("signal_id") == pl.col("module"))
            match = base if not base.is_empty() else module_rows.head(1)
        if match.is_empty():
            return None
        return match.row(0, named=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", metavar="MODULE")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--defaults",
        action="store_true",
        help="promote the module's curated default_params instead of a sweep rank",
    )
    parser.add_argument(
        "--variant",
        help="promote a named module.DASHBOARD_VARIANTS configuration",
    )
    parser.add_argument("--remove", metavar="SIGNAL_ID")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    reg = LiveRegistry()
    if args.promote:
        if args.variant:
            entry = reg.promote_variant(args.promote, args.variant)
        elif args.defaults:
            entry = reg.promote_defaults(args.promote)
        else:
            entry = reg.promote(args.promote, rank=args.rank)
        source = entry.get("selection_source", f"rank {args.rank}")
        print(
            f"promoted {entry['signal_id']} ({source}): "
            f"sharpe={entry['sharpe']:.2f}  n_trades={entry['n_trades']:.0f}  "
            f"hit_rate={entry['hit_rate']:.0%}"
        )
    elif args.remove:
        ok = reg.remove(args.remove)
        print(f"{'removed' if ok else 'not found:'} {args.remove}")
    elif args.list:
        df = reg.list()
        if df.is_empty():
            print("no live signals promoted yet")
        else:
            utils.pdf(
                df.select(
                    "signal_id", "module", "variant_label", "family", "sharpe", "n_trades",
                    "hit_rate", "promoted_at",
                )
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
