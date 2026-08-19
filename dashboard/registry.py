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
import textwrap
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from tabulate import tabulate

from .params import (
    PARAM_COLS,
    entry_short,
    exit_short,
    gate_short,
    input_label,
    model_label,
    params_from_row,
)


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


def _metric(value, fmt: str, suffix: str = "") -> str:
    if value is None:
        return "—"
    value = float(value)
    return "—" if value != value else format(value, fmt) + suffix


def _config_row(row: dict) -> dict:
    """What the signal does, plus the two facts you act on: the id to pass to
    --remove, and when it went live."""
    params = params_from_row(row)
    promoted = dt.datetime.fromisoformat(row["promoted_at"]).astimezone(
        ZoneInfo("America/New_York")
    )
    return {
        "target": row["target"],
        "input": input_label(row),
        "model": model_label(params),
        "entry": entry_short(params),
        "exit": exit_short(params),
        "stop": _metric(params.get("stop_loss_bps"), "g", "bps"),
        "gate": gate_short(params),
        "sharpe": _metric(row.get("sharpe"), ".2f"),
        "trades": _metric(row.get("n_trades"), ".0f"),
        "hit": _metric((row.get("hit_rate") or 0) * 100, ".1f", "%"),
        "max dd": _metric(row.get("max_drawdown_bps"), "+.1f"),
        "live since": promoted.strftime("%b %d"),
        # full id, not a shortened one: this is what --remove takes, so it has
        # to be pasteable straight out of the table
        "signal_id": row["signal_id"],
    }


def describe(frame: pl.DataFrame) -> str:
    """Promoted signals as one table per strategy family, plus provenance.

    Grouped by family because that is what makes two rows comparable: a curve
    signal and a vol signal share no metric worth reading side by side, while
    everything inside a family does. Target leads each row and orders the
    table, so signals on the same traded series sit together -- they are one
    risk position expressed several ways, and that has to be visible.

    Every field the per-signal blocks carried is here except the derived name
    and the sweep rank it came from; both stay in live_signals.parquet, and a
    rationale is printed underneath when one was recorded.
    """
    rows = list(frame.sort("family", "target", "signal_id").iter_rows(named=True))
    families = sorted({r["family"] for r in rows})
    targets = sorted({r["target"] for r in rows})
    plural = "s" if len(rows) != 1 else ""
    lines = [
        f"{len(rows)} live signal{plural} · {len(targets)} "
        f"target{'s' if len(targets) != 1 else ''} · "
        f"{len(families)} famil{'ies' if len(families) != 1 else 'y'}"
    ]

    for family in families:
        group = [r for r in rows if r["family"] == family]
        group_targets = sorted({r["target"] for r in group})
        table = [_config_row(r) for r in group]
        # target ascending, then best first within a target
        table.sort(key=lambda t: (
            t["target"],
            -(float(t["sharpe"]) if t["sharpe"] != "—" else -9e9),
        ))
        lines.append(
            f"\n{family.upper()} · {len(group)} signal"
            f"{'s' if len(group) != 1 else ''} · "
            f"{len(group_targets)} target{'s' if len(group_targets) != 1 else ''}"
        )
        lines.append(
            textwrap.indent(
                # disable_numparse: these cells are already formatted to a
                # fixed precision, and tabulate would re-parse and re-render
                # them (0.70 -> 0.7), breaking the column's decimal alignment
                tabulate(table, headers="keys", tablefmt="simple",
                         disable_numparse=True,
                         colalign=("left",) * 7 + ("right",) * 4 + ("left",) * 2),
                "  ",
            )
        )

    rationales = [r for r in rows if r.get("rationale")]
    if rationales:
        lines.append("")
        lines += [f"  {r['signal_id']}: {r['rationale']}" for r in rationales]
    return "\n".join(lines)


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
            print(describe(df))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
