"""Treasury futures calendar-roll mean reversion.

Question: is the front Treasury futures contract rich or cheap versus the
deferred contract?

The signal is the OU z-score of the front-minus-deferred generic futures roll:

    roll = front generic price - deferred generic price
    z high -> front rich -> SELL the roll (short front, long deferred)
    z low  -> front cheap -> BUY the roll (long front, short deferred)

The traded level is back-adjusted across generic contract-pair changes, so a
position held through a generic roll does not book the mechanical roll gap.

    python -m book.basis.fut_roll
    python -m book.basis.fut_roll --roots TY,US
    python -m book.basis.fut_roll --diagnose

main() returns a dict of state for interactive chaining: state = main().
"""

from __future__ import annotations

import argparse

import polars as pl

import utils
from backtest import (
    BacktestConfig,
    Engine,
    SignalConfig,
    SignalPipeline,
    TradeDef,
    print_summary,
    trade_log,
)
from stats import horizon_backtest, roll_ou_features
from utils.basis import DELIVERABLE_ROOTS, futures_roll, futures_roll_panel


STRATEGY_FAMILY = "basis"
SIGNAL_NAME = "fut_roll"
START = "2010-01-01"

# Keep the final delivery week out of the signal. Generic prices can be noisy
# there and the roll series changes identity shortly afterward anyway.
MIN_DAYS_TO_FRONT_DELIVERY = 5

DEFAULT_PARAMS = {
    "ou_lb": 252,
    "entry_z": 1.75,
    "exit_z": 0.25,
    "stop_loss_px": 2.0,
    "time_stop_bars": 40,
}

# Units note: engine "bps" are futures price points here. PnL is therefore not
# directly comparable to yield-bps strategies.
TRANSACTION_COST_PX = 0.015625


def load_data(
    roots: list[str],
    start: str = START,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Futures-roll long frame and its wide trading panel."""
    roll = futures_roll(roots, start=start, min_days=MIN_DAYS_TO_FRONT_DELIVERY)
    return roll, futures_roll_panel(roll)


def compute(panel: pl.DataFrame, root: str, params: dict | None = None) -> pl.DataFrame:
    """Signal frame for one root: OU z-score of the futures roll."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    ou = roll_ou_features(panel[f"{root}_roll"], lookback=p["ou_lb"])
    return pl.DataFrame(
        {
            "signal": ou["ou_z"],
            "roll": panel[f"{root}_roll"],
            "ou_mean": ou["ou_mean"],
            "ou_sigma": ou["ou_sigma"],
            "half_life": ou["half_life"],
        }
    )


def make_pipeline(root: str, params: dict | None = None) -> SignalPipeline:
    """One root's calendar-roll pipeline."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    return SignalPipeline(
        name=f"{SIGNAL_NAME}_{root}",
        trade_def=TradeDef.outright(root, f"{root}_level"),
        compute_fn=lambda panel: compute(panel, root, params=p),
        config=SignalConfig(
            entry_long=-p["entry_z"],
            entry_short=p["entry_z"],
            exit_long=p["exit_z"],
            exit_short=-p["exit_z"],
            stop_loss_bps=p["stop_loss_px"],
            time_stop_bars=p["time_stop_bars"],
        ),
    )


def coverage(roll: pl.DataFrame) -> pl.DataFrame:
    """Per-root sample, contract-pair turnover, and roll distribution."""
    return roll.group_by("root").agg(
        pl.len().alias("n_days"),
        pl.col("ts").min().alias("first"),
        pl.col("ts").max().alias("last"),
        pl.concat_str(["front_contract", "deferred_contract"], separator="/")
        .n_unique()
        .alias("n_pairs"),
        pl.col("roll").median().round(4).alias("med_roll"),
        pl.col("roll").std().round(4).alias("sd_roll"),
        pl.col("roll").quantile(0.05).round(4).alias("p05_roll"),
        pl.col("roll").quantile(0.95).round(4).alias("p95_roll"),
    ).sort("root")


def reversion(roll: pl.DataFrame) -> pl.DataFrame:
    """Does fading the futures roll pay in the raw roll units?"""
    rows = []
    for root in roll["root"].unique(maintain_order=True):
        series = roll.filter(pl.col("root") == root)["roll"]
        diag = horizon_backtest(series, horizons=(5, 10, 20, 40))
        rows.append(diag.insert_column(0, pl.Series("root", [root] * len(diag))))
    return pl.concat(rows)


def latest(roll: pl.DataFrame, panel: pl.DataFrame, params: dict) -> pl.DataFrame:
    """Latest read per root."""
    rows = []
    for root in roll["root"].unique(maintain_order=True):
        sig = compute(panel, root, params=params)
        last_row = roll.filter(pl.col("root") == root).row(-1, named=True)
        z = sig["signal"][-1]
        if z is None or z != z:
            action = "warmup"
        elif z >= params["entry_z"]:
            action = "SELL roll"
        elif z <= -params["entry_z"]:
            action = "BUY roll"
        else:
            action = "flat"
        rows.append(
            {
                "root": root,
                "ts": last_row["ts"],
                "front": last_row["front_contract"],
                "deferred": last_row["deferred_contract"],
                "n_days": last_row["n_days"],
                "roll": round(last_row["roll"], 4),
                "ou_z": None if z is None or z != z else round(float(z), 2),
                "action": action,
            }
        )
    return pl.DataFrame(rows)


def main(
    roots: list[str] | None = None,
    params: dict | None = None,
) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    roots = list(roots or DELIVERABLE_ROOTS)

    roll, panel = load_data(roots)
    print(f"roots={roots}  rows={len(roll)}  panel={panel.shape}  params={p}")

    cover = coverage(roll)
    print("\ncoverage / roll distribution (front - deferred, price points):")
    utils.pdf(cover)

    revert = reversion(roll)
    print("\nfutures roll horizon backtest (IC / hit / Sharpe of fading it):")
    utils.pdf(revert)

    engine = Engine(
        BacktestConfig(
            transaction_cost_bps=TRANSACTION_COST_PX,
            max_total_positions=len(roots),
        )
    )
    for root in roots:
        engine.add_signal(make_pipeline(root, p))
    result = engine.run(panel)
    print_summary(result)

    trades = trade_log(result.closed_trades)
    if not trades.is_empty():
        print("\nper-root trade summary:")
        utils.pdf(
            trades.group_by("trade_name")
            .agg(
                pl.len().alias("n_trades"),
                pl.col("pnl_bps").sum().round(3).alias("total_px"),
                pl.col("pnl_bps").median().round(4).alias("median_px"),
                (pl.col("pnl_bps") > 0).mean().round(3).alias("hit_rate"),
                pl.col("bars_held").median().alias("median_hold"),
            )
            .sort("trade_name")
        )

    live = latest(roll, panel, p)
    print("\nlatest signal:")
    utils.pdf(live)

    return {
        "roll": roll,
        "panel": panel,
        "coverage": cover,
        "reversion": revert,
        "result": result,
        "trades": trades,
        "latest": live,
    }


def diagnose(roots: list[str] | None = None) -> dict:
    """Futures-roll construction only; no trading."""
    roots = list(roots or DELIVERABLE_ROOTS)
    roll, panel = load_data(roots)
    print(f"roots={roots}  rows={len(roll)}  panel={panel.shape}")
    print("\ncoverage / roll distribution:")
    utils.pdf(coverage(roll))
    print("\nmedian roll by root and year:")
    utils.pdf(
        roll.with_columns(pl.col("ts").dt.year().alias("year"))
        .group_by("root", "year")
        .agg(pl.col("roll").median().round(4).alias("roll"))
        .pivot(index="year", on="root", values="roll")
        .sort("year")
    )
    return {"roll": roll, "panel": panel}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        help=f"comma-separated contract roots (default: {','.join(DELIVERABLE_ROOTS)})",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="roll construction diagnostics only, no backtest",
    )
    args = parser.parse_args()
    selected = args.roots.split(",") if args.roots else None

    state = diagnose(selected) if args.diagnose else main(selected)
