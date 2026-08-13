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
    python -m book.basis.fut_roll --synthetic
    python -m book.basis.fut_roll --diagnose

main() returns a dict of state for interactive chaining: state = main().
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
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
    *,
    use_db: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Futures-roll long frame and its wide trading panel."""
    if use_db:
        roll = futures_roll(
            roots,
            start=start,
            min_days=MIN_DAYS_TO_FRONT_DELIVERY,
        )
    else:
        roll = synthetic_roll(roots)
    return roll, futures_roll_panel(roll)


def synthetic_roll(
    roots: list[str] | None = None,
    n: int = 1500,
    seed: int = 51,
) -> pl.DataFrame:
    """Synthetic front/deferred futures roll with mean-reverting dislocations."""
    roots = list(roots or DELIVERABLE_ROOTS)
    rng = np.random.default_rng(seed)
    start_date = dt.date.fromisoformat(START)
    dates = pl.date_range(
        start_date,
        start_date + dt.timedelta(days=n - 1),
        interval="1d",
        eager=True,
    ).to_list()

    rows = []
    cycle = 63
    quarter_codes = ("H", "M", "U", "Z")
    for root_i, root in enumerate(roots):
        front_px = 108.0 + root_i * 1.5 + np.cumsum(rng.normal(0.0, 0.08, n))
        resid = np.zeros(n)
        for i in range(1, n):
            resid[i] = 0.93 * resid[i - 1] + rng.normal(0.0, 0.08)
        seasonal = 0.12 * np.sin(np.arange(n) / cycle * 2.0 * np.pi)
        roll = -0.25 - 0.03 * root_i + seasonal + resid

        for i, ts in enumerate(dates):
            pair = i // cycle
            front_code = quarter_codes[pair % 4]
            deferred_code = quarter_codes[(pair + 1) % 4]
            year_digit = (start_date.year + pair // 4) % 10
            front_contract = f"{root}{front_code}{year_digit}"
            deferred_contract = f"{root}{deferred_code}{year_digit}"
            front_delivery = start_date + dt.timedelta(days=(pair + 1) * cycle)
            deferred_delivery = start_date + dt.timedelta(days=(pair + 2) * cycle)
            rows.append(
                {
                    "ts": ts,
                    "root": root,
                    "front_contract": front_contract,
                    "deferred_contract": deferred_contract,
                    "front_delivery": front_delivery,
                    "deferred_delivery": deferred_delivery,
                    "n_days": (front_delivery - ts).days,
                    "deferred_n_days": (deferred_delivery - ts).days,
                    "front_px": float(front_px[i]),
                    "deferred_px": float(front_px[i] - roll[i]),
                    "roll": float(roll[i]),
                }
            )

    return pl.DataFrame(rows).sort(["root", "ts"])


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
    *,
    use_db: bool = True,
) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    roots = list(roots or DELIVERABLE_ROOTS)

    roll, panel = load_data(roots, use_db=use_db)
    source = "db" if use_db else "synthetic"
    print(f"roots={roots}  rows={len(roll)}  panel={panel.shape}  source={source}  params={p}")

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


def diagnose(roots: list[str] | None = None, *, use_db: bool = True) -> dict:
    """Futures-roll construction only; no trading."""
    roots = list(roots or DELIVERABLE_ROOTS)
    roll, panel = load_data(roots, use_db=use_db)
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
    parser.add_argument("--synthetic", action="store_true", help="use synthetic data")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="roll construction diagnostics only, no backtest",
    )
    args = parser.parse_args()
    selected = args.roots.split(",") if args.roots else None
    kwargs = {"use_db": not args.synthetic}

    state = diagnose(selected, **kwargs) if args.diagnose else main(selected, **kwargs)
