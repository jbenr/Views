"""Cash-versus-futures basis across the Treasury futures complex.

Question: is the CTD's net basis rich or cheap versus its own recent range?

Net basis is what a basis buyer (long cash, short futures) pays for the
short's delivery options. It is bounded below by roughly zero and above by
the option value, so it is a level to fade rather than a spread to regress:
this module trades the OU z-score of net basis directly and does not fit a
hedge-ratio model. That is why it runs against the Engine rather than
backtest.strategy.Strategy, whose funnel assumes a (feature -> target)
residual.

    signal = OU z-score of the CTD net basis in 32nds
    z high  -> basis rich  -> SELL the basis (sell cash, buy futures)
    z low   -> basis cheap -> BUY  the basis (buy cash, sell futures)

Six contracts trade as one book, one pipeline each, sharing thresholds. The
z-score is what makes that legitimate: a 32nd of WN basis and a 32nd of TU
basis are not the same risk, but two standard deviations of each are
comparable.

The measure, its two financing/delivery approximations, and the choice of
32nds over implied-repo bps all live in utils.basis — read that docstring
before trusting a number out of this one.

    python -m book.basis.cash_futures            # backtest the book
    python -m book.basis.cash_futures --roots TY,US
    python -m book.basis.cash_futures --diagnose # basis diagnostics, no trading

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
from utils.basis import DELIVERABLE_ROOTS, basis_panel, net_basis

# ── config (identity — what this strategy is, not how it's tuned) ─────────────

STRATEGY_FAMILY = "basis"
SIGNAL_NAME = "cash_futures_basis"

# UXY only starts in 2016 and WN in 2010; each root simply carries its own
# history rather than trimming the panel to the shortest one.
START = "2010-01-01"

# Roll to the deferred contract this many days before delivery. Implied repo
# divides by days-to-delivery, so the last fortnight of a cycle is mostly
# amplified quote noise.
MIN_DAYS_TO_DELIVERY = 15


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data(roots: list[str], start: str = START) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Net basis long frame (one row per ts/root) and its wide trading panel."""
    nb = net_basis(roots, start=start, min_days=MIN_DAYS_TO_DELIVERY)
    return nb, basis_panel(nb)


def compute(panel: pl.DataFrame, root: str, params: dict | None = None) -> pl.DataFrame:
    """Signal frame for one root: OU z-score of its net basis in 32nds.

    Returns the engine's ``signal`` column plus the OU state as extras, which
    the engine carries into the trade log at entry and exit.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    ou = roll_ou_features(panel[f"{root}_nb32"], lookback=p["ou_lb"])
    frame = pl.DataFrame(
        {
            "signal": ou["ou_z"],
            "nb32": panel[f"{root}_nb32"],
            "ou_mean": ou["ou_mean"],
            "ou_sigma": ou["ou_sigma"],
            "half_life": ou["half_life"],
        }
    )
    if p["time_stop_style"] == "half_life":
        # Hold for a multiple of the half-life measured at entry rather than a
        # fixed horizon: basis mean-reverts on the delivery cycle's clock, not
        # the calendar's.
        frame = frame.with_columns(
            (ou["half_life"] * p["time_stop_param"]).alias("time_stop")
        )
    return frame


def make_pipeline(root: str, params: dict | None = None) -> SignalPipeline:
    """One root's pipeline. The traded level is the roll-stitched net basis,
    so marking a position through a contract roll never books the roll gap."""
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
            stop_loss_bps=p["stop_loss_32"],
            time_stop_bars=None,
        ),
    )


def coverage(nb: pl.DataFrame) -> pl.DataFrame:
    """Per-root sample, CTD turnover and net basis distribution."""
    return nb.group_by("root").agg(
        pl.len().alias("n_days"),
        pl.col("ts").min().alias("first"),
        pl.col("ts").max().alias("last"),
        pl.col("contract").n_unique().alias("n_contracts"),
        pl.col("ctd").n_unique().alias("n_ctd"),
        pl.col("gross_basis_32").median().round(2).alias("med_gross_32"),
        pl.col("net_basis_32").median().round(2).alias("med_net_32"),
        pl.col("net_basis_32").std().round(2).alias("sd_net_32"),
        (pl.col("implied_repo") - pl.col("financing")).median().round(3).alias("med_irr_less_fin"),
    ).sort("root")


def reversion(nb: pl.DataFrame) -> pl.DataFrame:
    """Does fading net basis pay? IC / hit / Sharpe per root, per horizon."""
    rows = []
    for root in nb["root"].unique(maintain_order=True):
        series = nb.filter(pl.col("root") == root)["net_basis_32"]
        diag = horizon_backtest(series, horizons=(5, 10, 20, 40))
        rows.append(diag.insert_column(0, pl.Series("root", [root] * len(diag))))
    return pl.concat(rows)


def latest(nb: pl.DataFrame, panel: pl.DataFrame, params: dict) -> pl.DataFrame:
    """Today's read per root: where the basis sits and what it implies."""
    rows = []
    for root in nb["root"].unique(maintain_order=True):
        sig = compute(panel, root, params=params)
        last_row = nb.filter(pl.col("root") == root).row(-1, named=True)
        z = sig["signal"][-1]
        if z is None or z != z:
            action = "warmup"
        elif z >= params["entry_z"]:
            action = "SELL basis"
        elif z <= -params["entry_z"]:
            action = "BUY basis"
        else:
            action = "flat"
        rows.append(
            {
                "root": root,
                "ts": last_row["ts"],
                "contract": last_row["contract"],
                "ctd": last_row["ctd"],
                "n_days": last_row["n_days"],
                "net_basis_32": round(last_row["net_basis_32"], 2),
                "implied_repo": round(last_row["implied_repo"], 3),
                "financing": round(last_row["financing"], 3),
                "ou_z": None if z is None or z != z else round(float(z), 2),
                "action": action,
            }
        )
    return pl.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────
# Backtest parameters live HERE — the first thing you touch when scripting
# runs. main(params={...}) overrides ad hoc.

DEFAULT_PARAMS = {
    "ou_lb": 252,             # OU state lookback (days), ~4 delivery cycles
    "entry_z": 2.0,           # fade |z| beyond this
    "exit_z": 0.5,            # flatten once z is back inside this band
    "stop_loss_32": 8.0,      # hard stop, 32nds of adverse move
    "time_stop_style": "half_life",   # "half_life" | "none"
    "time_stop_param": 2.0,   # time stop at this x the half-life measured at entry
}

# Units note: the engine's "bps" are 32nds of a point here, because the traded
# level is net basis in 32nds. PnL from this module is therefore NOT comparable
# to the curve or duration strategies, whose levels are yields in bps.
TRANSACTION_COST_32 = 0.5


def main(roots: list[str] | None = None, params: dict | None = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    roots = list(roots or DELIVERABLE_ROOTS)

    # net basis per root, plus the wide panel the engine marks against
    nb, panel = load_data(roots)
    print(f"roots={roots}  rows={len(nb)}  panel={panel.shape}  params={p}")

    # sample, CTD turnover and where the basis has actually sat
    cover = coverage(nb)
    print("\ncoverage / basis distribution (32nds; irr_less_fin in %):")
    utils.pdf(cover)

    # the prior the whole strategy rests on: does net basis revert?
    revert = reversion(nb)
    print("\nnet basis horizon backtest (IC / hit / Sharpe of fading it):")
    utils.pdf(revert)

    # backtest the six roots as one book through the shared engine
    engine = Engine(BacktestConfig(transaction_cost_bps=TRANSACTION_COST_32,
                                   max_total_positions=len(roots)))
    for root in roots:
        engine.add_signal(make_pipeline(root, p))
    result = engine.run(panel)
    print_summary(result)

    trades = trade_log(result.closed_trades)
    if not trades.is_empty():
        print("\nper-root trade summary:")
        utils.pdf(
            trades.group_by("trade_name").agg(
                pl.len().alias("n_trades"),
                pl.col("pnl_bps").sum().round(1).alias("total_32"),
                pl.col("pnl_bps").median().round(2).alias("median_32"),
                (pl.col("pnl_bps") > 0).mean().round(3).alias("hit_rate"),
                pl.col("bars_held").median().alias("median_hold"),
            ).sort("trade_name")
        )

    # today's live read
    live = latest(nb, panel, p)
    print("\nlatest signal:")
    utils.pdf(live)

    return {
        "nb": nb,
        "panel": panel,
        "coverage": cover,
        "reversion": revert,
        "result": result,
        "trades": trades,
        "latest": live,
    }


def diagnose(roots: list[str] | None = None) -> dict:
    """Basis construction only — no trading. Use when checking the data."""
    roots = list(roots or DELIVERABLE_ROOTS)
    nb, panel = load_data(roots)

    print(f"roots={roots}  rows={len(nb)}")
    print("\ncoverage / basis distribution:")
    utils.pdf(coverage(nb))

    print("\nmedian net basis (32nds) by root and year:")
    utils.pdf(
        nb.with_columns(pl.col("ts").dt.year().alias("year"))
        .group_by("root", "year")
        .agg(pl.col("net_basis_32").median().round(2).alias("nb32"))
        .pivot(index="year", on="root", values="nb32")
        .sort("year")
    )

    # implied repo is the readable cross-check: it should sit near financing
    print("\nimplied repo less financing (%), by root:")
    utils.pdf(
        nb.with_columns(
            (pl.col("implied_repo") - pl.col("financing")).alias("spread")
        )
        .group_by("root")
        .agg(
            pl.col("spread").quantile(0.25).round(3).alias("p25"),
            pl.col("spread").median().round(3).alias("median"),
            pl.col("spread").quantile(0.75).round(3).alias("p75"),
        )
        .sort("root")
    )

    return {"nb": nb, "panel": panel}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        help=f"comma-separated contract roots (default: {','.join(DELIVERABLE_ROOTS)})",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="basis construction diagnostics only, no backtest",
    )
    args = parser.parse_args()
    selected = args.roots.split(",") if args.roots else None

    state = diagnose(selected) if args.diagnose else main(selected)
