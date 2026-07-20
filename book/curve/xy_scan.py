"""Cross-pair predictability explorer - funnel step 0 for the curve book.

Question: which (x feature, y curve) pairs show ANY corroborated forward
predictability? Predict-only: for every pair, the same residual / OU-z
signal grid and gate overlay as tens_10s30s --predict, scored on raw IC and
selected by neighborhood IC. No trading mechanics.

X candidates include the 10Y level, real 10Y, breakevens, both 5y5y flavors
(nominal 2*10y-5y and inflation 2*be10-be5), and point-in-time PC1 of the
yield panel at several PCA lookbacks (PC1_LBS) - rolling, sign-fixed, no
lookahead.

Winners graduate: clone book/curve/tens_10s30s.py for the promising pair
and run the full --predict/--exit/--sweep funnel there. The top setups per
pair are saved to XY_SETUPS_FILE for reference.

    python -m book.curve.xy_scan              # live DB (GPU-friendly)
    python -m book.curve.xy_scan --synthetic  # no DB
    --cpu / --gpu force the scan device (default: auto).
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

import utils
from backtest import gate_variant_count, neighbor_ic_stats, predict_scan, signal_matrix
from stats import roll_pc1_score
from utils.market_data import align_columns, load_wide
from utils.rates import linear_5y5y_forward

# -- config -----------------------------------------------------------------

SCAN_NAME = "xy_scan"

START = "2010-01-01"

TICKERS = {
    # yield panel (PC1 inputs and 5y5y construction)
    "2y": "USGG2YR Index",
    "5y": "USGG5YR Index",
    "10y": "USGG10YR Index",
    "30y": "USGG30YR Index",
    # inflation / real
    "be5": "USGGBE05 Index",
    "be10": "USGGBE10 Index",
    "real10y": "USGGT10Y Index",
    # y candidates: curve generics, already quoted in bps
    "2s5s": "USYC2Y5Y Index",
    "2s10s": "USYC2Y10 Index",
    "2s30s": "USYC2Y30 Index",
    "5s10s": "USYC5Y10 Index",
    "5s30s": "USYC5Y30 Index",
    "10s30s": "USYC1030 Index",
}
BPS_COLS = ["2y", "5y", "10y", "30y", "be5", "be10", "real10y"]

PC1_COLS = ["2y", "5y", "10y", "30y"]
PC1_LBS = [126, 252, 504]  # each lookback is its own x candidate (pc1_126, ...)

YS = ["2s5s", "2s10s", "2s30s", "5s10s", "5s30s", "10s30s"]
XS = ["10y", "real10y", "be10", "5y5y", "5y5y_infl"] + [
    f"pc1_{lb}" for lb in PC1_LBS
]

# the tens_10s30s predict grid at half resolution (step 20 not 10) so the
# full pair sweep stays in GPU-minutes; neighborhood steps follow these lists
XY_ENTRY_SIGNALS = ["residual", "ou_z"]
XY_BETA_LBS = list(range(20, 501, 20))
XY_OU_LBS = list(range(20, 501, 20))
XY_HORIZONS = [5, 10, 20, 40, 60, 100]
XY_RESID_THRESHOLDS_BPS = list(range(11, 31, 2))
XY_OU_Z_THRESHOLDS = np.arange(0.5, 3.1, 0.2).tolist()
XY_GATE_BUCKETS = "regime"
XY_MIN_OBS = 30  # ignore cells with fewer threshold-crossing events
XY_MIN_ROWS = 750  # skip pairs with less aligned history (~3y)
XY_TOP_N_PER_PAIR = 3  # setups kept per (x, y) pair
XY_MIN_NEIGHBORS = 3  # corroborating grid neighbors a setup needs

XY_SETUPS_FILE = Path(__file__).with_name(f"{SCAN_NAME}_setups.parquet")


# -- helpers ----------------------------------------------------------------


def load_data(start: str = START) -> pl.DataFrame:
    """Load the full ticker panel from md.index_eod."""
    return load_wide(TICKERS, start=start, bps_cols=BPS_COLS)


def synthetic_data(n: int = 1500, seed: int = 7) -> pl.DataFrame:
    """Synthetic panel: correlated yield levels, curves as spreads plus OU
    residuals, breakevens as their own walk."""
    rng = np.random.default_rng(seed)

    def walk(start_level, vol):
        return start_level + np.cumsum(rng.normal(0.0, vol, n))

    level = np.cumsum(rng.normal(0.0, 2.0, n))  # common level factor
    yields = {
        "2y": 150.0 + level + walk(0.0, 1.0),
        "5y": 250.0 + level + walk(0.0, 0.8),
        "10y": 350.0 + level + walk(0.0, 0.6),
        "30y": 400.0 + level + walk(0.0, 0.6),
    }
    be = {"be5": 200.0 + walk(0.0, 1.0), "be10": 220.0 + walk(0.0, 0.8)}

    def ou(sigma=2.0, theta=0.05):
        r = np.zeros(n)
        for i in range(1, n):
            r[i] = r[i - 1] * (1 - theta) + rng.normal(0.0, sigma)
        return r

    curves = {
        "2s5s": yields["5y"] - yields["2y"] + ou(),
        "2s10s": yields["10y"] - yields["2y"] + ou(),
        "2s30s": yields["30y"] - yields["2y"] + ou(),
        "5s10s": yields["10y"] - yields["5y"] + ou(),
        "5s30s": yields["30y"] - yields["5y"] + ou(),
        "10s30s": yields["30y"] - yields["10y"] + ou(),
    }

    start_date = dt.date.fromisoformat(START)
    ts = pl.date_range(
        start_date, start_date + dt.timedelta(days=2 * n), interval="1d", eager=True
    )
    ts = ts.filter(ts.dt.weekday() <= 5)[:n]

    return pl.DataFrame({
        "ts": ts,
        **yields,
        **be,
        "real10y": yields["10y"] - be["be10"],
        **curves,
    })


def add_features(data: pl.DataFrame) -> pl.DataFrame:
    """Derived x candidates: both 5y5y flavors and rolling sign-fixed PC1
    scores at each PC1_LBS lookback."""
    out = data.with_columns(
        linear_5y5y_forward(pl.col("5y"), pl.col("10y")).alias("5y5y"),
        linear_5y5y_forward(pl.col("be5"), pl.col("be10")).alias("5y5y_infl"),
    )
    panel = align_columns(out, PC1_COLS)
    scores = panel.select("ts")
    for lb in PC1_LBS:
        scores = scores.with_columns(
            roll_pc1_score(panel.select(PC1_COLS), lookback=lb).alias(f"pc1_{lb}")
        )
    return out.join(scores, on="ts", how="left")


def _pair_scan(
    frame: pl.DataFrame, x: str, y: str, device: str
) -> list[pl.DataFrame]:
    """tens_10s30s-style predict scan (residual + ou_z, gates overlaid) for
    one aligned (x, y) frame, tagged with pair identity."""
    level = frame[y].to_numpy()
    blocks = []
    for entry_signal, ou_lbs, thresholds, kind, units in [
        ("residual", [0], XY_RESID_THRESHOLDS_BPS, "residual", "bps"),
        ("ou_z", XY_OU_LBS, XY_OU_Z_THRESHOLDS, "ou_zscore", "z"),
    ]:
        if entry_signal not in XY_ENTRY_SIGNALS:
            continue
        matrix, combos, conditions = signal_matrix(
            frame[x],
            frame[y],
            XY_BETA_LBS,
            ou_lbs,
            return_conditions=True,
            signal_kind=kind,
            lookback_name="ou_lb",
        )
        block = predict_scan(
            matrix,
            level,
            entries=thresholds,
            horizons=XY_HORIZONS,
            combos=combos,
            gates=conditions,
            gate_buckets=XY_GATE_BUCKETS,
            device=device,
            entry_col="entry_threshold",
        ).with_columns(
            pl.lit(x).alias("x"),
            pl.lit(y).alias("y"),
            pl.lit(entry_signal).alias("entry_signal"),
            pl.lit(units).alias("threshold_units"),
        )
        if entry_signal == "residual":
            block = block.with_columns(pl.lit(None, dtype=pl.Int64).alias("ou_lb"))
        blocks.append(block)
    return blocks


def _setup_name(row: dict) -> str:
    """Pair-qualified setup label, e.g. '5s30s~pc1_252 ou60/240/e1.3 h60 r2:low_25'."""
    if row["entry_signal"] == "residual":
        base = f"res{row['beta_lb']}/e{row['entry_threshold']:g}"
    else:
        base = f"ou{row['beta_lb']}/{row['ou_lb']}/e{row['entry_threshold']:g}"
    name = f"{row['y']}~{row['x']} {base} h{row['predict_horizon']}"
    if row["gate"] != "(none)":
        name += f" {row['gate']}:{row['gate_bucket']}"
    return name


def _select_setups(valid: pl.DataFrame) -> pl.DataFrame:
    """Top XY_TOP_N_PER_PAIR setups per (x, y) pair by neighborhood IC.

    Same discipline as tens_10s30s: rank on nbr_ic (neighborhoods never
    cross pairs), require XY_MIN_NEIGHBORS corroborating neighbors, dedupe
    threshold/horizon variants of the same cell."""
    best = (
        neighbor_ic_stats(
            valid,
            beta_lbs=XY_BETA_LBS,
            ou_lbs=XY_OU_LBS,
            resid_thresholds=XY_RESID_THRESHOLDS_BPS,
            z_thresholds=XY_OU_Z_THRESHOLDS,
            pool_size=100 * len(YS) * len(XS),
            extra_keys=("x", "y"),
        )
        .filter(pl.col("n_nbr") >= XY_MIN_NEIGHBORS)
        .sort("nbr_ic", descending=True)
        .unique(
            subset=["x", "y", "entry_signal", "beta_lb", "ou_lb", "gate",
                    "gate_bucket"],
            keep="first",
            maintain_order=True,
        )
        .group_by("x", "y", maintain_order=True)
        .head(XY_TOP_N_PER_PAIR)
        .sort("nbr_ic", descending=True)
        .rename({"horizon": "predict_horizon"})
        .select(
            "x", "y", "entry_signal", "beta_lb", "ou_lb", "entry_threshold",
            "predict_horizon", "gate", "gate_bucket",
            "ic", "nbr_ic", "n_nbr", "hit_rate", "fire_rate", "n_obs",
        )
    )
    names = [_setup_name(r) for r in best.iter_rows(named=True)]
    return best.insert_column(0, pl.Series("name", names))


# -- mode -------------------------------------------------------------------


def main(use_db: bool = True, device: str = "auto") -> dict:
    data = add_features(load_data() if use_db else synthetic_data())

    pairs = [
        (y, x) for y in YS for x in XS if y != x
    ]
    n_variants = 1 + 11 * gate_variant_count(XY_GATE_BUCKETS)
    print(
        f"xy scan: {len(XS)} x-features vs {len(YS)} curves = {len(pairs)} pairs  "
        f"horizons={XY_HORIZONS}  gate variants~{n_variants}  (device={device})"
    )

    t0 = time.time()
    result_blocks = []
    skipped = []
    for i, (y, x) in enumerate(pairs, 1):
        frame = align_columns(data, [y, x])
        if len(frame) < XY_MIN_ROWS:
            skipped.append((y, x, len(frame)))
            continue
        bt = time.time()
        result_blocks.extend(_pair_scan(frame, x, y, device))
        print(
            f"\r  [{i}/{len(pairs)}] y={y} x={x} rows={len(frame)} "
            f"({time.time() - bt:.1f}s, total {time.time() - t0:.0f}s)   ",
            end="",
            flush=True,
        )
    print(f"\n  scan done in {time.time() - t0:.1f}s")
    if skipped:
        print(f"  skipped {len(skipped)} pairs with < {XY_MIN_ROWS} aligned rows")

    results = pl.concat(result_blocks, how="diagonal_relaxed").sort(
        "ic", descending=True, nulls_last=True
    )
    valid = results.filter(
        (pl.col("n_obs") >= XY_MIN_OBS) & pl.col("ic").is_finite()
    )

    show = [
        "x", "y", "entry_signal", "beta_lb", "ou_lb", "entry_threshold",
        "horizon", "gate", "gate_bucket", "ic", "hit_rate", "fire_rate",
        "n_obs",
    ]
    print(f"\ntop 20 cells by raw IC, any pair (n_obs >= {XY_MIN_OBS}):")
    utils.pdf(valid.select([c for c in show if c in valid.columns]).head(20))

    setups = _select_setups(valid)
    setups.write_parquet(XY_SETUPS_FILE)
    print(
        f"\ntop {XY_TOP_N_PER_PAIR} setups per pair by neighborhood IC "
        f"(>= {XY_MIN_NEIGHBORS} neighbors), saved -> {XY_SETUPS_FILE}:"
    )
    utils.pdf(setups.head(25))

    print("\nwhich x predicts which y (best nbr_ic per pair; blank = nothing"
          " corroborated):")
    grid = (
        setups.group_by("y", "x")
        .agg(pl.col("nbr_ic").max().round(3))
        .pivot(on="x", index="y", values="nbr_ic")
        .sort("y")
    )
    utils.pdf(grid.select(["y", *[c for c in XS if c in grid.columns]]))

    return {"data": data, "results": results, "valid": valid, "setups": setups}


if __name__ == "__main__":
    args = set(sys.argv[1:])
    known = {"--synthetic", "--cpu", "--gpu"}
    unknown = args - known
    if unknown:
        sys.exit(
            f"unknown argument(s): {sorted(unknown)}\n"
            "flags: --synthetic --cpu --gpu"
        )
    use_db = "--synthetic" not in args
    device = "cpu" if "--cpu" in args else ("gpu" if "--gpu" in args else "auto")
    state = main(use_db=use_db, device=device)
