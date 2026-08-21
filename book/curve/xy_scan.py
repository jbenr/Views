"""Cross-pair predictability explorer - funnel step 0 for the curve book.

Question: which (x feature, y curve) pairs show ANY corroborated forward
predictability? Predict-only: for every pair, the same residual / OU-z
signal grid and gate overlay as tens_10s30s --predict, scored on raw IC,
selected by neighborhood IC, and screened for independent forecast windows
the same way Strategy.predict screens them. No trading mechanics.

X candidates include the 2Y and 10Y levels, real 10Y, breakevens, several
forwards (nominal 5y5y, inflation 5y5y, SOFR 1y1y / 5y5y / 20y10y and a
synthetic real 5y5y), and point-in-time PC1 of the yield panel at several
PCA lookbacks (PC1_LBS) - rolling, sign-fixed, no lookahead.

A forward spanning two tenors is a linear combination of them, so a forward
that straddles the curve being modelled contains that curve outright: nominal
5y5y carries 1.5x the 5s10s spread. leak_matrix() prints every such
construction overlap before the scan runs, so a headline IC that is really a
target explaining itself is visible rather than inferred.

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
from backtest import (
    event_overlap_diagnostics,
    gate_allow_mask,
    gate_variant_count,
    neighbor_ic_stats,
    predict_scan,
    signal_matrix,
)
from stats import roll_pc1_score
from utils.market_data import align_columns, load_wide
from utils.rates import linear_forward, synthetic_5y5y_real

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
    # forward inputs are built from SOFR OIS and inflation swaps rather than
    # the Treasury panel: a forward spanning two Treasury tenors is a linear
    # combination of them, so it carries any curve between them straight into
    # the explanatory variable (see X_PANEL_WEIGHTS / leak_matrix below)
    "ois1": "USOSFR1 Curncy",
    "ois2": "USOSFR2 Curncy",
    "ois3": "USOSFR3 Curncy",
    "ois5": "USOSFR5 Curncy",
    "ois10": "USOSFR10 Curncy",
    "ois20": "USOSFR20 Curncy",
    "ois30": "USOSFR30 Curncy",
    "zc5": "USSWIT5 Curncy",
    "zc10": "USSWIT10 Curncy",
    # y candidates: curve generics, already quoted in bps
    "2s5s": "USYC2Y5Y Index",
    "2s10s": "USYC2Y10 Index",
    "2s30s": "USYC2Y30 Index",
    "5s10s": "USYC5Y10 Index",
    "5s30s": "USYC5Y30 Index",
    "10s30s": "USYC1030 Index",
}
BPS_COLS = [
    "2y",
    "5y",
    "10y",
    "30y",
    "be5",
    "be10",
    "real10y",
    "ois1",
    "ois2",
    "ois3",
    "ois5",
    "ois10",
    "ois20",
    "ois30",
    "zc5",
    "zc10",
]

PC1_COLS = ["2y", "5y", "10y", "30y"]
PC1_LBS = [126, 252, 504]  # each lookback is its own x candidate (pc1_126, ...)

YS = ["2s5s", "2s10s", "2s30s", "5s10s", "5s30s", "10s30s"]
XS = [
    "2y",
    "10y",
    "real10y",
    "be10",
    "5y5y",
    "5y5y_infl",
    # forwards: 1y1y is the policy path, 20y10y the long-end term premium --
    # both sit outside every YS target's legs. 5y5y_sfr and 5y5y_real span
    # 5y-10y but on a different instrument, so they carry no arithmetic leak
    # (they are still ~0.95 correlated with the nominal curve; a residual
    # against 5y5y_sfr is part swap-spread trade).
    "1y1y",
    "3y2y",
    "2y3y",
    "20y10y",
    "5y5y_sfr",
    "5y5y_real",
] + [f"pc1_{lb}" for lb in PC1_LBS]

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
# Non-overlapping forecast windows a setup needs, matching Strategy's
# predict_min_independent_events. n_obs counts threshold crossings, which at
# h100 can be 30+ events sharing almost the same 100-day forward window -- an
# IC computed over those is mostly the autocorrelation of overlapping labels,
# and it ranks the longest horizon top of the board every time. Screening on
# independent episodes is what makes this explorer agree with the per-strategy
# funnel that setups graduate into; without it the scan promotes cells that
# --predict then rejects.
XY_MIN_INDEPENDENT_EVENTS = 8
# candidates shortlisted per pair BEFORE the independence screen, so the
# XY_TOP_N_PER_PAIR survivors come from a real pool rather than whatever three
# cells happened to top the raw board (mirrors Strategy's predict_top_n * 10)
XY_CANDIDATE_LIMIT = 30
XY_KEEP_PER_PAIR = 200  # top valid rows kept per pair for the cross-pair board
# (each pair produces ~7M result rows; they are reduced per pair - filtered,
# selected, top slice kept - so 48 pairs never accumulate ~40GB in RAM)

XY_DATA_DIR = Path(__file__).with_name("data")
XY_SETUPS_FILE = XY_DATA_DIR / f"{SCAN_NAME}_setups.parquet"


# Weights each x places on the NOMINAL TREASURY legs that YS targets are built
# from. Anything drawn from another instrument (TIPS, breakevens, SOFR OIS,
# inflation swaps) contributes nothing here: it may be highly correlated with
# the curve, but it is not arithmetically made of it. pc1_* is absent because
# its weights are fitted per bar rather than fixed -- measure those with
# stats.pc1_self_weight instead.
X_PANEL_WEIGHTS = {
    "2y": {"2y": 1.0},
    "10y": {"10y": 1.0},
    "5y5y": {"5y": -1.0, "10y": 2.0},
}

# y -> (short leg, long leg)
Y_LEGS = {
    "2s5s": ("2y", "5y"),
    "2s10s": ("2y", "10y"),
    "2s30s": ("2y", "30y"),
    "5s10s": ("5y", "10y"),
    "5s30s": ("5y", "30y"),
    "10s30s": ("10y", "30y"),
}


# -- helpers ----------------------------------------------------------------


def leak_coefficient(x: str, y: str) -> float | None:
    """How much of curve `y` the feature `x` contains, by construction.

    Writing the target's legs as their mean m and spread s = long - short, a
    feature with weights w carries s with coefficient (w_long - w_short)/2 --
    the same quantity stats.pc1_self_weight measures for a fitted PC1. It is
    exact here because these weights are fixed.

    Non-zero does not mean invalid: hedging a curve on its own short leg is a
    deliberate design (tens_10s30s) and lands at 0.5. It means the regression
    is partly explaining the target with itself, so the residual shrinks and
    reverts more readily than the market did. Magnitude is what matters --
    5y5y against 5s10s is 1.5, an order above any leg hedge.

    None for features whose weights are fitted rather than declared.
    """
    if x.startswith("pc1_"):
        return None
    weights = X_PANEL_WEIGHTS.get(x, {})
    short, long = Y_LEGS[y]
    return (weights.get(long, 0.0) - weights.get(short, 0.0)) / 2.0


def leak_matrix() -> pl.DataFrame:
    """Every (x, y) construction leak, as an x-by-y table."""
    return pl.DataFrame(
        [{"x": x, **{y: leak_coefficient(x, y) for y in YS}} for x in XS]
    )


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

    # Swap-market stand-ins: same level factor, own idiosyncratic noise, so
    # they correlate with the Treasury panel without being made of it. Tenors
    # are read off TICKERS rather than listed again here -- a hardcoded list
    # silently drops any swap ticker added later, and --synthetic then fails
    # on a missing column well away from the edit that caused it.
    def _anchor(tenor: int) -> str:
        return f"{min(yields, key=lambda k: abs(int(k[:-1]) - tenor))}"

    ois = {
        k: yields[_anchor(int(k[3:]))] + walk(0.0, 0.3)
        for k in TICKERS if k.startswith("ois")
    }
    zc = {
        k: 200.0 + 2.0 * int(k[2:]) + walk(0.0, 0.6)
        for k in TICKERS if k.startswith("zc")
    }

    return pl.DataFrame(
        {
            "ts": ts,
            **yields,
            **be,
            **ois,
            **zc,
            "real10y": yields["10y"] - be["be10"],
            **curves,
        }
    )


def add_features(data: pl.DataFrame) -> pl.DataFrame:
    """Derived x candidates: both 5y5y flavors and rolling sign-fixed PC1
    scores at each PC1_LBS lookback."""
    out = data.with_columns(
        # <a>y<b>y = the b-year rate starting a years forward, so the legs are
        # a and a+b -- not a and b. linear_forward is symmetric in its
        # (rate, tenor) pairs, so passing the same two tenors in either order
        # yields the same series: 2y3y and 3y2y must differ in their LEGS.
        linear_forward(pl.col("ois1"), 1, pl.col("ois2"), 2).alias("1y1y"),
        linear_forward(pl.col("ois2"), 2, pl.col("ois5"), 5).alias("2y3y"),
        linear_forward(pl.col("ois3"), 3, pl.col("ois5"), 5).alias("3y2y"),
        linear_forward(pl.col("5y"), 5, pl.col("10y"), 10).alias("5y5y"),
        linear_forward(pl.col("be5"), 5, pl.col("be10"), 10).alias("5y5y_infl"),
        linear_forward(pl.col("ois20"), 20, pl.col("ois30"), 30).alias("20y10y"),
        linear_forward(pl.col("ois5"), 5, pl.col("ois10"), 10).alias("5y5y_sfr"),
        synthetic_5y5y_real(
            pl.col("ois5"), pl.col("ois10"), pl.col("zc5"), pl.col("zc10")
        ).alias("5y5y_real"),
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
) -> tuple[list[pl.DataFrame], list[dict]]:
    """tens_10s30s-style predict scan (residual + ou_z, gates overlaid) for
    one aligned (x, y) frame, tagged with pair identity.

    Returns the result blocks and the underlying signal matrices, so the
    independence screen can re-derive each shortlisted setup's entry events
    without rebuilding the scan.
    """
    level = frame[y].to_numpy()
    blocks = []
    scans = []
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
        scans.append(
            {
                "entry_signal": entry_signal,
                "matrix": matrix,
                "combos": combos,
                "conditions": conditions,
            }
        )
    return blocks, scans


def _overlap_diagnostics(setups: pl.DataFrame, scans: list[dict]) -> pl.DataFrame:
    """Count each setup's non-overlapping forecast windows.

    Replays the entry rule the scan scored -- first bar of each threshold
    excursion, gate applied, forward window in sample -- against the cached
    signal matrix, then measures how many of those events carry independent
    labels. Same construction as Strategy._add_overlap_diagnostics, so a
    setup's count here is the count it will get on graduation.
    """
    lookup = {}
    for scan in scans:
        for column, combo in enumerate(scan["combos"]):
            key = (
                scan["entry_signal"],
                int(combo["beta_lb"]),
                int(combo.get("ou_lb", 0)),
            )
            lookup[key] = (scan, column)

    rows = []
    for setup in setups.iter_rows(named=True):
        entry_signal = setup["entry_signal"]
        ou_lb = 0 if entry_signal == "residual" else int(setup["ou_lb"])
        scan, column = lookup[(entry_signal, int(setup["beta_lb"]), ou_lb)]
        signal = scan["matrix"][:, column]

        previous = np.concatenate([[np.nan], signal[:-1]])
        entry = float(setup["entry_threshold"])
        crossed = ((signal >= entry) & ~(previous >= entry)) | (
            (signal <= -entry) & ~(previous <= -entry)
        )
        gate = setup["gate"]
        if gate in (None, "(none)"):
            gate_ok = np.ones(len(signal), dtype=bool)
        else:
            # the scan took predict_scan's gate defaults (252-bar warmup,
            # expanding percentile), so the replay must too
            gate_ok = gate_allow_mask(
                scan["conditions"][gate][:, column], (gate, setup["gate_bucket"])
            )
        horizon = int(setup["predict_horizon"])
        valid_forward = np.arange(len(signal)) < len(signal) - horizon
        indices = np.flatnonzero(crossed & gate_ok & valid_forward)
        rows.append(event_overlap_diagnostics(indices, horizon))

    return setups.with_columns(
        pl.Series(
            "n_non_overlapping", [r["n_non_overlapping"] for r in rows], dtype=pl.Int64
        ),
        pl.Series(
            "overlap_fraction", [r["overlap_fraction"] for r in rows], dtype=pl.Float64
        ),
    )


def _fmt_secs(seconds: float) -> str:
    """Compact duration: 45s, 3m10s, 1h04m."""
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _device_label(device: str) -> str:
    """Which array backend the scan will actually use, not which was asked for.

    device='auto' silently falls back to numpy when CUDA is unusable, and the
    difference is minutes per pair -- worth stating outright rather than
    leaving the reader to infer it from a cupy import warning.
    """
    from backtest.lab import _get_xp  # authoritative: also test-compiles a kernel

    resolved = "gpu" if _get_xp(device).__name__ == "cupy" else "cpu"
    return resolved if resolved == device else f"{resolved}  (requested {device})"


def _pair_line(
    i: int,
    n: int,
    y: str,
    x: str,
    rows: int,
    best_ic: float | None,
    n_kept: int,
    n_candidates: int,
    pair_secs: float,
    total_secs: float,
) -> str:
    """One finished pair as a permanent line: what it found, and when the run ends.

    Printed per pair rather than overwritten in place, so the scan leaves a
    readable history -- which pairs produced setups is the actual output of
    this stage, and waiting for the final table to learn it wastes the run.
    """
    ic = f"{best_ic:+.3f}" if best_ic is not None and np.isfinite(best_ic) else "  --  "
    eta = _fmt_secs((total_secs / i) * (n - i)) if i < n else "done"
    return (
        f"  [{i:>2}/{n}] {y:>6}~{x:<9} rows {rows:>5}  ic {ic}  "
        f"kept {n_kept}/{n_candidates:<2}  {pair_secs:5.1f}s  eta {eta:>6}"
    )


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


def _select_setups(valid: pl.DataFrame, limit: int) -> pl.DataFrame:
    """Top `limit` setups per (x, y) pair by neighborhood IC.

    Called per pair (memory discipline), but keyed on x/y regardless so
    neighborhoods can never cross pairs. Same discipline as tens_10s30s:
    rank on nbr_ic, require XY_MIN_NEIGHBORS corroborating neighbors,
    dedupe threshold/horizon variants of the same cell."""
    best = (
        neighbor_ic_stats(
            valid,
            beta_lbs=XY_BETA_LBS,
            ou_lbs=XY_OU_LBS,
            resid_thresholds=XY_RESID_THRESHOLDS_BPS,
            z_thresholds=XY_OU_Z_THRESHOLDS,
            pool_size=300,
            extra_keys=("x", "y"),
        )
        .filter(pl.col("n_nbr") >= XY_MIN_NEIGHBORS)
        .sort("nbr_ic", descending=True)
        .unique(
            subset=[
                "x",
                "y",
                "entry_signal",
                "beta_lb",
                "ou_lb",
                "gate",
                "gate_bucket",
            ],
            keep="first",
            maintain_order=True,
        )
        .group_by("x", "y", maintain_order=True)
        .head(limit)
        .sort("nbr_ic", descending=True)
        .rename({"horizon": "predict_horizon"})
        .select(
            "x",
            "y",
            "entry_signal",
            "beta_lb",
            "ou_lb",
            "entry_threshold",
            "predict_horizon",
            "gate",
            "gate_bucket",
            "ic",
            "nbr_ic",
            "n_nbr",
            "hit_rate",
            "fire_rate",
            "n_obs",
        )
    )
    names = [_setup_name(r) for r in best.iter_rows(named=True)]
    return best.insert_column(0, pl.Series("name", names))


# -- mode -------------------------------------------------------------------


def main(use_db: bool = True, device: str = "auto") -> dict:
    data = add_features(load_data() if use_db else synthetic_data())

    pairs = [(y, x) for y in YS for x in XS if y != x]
    n_variants = 1 + 11 * gate_variant_count(XY_GATE_BUCKETS)
    print(
        f"xy scan  {len(XS)} x-features x {len(YS)} curves = {len(pairs)} pairs  "
        f"horizons={XY_HORIZONS}  gate variants~{n_variants}\n"
        f"data     {len(data)} rows  {data['ts'][0]} -> {data['ts'][-1]}  "
        f"{len(TICKERS)} tickers  ({'db' if use_db else 'synthetic'})\n"
        f"device   {_device_label(device)}\n"
    )

    print("construction leak - how much of y each x contains by arithmetic")
    print("(0 = none; 0.5 = hedging on the curve's own leg; null = fitted PC1)")
    utils.pdf(leak_matrix())
    print()

    t0 = time.time()
    top_blocks: list[pl.DataFrame] = []
    setup_blocks: list[pl.DataFrame] = []
    skipped = []
    n_rejected = 0
    for i, (y, x) in enumerate(pairs, 1):
        frame = align_columns(data, [y, x])
        if len(frame) < XY_MIN_ROWS:
            skipped.append((y, x, len(frame)))
            continue
        bt = time.time()
        # transient: a pair takes tens of seconds, so say what is running
        print(
            f"\r  [{i:>2}/{len(pairs)}] {y:>6}~{x:<9} scanning...", end="", flush=True
        )
        # reduce per pair: filter, select setups, keep a top slice - the raw
        # ~7M-row pair frame is dropped before the next pair starts
        blocks, scans = _pair_scan(frame, x, y, device)
        pair_valid = (
            pl.concat(blocks, how="diagonal_relaxed")
            .filter((pl.col("n_obs") >= XY_MIN_OBS) & pl.col("ic").is_finite())
            .sort("ic", descending=True, nulls_last=True)
        )
        top_blocks.append(pair_valid.head(XY_KEEP_PER_PAIR))
        candidates = _select_setups(pair_valid, limit=XY_CANDIDATE_LIMIT)
        n_kept = 0
        if not candidates.is_empty():
            candidates = _overlap_diagnostics(candidates, scans)
            n_rejected += int(
                (candidates["n_non_overlapping"] < XY_MIN_INDEPENDENT_EVENTS).sum()
            )
            pair_setups = candidates.filter(
                pl.col("n_non_overlapping") >= XY_MIN_INDEPENDENT_EVENTS
            ).head(XY_TOP_N_PER_PAIR)
            n_kept = len(pair_setups)
            if not pair_setups.is_empty():
                setup_blocks.append(pair_setups)
        # \r + pad: overwrite the transient line, then keep this one
        print(
            "\r"
            + _pair_line(
                i,
                len(pairs),
                y,
                x,
                len(frame),
                pair_valid["ic"].max() if not pair_valid.is_empty() else None,
                n_kept,
                len(candidates),
                time.time() - bt,
                time.time() - t0,
            ).ljust(88)
        )
    print(
        f"\n  scanned {len(pairs) - len(skipped)}/{len(pairs)} pairs in "
        f"{_fmt_secs(time.time() - t0)}  |  {len(setup_blocks)} pairs produced "
        f"setups, {sum(len(b) for b in setup_blocks)} setups total"
    )
    if skipped:
        print(f"  skipped {len(skipped)} pairs with < {XY_MIN_ROWS} aligned rows")
    if n_rejected:
        print(
            f"  rejected {n_rejected} shortlisted cells with < "
            f"{XY_MIN_INDEPENDENT_EVENTS} independent forecast windows"
        )

    results = pl.concat(top_blocks, how="diagonal_relaxed").sort(
        "ic", descending=True, nulls_last=True
    )

    show = [
        "x",
        "y",
        "entry_signal",
        "beta_lb",
        "ou_lb",
        "entry_threshold",
        "horizon",
        "gate",
        "gate_bucket",
        "ic",
        "hit_rate",
        "fire_rate",
        "n_obs",
    ]
    print(f"\ntop 20 cells by raw IC, any pair (n_obs >= {XY_MIN_OBS}):")
    utils.pdf(results.select([c for c in show if c in results.columns]).head(20))

    setups = (
        pl.concat(setup_blocks, how="diagonal_relaxed").sort(
            "nbr_ic", descending=True, nulls_last=True
        )
        if setup_blocks
        else pl.DataFrame()
    )
    XY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    setups.write_parquet(XY_SETUPS_FILE)
    print(
        f"\ntop {XY_TOP_N_PER_PAIR} setups per pair by neighborhood IC "
        f"(>= {XY_MIN_NEIGHBORS} neighbors, >= {XY_MIN_INDEPENDENT_EVENTS} "
        f"independent forecast windows), saved -> {XY_SETUPS_FILE}:"
    )
    utils.pdf(setups.head(25))

    if not setups.is_empty():
        print(
            "\nwhich x predicts which y (best nbr_ic per pair; blank = "
            "nothing corroborated):"
        )
        grid = (
            setups.group_by("y", "x")
            .agg(pl.col("nbr_ic").max().round(3))
            .pivot(on="x", index="y", values="nbr_ic")
            .sort("y")
        )
        utils.pdf(grid.select(["y", *[c for c in XS if c in grid.columns]]))

    return {"data": data, "results": results, "setups": setups}


if __name__ == "__main__":
    args = set(sys.argv[1:])
    known = {"--synthetic", "--cpu", "--gpu"}
    unknown = args - known
    if unknown:
        sys.exit(
            f"unknown argument(s): {sorted(unknown)}\nflags: --synthetic --cpu --gpu"
        )
    use_db = "--synthetic" not in args
    device = "cpu" if "--cpu" in args else ("gpu" if "--gpu" in args else "auto")
    state = main(use_db=use_db, device=device)
