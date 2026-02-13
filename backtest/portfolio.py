"""Cross-sectional spread portfolio — rank signals, allocate risk.

The big Sharpe multiplier: instead of trading one spread, score ALL
spreads each day by |z| × confidence, allocate to top-K with risk
parity (inverse-vol sizing). Diversification is mechanical Sharpe
improvement.

Usage:
    book = SpreadBook(BookConfig(max_spreads=3))
    book.add_spread("2s10s",  gen_x="gen_2Y",  gen_y="gen_2s10s",
                     otr_short="otr_2Y", otr_long="otr_10Y")
    book.add_spread("10s30s", gen_x="gen_10Y", gen_y="gen_1030",
                     otr_short="otr_10Y", otr_long="otr_30Y")
    result = book.run(data, lookback=50, z_entry=2.0)
"""

from __future__ import annotations
from dataclasses import dataclass
import polars as pl
import numpy as np

from .trades import TradeDef
from .signals import SignalPipeline, SignalConfig
from .engine import Engine, BacktestConfig, BacktestResult
from .sizing import DV01Map


@dataclass
class SpreadDef:
    """One spread to include in the cross-sectional book."""

    name: str
    gen_x: str       # generic column for regression x (e.g. "gen_10Y")
    gen_y: str       # generic column for regression y (e.g. "gen_1030")
    otr_short: str   # OTR column for short leg PnL (e.g. "otr_10Y")
    otr_long: str    # OTR column for long leg PnL  (e.g. "otr_30Y")
    dv01_short: float = 1.0
    dv01_long: float = 1.0


@dataclass
class BookConfig:
    """Configuration for the cross-sectional spread book."""

    max_spreads: int = 3
    allocation: str = "risk_parity"   # "risk_parity", "equal_weight", "rank_weighted"
    total_risk_budget: float = 1.0
    transaction_cost_bps: float = 0.1
    slippage_bps: float = 0.0


def _compute_spread_signals(
    data: pl.DataFrame, gen_x: str, gen_y: str,
    lookback: int, z_entry: float,
) -> pl.DataFrame:
    """Compute signal, time_stop, size, confidence, vol for one spread."""
    from stats import roll_lr, ou_zscore, roll_half_life

    reg = roll_lr(data[gen_x], data[gen_y], lookback=lookback)
    z = ou_zscore(reg["resid"], lookback=lookback)
    rhl = roll_half_life(reg["resid"], lookback=lookback)

    # Confidence: R² × beta stability
    r2 = reg["r2"].fill_null(0).clip(0, 1)
    beta_cv = (
        reg["beta"].rolling_std(lookback)
        / reg["beta"].rolling_mean(lookback).abs()
    ).fill_null(2.0).clip(0, 2)
    conf = (r2 * (1 - beta_cv / 2)).clip(0.05, 1.0)

    # Vol scaling
    resid_vol = reg["resid"].diff().rolling_std(20)
    vol_target = resid_vol.rolling_mean(252)
    vol_scale = (vol_target / resid_vol).fill_null(1.0).clip(0.2, 5.0)

    # Continuous sizing
    raw_size = z.abs() / z_entry
    size = (raw_size * vol_scale * conf).clip(0.1, 3.0)

    out = pl.DataFrame({
        "signal": z, "time_stop": rhl, "size": size,
        "confidence": conf, "vol": resid_vol,
    })

    # Suppress when not mean-reverting
    return out.with_columns(
        pl.when(pl.col("time_stop").is_not_null())
        .then(pl.col("signal"))
        .otherwise(None)
        .alias("signal")
    )


class SpreadBook:
    """Cross-sectional spread portfolio with ranking and allocation."""

    def __init__(self, config: BookConfig = None):
        self.config = config or BookConfig()
        self.spreads: list[SpreadDef] = []

    def add_spread(self, name: str, gen_x: str, gen_y: str,
                   otr_short: str, otr_long: str,
                   dv01_short: float = 1.0,
                   dv01_long: float = 1.0) -> SpreadBook:
        self.spreads.append(SpreadDef(
            name=name, gen_x=gen_x, gen_y=gen_y,
            otr_short=otr_short, otr_long=otr_long,
            dv01_short=dv01_short, dv01_long=dv01_long,
        ))
        return self

    def run(self, data: pl.DataFrame, lookback: int = 50,
            z_entry: float = 2.0) -> BacktestResult:
        """Run cross-sectional backtest.

        1. Compute signals for all spreads
        2. Cross-sectional rank by |z| × confidence
        3. Allocate top-K with risk parity
        4. Feed modified signals into Engine
        """
        names = [s.name for s in self.spreads]
        n = data.shape[0]

        # ── 1. Pre-compute raw signals for every spread ───────────────
        raw_frames: dict[str, pl.DataFrame] = {}
        for s in self.spreads:
            raw_frames[s.name] = _compute_spread_signals(
                data, s.gen_x, s.gen_y, lookback, z_entry,
            )

        # ── 2. Extract to mutable numpy for cross-sectional ranking ───
        sig = {nm: raw_frames[nm]["signal"].to_numpy() for nm in names}
        sz = {nm: raw_frames[nm]["size"].to_numpy() for nm in names}
        conf = {nm: raw_frames[nm]["confidence"].to_numpy() for nm in names}
        vol = {nm: raw_frames[nm]["vol"].to_numpy() for nm in names}

        # ── 3. Row-by-row cross-sectional ranking ─────────────────────
        for i in range(n):
            # Score each spread: |z| × confidence
            scores = {}
            for nm in names:
                s_val = sig[nm][i]
                c_val = conf[nm][i]
                if np.isnan(s_val) or np.isnan(c_val):
                    continue
                scores[nm] = abs(s_val) * c_val

            if not scores:
                continue

            # Top-K selection
            ranked = sorted(scores, key=scores.get, reverse=True)
            top_k = set(ranked[:self.config.max_spreads])

            # Allocate risk budget
            if self.config.allocation == "risk_parity":
                vols = {}
                for nm in top_k:
                    v = vol[nm][i]
                    vols[nm] = v if not np.isnan(v) and v > 0 else 1.0
                inv_vol_sum = sum(1.0 / v for v in vols.values())
                weights = {nm: (1.0 / vols[nm]) / inv_vol_sum for nm in top_k}
            elif self.config.allocation == "rank_weighted":
                total_score = sum(scores[nm] for nm in top_k)
                weights = {nm: scores[nm] / total_score for nm in top_k} if total_score > 0 else {nm: 1.0 / len(top_k) for nm in top_k}
            else:  # equal_weight
                weights = {nm: 1.0 / len(top_k) for nm in top_k}

            budget = self.config.total_risk_budget

            # Apply: scale top-K sizes, suppress non-top-K signals
            for nm in names:
                if nm in top_k:
                    if not np.isnan(sz[nm][i]):
                        sz[nm][i] *= weights[nm] * budget
                else:
                    sig[nm][i] = np.nan  # engine skips NaN signals

        # ── 4. Rebuild modified signal frames ─────────────────────────
        modified_frames: dict[str, pl.DataFrame] = {}
        for nm in names:
            rf = raw_frames[nm]
            modified_frames[nm] = rf.with_columns(
                pl.Series("signal", sig[nm]),
                pl.Series("size", sz[nm]),
            )

        # ── 5. Build Engine with wrapper pipelines ────────────────────
        dv01_map = {}
        for s in self.spreads:
            dv01_map[s.otr_short] = s.dv01_short
            dv01_map[s.otr_long] = s.dv01_long

        engine = Engine(BacktestConfig(
            transaction_cost_bps=self.config.transaction_cost_bps,
            slippage_bps=self.config.slippage_bps,
            max_total_positions=self.config.max_spreads * 2,
            dv01_map=DV01Map(dv01=dv01_map) if dv01_map else None,
        ))

        # Create pipelines whose compute_fn returns the pre-computed frame.
        # Pipeline.run() will extract "signal", run generate_signals(), then
        # re-attach extras (time_stop, size, etc.). This is correct because
        # we modified the signal column to suppress non-top-K entries.
        for s in self.spreads:
            frame = modified_frames[s.name]

            def _make_fn(f):
                def fn(d):
                    return f
                return fn

            trade_def = TradeDef.spread(s.name, s.otr_short, s.otr_long)
            pipeline = SignalPipeline(
                name=s.name,
                trade_def=trade_def,
                compute_fn=_make_fn(frame),
                config=SignalConfig(
                    entry_long=-z_entry,
                    entry_short=z_entry,
                    exit_long=None,
                    exit_short=None,
                    time_stop_bars=None,
                    max_positions=1,
                ),
            )
            engine.add_signal(pipeline)

        return engine.run(data)
