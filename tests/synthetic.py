"""Synthetic panel generators — test fixtures only.

Production code has no synthetic path: strategy modules load from the DB and
nothing else. This module is the only place fake data is constructed, so a
test injects it by replacing the loader rather than by flipping a flag:

    monkeypatch.setattr(mod.STRATEGY, "load_data", lambda: pair_panel(mod))

Each generator returns a frame whose schema matches what the corresponding
load_data() returns. Structure is deliberate, not noise — a feature plus an
OU residual with a ~14d half-life — so a signal designed to fade that residual
demonstrably trades.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

START = "2010-01-01"


def _weekdays(n: int, start: str = START) -> pl.Series:
    """n weekday timestamps starting at `start`."""
    start_date = dt.date.fromisoformat(start)
    ts = pl.date_range(
        start_date, start_date + dt.timedelta(days=2 * n), interval="1d", eager=True
    )
    return ts.filter(ts.dt.weekday() <= 5)[:n]


def _ou(rng, n: int, sigma: float = 2.0, theta: float = 0.05) -> np.ndarray:
    """OU residual path; half-life is ln2/theta, ~14d at the default theta."""
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = r[i - 1] * (1 - theta) + rng.normal(0.0, sigma)
    return r


def synthetic_pair(
    target: str,
    feature: str,
    n: int = 1500,
    seed: int = 21,
    start: str = START,
    feature_level: float = 350.0,
    target_base: float = 50.0,
    beta: float = 0.25,
) -> pl.DataFrame:
    """Standard (feature, target) pair: target explained by the feature plus
    an OU residual."""
    rng = np.random.default_rng(seed)
    x = feature_level + np.cumsum(rng.normal(0.0, 2.0, n))
    y = target_base + beta * (x - feature_level) + _ou(rng, n)
    return pl.DataFrame({"ts": _weekdays(n, start), feature: x, target: y}).with_columns(
        pl.col([feature, target]).round(2)
    )


# Per-module pair overrides: seed, and the levels the module's legs sit at.
# Modules absent from this table use the synthetic_pair defaults.
PAIR_SPECS: dict[str, dict] = {
    "book.curve.tens_10s30s": {"seed": 21},
    "book.curve.fives_5s7s": {"seed": 35},
    "book.curve.fives_5s10s": {"seed": 36},
    "book.curve.fives_5s30s": {"seed": 37},
    "book.curve.sevens_7s10s": {"seed": 38},
    "book.curve.twenties_20s30s": {"seed": 40},
    "book.curve.twos_2s5s": {"seed": 31},
    "book.curve.twos_2s7s": {"seed": 32},
    "book.curve.twos_2s10s": {"seed": 33},
    "book.curve.twos_2s30s": {"seed": 34},
    "book.curve.twos_10s30s": {"seed": 31, "feature_level": 250.0, "target_base": 50.0},
    "book.curve.real10y_2s10s": {
        "seed": 11,
        "feature_level": 150.0,
        "target_base": 80.0,
    },
}


def pair_panel(mod, n: int = 1500, **overrides) -> pl.DataFrame:
    """Pair panel matching a single-pair strategy module's schema."""
    spec = {**PAIR_SPECS.get(mod.__name__, {}), **overrides}
    return synthetic_pair(mod.TARGET, mod.FEATURE, n=n, **spec)


def panel_for(mod, n: int = 1500) -> pl.DataFrame:
    """Raw panel for any strategy module — the shape its load_data() returns,
    before the feature hook runs."""
    if mod.__name__ == "book.curve.pc1_10s30s":
        return pc1_panel(n=n)
    return pair_panel(mod, n=n)


def use(monkeypatch, strategy, panel: pl.DataFrame) -> None:
    """Point a Strategy's loader at a synthetic panel.

    Mirrors Strategy.load_data(): the feature hook and the rounding applied to
    DB data are applied here too, so the test sees the same frame the live
    path would produce.
    """

    def _load(start=None):
        data = strategy.feature_fn(panel) if strategy.feature_fn is not None else panel
        return data.with_columns(pl.col(strategy.model_columns).round(2))

    monkeypatch.setattr(strategy, "load_data", _load)


def pc1_panel(n: int = 1500, seed: int = 13) -> pl.DataFrame:
    """Correlated yield panel with a common level factor; 10s30s explained by
    that factor plus an OU residual. Matches book.curve.pc1_10s30s."""
    rng = np.random.default_rng(seed)
    level = np.cumsum(rng.normal(0.0, 2.0, n))
    yields = {
        "2y": 150.0 + level + np.cumsum(rng.normal(0.0, 0.5, n)),
        "5y": 250.0 + level + np.cumsum(rng.normal(0.0, 0.5, n)),
        "10y": 350.0 + level + np.cumsum(rng.normal(0.0, 0.5, n)),
        "30y": 400.0 + level + np.cumsum(rng.normal(0.0, 0.5, n)),
    }
    slope = 50.0 + 0.25 * level + _ou(rng, n)
    return pl.DataFrame({"ts": _weekdays(n), **yields, "10s30s": slope})


def template_panel(n: int = 1500, seed: int = 7) -> pl.DataFrame:
    """Anchor random walk plus a beta-linked target with an OU residual.
    Matches book.rate_vol.template."""
    rng = np.random.default_rng(seed)
    anchor = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    target = 20.0 + 0.8 * anchor + _ou(rng, n, sigma=1.0)
    return pl.DataFrame(
        {"ts": _weekdays(n, "2015-01-01"), "target": target, "anchor": anchor}
    )


def xy_panel(tickers: dict, n: int = 1500, seed: int = 7) -> pl.DataFrame:
    """Wide panel for book.curve.xy_scan: correlated yield levels, curves as
    spreads plus OU residuals, breakevens as their own walk.

    Swap tenors are read off `tickers` rather than listed here, so a ticker
    added later cannot leave this panel silently short a column.
    """
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
    curves = {
        "2s5s": yields["5y"] - yields["2y"] + _ou(rng, n),
        "2s10s": yields["10y"] - yields["2y"] + _ou(rng, n),
        "2s30s": yields["30y"] - yields["2y"] + _ou(rng, n),
        "5s10s": yields["10y"] - yields["5y"] + _ou(rng, n),
        "5s30s": yields["30y"] - yields["5y"] + _ou(rng, n),
        "10s30s": yields["30y"] - yields["10y"] + _ou(rng, n),
    }

    def _anchor(tenor: int) -> str:
        return min(yields, key=lambda k: abs(int(k[:-1]) - tenor))

    ois = {
        k: yields[_anchor(int(k[3:]))] + walk(0.0, 0.3)
        for k in tickers if k.startswith("ois")
    }
    zc = {
        k: 200.0 + 2.0 * int(k[2:]) + walk(0.0, 0.6)
        for k in tickers if k.startswith("zc")
    }

    return pl.DataFrame(
        {
            "ts": _weekdays(n),
            **yields,
            **be,
            **ois,
            **zc,
            "real10y": yields["10y"] - be["be10"],
            **curves,
        }
    )


def research_panel(n: int = 1500, seed: int = 41) -> pl.DataFrame:
    """Causal panel with conditional, RV, and fair-value structure, for the
    10s30s research files.

    It only proves that those files wire the intended calculations together.
    It is not evidence for a live 10s30s relationship.
    """
    rng = np.random.default_rng(seed)
    level = 350.0 + np.cumsum(rng.normal(0.0, 1.8, n))
    inflation = 220.0 + np.cumsum(rng.normal(0.0, 0.45, n))
    move = np.empty(n)
    move[0] = 100.0
    for i in range(1, n):
        move[i] = 100.0 + 0.93 * (move[i - 1] - 100.0) + rng.normal(0.0, 2.0)

    # Curve-specific state: the common building block all three methods should
    # be able to see from a different angle.
    residual = np.zeros(n)
    for i in range(1, n):
        residual[i] = 0.94 * residual[i - 1] + rng.normal(0.0, 1.8)

    be5 = inflation - 8.0 + np.cumsum(rng.normal(0.0, 0.15, n))
    be10 = inflation + np.cumsum(rng.normal(0.0, 0.15, n))
    fivey_fivey_infl = 2.0 * be10 - be5
    curve = 45.0 + 0.12 * level + 0.10 * fivey_fivey_infl + 0.08 * move + residual

    return pl.DataFrame(
        {
            "ts": _weekdays(n),
            "10y": level,
            "10s30s": curve,
            "real10y": level - be10,
            "be5": be5,
            "be10": be10,
            "move": move,
            "5y5y_infl": fivey_fivey_infl,
        }
    )


def roll_panel(roots: list[str], n: int = 1500, seed: int = 51) -> pl.DataFrame:
    """Front/deferred futures roll with mean-reverting dislocations, in the
    long shape book.basis.fut_roll.futures_roll() returns."""
    rng = np.random.default_rng(seed)
    start_date = dt.date.fromisoformat(START)
    dates = pl.date_range(
        start_date, start_date + dt.timedelta(days=n - 1), interval="1d", eager=True
    ).to_list()

    rows = []
    cycle = 63
    quarter_codes = ("H", "M", "U", "Z")
    for root_i, root in enumerate(roots):
        front_px = 108.0 + root_i * 1.5 + np.cumsum(rng.normal(0.0, 0.08, n))
        resid = _ou(rng, n, sigma=0.08, theta=0.07)
        seasonal = 0.12 * np.sin(np.arange(n) / cycle * 2.0 * np.pi)
        roll = -0.25 - 0.03 * root_i + seasonal + resid

        for i, ts in enumerate(dates):
            pair = i // cycle
            year_digit = (start_date.year + pair // 4) % 10
            front_delivery = start_date + dt.timedelta(days=(pair + 1) * cycle)
            deferred_delivery = start_date + dt.timedelta(days=(pair + 2) * cycle)
            rows.append(
                {
                    "ts": ts,
                    "root": root,
                    "front_contract": f"{root}{quarter_codes[pair % 4]}{year_digit}",
                    "deferred_contract": (
                        f"{root}{quarter_codes[(pair + 1) % 4]}{year_digit}"
                    ),
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
