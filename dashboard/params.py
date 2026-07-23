"""Shared param plumbing between the live registry and the runner.

Registry rows store a strategy's sweep_results columns verbatim, including
`gate` as the stringified tuple repr the sweep grid produces (e.g.
"('beta_cv', 'tails_25_75')" or "(none)"). params_from_row() reconstructs
the dict Strategy.compute() expects.
"""

from __future__ import annotations

import ast

PARAM_COLS = [
    "entry_signal", "beta_lb", "ou_lb", "entry_threshold",
    "exit_style", "exit_param", "gate", "z_gate",
    "half_life_min", "half_life_max", "stop_loss_bps",
]


def unpack_gate(value):
    if value is None or value == "(none)":
        return None
    return ast.literal_eval(value)


def params_from_row(row: dict) -> dict:
    """Rebuild a Strategy.compute()-ready params dict from a registry row."""
    params = {c: row.get(c) for c in PARAM_COLS if row.get(c) is not None}
    if "gate" in params:
        params["gate"] = unpack_gate(params["gate"])
    if "beta_lb" in params:
        params["beta_lb"] = int(params["beta_lb"])
    if "ou_lb" in params:
        params["ou_lb"] = int(params["ou_lb"])
    return params
