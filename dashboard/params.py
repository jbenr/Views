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
    "exit_style", "exit_param", "gate", "gate_window", "z_gate",
    "half_life_min", "half_life_max", "stop_loss_bps",
]

# Null is a meaningful, explicit "disabled" value only for these filters.
# Other null registry cells commonly come from schema-unioning heterogeneous
# signals and must not override a strategy's required/default parameter.
NULLABLE_FILTER_COLS = {
    "gate",
    # explicit null = expanding percentile, which must not fall back to a
    # module default that happens to name a rolling window
    "gate_window",
    "z_gate",
    "half_life_min",
    "half_life_max",
}


def unpack_gate(value):
    if value is None or value == "(none)":
        return None
    return ast.literal_eval(value)


def params_from_row(row: dict) -> dict:
    """Rebuild a Strategy.compute()-ready params dict from a registry row."""
    # Presence and value are distinct here. An explicit null disables optional
    # filters such as z_gate and the half-life bounds; dropping that key would
    # let Strategy._params() silently restore the module default.
    params = {
        c: row.get(c)
        for c in PARAM_COLS
        if c in row and (row.get(c) is not None or c in NULLABLE_FILTER_COLS)
    }
    if "gate" in params:
        params["gate"] = unpack_gate(params["gate"])
    if params.get("beta_lb") is not None:
        params["beta_lb"] = int(params["beta_lb"])
    if params.get("ou_lb") is not None:
        params["ou_lb"] = int(params["ou_lb"])
    if params.get("gate_window") is not None:
        params["gate_window"] = int(params["gate_window"])
    return params
