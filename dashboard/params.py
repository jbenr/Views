"""Shared param plumbing between the live registry and the runner.

Registry rows store a strategy's sweep_results columns verbatim, including
`gate` as the stringified tuple repr the sweep grid produces (e.g.
"('beta_cv', 'tails_25_75')" or "(none)"). params_from_row() reconstructs
the dict Strategy.compute() expects.

The label helpers live here rather than in the dashboard so the registry CLI
can describe a promoted signal without importing Dash, and so one signal reads
the same in `registry --list` as it does on the app.
"""

from __future__ import annotations

import ast

PARAM_COLS = [
    "entry_signal", "beta_lb", "ou_lb", "entry_threshold",
    "exit_style", "exit_param", "gate", "gate_window", "z_gate",
    "stop_loss_bps",
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
}


def unpack_gate(value):
    if value is None or value == "(none)":
        return None
    return ast.literal_eval(value)


def params_from_row(row: dict) -> dict:
    """Rebuild a Strategy.compute()-ready params dict from a registry row."""
    # Presence and value are distinct here. An explicit null disables an
    # optional filter such as z_gate; dropping that key would let
    # Strategy._params() silently restore the module default.
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


# ── labels ───────────────────────────────────────────────────────────────────

def auto_label(row: dict) -> str:
    """Params-derived label for a promotion with no curated variant name, in
    the same shorthand curated labels use: the lookback that defines the
    signal plus its entry threshold."""
    if row["entry_signal"] == "residual":
        return f"RES{int(row['beta_lb'])} · {float(row['entry_threshold']):g}bps"
    return f"OU{int(row['ou_lb'])} · {float(row['entry_threshold']):g}z"


def signal_label(row: dict) -> str:
    """Canonical user-facing signal name: input feature first, traded target
    second -- the same `<input>_<target>` order the strategy modules use, so
    a name reads identically on the app and in `registry --list`. Always
    suffixed with a label: curated variants use their own, every other
    promotion gets one derived from its frozen params."""
    base = f"{row['feature']}_{row['target']}"
    return f"{base} · {row.get('variant_label') or auto_label(row)}"


def entry_label(params: dict) -> str:
    """The trigger, in the units the signal is measured in."""
    units = "bps" if params["entry_signal"] == "residual" else "z"
    return f"|{params['entry_signal']}| >= {float(params['entry_threshold']):g}{units}"


def exit_label(params: dict) -> str:
    """The primary exit rule, spelled out."""
    style = params["exit_style"]
    param = float(params["exit_param"])
    if style == "revert_frac":
        return (
            f"revert_frac={param:g} "
            f"(after {param:.0%} of entry signal reverts toward zero)"
        )
    if style == "half_life_frac":
        return f"half_life_frac={param:g} ({param:g}× entry half-life)"
    if style == "band":
        units = "bps" if params["entry_signal"] == "residual" else "z"
        return f"band={param:g}{units} (inside ±{param:g}{units})"
    return f"{style}={param:g}"


def gate_label(params: dict) -> str:
    """What the gate tests: condition, bucket, and what the percentile is
    measured against. Without the basis the same condition and bucket can
    mean two different gates."""
    gate = params.get("gate")
    if gate is None:
        return "none"
    condition, bucket = gate[0], gate[1]
    window = params.get("gate_window")
    basis = "expanding" if window is None else f"roll {int(window)}d"
    return f"{condition} · {bucket} · {basis}"


def filter_label(params: dict) -> str:
    """The optional entry filter, including when it is switched off -- an
    audit record has to state what is not being applied too."""
    z_gate = params.get("z_gate")
    return f"z_gate={'off' if z_gate is None else format(float(z_gate), 'g')}"
