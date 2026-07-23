"""Live signal dashboard -- one card per promoted signal.

Static on load: shows whatever was last pulled / analyzed. Two buttons per
card drive everything else:

    Re-pull data       fresh Strategy.load_data(), cached to disk.
                        No ledger write.
    Re-run analysis     Strategy.compute() on the cached data, logged as
                        one timestamped row in the signal ledger.

No background loop or auto-refresh -- nothing changes until you click.

    python -m dashboard.registry --promote book.curve.tens_10s30s
    mamba run -n 2s10s python -m dashboard.app
    open http://127.0.0.1:8052

See README.md for the full workflow.
"""

from __future__ import annotations

import argparse
import re

import dash
from dash import Input, Output, ctx, html

from dashboard import runner
from dashboard.charts import level_chart, signal_chart
from dashboard.ledger import SignalLedger
from dashboard.registry import LiveRegistry
from utils.research_app import BORDER, DIM, ORANGE, PANEL, TEXT, make_app, run, stat_block

REGISTRY = LiveRegistry()
LEDGER = SignalLedger()


def _slug(module: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", module.lower()).strip("-")


def _param_summary(row: dict) -> str:
    bits = [f"{row['entry_signal']} beta_lb={row['beta_lb']}"]
    if row.get("ou_lb"):
        bits.append(f"ou_lb={row['ou_lb']}")
    bits.append(f"entry={row['entry_threshold']:g}")
    bits.append(f"exit={row['exit_style']}/{row['exit_param']:g}")
    if row.get("gate") and row["gate"] != "(none)":
        bits.append(f"gate={row['gate']}")
    return "  ·  ".join(bits)


def _card_body(module: str) -> html.Div:
    row = REGISTRY.get(module)
    if row is None:
        return html.Div(f"{module}: no longer promoted", style={"color": DIM})

    ledger_row = LEDGER.latest(module)
    try:
        state = runner.compute_signal(module)
        error = None
    except RuntimeError as exc:
        state = None
        error = str(exc)

    stats = [
        stat_block("data as-of", str(state["data_asof"]) if state else "—"),
        stat_block("last analysis run", ledger_row["run_ts"] if ledger_row else "never"),
        stat_block(
            "reading",
            state["fired"] if state else "—",
            alert=bool(state and state["fired"] not in ("flat", "flat (gated)")),
        ),
        stat_block("params", _param_summary(row)),
    ]

    if error:
        body = [html.Div(error, style={"color": ORANGE, "padding": "12px 0"})]
    else:
        level_png = level_chart(state["data"], state["strategy"].target)
        sig_png = signal_chart(
            state["data"], state["signal_frame"],
            state["params"]["entry_signal"], state["params"]["entry_threshold"],
        )
        body = [
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": 10},
                children=[
                    html.Img(src=f"data:image/png;base64,{level_png}",
                              style={"width": "100%", "border": f"1px solid {BORDER}"}),
                    html.Img(src=f"data:image/png;base64,{sig_png}",
                              style={"width": "100%", "border": f"1px solid {BORDER}"}),
                ],
            )
        ]

    return html.Div([
        html.Div(stats, style={"display": "flex", "gap": 28, "flexWrap": "wrap",
                                "padding": "8px 2px 14px"}),
        *body,
    ])


def _btn_style(primary: bool = False) -> dict:
    return {
        "padding": "6px 14px", "fontSize": 12, "cursor": "pointer",
        "border": f"1px solid {ORANGE if primary else BORDER}",
        "background": ORANGE if primary else "#FFFFFF",
        "color": "#FFFFFF" if primary else TEXT,
        "borderRadius": 3,
    }


def _card(row: dict) -> html.Div:
    module = row["module"]
    slug = _slug(module)
    return html.Div(
        style={"border": f"1px solid {BORDER}", "background": PANEL,
               "padding": "14px 18px", "marginBottom": 18},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "baseline", "gap": 12,
                       "marginBottom": 4},
                children=[
                    html.Span(row["name"], style={"fontSize": 15, "fontWeight": "bold",
                                                    "color": ORANGE}),
                    html.Span(f"{row['target']} ~ {row['feature']}",
                              style={"fontSize": 12, "color": DIM}),
                    html.Span(module, style={"fontSize": 11, "color": DIM,
                                              "marginLeft": "auto", "fontStyle": "italic"}),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": 8, "marginBottom": 4},
                children=[
                    html.Button("Re-pull data", id=f"pull-{slug}", n_clicks=0,
                                style=_btn_style()),
                    html.Button("Re-run analysis", id=f"run-{slug}", n_clicks=0,
                                style=_btn_style(primary=True)),
                ],
            ),
            html.Div(id=f"card-body-{slug}", children=_card_body(module)),
        ],
    )


def build_app() -> dash.Dash:
    reg_df = REGISTRY.list()
    rows = [] if reg_df.is_empty() else reg_df.sort("family", "name").to_dicts()

    body = html.Div(
        style={"padding": "18px 24px"},
        children=(
            [html.Div(
                "No signals promoted yet -- "
                "python -m dashboard.registry --promote <module>",
                style={"color": DIM, "padding": "24px 0"},
            )]
            if not rows else [_card(row) for row in rows]
        ),
    )

    app = make_app(
        title="LIVE",
        subtitle="promoted signals -- data as-of / last analysis run / current reading",
        data_info=f"{len(rows)} live signal{'s' if len(rows) != 1 else ''}",
        sliders=[],
        body=body,
    )

    for row in rows:
        module = row["module"]
        slug = _slug(module)

        def _update(_pull_clicks, _run_clicks, module=module):
            trigger = ctx.triggered_id or ""
            if trigger.startswith("pull-"):
                runner.pull_data(module)
            elif trigger.startswith("run-"):
                runner.run_analysis(module)
            return _card_body(module)

        app.callback(
            Output(f"card-body-{slug}", "children"),
            Input(f"pull-{slug}", "n_clicks"),
            Input(f"run-{slug}", "n_clicks"),
            prevent_initial_call=True,
        )(_update)

    return app


app = build_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    run(app, port=args.port, host=args.host)
