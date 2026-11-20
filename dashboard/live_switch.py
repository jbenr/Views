"""The paper/live switch at the top of the Portfolio tab: flips the dashboard's IBKR connection between the
paper and the live account. Going live takes the passcode (VIEWS_LIVE_PASSCODE in the
environment or .env) typed on a keypad; five misses lock the switch for 15 minutes.
Back to paper needs no code. Live is read-only: the session never trades there."""
from __future__ import annotations

import hmac
import time

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from dashboard.portfolio import SESSION
from utils.helpers import env_value

PASSCODE_ENV = "VIEWS_LIVE_PASSCODE"
MAX_TRIES, LOCKOUT, MAX_DIGITS = 5, 15 * 60, 12
KEY_ROWS = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["clear", "0", "back"]]  # a phone's keypad
LIVE, PAPER = "#E74C3C", "#27AE60"
HIDDEN, SHOWN = {"display": "none"}, {"display": "flex"}


class PasscodeGate:
    """Checks the live passcode, counting misses across every browser on this server."""

    def __init__(self, clock=time.monotonic):
        self._clock, self.misses, self.locked_until = clock, 0, 0.0

    def locked_for(self) -> float:
        """Seconds left on a lockout, 0 when open."""
        return max(0.0, self.locked_until - self._clock())

    def check(self, typed: str) -> tuple[bool, str]:
        """(accepted, what to tell the person at the keypad)."""
        if self.locked_for():
            return False, f"Locked: try again in {self.locked_for() / 60:.0f} min."
        expected = env_value(PASSCODE_ENV)
        if not expected:
            return False, f"No live passcode set: add {PASSCODE_ENV}=<digits> to .env."
        if hmac.compare_digest(typed.encode(), expected.encode()):
            self.misses = 0
            return True, ""
        self.misses += 1
        if self.misses >= MAX_TRIES:
            self.misses, self.locked_until = 0, self._clock() + LOCKOUT
            return False, f"Wrong passcode. Locked for {LOCKOUT // 60} min."
        return False, f"Wrong passcode: {MAX_TRIES - self.misses} tries left."


GATE = PasscodeGate()


def press(digits: str, key: str) -> str:
    """The typed digits after one keypad key."""
    if key == "clear":
        return ""
    if key == "back":
        return digits[:-1]
    return (digits + key)[:MAX_DIGITS]


def _switch(mode: str) -> list:
    """The mode as it stands and the button that flips it."""
    live = mode == "live"
    return [html.Span("LIVE · read-only" if live else "PAPER", className="sd-mode-label",
                      style={"color": LIVE if live else PAPER}),
            html.Button("Switch to paper" if live else "Switch to live", id="mode-flip", n_clicks=0,
                        className="ref-btn sd-btn")]


def _keypad() -> html.Div:
    return html.Div([
        html.Div("GO LIVE", className="sd-bar-title"),
        html.Div("Enter the live passcode. Live is read-only: orders still go to paper only.", className="sd-note"),
        html.Div("", id="keypad-display", className="sd-keypad-display", **{"aria-live": "polite"}),
        html.Div([html.Div([html.Button({"clear": "C", "back": "⌫"}.get(key, key), id={"type": "keypad", "key": key},
                                        n_clicks=0, className="sd-key", **{"aria-label": key}) for key in row],
                           className="sd-keypad-row") for row in KEY_ROWS],
                 className="sd-keypad"),
        html.Div(className="sd-bar", children=[
            html.Button("Enter", id="keypad-enter", n_clicks=0, className="ref-btn sd-btn"),
            html.Button("Cancel", id="keypad-cancel", n_clicks=0, className="ref-btn sd-btn"),
        ]),
        html.Div(id="keypad-message", role="status", className="sd-note"),
    ], className="sd-modal-box")


def layout() -> html.Div:
    return html.Div([
        html.Div(_switch(SESSION.mode), id="mode-switch", className="sd-mode"),
        html.Div(id="mode-message", role="status", className="sd-note"),
        dcc.Store(id="keypad-digits", data=""),
        html.Div(_keypad(), id="keypad-modal", className="sd-modal sd-modal--center", style=HIDDEN),
    ], className="sd-mode-bar")


def register_callbacks(app) -> None:
    @app.callback(Output("keypad-modal", "style"), Output("mode-switch", "children"),
                  Output("mode-message", "children"), Output("keypad-digits", "data"),
                  Output("keypad-message", "children"),
                  Input("mode-flip", "n_clicks"), Input("keypad-cancel", "n_clicks"), prevent_initial_call=True)
    def flip(clicks, _cancel):
        """Paper -> the keypad; live -> straight back to paper; cancel closes the keypad."""
        if ctx.triggered_id == "keypad-cancel":
            return HIDDEN, no_update, "", "", ""
        if not clicks:
            return (no_update,) * 5
        if SESSION.mode == "paper":
            return SHOWN, no_update, "", "", ""
        try:
            SESSION.set_mode("paper")
        except Exception as exc:
            return no_update, _switch(SESSION.mode), f"Not switched ({type(exc).__name__}: {exc})", "", ""
        return HIDDEN, _switch(SESSION.mode), "Paper: reconnecting to the paper account.", "", ""

    @app.callback(Output("keypad-digits", "data", allow_duplicate=True),
                  Input({"type": "keypad", "key": ALL}, "n_clicks"), State("keypad-digits", "data"),
                  prevent_initial_call=True)
    def key(_clicks, digits):
        if not isinstance(ctx.triggered_id, dict) or not ctx.triggered[0]["value"]:
            return no_update
        return press(digits or "", ctx.triggered_id["key"])

    @app.callback(Output("keypad-display", "children"), Input("keypad-digits", "data"))
    def dots(digits):
        return "•" * len(digits or "")

    @app.callback(Output("keypad-modal", "style", allow_duplicate=True),
                  Output("mode-switch", "children", allow_duplicate=True),
                  Output("mode-message", "children", allow_duplicate=True),
                  Output("keypad-digits", "data", allow_duplicate=True),
                  Output("keypad-message", "children", allow_duplicate=True),
                  Input("keypad-enter", "n_clicks"), State("keypad-digits", "data"), prevent_initial_call=True)
    def enter(clicks, digits):
        if not clicks:
            return (no_update,) * 5
        accepted, why = GATE.check(digits or "")
        if not accepted:
            return no_update, no_update, no_update, "", why
        try:
            SESSION.set_mode("live")
        except Exception as exc:
            return HIDDEN, _switch(SESSION.mode), f"Not switched ({type(exc).__name__}: {exc})", "", ""
        return (HIDDEN, _switch(SESSION.mode),
                "Live: IB Gateway is logging in to the live account; approve 2FA on your phone. Read-only.", "", "")
