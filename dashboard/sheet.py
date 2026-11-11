"""The dashboard's spreadsheet pieces: plain gridded ranges, styled in utils/assets/dashboard.css.

Shared by the tabs in dashboard.app and the broker tab in dashboard.portfolio, so a
label | value | note range or a gridded table reads the same on every tab.
"""

from __future__ import annotations

from dash import html

from utils.viz import table_div


def usd(value) -> str:
    """Dollars with a sign and no cents: '+$1,704', '-$26'; '—' when missing."""
    if value is None or value != value:
        return "—"
    return f"{'-' if value < 0 else '+'}${abs(value):,.0f}"


def heading(title: str, note_text: str = "") -> html.Div:
    """A section's title on a rule, with a quiet note beside it."""
    return html.Div([title, *([html.Span(note_text, className="sd-h-note")] if note_text else [])],
                    className="sd-h")


def kv(label: str, value, note_text: str = "", colour: str | None = None) -> html.Div:
    """One row of a label | value | note range (see kv_range)."""
    return html.Div([
        html.Span(label, className="sd-k"),
        html.Span(value, className="sd-v", style={"color": colour} if colour else None),
        html.Span(note_text, className="sd-n"),
    ])


def kv_range(rows: list) -> html.Div:
    """kv rows as one gridded range, the way a block of spreadsheet cells reads."""
    return html.Div(rows, className="sd-kv")


def grid_table(frame, **kwargs) -> html.Div:
    """The house table with every cell ruled, sized to its content; it scrolls sideways
    in its own box when the screen is narrower. ``kwargs`` go to utils.viz.table_div."""
    kwargs.setdefault("max_rows", len(frame))
    return html.Div(table_div(frame, **kwargs), className="sd-grid")
