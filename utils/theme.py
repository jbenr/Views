"""Dark-mode palette for the Dash apps, loaded from utils/theme.toml.

The TOML file is the one place dark mode's colours are defined. ``DARK`` holds
the page palette and ``DARK_CHART`` the matplotlib restyle; ``css_variables``
renders the page palette as the CSS custom properties theme.css reads.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

with open(Path(__file__).with_name("theme.toml"), "rb") as _f:
    _CONFIG = tomllib.load(_f)

DARK_CHART: dict[str, str] = _CONFIG["dark"]["chart"]
DARK: dict[str, str] = {k: v for k, v in _CONFIG["dark"].items() if k != "chart"}


def css_variables() -> str:
    """The dark palette as custom properties, set while the page is in dark mode."""
    props = " ".join(f"--theme-{k.replace('_', '-')}: {v};" for k, v in DARK.items())
    return f'html[data-theme="dark"] {{ {props} }}'
