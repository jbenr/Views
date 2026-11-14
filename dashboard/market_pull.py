"""One background Bloomberg live pull, with a bounded console for the Market tab."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from uuid import uuid4

from dash import Input, Output, State, ctx, dcc, html, no_update

from dashboard import markets

ROOT = Path(__file__).resolve().parents[1]
COMMAND = [sys.executable, "-u", str(ROOT / "data_pull.ref"), "--live"]
MAX_OUTPUT = 64_000
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def clean_output(text):
    """Keep actual output as plain text, with connection credentials redacted."""
    text = _ANSI.sub("", text)
    text = re.sub(r"(?i)(postgres(?:ql)?://)[^\s/'\"]+@", r"\1[redacted]@", text)
    text = re.sub(r"(?i)(password\s*=\s*)(?:'[^']*'|\"[^\"]*\"|[^\s;]+)", r"\1[redacted]", text)
    return text


class LivePull:
    """Shared by browser sessions in this dashboard process; no automatic retries."""
    def __init__(self):
        self.lock = threading.Lock()
        self.state = dict(id=None, status="idle", output="", started=None, finished=None, exit_code=None)

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def start(self):
        with self.lock:
            if self.state["status"] == "running":
                return
            self.state = dict(id=uuid4().hex, status="running", output="", started=time.time(),
                              finished=None, exit_code=None)
        # A normal app shutdown waits for this job instead of abandoning its child.
        threading.Thread(target=self._run, name="bloomberg-live-pull", daemon=False).start()

    def _append(self, text, log):
        text = clean_output(text)
        log.write(text)
        log.flush()
        with self.lock:
            self.state["output"] = (self.state["output"] + text)[-MAX_OUTPUT:]

    def _run(self):
        process = None
        status = "failed"
        try:
            store = Path(os.getenv("VIEWS_STORE_DIR", str(ROOT / "store")))
            logs = store / "market_pulls"
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / f"{self.state['id']}.log").open("w", encoding="utf-8") as log:
                self._append("$ python data_pull.ref --live\n", log)
                process = subprocess.Popen(
                    COMMAND, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                with process.stdout:
                    for line in process.stdout:
                        self._append(line, log)
                code = process.wait()
                with self.lock:
                    self.state["exit_code"] = code
                if code:
                    self._append(f"\n[failed] Bloomberg pull exited with code {code}.\n", log)
                    status = "failed"
                else:
                    self._append("\nRefreshing Market board from the updated database...\n", log)
                    try:
                        markets.pull()
                    except Exception as exc:
                        self._append(f"[failed] Pull completed, but Market board refresh failed: {type(exc).__name__}: {exc}\n", log)
                        status = "failed"
                    else:
                        self._append("[succeeded] Bloomberg live pull and Market board refresh completed.\n", log)
                        status = "succeeded"
        except Exception as exc:
            with self.lock:
                self.state["output"] = (self.state["output"] + clean_output(
                    f"\n[failed] {type(exc).__name__}: {exc}\n"))[-MAX_OUTPUT:]
            status = "failed"
        finally:
            if process is not None and process.poll() is None:
                # Preserve ownership of the pull until it finishes, even if logging failed.
                process.wait()
            with self.lock:
                self.state.update(status=status, finished=time.time())


PULL = LivePull()


def layout():
    return html.Div([
        html.Div([
            html.Button("Pull Bloomberg live", id="market-bbg-pull", n_clicks=0, className="ref-btn"),
            html.Span(id="market-bbg-status", role="status"),
        ], className="sd-bbg-controls"),
        html.Details([
            html.Summary("Pull output"),
            html.Pre(id="market-bbg-output", className="sd-bbg-console"),
        ], id="market-bbg-console-panel", className="sd-bbg-output", open=False,
           style={"display": "none"}),
        dcc.Interval(id="market-bbg-poll", interval=1000),
        dcc.Store(id="market-bbg-seen"),
        dcc.Store(id="market-bbg-phase"),
    ], className="sd-bbg-panel")


def register_callbacks(app, render_market):
    @app.callback(
        Output("market-bbg-output", "children"), Output("market-bbg-status", "children"),
        Output("market-bbg-pull", "disabled"), Output("market-bbg-seen", "data"),
        Output("market-content", "children", allow_duplicate=True),
        Output("market-bbg-console-panel", "style"), Output("market-bbg-console-panel", "open"),
        Output("market-bbg-phase", "data"),
        Input("market-bbg-pull", "n_clicks"), Input("market-bbg-poll", "n_intervals"),
        State("market-bbg-seen", "data"), State("market-bbg-phase", "data"), prevent_initial_call=True,
    )
    def update(_clicks, _ticks, seen, previous_phase):
        if ctx.triggered_id == "market-bbg-pull" and _clicks:
            PULL.start()
        state = PULL.snapshot()
        running = state["status"] == "running"
        elapsed = (state["finished"] or time.time()) - state["started"] if state["started"] else 0
        label = "" if state["status"] == "idle" else f"{state['status'].capitalize()} · {elapsed:.0f}s"
        if state["exit_code"] is not None:
            label += f" · exit {state['exit_code']}"
        refresh = state["status"] == "succeeded" and seen != state["id"]
        output = state["output"] or ("Waiting for Bloomberg output..." if running else "Click Pull Bloomberg live to start.")
        phase = [state["id"], state["status"]]
        # Only change expansion at run transitions; polling must preserve manual toggles.
        expanded = running if phase != previous_phase else no_update
        return (output,
                label, running, state["id"] if refresh else seen,
                render_market() if refresh else no_update,
                {"display": "none"} if state["status"] == "idle" else {}, expanded, phase)
