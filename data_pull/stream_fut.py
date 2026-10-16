#!/usr/bin/env python3
"""
Live top-of-book stream for front-month tsy futures (//blp/mktdata).

Display only -- nothing is written to the DB. Redraws a board in place
until Ctrl-C.

Usage:
    python stream_fut.py
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

if os.name == "nt":
    os.system("")

_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_CYAN  = "\033[96m"
_RESET = "\033[0m"
_HOME  = "\033[H\033[J"

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # berg lives here; needed under `python -m`
from berg import Bbg
from utils import format_tsy_price

# front-month tsy futures, with min tick in 32nds
TSY_TICKS = {"TU": 0.125, "FV": 0.25, "TY": 0.5, "UXY": 0.5, "US": 1, "WN": 1}
FIELDS    = ["BID", "ASK", "BID_SIZE", "ASK_SIZE", "LAST_PRICE"]
REDRAW_S  = 0.25


def _px(v, tick: float) -> str:
    return format_tsy_price(v, is_ticks=False, tick_32nds=tick) if v is not None else "n/a"


def _sz(v) -> str:
    return f"{int(v):,}" if v is not None else "-"


def render(book: dict[str, dict], last_tick: dt.datetime | None) -> str:
    stamp = last_tick.astimezone(ET).strftime("%H:%M:%S") if last_tick else "--:--:--"
    lines = [
        f"  {_BOLD}tsy futs{_RESET}  {_DIM}last tick {stamp} ET  ·  Ctrl-C to quit{_RESET}",
        "",
        f"  {_DIM}{'':<4}{'bsz':>7}  {'bid':>7}   {'ask':<7}  {'asz':<7} {'last':>7}{_RESET}",
    ]
    for root, tick in TSY_TICKS.items():
        q = book.get(f"{root}1 Comdty", {})
        lines.append(
            f"  {_CYAN}{root:<4}{_RESET}"
            f"{_DIM}{_sz(q.get('BID_SIZE')):>7}{_RESET}  "
            f"{_BOLD}{_px(q.get('BID'), tick):>7} / {_px(q.get('ASK'), tick):<7}{_RESET}  "
            f"{_DIM}{_sz(q.get('ASK_SIZE')):<7}{_RESET} "
            f"{_px(q.get('LAST_PRICE'), tick):>7}"
        )
    return "\n".join(lines)


def main():
    import argparse
    argparse.ArgumentParser().parse_args()

    tickers = [f"{root}1 Comdty" for root in TSY_TICKS]
    book: dict[str, dict] = {t: {} for t in tickers}
    lock = threading.Lock()
    last_tick: list[dt.datetime | None] = [None]

    # subscription messages only carry the fields that changed, so merge
    def on_data(sec: str, payload: dict, _ts: dt.datetime) -> None:
        upd = {k: payload[k] for k in FIELDS if k in payload}
        if upd:
            with lock:
                book[sec].update(upd)
                last_tick[0] = dt.datetime.now(ET)

    bbg = Bbg()
    feed = bbg.subscribe(tickers, FIELDS, on_data=on_data)
    try:
        with feed:
            while True:
                with lock:
                    frame = render(book, last_tick[0])
                print(_HOME + frame, flush=True)
                time.sleep(REDRAW_S)
    except KeyboardInterrupt:
        pass
    finally:
        bbg.session.stop()


if __name__ == '__main__':
    main()
