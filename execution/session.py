"""One long-lived IBKR connection, starting IB Gateway when it is not up.

Dashboard requests come and go on short-lived threads, so the connection lives on its
own event-loop thread and callers hand it work. It is read-only except on a paper
port, where submit() can send limit orders to a paper account and nothing else.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import os
import re
import subprocess
import threading
from pathlib import Path

from execution.ibkr import IBKRConfig
from execution.journal import TradeJournal
from execution.portfolio import collect, connect, new_ib
from execution.quotes import FUTURES, SPREADS, quote_row, spread_contract, spread_row, subscribe

GATEWAY_ROOT = Path("C:/Jts/ibgateway")
IBC_ROOT = Path("C:/IBC")  # https://github.com/IbcAlpha/IBC, optional: automates the login form
IBC_CONFIG = Path.home() / "Documents" / "IBC" / "config.ini"  # holds IbLoginId / IbPassword
GATEWAY_TASK = "IB Gateway"  # scheduled task that runs GATEWAY_LAUNCHER on the logged-on desktop
GATEWAY_LAUNCHER = IBC_ROOT / "launch_gateway.cmd"
PAPER_PORTS = (4002, 7497)  # IB Gateway and TWS paper trading: the only ports orders are sent on
LOGIN_WAIT = 300 # seconds allowed for typing the password and approving 2FA
POLL = 2
BUSY = ("waiting", "connecting")


def gateway_exe():
    """The newest installed IB Gateway launcher."""
    # IBC renames the launcher to ibgateway1.exe so Gateway cannot restart without it
    found = sorted(GATEWAY_ROOT.glob("*/ibgateway*.exe"), key=lambda p: p.parent.name.zfill(8))
    if not found:
        raise FileNotFoundError(f"No IB Gateway under {GATEWAY_ROOT}: install it, or start TWS yourself.")
    return found[-1]


def gateway_running():
    """True for Gateway's own launcher or for the java.exe that IBC runs it under."""
    query = ("@(Get-CimInstance Win32_Process -Filter \"Name='ibgateway.exe' OR Name='java.exe'\" | "
             "Where-Object { $_.Name -eq 'ibgateway.exe' -or $_.CommandLine -match 'ibcalpha' }).Count -gt 0")
    return _powershell(query).stdout.strip() == "True"


def auto_login():
    """IBC is installed and its config names a login, so it can fill in Gateway's login form."""
    return ((IBC_ROOT / "scripts" / "DisplayBannerAndLaunch.bat").exists() and IBC_CONFIG.exists()
            and re.search(r"(?m)^IbLoginId=\S", IBC_CONFIG.read_text(encoding="utf-8")) is not None)


def _powershell(command):
    return subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)


def on_desktop():
    """True when this process can open windows: it runs in the Windows session a user is logged on to.
    A dashboard started over SSH or by a service runs in session 0, which has no desktop."""
    return _powershell("(Get-Process -Id $PID).SessionId -ne 0").stdout.strip() == "True"


def launch_gateway(port):
    exe = gateway_exe()
    # `start` hands the window to a shell that exits at once, so Gateway is not a child
    # of the dashboard and survives its reloads and tree-kills.
    if not auto_login():
        subprocess.Popen(["cmd", "/c", "start", "", str(exe)], cwd=exe.parent)
        return
    # IBC reads the credentials from its own config; they never pass through this process.
    settings = dict(APP="GATEWAY", INLINE="0", TWS_MAJOR_VRSN=exe.parent.name,
                    TWS_PATH=str(GATEWAY_ROOT.parent), TWS_SETTINGS_PATH=str(exe.parent),
                    IBC_PATH=str(IBC_ROOT), CONFIG=str(IBC_CONFIG), LOG_PATH=str(IBC_ROOT / "Logs"),
                    TRADING_MODE="paper" if port in PAPER_PORTS else "live", TWOFA_TIMEOUT_ACTION="exit")
    banner = str(IBC_ROOT / "scripts" / "DisplayBannerAndLaunch.bat")
    if on_desktop():
        subprocess.Popen(["cmd", "/c", "start", "", "/min", banner], env=dict(os.environ, **settings))
        return
    # No desktop here, and Gateway's login window needs one: without it IBC waits 60s for the
    # window, exits and retries for ever. The scheduled task opens it on the logged-on desktop.
    if _powershell("@(Get-Process explorer -ErrorAction SilentlyContinue).Count -gt 0").stdout.strip() != "True":
        raise RuntimeError("Nobody is logged on to Windows on the dashboard machine, so IB Gateway has no desktop "
                           "to open on. Log on there (Remote Desktop is fine: disconnect, do not sign out) and "
                           "click Connect again.")
    GATEWAY_LAUNCHER.write_text("@echo off\n" + "".join(f"set {k}={v}\n" for k, v in settings.items())
                                + f'start "" /min "{banner}"\n', encoding="ascii")
    started = subprocess.run(["schtasks", "/run", "/tn", GATEWAY_TASK], capture_output=True, text=True)
    if started.returncode:
        raise RuntimeError(f'Scheduled task "{GATEWAY_TASK}" did not start ({started.stderr.strip()}): '
                           "create it once, see execution/README.md.")


async def port_open(host, port):
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), 1)
    except (OSError, TimeoutError):
        return False
    writer.close()
    return True


class Session:
    """state is one of: disconnected, waiting (for the Gateway login), connecting, connected, failed."""

    def __init__(self, factory=None):
        self.state, self.detail, self.account, self.snapshot_as_of = "disconnected", "", None, None
        self._factory = factory
        self._ib = self._loop = self._task = self._port = None
        self._tickers, self._spreads, self.quote_note = {}, {}, ""
        self._lock = threading.Lock()

    def status(self):
        return dict(state=self.state, detail=self.detail, account=self.account, snapshot=self.snapshot_as_of)

    def _set(self, state, detail=""):
        self.state, self.detail = state, detail

    def _run(self, coro):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            threading.Thread(target=self._loop.run_forever, name="ibkr-session", daemon=True).start()
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def connect(self):
        """Start connecting in the background; poll status() for the outcome."""
        with self._lock:
            if self.state == "connected" or self.state in BUSY:
                return
            self._set("connecting", "Connecting…")
            self._task = self._run(self._connect(IBKRConfig.from_env()))

    def disconnect(self):
        """Close the connection, or abandon a connect that is still waiting. Gateway stays up."""
        with self._lock:
            if self._task:
                self._task.cancel()
                self._run(self._close()).result(10)

    def refresh(self):
        """Save a fresh snapshot of the account to the journal."""
        if self.state != "connected":
            raise ConnectionError("Not connected to IBKR.")
        self._run(self._refresh()).result(60)

    def quotes(self):
        """Latest futures quotes, and what IBKR said if it is withholding them."""
        return dict(rows=[quote_row(symbol, ticker) for symbol, ticker in self._tickers.items()],
                    note=self.quote_note)

    def can_trade(self):
        """Orders go only to a paper account (IBKR numbers them D...) on a paper port."""
        return (self.state == "connected" and self._port in PAPER_PORTS
                and bool(self.account) and self.account.startswith("D"))

    def submit(self, legs, order_ref):
        """Send each leg -- dict(root, action, quantity, limit) -- as a day limit order on that
        root's streamed contract, tagged order_ref. All legs are checked before any is sent.
        Returns a row per leg with IBKR's order id and first status."""
        if not self.can_trade():
            raise PermissionError("Orders are only sent to a connected paper account.")
        return self._run(self._submit(legs, order_ref)).result(30)

    async def _submit(self, legs, order_ref):
        from ib_async import LimitOrder
        contracts = {FUTURES[symbol][0]: (ticker.contract, FUTURES[symbol][1] / 32.0)
                     for symbol, ticker in self._tickers.items()}
        for leg in legs:
            if leg["root"] not in contracts:
                raise ValueError(f"No streamed contract for {leg['root']}.")
            ticks = float(leg["limit"]) / contracts[leg["root"]][1]
            if leg["action"] not in ("BUY", "SELL") or int(leg["quantity"]) != leg["quantity"] or leg["quantity"] <= 0:
                raise ValueError(f"{leg['root']}: side must be BUY or SELL and quantity a whole number above zero.")
            if leg["limit"] <= 0 or abs(ticks - round(ticks)) > 1e-6:
                raise ValueError(f"{leg['root']}: limit {leg['limit']} is not on the contract's price tick.")
        tickers = {FUTURES[symbol][0]: ticker for symbol, ticker in self._tickers.items()}

        def seen(value):  # IBKR's "no quote" is NaN or -1; the journal wants None
            return float(value) if value is not None and value == value and value > 0 else None

        placed, books = [], []
        for leg in legs:
            ticker = tickers[leg["root"]]
            # the book each leg was sent into, read the instant before it went
            books.append(dict(sent=dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
                              bid=seen(ticker.bid), ask=seen(ticker.ask), bid_size=seen(ticker.bidSize),
                              ask_size=seen(ticker.askSize), last=seen(ticker.last), account=self.account))
            placed.append(self._ib.placeOrder(contracts[leg["root"]][0], LimitOrder(
                leg["action"], int(leg["quantity"]), float(leg["limit"]), orderRef=order_ref, account=self.account,
                tif="DAY")))
        await asyncio.sleep(2)  # the first status of each
        rows = [dict(leg, **book, contract=trade.contract.localSymbol, order_id=trade.order.orderId,
                     status=trade.orderStatus.status, filled=float(trade.orderStatus.filled),
                     message=trade.log[-1].message if trade.log else "")
                for leg, book, trade in zip(legs, books, placed)]
        try:
            await self._refresh()
        except Exception:
            pass  # the orders are in; the snapshot catches up on the next refresh
        return rows

    def set_spreads(self, names):
        """Stream exactly these exchange-listed spreads: subscribe to the new, drop the rest."""
        if self.state == "connected" and set(names) != set(self._spreads):
            self._run(self._set_spreads([name for name in names if name in SPREADS])).result(30)

    def spread_quotes(self):
        return [spread_row(name, ticker) for name, ticker in self._spreads.items()]

    async def _set_spreads(self, names):
        for name in set(self._spreads) - set(names):
            self._ib.cancelMktData(self._spreads.pop(name).contract)
        for name in names:
            if name not in self._spreads:
                self._spreads[name] = self._ib.reqMktData(await spread_contract(self._ib, name))

    def _quote_error(self, _id, code, message, contract):
        streams = [*self._tickers.values(), *self._spreads.values()]
        if contract is not None and any(t.contract.conId == contract.conId for t in streams):
            self.quote_note = f"IBKR {code}: {message.split(', contract:')[0]}"

    async def _connect(self, config):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + LOGIN_WAIT
        ib, launched, retry = None, False, ""
        try:
            while True:
                if await port_open(config.host, config.port):
                    self._set("connecting", f"Connecting to {config.host}:{config.port}…{retry}")
                    ib = new_ib(self._factory)
                    said = []  # Gateway's own words say more than the timeout they end in
                    ib.errorEvent += lambda _id, code, message, _contract: said.append(f"{code} {message}")
                    try:
                        # a trading session on a paper port, read-only everywhere else
                        account = await connect(ib, config, readonly=config.port not in PAPER_PORTS)
                        break
                    except (ConnectionError, TimeoutError, OSError) as exc:
                        # the port can open moments before Gateway accepts API clients
                        ib.disconnect()
                        retry = f" retrying after {said[-1] if said else f'{type(exc).__name__}: {exc}'}"
                elif config.host not in ("127.0.0.1", "localhost"):
                    raise ConnectionError(f"Nothing is listening on {config.host}:{config.port}, and Gateway "
                                          "can only be started on the machine running the dashboard.")
                else:
                    if not launched and not gateway_running():
                        launch_gateway(config.port)
                    launched = True
                    self._set("waiting", "IB Gateway is logging in: approve the 2FA prompt if your phone gets one."
                              if auto_login() else
                              "IB Gateway is open on the dashboard machine: log in there and "
                              "approve the 2FA prompt on your phone.")
                if loop.time() > deadline:
                    raise TimeoutError(f"No API connection on {config.host}:{config.port} after {LOGIN_WAIT}s."
                                       f"{retry}")
                await asyncio.sleep(POLL)
            ib.disconnectedEvent += self._dropped
            self._ib, self.account, self._port = ib, account, config.port
            self._set("connected")
        except asyncio.CancelledError:
            if ib:
                ib.disconnect()
            raise
        except Exception as exc:
            if ib:
                ib.disconnect()
            self._set("failed", f"{type(exc).__name__}: {exc}")
            return
        try:
            await self._refresh()
        except Exception as exc:
            self.detail = f"first snapshot failed ({type(exc).__name__}: {exc}); refresh to retry"
        try:
            self.quote_note = ""
            ib.errorEvent += self._quote_error
            self._tickers = await subscribe(ib)
        except Exception as exc:
            self.quote_note = f"futures quotes unavailable ({type(exc).__name__}: {exc})"

    async def _refresh(self):
        snapshot = await asyncio.wait_for(collect(self._ib, self.account), 45)
        TradeJournal().save_snapshot(snapshot)
        self.snapshot_as_of = snapshot["as_of"]

    async def _close(self):
        ib, self._ib, self._tickers, self._spreads = self._ib, None, {}, {}
        self._set("disconnected")
        if ib:
            ib.disconnect()

    def _dropped(self):
        # Gateway restarts daily and logs out weekly; reconnecting is a click, not automatic.
        if self.state == "connected":
            self._ib, self._tickers, self._spreads = None, {}, {}
            self._set("disconnected", "IBKR closed the connection.")
