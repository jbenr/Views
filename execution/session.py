"""One long-lived IBKR connection, starting IB Gateway when it is not up.

Dashboard requests come and go on short-lived threads, so the connection lives on its
own event-loop thread and callers hand it work. It is read-only except on a paper
port, where submit() can send limit orders to a paper account and nothing else.
set_mode() flips it between the paper and the live account; live is read-only.
"""
from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from execution.ibkr import IBKRConfig
from utils.helpers import env_value
from execution.journal import TradeJournal
from execution.legging import hedge_due, hedge_price
from execution.portfolio import collect, connect, new_ib
from execution.quotes import (FUTURES, SPREADS, listed_spreads, quote_row, spread_contract, spread_ratio,
                              spread_row, subscribe)

GATEWAY_ROOT = Path("C:/Jts/ibgateway")
IBC_ROOT = Path("C:/IBC")  # https://github.com/IbcAlpha/IBC, optional: automates the login form
IBC_CONFIG = Path.home() / "Documents" / "IBC" / "config.ini"  # holds IbLoginId / IbPassword (paper)
IBC_LIVE_CONFIG = Path.home() / "Documents" / "IBC" / "config-live.ini"  # written per live launch, then wiped
LIVE_USER_ENV, LIVE_PASS_ENV = "VIEWS_IB_LIVE_USER", "VIEWS_IB_LIVE_PASS"  # the live login, in .env
GATEWAY_TASK = "IB Gateway"  # scheduled task that runs GATEWAY_LAUNCHER on the logged-on desktop
GATEWAY_LAUNCHER = IBC_ROOT / "launch_gateway.cmd"
PAPER_PORTS = (4002, 7497)  # IB Gateway and TWS paper trading: the only ports orders are sent on
LIVE_PORT = 4001  # IB Gateway live: always connected read-only
SPREAD_BOOK_WAIT = 1.0  # seconds a spread snapshot may take before the ticket opens without it
ACCEPT_WAIT = 5.0  # seconds IBKR may take to take or refuse a first order before the rest are held
PENDING = ("PendingSubmit", "ApiPending", "")  # IBKR has not answered yet
ACCEPTED = ("PreSubmitted", "Submitted", "Filled")  # working or done at the exchange
DONE = ("Filled", "Cancelled", "ApiCancelled", "Inactive")  # an order that will fill no further
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


def ibc_config(port):
    """The IBC config holding the login for this port's account: live or paper."""
    return IBC_CONFIG if port in PAPER_PORTS else IBC_LIVE_CONFIG


def auto_login(config=IBC_CONFIG):
    """IBC is installed and this config names a login, so it can fill in Gateway's login form."""
    return ((IBC_ROOT / "scripts" / "DisplayBannerAndLaunch.bat").exists() and config.exists()
            and re.search(r"(?m)^IbLoginId=\S", config.read_text(encoding="utf-8")) is not None)


def live_login():
    """The live username and password from the environment or .env; RuntimeError naming
    what is missing. Never logged or shown."""
    user, password = env_value(LIVE_USER_ENV), env_value(LIVE_PASS_ENV)
    missing = [name for name, value in ((LIVE_USER_ENV, user), (LIVE_PASS_ENV, password)) if not value]
    if missing:
        raise RuntimeError(f"No live login: add {' and '.join(missing)} to .env.")
    return user, password


def _login_lines(text, user, password, mode=None):
    text = re.sub(r"(?m)^IbLoginId=.*$", lambda _: f"IbLoginId={user}", text)
    text = re.sub(r"(?m)^IbPassword=.*$", lambda _: f"IbPassword={password}", text)
    return re.sub(r"(?m)^TradingMode=.*$", f"TradingMode={mode}", text) if mode else text


def write_live_config():
    """IBC's live config for one launch: the paper config's settings with the live login
    from .env. IBC reads its login only from this file; scrub_live_config() wipes it."""
    user, password = live_login()
    IBC_LIVE_CONFIG.write_text(_login_lines(IBC_CONFIG.read_text(encoding="utf-8"), user, password, "live"),
                               encoding="utf-8")


def scrub_live_config():
    """Empty the live config's login lines once Gateway no longer needs them."""
    if IBC_LIVE_CONFIG.exists():
        IBC_LIVE_CONFIG.write_text(_login_lines(IBC_LIVE_CONFIG.read_text(encoding="utf-8"), "", ""),
                                   encoding="utf-8")


def stop_gateway(wait=30):
    """Close IB Gateway (it is logged in to one account, paper or live) and wait until it is gone."""
    _powershell("Get-CimInstance Win32_Process -Filter \"Name='ibgateway.exe' OR Name='java.exe'\" | "
                "Where-Object { $_.Name -eq 'ibgateway.exe' -or $_.CommandLine -match 'ibcalpha' } | "
                "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
    deadline = time.monotonic() + wait
    while gateway_running():
        if time.monotonic() > deadline:
            raise RuntimeError("IB Gateway did not close: close it on the dashboard machine and try again.")
        time.sleep(1)


def _powershell(command):
    return subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)


def on_desktop():
    """True when this process can open windows: it runs in the Windows session a user is logged on to.
    A dashboard started over SSH or by a service runs in session 0, which has no desktop."""
    return _powershell("(Get-Process -Id $PID).SessionId -ne 0").stdout.strip() == "True"


def launch_gateway(port):
    exe = gateway_exe()
    config = ibc_config(port)
    if port not in PAPER_PORTS:
        write_live_config()  # the live login from .env, for IBC to fill in
    # `start` hands the window to a shell that exits at once, so Gateway is not a child
    # of the dashboard and survives its reloads and tree-kills.
    if not auto_login(config):
        subprocess.Popen(["cmd", "/c", "start", "", str(exe)], cwd=exe.parent)
        return
    # IBC reads the credentials from its own config; they never pass through this process.
    settings = dict(APP="GATEWAY", INLINE="0", TWS_MAJOR_VRSN=exe.parent.name,
                    TWS_PATH=str(GATEWAY_ROOT.parent), TWS_SETTINGS_PATH=str(exe.parent),
                    IBC_PATH=str(IBC_ROOT), CONFIG=str(config), LOG_PATH=str(IBC_ROOT / "Logs"),
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


def _seen(value):
    """IBKR's "no quote" is NaN or -1; the journal wants None."""
    return float(value) if value is not None and value == value and value > 0 else None


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _quote(value):
    """A spread price can be zero or negative, so only NaN reads as no quote."""
    return float(value) if value is not None and value == value else None


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
        self.mode = "paper"  # which account connect() logs in to: "paper" or "live"
        self._factory = factory
        self._ib = self._loop = self._task = self._port = None
        self._tickers, self._spreads, self._listings, self.quote_note = {}, {}, {}, ""
        self._lock = threading.Lock()

    def status(self):
        return dict(state=self.state, detail=self.detail, account=self.account, snapshot=self.snapshot_as_of,
                    mode=self.mode)

    def config(self):
        """The connection settings for the current mode: the configured (paper) ones, or
        Gateway's live port with the live account (VIEWS_IB_LIVE_ACCOUNT, else the only one)."""
        config = IBKRConfig.from_env()
        if self.mode == "live":
            return dataclasses.replace(config, port=LIVE_PORT, account=os.getenv("VIEWS_IB_LIVE_ACCOUNT") or None)
        return config

    def set_mode(self, mode):
        """Flip between the paper and the live account: drop the connection, close Gateway
        (it is logged in to one or the other) and connect again on the other. The caller
        gates the switch to live; live connects read-only and never trades."""
        if mode not in ("paper", "live"):
            raise ValueError(f"mode is paper or live, not {mode!r}")
        if mode == self.mode:
            return
        if mode == "live":
            try:  # checked before the paper session is dropped
                live_login()
            except RuntimeError as exc:
                raise RuntimeError(f"{exc} Still on paper.") from None
        self.disconnect()
        stop_gateway()
        self.mode = mode
        self.connect()

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
            self._task = self._run(self._connect(self.config()))

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
        return (self.state == "connected" and self.mode == "paper" and self._port in PAPER_PORTS
                and bool(self.account) and self.account.startswith("D"))

    def submit(self, legs, order_ref):
        """Send each leg as a day limit order tagged order_ref: dict(root, action, quantity,
        limit) on that root's streamed contract, or dict(kind="spread", contract, action,
        quantity, limit) on that exchange-listed spread (contract is its localSymbol, from
        spread_listing). All legs are checked before any is sent. Returns a row per leg
        with IBKR's order id and first status."""
        if not self.can_trade():
            raise PermissionError("Orders are only sent to a connected paper account.")
        return self._run(self._submit(legs, order_ref)).result(30)

    def spread_listing(self, month):
        """The listed spreads of one contract month ('Z6'), by name: contract (its localSymbol),
        front and back ratio, and price tick. Looked up once per month per connection."""
        if self.state != "connected":
            return {}
        if month not in self._listings:
            self._listings[month] = self._run(self._spread_listing(month)).result(60)
        return {name: dict(info) for name, (_contract, info) in self._listings[month].items()}

    async def _spread_listing(self, month):
        out = {}
        # all twelve lookups at once: one after another they take ~1.7s, together ~0.3s
        found = await asyncio.gather(*[listed_spreads(self._ib, name) for name in SPREADS])
        for name, listed in zip(SPREADS, found):
            for details in listed:
                front, back, listed_month = spread_ratio(details.contract.localSymbol)
                if listed_month == month:
                    out[name] = details.contract, dict(contract=details.contract.localSymbol, front_ratio=front,
                                                       back_ratio=back, tick=details.minTick)
        return out

    def _listed(self, local_symbol):
        """A spread contract from the listings, and its tick, by localSymbol."""
        for listing in self._listings.values():
            for contract, info in listing.values():
                if contract.localSymbol == local_symbol:
                    return contract, info["tick"]
        raise ValueError(f"No listed spread {local_symbol}.")

    def spread_book(self, local_symbols):
        """Bid and ask per listed spread, None where IBKR quotes none."""
        if self.state != "connected" or not local_symbols:
            return {}
        return self._run(self._spread_book(local_symbols)).result(15)

    async def _spread_book(self, local_symbols):
        streamed = {ticker.contract.localSymbol: ticker for ticker in self._spreads.values()}
        # one snapshot each, unless the spreads fold is already streaming it
        tickers = {symbol: streamed.get(symbol) or self._ib.reqMktData(self._listed(symbol)[0], snapshot=True)
                   for symbol in local_symbols}
        # snapshots land within a moment when IBKR has a book; stop waiting as soon as each has one
        deadline = asyncio.get_running_loop().time() + SPREAD_BOOK_WAIT
        while (any(_quote(t.bid) is None and _quote(t.ask) is None for t in tickers.values())
               and asyncio.get_running_loop().time() < deadline):
            await asyncio.sleep(0.05)
        return {symbol: dict(bid=_quote(t.bid), ask=_quote(t.ask), bid_size=_seen(t.bidSize),
                             ask_size=_seen(t.askSize), last=_quote(t.last)) for symbol, t in tickers.items()}

    def _futures(self):
        """Each streamed root's contract and price tick (decimal points)."""
        return {FUTURES[symbol][0]: (ticker.contract, FUTURES[symbol][1] / 32.0)
                for symbol, ticker in self._tickers.items()}

    def _check(self, leg, contracts, price_key="limit"):
        """Raise ValueError unless the leg's side, size and price (its limit, or a hedge's cap)
        are sendable; spreads may price at or below zero, futures may not."""
        spread = leg.get("kind") == "spread"
        label = leg["contract"] if spread else leg["root"]
        if spread:
            tick = self._listed(leg["contract"])[1]
        elif leg["root"] not in contracts:
            raise ValueError(f"No streamed contract for {leg['root']}.")
        else:
            tick = contracts[leg["root"]][1]
        ticks = float(leg[price_key]) / tick
        if leg["action"] not in ("BUY", "SELL") or int(leg["quantity"]) != leg["quantity"] or leg["quantity"] <= 0:
            raise ValueError(f"{label}: side must be BUY or SELL and quantity a whole number above zero.")
        if (not spread and leg[price_key] <= 0) or abs(ticks - round(ticks)) > 1e-6:
            raise ValueError(f"{label}: {price_key} {leg[price_key]} is not on the contract's price tick.")

    async def _submit(self, legs, order_ref):
        from ib_async import LimitOrder
        contracts = self._futures()
        for leg in legs:
            self._check(leg, contracts)
        tickers = {FUTURES[symbol][0]: ticker for symbol, ticker in self._tickers.items()}
        spreads = [leg for leg in legs if leg.get("kind") == "spread"]
        outrights = [leg for leg in legs if leg.get("kind") != "spread"]
        spread_books = await self._spread_book([leg["contract"] for leg in spreads]) if spreads else {}
        said = {}  # IBKR's words per order id: a rejection arrives as an error, not in the order's status
        heard = lambda order_id, code, message, _contract: said.__setitem__(order_id, f"{code} {message}")  # noqa: E731
        self._ib.errorEvent += heard

        def send(leg):
            """Place one order; the book it went into, read the instant before it went."""
            sent = dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
            if leg.get("kind") == "spread":
                contract = self._listed(leg["contract"])[0]
                book = dict(sent=sent, **spread_books[leg["contract"]], account=self.account)
            else:
                contract, ticker = contracts[leg["root"]][0], tickers[leg["root"]]
                book = dict(sent=sent, bid=_seen(ticker.bid), ask=_seen(ticker.ask), bid_size=_seen(ticker.bidSize),
                            ask_size=_seen(ticker.askSize), last=_seen(ticker.last), account=self.account)
            return leg, book, self._ib.placeOrder(contract, LimitOrder(
                leg["action"], int(leg["quantity"]), float(leg["limit"]), orderRef=order_ref, account=self.account,
                tif="DAY"))

        def row(leg, book, trade):
            return dict(leg, **book, contract=trade.contract.localSymbol, order_id=trade.order.orderId,
                        status=trade.orderStatus.status, filled=float(trade.orderStatus.filled),
                        message=said.get(trade.order.orderId) or (trade.log[-1].message if trade.log else ""))

        try:
            # spreads first: the outrights go only once IBKR has taken every spread, so a refused
            # spread cannot leave the package's other legs filled without it
            placed = [send(leg) for leg in spreads]
            deadline = asyncio.get_running_loop().time() + ACCEPT_WAIT
            while (any(trade.orderStatus.status in PENDING for *_, trade in placed)
                   and asyncio.get_running_loop().time() < deadline):
                await asyncio.sleep(0.05)
            refused = [trade.contract.localSymbol for *_, trade in placed if trade.orderStatus.status not in ACCEPTED]
            held = []
            if refused:
                held = [dict(leg, contract=contracts[leg["root"]][0].localSymbol, order_id=None, status="Not sent",
                             filled=0.0, message=f"held back: {', '.join(refused)} not accepted by IBKR")
                        for leg in outrights]
            else:
                placed += [send(leg) for leg in outrights]
            await asyncio.sleep(2)  # the first status of each
            rows = [row(*sent) for sent in placed] + held
        finally:
            self._ib.errorEvent -= heard
        try:
            await self._refresh()
        except Exception:
            pass  # the orders are in; the snapshot catches up on the next refresh
        return rows

    def submit_contingent(self, lead, hedges, order_ref, intent_id, signal_id):
        """Work a package by contingent legging (execution.legging): the lead -- dict(root,
        action, quantity, limit) -- rests as a day limit; each hedge -- dict(root, action,
        quantity, cap) -- goes out in proportion as the lead fills, as a marketable limit no
        worse than its cap, and is journaled under intent_id as it goes. The plan is saved,
        so a reconnect catches up on fills it missed. Returns the lead's row and a row per
        hedge (sent later; or held back when IBKR does not take the lead)."""
        if not self.can_trade():
            raise PermissionError("Orders are only sent to a connected paper account.")
        return self._run(self._submit_contingent(lead, hedges, order_ref, intent_id, signal_id)).result(30)

    async def _submit_contingent(self, lead, hedges, order_ref, intent_id, signal_id):
        from ib_async import LimitOrder
        contracts = self._futures()
        self._check(lead, contracts)
        for hedge in hedges:
            self._check(hedge, contracts, "cap")
        ticker = {FUTURES[symbol][0]: t for symbol, t in self._tickers.items()}[lead["root"]]
        said = {}
        heard = lambda order_id, code, message, _contract: said.__setitem__(order_id, f"{code} {message}")  # noqa: E731
        self._ib.errorEvent += heard
        try:
            book = dict(sent=_now(), bid=_seen(ticker.bid), ask=_seen(ticker.ask), bid_size=_seen(ticker.bidSize),
                        ask_size=_seen(ticker.askSize), last=_seen(ticker.last), account=self.account)
            trade = self._ib.placeOrder(contracts[lead["root"]][0], LimitOrder(
                lead["action"], int(lead["quantity"]), float(lead["limit"]), orderRef=order_ref,
                account=self.account, tif="DAY"))
            deadline = asyncio.get_running_loop().time() + ACCEPT_WAIT
            while trade.orderStatus.status in PENDING and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
        finally:
            self._ib.errorEvent -= heard
        order_id = trade.order.orderId
        lead_row = dict(lead, **book, role="lead", contract=trade.contract.localSymbol, order_id=order_id,
                        status=trade.orderStatus.status, filled=float(trade.orderStatus.filled),
                        message=said.get(order_id) or (trade.log[-1].message if trade.log else ""))
        waiting = dict(order_id=None, filled=0.0)
        if trade.orderStatus.status not in ACCEPTED:
            return [lead_row] + [dict(hedge, **waiting, role="hedge", contract=contracts[hedge["root"]][0].localSymbol,
                                      limit=hedge["cap"], status="Not sent",
                                      message=f"held back: the lead {lead_row['contract']} was not accepted")
                                 for hedge in hedges]
        plan = dict(key=f"{order_ref}:{order_id}", order_ref=order_ref, intent_id=intent_id, signal_id=signal_id,
                    account=self.account, lead_order_id=order_id, lead_perm_id=getattr(trade.order, "permId", 0),
                    lead=dict(lead, contract=lead_row["contract"]), done=False,
                    hedges=[dict(hedge, contract=contracts[hedge["root"]][0].localSymbol, sent=0) for hedge in hedges])
        TradeJournal().save_plan(plan["key"], plan)
        self._watch(trade, plan)
        return [lead_row] + [dict(hedge, **waiting, role="hedge", contract=h["contract"], limit=hedge["cap"],
                                  status="Waiting for lead", message="sent in proportion as the lead fills")
                             for hedge, h in zip(hedges, plan["hedges"])]

    def _watch(self, trade, plan):
        """Hedge on every fill of the lead, and on any it took before we were watching."""
        def filled():
            # IBKR can resend an execution (a correction): count each execution id once
            return sum({f.execution.execId: f.execution.shares for f in trade.fills}.values())

        def on_fill(*_):
            self._hedge(plan, filled(), done=trade.orderStatus.status in DONE)

        trade.fillEvent += on_fill
        trade.statusEvent += on_fill
        on_fill()

    def _hedge(self, plan, lead_filled, done=False):
        """Send each hedge's share of the lead's fills so far; journal what went and the plan."""
        from ib_async import LimitOrder
        contracts = self._futures()
        tickers = {FUTURES[symbol][0]: t for symbol, t in self._tickers.items()}
        rows = []
        for hedge in plan["hedges"]:
            due = hedge_due(lead_filled, plan["lead"]["quantity"], hedge["quantity"], hedge["sent"])
            if not due:
                continue
            contract, tick = contracts[hedge["root"]]
            ticker = tickers[hedge["root"]]
            bid, ask = _seen(ticker.bid), _seen(ticker.ask)
            price = round(round(hedge_price(hedge["action"], bid, ask, tick, hedge["cap"]) / tick) * tick, 10)
            sent = _now()
            trade = self._ib.placeOrder(contract, LimitOrder(hedge["action"], due, price, orderRef=plan["order_ref"],
                                                             account=plan["account"], tif="DAY"))
            hedge["sent"] += due
            rows.append(dict(root=hedge["root"], action=hedge["action"], quantity=due, limit=price, role="hedge",
                             sent=sent, bid=bid, ask=ask, bid_size=_seen(ticker.bidSize),
                             ask_size=_seen(ticker.askSize), last=_seen(ticker.last), account=plan["account"],
                             contract=contract.localSymbol, order_id=trade.order.orderId,
                             status=trade.orderStatus.status, filled=0.0,
                             message=f"hedge for lead fill {lead_filled:g}/{plan['lead']['quantity']}"))
        plan["done"] = done or lead_filled >= plan["lead"]["quantity"]
        if rows or plan["done"]:
            journal = TradeJournal()
            if rows:
                journal.save_orders(plan["intent_id"], plan["signal_id"], rows)
            journal.save_plan(plan["key"], plan)

    def _resume(self):
        """After a (re)connect: catch up each open plan of this account on lead fills it
        missed, and watch its lead again while that still works."""
        for plan in TradeJournal().records("legging"):
            if plan["done"] or plan["account"] != self.account:
                continue
            def is_lead(ids):
                perm, order_id = ids
                return (plan["lead_perm_id"] and perm == plan["lead_perm_id"]) or order_id == plan["lead_order_id"]

            live = next((t for t in self._ib.openTrades() if is_lead((t.order.permId, t.order.orderId))), None)
            if live is not None:
                self._watch(live, plan)
                continue
            fills = {f.execution.execId: f.execution.shares for f in self._ib.fills()
                     if is_lead((f.execution.permId, f.execution.orderId))}
            self._hedge(plan, sum(fills.values()), done=True)  # the lead is no longer working

    def plans(self):
        """Contingent plans still waiting on their lead, for this account."""
        return [p for p in TradeJournal().records("legging") if not p["done"] and p["account"] == self.account]

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
                              if auto_login(ibc_config(config.port)) else
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
        finally:
            if config.port == LIVE_PORT:
                scrub_live_config()  # logged in or given up: the password need not sit on disk
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
        if self.can_trade():
            try:  # hedges owed on lead fills that landed while we were away
                await asyncio.sleep(1)  # a moment for the first quotes, which the hedges price off
                self._resume()
            except Exception as exc:
                self.detail = f"contingent plans not resumed ({type(exc).__name__}: {exc}): check Open orders"

    async def _refresh(self):
        snapshot = await asyncio.wait_for(collect(self._ib, self.account), 45)
        TradeJournal().save_snapshot(snapshot)
        self.snapshot_as_of = snapshot["as_of"]

    async def _close(self):
        ib, self._ib, self._tickers, self._spreads, self._listings = self._ib, None, {}, {}, {}
        self._set("disconnected")
        if ib:
            ib.disconnect()

    def _dropped(self):
        # Gateway restarts daily and logs out weekly; reconnecting is a click, not automatic.
        if self.state == "connected":
            self._ib, self._tickers, self._spreads, self._listings = None, {}, {}, {}
            self._set("disconnected", "IBKR closed the connection.")
