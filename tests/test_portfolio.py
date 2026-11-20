import asyncio
import os
import time
from types import SimpleNamespace as NS

import pytest

from execution.ibkr import IBKRConfig
from execution.journal import TradeJournal
from execution.portfolio import fetch_snapshot


class FakeIB:
    accounts = ["DU1"]
    fail = False
    def __init__(self):
        self.closed = False
    async def connectAsync(self, *args, **kwargs):
        assert kwargs["readonly"] is True
        if self.fail:
            raise ConnectionError("offline")
    def managedAccounts(self):
        return self.accounts
    async def reqAccountSummaryAsync(self):
        pass
    async def accountSummaryAsync(self):
        return [NS(account=a, tag="NetLiquidation", value="100", currency="USD") for a in ["DU1", "DU2"]]
    async def reqPositionsAsync(self):
        c = NS(conId=123, localSymbol="ZNZ6", symbol="ZN", secType="FUT", currency="USD",
               lastTradeDateOrContractMonth="202612", multiplier="1000")
        return [NS(account=a, contract=c, position=2, avgCost=110000) for a in ["DU1", "DU2"]]
    async def reqAllOpenOrdersAsync(self):
        return []
    async def reqAccountUpdatesAsync(self, account):
        pass
    def portfolio(self, account):
        c = NS(conId=123)
        return [NS(account=account, contract=c, position=2, marketPrice=111, unrealizedPNL=2000)]
    async def reqCompletedOrdersAsync(self, apiOnly):
        return []
    async def reqExecutionsAsync(self):
        return []
    def disconnect(self):
        self.closed = True


def test_snapshot_filtered_and_disconnected():
    ib = FakeIB()
    result = fetch_snapshot(IBKRConfig(account="DU1"), lambda: ib)
    assert ib.closed
    assert len(result["positions"]) == len(result["balances"]) == 1
    assert result["positions"][0]["strategy"] == "Unattributed"
    assert result["positions"][0]["avg_cost"] == 110
    assert result["positions"][0]["current_price"] == 111
    assert result["positions"][0]["unrealized_pnl"] == 2000
    assert result["as_of"].endswith("+00:00")


def test_account_selection_and_disconnect_on_error():
    ib = FakeIB()
    ib.accounts = ["DU1", "DU2"]
    with pytest.raises(ValueError, match="exactly one"):
        fetch_snapshot(IBKRConfig(), lambda: ib)
    assert ib.closed


def test_history_timeout_preserves_portfolio():
    class MissingHistoryIB(FakeIB):
        async def reqCompletedOrdersAsync(self, apiOnly):
            raise TimeoutError()
    ib = MissingHistoryIB()
    result = fetch_snapshot(IBKRConfig(account="DU1"), lambda: ib)
    assert len(result["positions"]) == 1
    assert result["completed_orders"] == []
    assert "Completed-order history" in result["warnings"][0]
    assert ib.closed
    ib.fail = True
    with pytest.raises(ConnectionError):
        fetch_snapshot(IBKRConfig(), lambda: ib)
    assert ib.closed


def test_reject_binding_client():
    with pytest.raises(ValueError, match="positive"):
        fetch_snapshot(IBKRConfig(client_id=0), FakeIB)


def test_refused_connection_reports_endpoint_and_setup():
    class RefusedIB(FakeIB):
        async def connectAsync(self, *args, **kwargs):
            raise ConnectionRefusedError("refused")
    ib = RefusedIB()
    with pytest.raises(ConnectionError, match="127.0.0.1:4002") as error:
        fetch_snapshot(IBKRConfig(port=4002), lambda: ib)
    assert "Socket Port" in str(error.value)
    assert ib.closed


def test_local_config_and_environment_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("VIEWS_STORE_DIR", str(tmp_path))
    for key in ("VIEWS_IB_HOST", "VIEWS_IB_PORT", "VIEWS_IB_CLIENT_ID", "VIEWS_IB_ACCOUNT"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "ibkr.json"
    path.write_text('{"port": 4001}', encoding="utf-8")
    assert IBKRConfig.from_env().port == 4001
    path.write_text('{"port": 4002}', encoding="utf-8")
    assert IBKRConfig.from_env().port == 4002
    monkeypatch.setenv("VIEWS_IB_PORT", "7497")
    assert IBKRConfig.from_env().port == 7497


def test_durable_flags_and_snapshot_history(tmp_path):
    journal = TradeJournal(tmp_path / "execution.sqlite3")
    first = journal.flag("strategy::variant", "strategy", "2026-11-04")
    assert journal.flag("strategy::variant", "strategy", "2026-11-04") == first
    assert len(journal.records("intent")) == 1
    journal.flag("strategy::other", "strategy", "2026-11-04")
    assert len(journal.records("intent")) == 2
    journal.save_snapshot({"account": "DU1", "as_of": "now"})
    assert TradeJournal(journal.path).latest_snapshot()["account"] == "DU1"
    assert journal.records("intent")[0]["status"] == "pending_review"


def test_position_time_groups_use_latest_fill_and_fixed_window():
    from execution.grouping import group_positions
    positions = [dict(con_id=i) for i in range(1, 6)]
    fills = [dict(con_id=1, time="2026-11-04 14:00:00+00:00"),
             dict(con_id=1, time="2026-11-04 15:00:00+00:00"),
             dict(con_id=2, time="2026-11-04 10:59:00-04:00"),
             dict(con_id=3, time="2026-11-04 14:57:00+00:00"),
             dict(con_id=4, time="invalid"),
             dict(con_id=5, time="2026-11-04 15:00:00")]
    groups = group_positions(positions, fills)
    assert [[p["con_id"] for p in rows] for _, rows in groups] == [[1, 2], [3], [4, 5]]
    assert groups[-1][0] is None
    assert group_positions([], fills) == []


def test_fills_persist_across_refreshes_and_stay_account_scoped(tmp_path):
    journal = TradeJournal(tmp_path / "journal.sqlite3")
    fill = dict(exec_id="e1", con_id=123, time="2026-11-04 15:00:00+00:00")
    for account in ("DU1", "DU1", "DU2"):
        journal.save_snapshot(dict(account=account, fills=[fill]))
    journal.save_snapshot(dict(account="DU1", fills=[]))
    assert len(journal.saved_fills("DU1")) == 1
    assert len(journal.saved_fills("DU2")) == 1
    assert journal.saved_fills("DU3") == []


class _Handlers(list):
    """Stands in for an ib_async event: `+=` a handler onto it, fire() calls them."""
    def __iadd__(self, handler):
        self.append(handler)
        return self

    def __isub__(self, handler):
        self.remove(handler)
        return self

    def fire(self):
        for handler in self:
            handler()


class SessionIB(FakeIB):
    readonly = False  # the default port is a paper one, where the session may trade

    def __init__(self):
        super().__init__()
        self.disconnectedEvent = _Handlers()
        self.errorEvent = _Handlers()
        self.placed = []
        self.refuse = set()  # localSymbols IBKR rejects, as it does spreads on a paper account
        self.trades = []

    async def connectAsync(self, *args, **kwargs):
        assert kwargs["readonly"] is self.readonly
        if self.fail:
            raise ConnectionError("offline")

    def placeOrder(self, contract, order):
        self.placed.append((contract.localSymbol, order.action, order.totalQuantity, order.lmtPrice, order.orderRef))
        order_id = len(self.placed)
        refused = contract.localSymbol in self.refuse
        if refused:  # IBKR says why as an error keyed by the order id
            for handler in list(self.errorEvent):
                handler(order_id, 201, "Order rejected - reason:Orders not supported for this contract type", contract)
        trade = NS(contract=contract, order=NS(orderId=order_id, permId=1000 + order_id), log=[], fills=[],
                   orderStatus=NS(status="Inactive" if refused else "Submitted", filled=0),
                   fillEvent=_Handlers(), statusEvent=_Handlers())
        self.trades.append(trade)
        return trade

    def openTrades(self):
        return [t for t in self.trades if t.orderStatus.status in ("PreSubmitted", "Submitted")]

    def fills(self):
        return [f for t in self.trades for f in t.fills]


def _wait(session, state, seconds=5):
    deadline = time.monotonic() + seconds
    while session.state != state and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session.state == state, session.status()


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """A machine whose Gateway API port opens only once `up` is set; counts launches."""
    from execution import session
    box = NS(up=False, running=False, launches=0)

    async def port_open(host, port):
        return box.up

    def launch(port):
        box.launches += 1
        box.running = True

    monkeypatch.setenv("VIEWS_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("VIEWS_IB_ACCOUNT", "DU1")
    monkeypatch.setattr(session, "port_open", port_open)
    monkeypatch.setattr(session, "gateway_running", lambda: box.running)
    monkeypatch.setattr(session, "launch_gateway", launch)
    monkeypatch.setattr(session, "auto_login", lambda *_: False)
    monkeypatch.setattr(session, "IBC_LIVE_CONFIG", tmp_path / "config-live.ini")  # never the real one
    monkeypatch.setattr(session, "stop_gateway", lambda: setattr(box, "running", False) or setattr(box, "up", False))
    monkeypatch.setattr(session, "POLL", 0.01)
    return box


def test_session_launches_gateway_once_then_connects_and_snapshots(gateway):
    from execution.session import Session
    ib = SessionIB()
    session = Session(lambda: ib)
    session.connect()
    _wait(session, "waiting")
    session.connect()  # a second click while waiting does nothing
    assert gateway.launches == 1 and "log in" in session.detail
    gateway.up = True
    _wait(session, "connected")
    assert session.account == "DU1"
    deadline = time.monotonic() + 5
    while not session.snapshot_as_of and time.monotonic() < deadline:
        time.sleep(0.01)
    assert TradeJournal().latest_snapshot()["as_of"] == session.snapshot_as_of
    first = session.snapshot_as_of
    session.refresh()
    assert session.snapshot_as_of != first
    ib.disconnectedEvent.fire()  # Gateway drops us
    assert session.state == "disconnected" and "closed" in session.detail
    with pytest.raises(ConnectionError):
        session.refresh()


def test_session_uses_running_gateway_and_cancel_stops_waiting(gateway):
    from execution.session import Session
    gateway.running = True  # sitting at its login window
    ib = SessionIB()
    session = Session(lambda: ib)
    session.connect()
    _wait(session, "waiting")
    session.disconnect()
    assert session.state == "disconnected" and gateway.launches == 0
    gateway.up = True
    time.sleep(0.1)
    assert session.state == "disconnected"  # the abandoned connect did not carry on


def test_session_reports_failure_and_can_retry(gateway):
    from execution.session import Session
    gateway.up = True
    ib = SessionIB()
    ib.accounts = ["DU9"]  # not the configured account
    session = Session(lambda: ib)
    session.connect()
    _wait(session, "failed")
    assert "not available" in session.detail and ib.closed
    ib.accounts = ["DU1"]
    session.connect()
    _wait(session, "connected")
    session.disconnect()
    assert session.state == "disconnected"



def test_session_sends_orders_only_to_a_paper_account(gateway, monkeypatch):
    from execution.session import Session
    gateway.up = True
    ib = SessionIB()
    session = Session(lambda: ib)
    session.connect()
    _wait(session, "connected")
    session._tickers = {"ZN": NS(contract=NS(localSymbol="ZNZ6", conId=1), bid=110.5, ask=110.515625,
                                 bidSize=40, askSize=float("nan"), last=-1)}
    leg = dict(root="TY", action="BUY", quantity=2, limit=110.515625)
    with pytest.raises(ValueError, match="price tick"):
        session.submit([dict(leg, limit=110.51)], "views:x")
    with pytest.raises(ValueError, match="No streamed contract"):
        session.submit([dict(leg, root="WN")], "views:x")
    assert not ib.placed  # nothing goes out when any leg is bad
    [sent] = session.submit([leg], "views:x")
    assert ib.placed == [("ZNZ6", "BUY", 2, 110.515625, "views:x")] and sent["status"] == "Submitted"
    # the book it was sent into is kept with it, and fits the journal (no NaN)
    assert (sent["bid"], sent["ask"], sent["bid_size"], sent["ask_size"], sent["last"]) == (
        110.5, 110.515625, 40, None, None) and sent["sent"].endswith("+00:00")
    from dashboard.trades import _sent_table
    cells = str(_sent_table([sent], signal=False).to_plotly_json())
    assert "110-16" in cells and "Bid at send" in cells and "Signal" not in cells
    # an exchange-listed spread goes as one order on its own contract, priced off its own tick
    nob = NS(localSymbol="NOB 02-01 Z6", conId=9)
    session._listings = {"Z6": {"NOB": (nob, dict(contract="NOB 02-01 Z6", front_ratio=2, back_ratio=1,
                                                  tick=0.015625))}}
    session._spreads = {"NOB": NS(contract=nob, bid=-0.375, ask=-0.359375, bidSize=5, askSize=7, last=float("nan"))}
    spread = dict(kind="spread", contract="NOB 02-01 Z6", name="NOB", action="SELL", quantity=3, limit=-0.375)
    with pytest.raises(ValueError, match="price tick"):
        session.submit([dict(spread, limit=-0.37)], "views:y")
    [sent] = session.submit([spread], "views:y")
    assert ib.placed[-1] == ("NOB 02-01 Z6", "SELL", 3, -0.375, "views:y")
    assert (sent["bid"], sent["ask"], sent["last"]) == (-0.375, -0.359375, None)
    assert "-12.000" in str(_sent_table([sent], signal=False).to_plotly_json())
    # a refused spread holds back the package's outrights: nothing goes on half done
    ib.refuse.add("NOB 02-01 Z6")
    before, listeners = len(ib.placed), len(ib.errorEvent)
    nob_sent, held = session.submit([spread, leg], "views:z")
    assert ib.placed[before:] == [("NOB 02-01 Z6", "SELL", 3, -0.375, "views:z")]  # the TY leg never went
    assert nob_sent["status"] == "Inactive" and nob_sent["message"].startswith("201 Order rejected")
    assert (held["status"], held["order_id"], held["contract"]) == ("Not sent", None, "ZNZ6")
    assert "NOB 02-01 Z6 not accepted" in held["message"] and len(ib.errorEvent) == listeners  # listener removed
    assert "Not sent" in str(_sent_table([nob_sent, held], signal=False).to_plotly_json())
    session.disconnect()

    monkeypatch.setenv("VIEWS_IB_PORT", "4001")  # a live port: read-only, and submit refuses
    live = SessionIB()
    live.readonly = True
    session = Session(lambda: live)
    session.connect()
    _wait(session, "connected")
    assert not session.can_trade()
    with pytest.raises(PermissionError):
        session.submit([leg], "views:x")
    assert not live.placed


def test_live_mode_reconnects_read_only_on_the_live_port(gateway, monkeypatch):
    from execution.session import LIVE_PORT, Session
    gateway.up = True
    ib = SessionIB()
    session = Session(lambda: ib)
    session.connect()
    _wait(session, "connected")
    assert session.can_trade() and session.config().port == 7497  # the configured paper port
    live = SessionIB()
    live.readonly = True  # connectAsync asserts the session asks for read-only
    live.accounts = ["U7"]
    session._factory = lambda: live
    monkeypatch.delenv("VIEWS_IB_ACCOUNT")
    from execution import session as session_module
    monkeypatch.setattr(session_module, "env_value", os.getenv)  # the environment only, never the real .env
    monkeypatch.delenv("VIEWS_IB_LIVE_USER", raising=False)
    monkeypatch.setenv("VIEWS_IB_LIVE_PASS", "pw")
    with pytest.raises(RuntimeError, match="add VIEWS_IB_LIVE_USER to .env. Still on paper"):
        session.set_mode("live")  # no live login: refused before paper is touched
    assert session.state == "connected" and session.mode == "paper" and gateway.up  # Gateway never stopped
    monkeypatch.setenv("VIEWS_IB_LIVE_USER", "liveuser")
    session.set_mode("live")  # closes Gateway: it is logged in to paper
    assert session.config().port == LIVE_PORT and session.status()["mode"] == "live"
    gateway.up = True
    _wait(session, "connected")
    assert session.account == "U7" and not session.can_trade()
    with pytest.raises(PermissionError):
        session.submit([dict(root="TY", action="BUY", quantity=1, limit=110.5)], "views:x")
    with pytest.raises(ValueError):
        session.set_mode("demo")
    session.disconnect()


def test_live_ibc_config_carries_the_env_login_only_until_used(tmp_path, monkeypatch):
    from execution import session
    paper, live = tmp_path / "config.ini", tmp_path / "config-live.ini"
    paper.write_text("IbLoginId=paperuser\nIbPassword=paperpw\nTradingMode=paper\nAcceptNonBrokerageAccountWarning=yes\n",
                     encoding="utf-8")
    monkeypatch.setattr(session, "IBC_CONFIG", paper)
    monkeypatch.setattr(session, "IBC_LIVE_CONFIG", live)
    monkeypatch.setattr(session, "env_value", {"VIEWS_IB_LIVE_USER": "me", "VIEWS_IB_LIVE_PASS": r"p$w\1"}.get)
    session.write_live_config()
    assert live.read_text(encoding="utf-8") == ("IbLoginId=me\nIbPassword=p$w\\1\nTradingMode=live\n"
                                                "AcceptNonBrokerageAccountWarning=yes\n")
    assert "paperpw" in paper.read_text(encoding="utf-8")  # the paper config is untouched
    session.scrub_live_config()
    assert live.read_text(encoding="utf-8").startswith("IbLoginId=\nIbPassword=\nTradingMode=live\n")
    monkeypatch.setattr(session, "env_value", {}.get)
    with pytest.raises(RuntimeError, match="VIEWS_IB_LIVE_USER and VIEWS_IB_LIVE_PASS"):
        session.write_live_config()


def test_live_passcode_gate_locks_after_five_misses(tmp_path, monkeypatch):
    from dashboard.live_switch import PASSCODE_ENV, PasscodeGate, press
    from utils import helpers
    env = tmp_path / ".env"
    monkeypatch.delenv(PASSCODE_ENV, raising=False)
    monkeypatch.setattr(helpers, "ENV_FILE", env)
    monkeypatch.setattr(helpers.env_value, "__defaults__", (env,))
    now = [0.0]
    gate = PasscodeGate(clock=lambda: now[0])
    accepted, why = gate.check("1234")
    assert not accepted and "No live passcode" in why and gate.misses == 0  # unset is not a miss
    env.write_text("# broker\nexport VIEWS_LIVE_PASSCODE='482913'\n", encoding="utf-8")
    assert gate.check("482913") == (True, "")
    for left in (4, 3, 2, 1):
        assert gate.check("0000") == (False, f"Wrong passcode: {left} tries left.")
    assert "Locked for 15 min" in gate.check("0000")[1]
    assert not gate.check("482913")[0]  # even the right code waits out the lockout
    now[0] += 15 * 60
    assert gate.check("482913") == (True, "")
    digits = ""
    for key in ["4", "8", "2", "back", "clear", "9", "0"]:
        digits = press(digits, key)
    assert digits == "90" and press("1" * 12, "5") == "1" * 12


def test_quote_rows_and_futures_table_render():
    import datetime as dt
    import math
    from dashboard import futures
    from execution.quotes import quote_row
    ticker = NS(contract=NS(localSymbol="ZNZ6", conId=1), bid=110.5, ask=110.515625, last=110.515625, close=110.25,
                bidSize=120, askSize=math.nan, volume=5000, marketDataType=3,
                time=dt.datetime(2026, 11, 11, 15, 0, tzinfo=dt.timezone.utc))
    row = quote_row("ZN", ticker)
    assert row["change"] == pytest.approx(0.265625) and row["ask_size"] is None and row["delayed"]
    cells = str(futures._table([row]).to_plotly_json())
    assert "110-16+" in cells and "+8.50" in cells and "10:00:00" in cells and row["root"] == "TY"
    silent = quote_row("ZB", NS(contract=NS(localSymbol="ZBZ6", conId=2), bid=math.nan, ask=math.nan, last=math.nan,
                                close=math.nan, bidSize=math.nan, askSize=math.nan, volume=math.nan,
                                marketDataType=1, time=None))
    assert silent["last"] is None and silent["change"] is None
    futures._table([silent])


def test_tsy_prices_read_back_from_32nds():
    from utils.formatting import format_tsy_price, parse_tsy_price
    assert parse_tsy_price("110-16+") == 110.515625 and parse_tsy_price("104-032") == pytest.approx(104 + 3.25 / 32)
    assert parse_tsy_price(" 110.5 ") == 110.5 and parse_tsy_price("110-32") is None and parse_tsy_price("x") is None
    for price, tick in [(103.9921875, 0.25), (104.0703125, 0.25), (109.50390625, 0.125), (121.3125, 1)]:
        assert parse_tsy_price(format_tsy_price(price, is_ticks=False, tick_32nds=tick)) == pytest.approx(price)


def test_order_ticket_stats_follow_the_typed_size_and_limits():
    from dashboard.trades import _ticket_stats
    # a 5s7s steepener: buy FV, sell TY, $750/bp; FV $45/bp each, TY $62.5/bp each
    ticket = dict(target=750, exp_bps=4.0, roots={"FV": dict(weight=-1.0, dv01=45.0), "TY": dict(weight=1.0, dv01=62.5)},
                  legs=[dict(root="FV", action="BUY", bid=104.0, ask=104.0078125),
                        dict(root="TY", action="SELL", bid=110.5, ask=110.515625)])
    legs, stats = _ticket_stats(ticket, [17, 12], ["104-002", "110-16"])  # both at the far touch
    assert legs == ["-$765", "+$750"]
    text = str(stats.to_plotly_json())
    # package DV01 (765 + 750) / 2 = 757.5; each leg pays half its spread over mid at $1,000 a point:
    # FV 1/8 32nd on 17 ($66.41), TY 1/4 32nd on 12 ($93.75), $160.16 = 0.21 bp of 757.5
    assert "+$758" in text and "target $750" in text and "'29'" in text
    assert "+$160" in text and "+0.21 bp" in text and "+$3,030" in text
    _legs, blank = _ticket_stats(ticket, [17, 12], ["104-002", ""])
    assert "cost vs mid" in str(blank.to_plotly_json()) and "+$160" not in str(blank.to_plotly_json())


def test_spread_orders_leave_the_least_dv01_legged():
    from execution.ticket import spread_orders
    dv01 = {"TU": 38.0, "FV": 45.0, "TY": 62.5, "UXY": 80.0, "US": 130.0}
    # a steepener: buy 7 TU, sell 4 UXY; TUX trades 5:2, so one spread and 2 of each outright
    orders, left = spread_orders({"TU": 7, "UXY": -4}, dv01, {"TUX": ("TU", "UXY", 5, 2)})
    assert [(o["name"], o["action"], o["quantity"]) for o in orders] == [("TUX", "BUY", 1)]
    assert orders[0]["parts"] == [("TU", 5), ("UXY", -2)] and left == {"TU": 2, "UXY": -2}
    # a fly sold through two spreads sharing the belly: nothing left to leg
    orders, left = spread_orders({"FV": -3, "TY": 4, "US": -1}, dv01,
                                 {"FYT": ("FV", "TY", 3, 2), "NOB": ("TY", "US", 2, 1), "FOB": ("FV", "US", 3, 1)})
    assert sorted((o["name"], o["action"], o["quantity"]) for o in orders) == [("FYT", "SELL", 1), ("NOB", "BUY", 1)]
    assert left == {"FV": 0, "TY": 0, "US": 0}
    # the belly cannot feed both spreads twice over: the choice is the one leaving the least DV01
    orders, left = spread_orders({"FV": -6, "TY": 4, "US": -2}, dv01,
                                 {"FYT": ("FV", "TY", 3, 2), "NOB": ("TY", "US", 2, 1)})
    assert all(left[r] * s >= 0 for r, s in {"FV": -6, "TY": 4, "US": -2}.items())
    assert sum(abs(n) * dv01[r] for r, n in left.items()) == min(
        (6 - 3 * a) * 45 + (4 - 2 * a - 2 * b) * 62.5 + (2 - b) * 130
        for a in range(3) for b in range(3) if 2 * a + 2 * b <= 4)
    # same-way legs, or too few contracts for one spread: everything outright
    assert spread_orders({"TU": 5, "FV": 5}, dv01, {"TUF": ("TU", "FV", 1, 1)}) == ([], {"TU": 5, "FV": 5})
    assert spread_orders({"TY": 1, "US": -1}, dv01, {"NOB": ("TY", "US", 2, 1)}) == ([], {"TY": 1, "US": -1})


def test_legging_rules_size_price_and_pick_the_lead():
    from execution.legging import hedge_due, hedge_price, pick_lead, thin_hedges
    # 12 TY hedge against a 5-lot lead, filled 1, then 3, then all 5
    assert [hedge_due(f, 5, 12, s) for f, s in [(1, 0), (3, 2), (5, 7)]] == [2, 5, 5]
    assert hedge_due(2.5, 5, 7, 0) == 4 and hedge_due(0, 5, 12, 0) == 0  # half rounds up; nothing before a fill
    tick = 1 / 64
    assert hedge_price("BUY", 105.0, 105.015625, tick, 106.0) == 105.015625 + 2 * tick  # 2 ticks through the ask
    assert hedge_price("SELL", 105.0, 105.015625, tick, 104.9921875) == 104.9921875  # never past the cap
    assert hedge_price("BUY", None, None, tick, 105.5) == 105.5  # no market: the cap
    legs = [dict(contract="TYZ6", action="SELL", quantity=12, bid_size=900.0, ask_size=1000.0),
            dict(contract="UXYZ6", action="BUY", quantity=17, bid_size=150.0, ask_size=40.0),
            dict(contract="WNZ6", action="SELL", quantity=5, bid_size=20.0, ask_size=30.0)]
    assert pick_lead(legs) == 1  # 17 of 40 at the ask is the hardest to fill
    assert thin_hedges(legs, 2) == ["UXYZ6: 17 to hedge vs 40 at the ask"][:0] and \
        thin_hedges([dict(legs[0], quantity=950)] + legs[1:], 1) == ["TYZ6: 950 to hedge vs 900 at the bid"]
    assert pick_lead([dict(legs[0], bid_size=None)] + legs[1:]) == 0  # no size showing: hardest


def test_contingent_legging_hedges_each_lead_fill_and_resumes(gateway, monkeypatch):
    from execution import session as session_module
    from execution.session import Session
    from execution.journal import TradeJournal
    gateway.up = True
    ib = SessionIB()
    q = dict(bid=105.0, ask=105.015625, bidSize=900, askSize=800, last=105.0)
    streams = {"ZN": NS(contract=NS(localSymbol="ZNZ6", conId=1), **q),
               "UB": NS(contract=NS(localSymbol="UBZ6", conId=2), bid=120.0, ask=120.03125, bidSize=50,
                        askSize=60, last=120.0)}

    async def subscribe(_ib):  # the futures stream, as each connect subscribes it
        return dict(streams)

    monkeypatch.setattr(session_module, "subscribe", subscribe)
    session = Session(lambda: ib)
    session.connect()
    _wait(session, "connected")
    session._tickers = dict(streams)
    lead = dict(root="WN", action="SELL", quantity=4, limit=120.03125)
    hedge = dict(root="TY", action="BUY", quantity=10, cap=105.0625)
    rows = session.submit_contingent(lead, [hedge], "views:legs", "intent-1", "sig-1")
    assert [r["status"] for r in rows] == ["Submitted", "Waiting for lead"] and len(ib.placed) == 1  # lead only
    trade = ib.trades[0]

    def fill(exec_id, shares):
        trade.fills.append(NS(execution=NS(execId=exec_id, shares=shares, permId=trade.order.permId,
                                           orderId=trade.order.orderId)))
        for handler in list(trade.fillEvent):
            handler(trade, trade.fills[-1])

    fill("e1", 1)  # 1 of 4: 10/4 = 2.5 rounds to 3 TY, at the ask + 2 ticks of 1/64
    assert ib.placed[-1] == ("ZNZ6", "BUY", 3, 105.046875, "views:legs")
    fill("e1", 1)  # IBKR resending the same execution hedges nothing more
    assert len(ib.placed) == 2
    streams["ZN"].ask = 105.25  # the market runs: the hedge stops at its cap
    fill("e2", 2)
    assert ib.placed[-1] == ("ZNZ6", "BUY", 5, 105.0625, "views:legs")  # 3 of 4 filled: 8 of 10, 3 sent
    [plan] = TradeJournal().records("legging")
    assert plan["hedges"][0]["sent"] == 8 and not plan["done"]
    logged = [o for sub in TradeJournal().records("orders") for o in sub["orders"]]
    assert sorted(o["quantity"] for o in logged) == [3, 5] and all(o["role"] == "hedge" for o in logged)

    # the dashboard restarts; the last lot fills while it is away; reconnecting catches up
    trade.fills.append(NS(execution=NS(execId="e3", shares=1, permId=trade.order.permId, orderId=trade.order.orderId)))
    trade.orderStatus.status = "Filled"
    session.disconnect()
    session.connect()
    _wait(session, "connected")
    deadline = time.monotonic() + 5
    while ib.placed[-1][2] != 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ib.placed[-1][:3] == ("ZNZ6", "BUY", 2)  # the last 2 of 10
    assert TradeJournal().records("legging")[0]["done"]
    session.disconnect()

    # a refused lead sends no hedge at all
    ib.refuse.add("UBZ6")
    session = Session(lambda: ib)
    session.connect()
    _wait(session, "connected")
    before = len(ib.placed)
    lead_row, held = session.submit_contingent(dict(lead, limit=105.0), [hedge], "views:no", "intent-2", "sig-1")
    assert lead_row["status"] == "Inactive" and held["status"] == "Not sent" and len(ib.placed) == before + 1
    session.disconnect()


def test_ticket_stats_count_a_spread_in_both_its_legs():
    from dashboard.sheet import usd
    from dashboard.trades import _ticket_stats
    ticket = dict(target=750, exp_bps=None, roots={"TY": dict(weight=1.0, dv01=62.5), "US": dict(weight=-1.0, dv01=130.0)},
                  legs=[dict(kind="spread", contract="NOB 02-01 Z6", action="SELL", parts=[["TY", -2], ["US", 1]],
                             tick=0.015625, bid=None, ask=None),
                        dict(root="TY", action="SELL", bid=110.5, ask=110.515625)])
    shown, stats = _ticket_stats(ticket, [3, 1], ["+12.500", "110-16"])
    # 3 spreads short 6 TY, long 3 US; the outright shorts one more TY
    assert shown == [usd(6 * 62.5 - 3 * 130), usd(62.5)]
    text = str(stats.to_plotly_json())
    assert "'10'" in text and "cost vs mid" in text and "+$" not in text.split("cost vs mid")[1].split("expected")[0]


def test_ticket_legs_net_the_hedge_and_round_to_contracts():
    from execution.ticket import contract_legs, yield_legs
    legs = yield_legs("7s10s30s", -1, "10y", 0.1)  # short the fly, belly hedge netted into the belly
    assert legs == pytest.approx({"7y": 1.0, "10y": -1.9, "30y": 1.0})
    assert yield_legs("5s7s", 1) == {"5y": -1.0, "7y": 1.0}
    dv01 = {"TY": dict(contract="TYZ6", dv01=50.0), "UXY": dict(contract="UXYZ6", dv01=80.0)}
    seven, ten, thirty = contract_legs(legs, 750, dv01)
    assert (seven["action"], seven["quantity"], seven["residual"]) == ("SELL", 15, 0.0)  # long the yield
    assert (ten["action"], ten["quantity"]) == ("BUY", 18) and ten["exact"] == pytest.approx(17.8125)
    assert ten["residual"] == pytest.approx(-1425 + 18 * 80)
    assert thirty["root"] == "WN" and thirty["quantity"] is None  # no $/bp for it: left unsized
