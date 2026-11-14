import asyncio
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

    def fire(self):
        for handler in self:
            handler()


class SessionIB(FakeIB):
    def __init__(self):
        super().__init__()
        self.disconnectedEvent = _Handlers()
        self.errorEvent = _Handlers()


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
    monkeypatch.setattr(session, "auto_login", lambda: False)
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
