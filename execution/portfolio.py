"""On-demand, read-only IBKR snapshots. No order mutation methods are exposed."""
from __future__ import annotations

import asyncio
import datetime as dt
import threading
import math

from execution.ibkr import IBKRConfig

_LOCK = threading.Lock()


def contract_row(c):
    return dict(con_id=c.conId, symbol=c.localSymbol or c.symbol,
                sec_type=c.secType, currency=c.currency,
                expiry=c.lastTradeDateOrContractMonth, multiplier=c.multiplier)


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and abs(value) < 1e100 else None
    except (ValueError, TypeError):
        return None


def position_row(p, marks):
    c = p.contract
    mark = marks.get((p.account, c.conId))
    # Ignore a mark for a different position size; never invent P&L from a stale cost.
    if mark is not None and mark.position != p.position:
        mark = None
    multiplier = _number(c.multiplier) if c.multiplier else 1.0
    cost = _number(p.avgCost)
    average = cost / multiplier if cost is not None and multiplier and multiplier > 0 else None
    return dict(**contract_row(c), quantity=str(p.position), avg_cost=average,
                current_price=_number(mark.marketPrice) if mark else None,
                unrealized_pnl=_number(mark.unrealizedPNL) if mark else None,
                strategy="Unattributed")


async def _snapshot(config, factory=None):
    if config.client_id <= 0:
        raise ValueError("Use a positive, dedicated VIEWS_IB_CLIENT_ID for read-only access.")
    if factory is None:
        try:
            from ib_async import IB
        except ImportError as exc:
            raise RuntimeError('Install the broker extra: pip install -e ".[ibkr]"') from exc
        factory = IB
    ib = factory()
    ib.RaiseRequestErrors = True
    try:
        try:
            await ib.connectAsync(config.host, config.port, clientId=config.client_id,
                                  readonly=True, timeout=10, account=config.account or "",
                                  raiseSyncErrors=True)
        except ConnectionRefusedError as exc:
            raise ConnectionError(
                f"IBKR refused the connection to {config.host}:{config.port}. "
                "Start TWS or IB Gateway and log in, enable ActiveX and Socket Clients "
                "under API Settings, and match its Socket Port to VIEWS_IB_PORT. "
                "Keep Read-Only API enabled. Restart the dashboard after changing "
                "environment variables."
            ) from exc
        accounts = ib.managedAccounts()
        if config.account and config.account not in accounts:
            raise ValueError("Configured IBKR account is not available in this session.")
        if not config.account and len(accounts) != 1:
            raise ValueError("Set VIEWS_IB_ACCOUNT to select exactly one managed account.")
        account = config.account or accounts[0]
        balances = await ib.accountSummaryAsync()
        positions = await ib.reqPositionsAsync()
        # connectAsync already downloads the portfolio for its selected account.
        # Re-requesting an existing subscription may not send another end marker.
        if positions and not ib.portfolio(account):
            await asyncio.wait_for(ib.reqAccountUpdatesAsync(account), 10)
        orders = await ib.reqAllOpenOrdersAsync()
        warnings = []
        async def optional_history(label, request):
            try:
                return await asyncio.wait_for(request, 5)
            except (TimeoutError, ConnectionError) as exc:
                warnings.append(f"{label} unavailable ({type(exc).__name__}); refresh to retry.")
                return []
        completed = await optional_history("Completed-order history", ib.reqCompletedOrdersAsync(apiOnly=False))
        fills = await optional_history("Execution history", ib.reqExecutionsAsync())
        marks = {(p.account, p.contract.conId): p for p in ib.portfolio(account)}
        def order_row(t):
            o, s = t.order, t.orderStatus
            return dict(account=o.account, **contract_row(t.contract), order_id=o.orderId,
                        client_id=o.clientId, perm_id=o.permId, order_ref=o.orderRef,
                        action=o.action, quantity=str(o.totalQuantity), status=s.status,
                        filled=str(s.filled), remaining=str(s.remaining), avg_fill_price=s.avgFillPrice)
        return dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(), account=account, warnings=warnings,
                    balances=[dict(tag=v.tag, value=v.value, currency=v.currency)
                              for v in balances if v.account == account],
                    positions=[position_row(p, marks)
                               for p in positions if p.account == account],
                    orders=[order_row(t) for t in orders if t.order.account == account],
                    completed_orders=[order_row(t) for t in completed if t.order.account == account],
                    fills=[dict(**contract_row(f.contract), exec_id=f.execution.execId,
                                perm_id=f.execution.permId, order_ref=f.execution.orderRef,
                                time=str(f.time), side=f.execution.side,
                                quantity=str(f.execution.shares), price=f.execution.price)
                           for f in fills if f.execution.acctNumber == account])
    finally:
        ib.disconnect()


def fetch_snapshot(config=None, factory=None):
    # Each click owns its event loop and connection; serialize duplicate client IDs.
    with _LOCK:
        async def bounded():
            return await asyncio.wait_for(_snapshot(config or IBKRConfig.from_env(), factory), 45)
        return asyncio.run(bounded())
