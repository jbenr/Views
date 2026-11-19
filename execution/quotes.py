"""Streaming Treasury futures quotes from an open IBKR connection."""
from __future__ import annotations

from execution.portfolio import _number

# CBOT symbol -> (the root it is known by, minimum tick in 32nds)
FUTURES = {
    "ZT": ("TU", 0.125), "ZF": ("FV", 0.25), "ZN": ("TY", 0.5),
    "TN": ("UXY", 0.5), "ZB": ("US", 1), "UB": ("WN", 1),
}

# CME inter-commodity spreads IBKR lists on those futures: name -> its two roots, front leg first.
# Each trades as one instrument at the exchange's own leg ratio, which IBKR's contract name carries.
SPREADS = {
    "TUF": ("TU", "FV"), "TUT": ("TU", "TY"), "TUX": ("TU", "UXY"), "TUB": ("TU", "US"), "TUL": ("TU", "WN"),
    "FYT": ("FV", "TY"), "FOB": ("FV", "US"), "FOL": ("FV", "WN"), "NOB": ("TY", "US"), "NOL": ("TY", "WN"),
    "NUB": ("UXY", "WN"), "BOB": ("US", "WN"),
}
MONTH_CODES = "FGHJKMNQUVXZ"


async def subscribe(ib):
    """Tickers that keep updating, for IBKR's lead contract of each future, keyed by symbol."""
    from ib_async import ContFuture, Future
    ib.reqMarketDataType(3)  # real time where the account is subscribed, delayed otherwise
    tickers = {}
    for symbol in FUTURES:
        [lead] = await ib.qualifyContractsAsync(ContFuture(symbol, "CBOT"))
        [contract] = await ib.qualifyContractsAsync(Future(conId=lead.conId, exchange="CBOT"))
        tickers[symbol] = ib.reqMktData(contract)
    return tickers


async def spread_contract(ib, name):
    """The nearest listed contract of a spread. Its localSymbol reads 'TUX 05-02 Z6':
    name, leg ratio, month."""
    from ib_async import Contract
    listed = await ib.reqContractDetailsAsync(Contract(secType="ICS", symbol=name, exchange="CBOT", currency="USD"))

    def month(details):
        code = details.contract.localSymbol.split()[-1]
        return int(code[1:]), MONTH_CODES.index(code[0])

    return min(listed, key=month).contract


def spread_row(name, ticker):
    """Prices are decimal points of the spread, as the exchange quotes it."""
    _, ratio, month = ticker.contract.localSymbol.split()
    front, back = (int(part) for part in ratio.split("-"))
    return dict(name=name, legs=f"{front} {SPREADS[name][0]} : {back} {SPREADS[name][1]}", month=month,
                bid=_number(ticker.bid), ask=_number(ticker.ask), last=_number(ticker.last),
                bid_size=_number(ticker.bidSize), ask_size=_number(ticker.askSize),
                volume=_number(ticker.volume), time=ticker.time.isoformat() if ticker.time else None)


def quote_row(symbol, ticker):
    """Prices are decimal points; change is against the prior settlement."""
    root, tick = FUTURES[symbol]
    last, close = _number(ticker.last), _number(ticker.close)
    return dict(symbol=symbol, root=root, tick=tick, contract=ticker.contract.localSymbol,
                bid=_number(ticker.bid), ask=_number(ticker.ask), last=last, close=close,
                change=last - close if last is not None and close is not None else None,
                bid_size=_number(ticker.bidSize), ask_size=_number(ticker.askSize),
                volume=_number(ticker.volume), time=ticker.time.isoformat() if ticker.time else None,
                delayed=ticker.marketDataType in (3, 4))
