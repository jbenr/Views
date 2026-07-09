"""Interactive Brokers execution adapter — boilerplate, dry-run by default.

This module translates strategy targets into order previews. It does NOT
place live orders: `IBKRExecutor.submit()` is stubbed and double-gated
(constructor opt-in AND environment variable) so nothing can trade by
accident while the ledger/reconciliation layers are being built.

Safety model:
  1. `IBKRExecutor(dry_run=True)` is the default — submit() just logs.
  2. Passing `dry_run=False` still refuses unless the environment variable
     VIEWS_IB_ALLOW_LIVE=1 is set.
  3. Even then, the actual API call is a NotImplementedError stub — wiring
     ib_insync/ibapi in is a deliberate, reviewed change.

Configuration comes from environment variables (never commit credentials):
  VIEWS_IB_HOST       default 127.0.0.1
  VIEWS_IB_PORT       default 7497 (TWS paper). 4002 = IB Gateway paper.
  VIEWS_IB_CLIENT_ID  default 1
  VIEWS_IB_ACCOUNT    no default — required only for live submission

See execution/README.md for IB Gateway / TWS setup instructions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ── configuration ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IBKRConfig:
    """Connection settings for TWS / IB Gateway. Build via from_env()."""

    host: str = "127.0.0.1"
    port: int = 7497          # 7497 TWS paper, 7496 TWS live, 4002/4001 Gateway
    client_id: int = 1
    account: str | None = None

    @classmethod
    def from_env(cls) -> IBKRConfig:
        return cls(
            host=os.getenv("VIEWS_IB_HOST", "127.0.0.1"),
            port=int(os.getenv("VIEWS_IB_PORT", "7497")),
            client_id=int(os.getenv("VIEWS_IB_CLIENT_ID", "1")),
            account=os.getenv("VIEWS_IB_ACCOUNT") or None,
        )


# ── instrument mapping ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContractSpec:
    """Minimal IB contract description (front-month future by default)."""

    symbol: str
    sec_type: str = "FUT"
    exchange: str = "CBOT"
    currency: str = "USD"


# Book instrument alias -> IB contract. Extend as strategies go live.
# Signals are researched on yields; the trade expression is the future.
# NOTE: for yield instruments, LONG DURATION (yields fall) = BUY the future.
CONTRACT_MAP: dict[str, ContractSpec] = {
    "2y":  ContractSpec("ZT"),
    "5y":  ContractSpec("ZF"),
    "10y": ContractSpec("ZN"),
    "30y": ContractSpec("ZB"),
}


# ── order translation ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TargetPosition:
    """What a strategy wants: an instrument, a direction, a size in risk units.

    direction is in PRICE space: +1 = buy the contract, -1 = sell. A strategy
    that is long duration on a yield signal maps to direction=+1 (buy future).
    """

    strategy: str
    instrument: str            # book alias, e.g. "10y"
    direction: int             # +1 buy, -1 sell
    risk_units: float = 1.0
    signal_value: float | None = None


@dataclass(frozen=True)
class OrderPreview:
    """A fully-resolved order that COULD be sent — human-readable, loggable."""

    contract: ContractSpec
    action: str                # "BUY" / "SELL"
    quantity: int
    order_type: str = "MKT"
    limit_price: float | None = None
    source: TargetPosition | None = field(default=None, repr=False)

    def describe(self) -> str:
        px = f" @ {self.limit_price}" if self.limit_price is not None else ""
        return (
            f"{self.action} {self.quantity} {self.contract.symbol} "
            f"{self.contract.sec_type} ({self.contract.exchange}) "
            f"{self.order_type}{px}"
        )


def target_to_preview(
    target: TargetPosition,
    contracts_per_risk_unit: float = 1.0,
) -> OrderPreview:
    """Translate a strategy target into an order preview.

    contracts_per_risk_unit is the sizing policy — how many contracts one
    standardized risk unit represents. Proper DV01-based sizing should come
    from backtest.sizing once strategies go live.
    """
    if target.instrument not in CONTRACT_MAP:
        raise KeyError(
            f"No IB contract mapped for '{target.instrument}' — add it to "
            "execution.ibkr.CONTRACT_MAP"
        )
    if target.direction not in (1, -1):
        raise ValueError(f"direction must be +1 or -1, got {target.direction}")

    quantity = max(1, round(abs(target.risk_units) * contracts_per_risk_unit))
    return OrderPreview(
        contract=CONTRACT_MAP[target.instrument],
        action="BUY" if target.direction == 1 else "SELL",
        quantity=quantity,
        source=target,
    )


# ── executor ─────────────────────────────────────────────────────────────────

class IBKRExecutor:
    """Order gateway. Dry-run by default; live submission is stubbed.

    Usage:
        executor = IBKRExecutor()                       # safe: dry-run
        previews = executor.preview(targets)
        executor.submit(previews[0])                    # logs, sends nothing
    """

    def __init__(self, config: IBKRConfig | None = None, dry_run: bool = True):
        self.config = config or IBKRConfig.from_env()
        self.dry_run = dry_run

    def preview(self, targets: list[TargetPosition]) -> list[OrderPreview]:
        return [target_to_preview(t) for t in targets]

    def submit(self, preview: OrderPreview) -> dict:
        """Submit an order. Dry-run logs and returns a record; live is gated.

        Live submission requires ALL of:
          - IBKRExecutor(dry_run=False)
          - environment variable VIEWS_IB_ALLOW_LIVE=1
          - an account id in the config
        and even then raises NotImplementedError until the IB API call is
        deliberately wired in.
        """
        if self.dry_run:
            record = {"status": "dry_run", "order": preview.describe()}
            print(f"[ibkr dry-run] would send: {preview.describe()}")
            return record

        if os.getenv("VIEWS_IB_ALLOW_LIVE") != "1":
            raise PermissionError(
                "Live submission blocked: set VIEWS_IB_ALLOW_LIVE=1 to opt in "
                "(and read execution/README.md first)."
            )
        if not self.config.account:
            raise PermissionError(
                "Live submission blocked: no account configured "
                "(set VIEWS_IB_ACCOUNT)."
            )

        # Deliberately unimplemented. When ready, wire ib_insync here:
        #   ib = IB(); ib.connect(self.config.host, self.config.port,
        #                         clientId=self.config.client_id)
        #   ... qualify contract, place order, return the trade record ...
        raise NotImplementedError(
            "Live order submission is not wired in yet — see execution/README.md"
        )
