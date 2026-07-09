# execution/ — Interactive Brokers adapter (boilerplate)

Translates strategy targets into order previews. **It does not and cannot
place live orders as shipped** — `submit()` is dry-run by default, live mode
is double-gated behind an env var, and the actual API call is a deliberate
`NotImplementedError` stub.

## What exists

- `IBKRConfig.from_env()` — connection settings from `VIEWS_IB_*` env vars.
  No credentials or account ids are committed to the repo.
- `TargetPosition` — what a strategy wants (instrument alias, ±1 direction
  in price space, risk units).
- `target_to_preview()` / `OrderPreview` — resolves the book alias through
  `CONTRACT_MAP` (e.g. `"10y"` → ZN future on CBOT) into a loggable order.
- `IBKRExecutor` — `preview()` and `submit()`. Dry-run prints and returns a
  record; nothing leaves the process.

```python
from execution import IBKRExecutor, TargetPosition

executor = IBKRExecutor()  # dry-run by default
targets = [TargetPosition(strategy="curve", instrument="10y", direction=1)]
for p in executor.preview(targets):
    executor.submit(p)     # "[ibkr dry-run] would send: BUY 1 ZN FUT (CBOT) MKT"
```

## Configuration (environment variables)

| Variable              | Default     | Notes                                   |
|-----------------------|-------------|-----------------------------------------|
| `VIEWS_IB_HOST`       | `127.0.0.1` |                                         |
| `VIEWS_IB_PORT`       | `7497`      | 7497 TWS paper, 7496 TWS live, 4002/4001 Gateway paper/live |
| `VIEWS_IB_CLIENT_ID`  | `1`         |                                         |
| `VIEWS_IB_ACCOUNT`    | *(none)*    | required only for live; never commit it |
| `VIEWS_IB_ALLOW_LIVE` | *(unset)*   | must be `1` for live mode to even be considered |

Put local values in your shell profile or a git-ignored `.env` — never in code.

## Setting up IB Gateway / TWS (when you're ready)

1. Install [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
   or TWS and log in to a **paper account** first.
2. In API settings: enable "ActiveX and Socket Clients", note the socket port,
   add `127.0.0.1` to trusted IPs, and keep "Read-Only API" ON until you have
   reconciliation working.
3. `pip install ib_insync` (or `ibapi`) — not a declared dependency on purpose.
4. Wire the connection into `IBKRExecutor.submit()` where the
   `NotImplementedError` stub is. Keep the env-var gate.

## Intended integration

Per `notes/PLAN.md`: signals → ledger (timestamped JSON) → model portfolio →
**then** execution. The adapter's job is only the last hop: take a
`TargetPosition` produced from a logged signal, preview it, and (eventually)
submit + record fills for broker reconciliation. Sizing should come from
`backtest.sizing` DV01 logic, not ad-hoc contract counts.
