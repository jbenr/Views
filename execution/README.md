# execution/ — Interactive Brokers connection and trade review

## Portfolio dashboard setup

Persistent local settings can be stored in `store/ibkr.json` (under
`VIEWS_STORE_DIR` when set), for example
`{"host": "127.0.0.1", "port": 4001, "client_id": 1}` for live Gateway.
This file is git-ignored and read on each refresh. `VIEWS_IB_*` environment
variables override these values. No credentials are needed in this file.

1. Install the optional connector in the dashboard environment: `pip install -e ".[ibkr]"`.
2. Log into TWS or IB Gateway (start with paper). Enable socket clients and
   keep **Read-Only API** enabled. The app never stores your IBKR password.
3. Set `VIEWS_IB_HOST`, `VIEWS_IB_PORT`, and a dedicated positive
   `VIEWS_IB_CLIENT_ID` in the shell launching the dashboard. Set
   `VIEWS_IB_ACCOUNT` if the login has multiple accounts. Paper TWS usually
   uses 7497; paper Gateway usually uses 4002. Check the actual API settings.
4. Launch `python -m dashboard.app`, open **Portfolio**, and click
   **Connect IBKR**. No network request occurs on page load.

**Connect IBKR** (`execution/session.py`) holds one read-only connection for
as long as the dashboard process runs. If nothing is listening on the API port
and the host is local, it starts the newest IB Gateway under `C:\Jts\ibgateway`
and waits up to five minutes for you to log in there (password and 2FA are
typed into Gateway, never into the dashboard), then connects and takes a
snapshot. The button becomes **Cancel** while waiting and **Disconnect** once
connected; Gateway itself is left running. A dropped connection (Gateway's
daily restart) shows as not connected and is one click to restore. A dashboard
restart or `--reload` drops the connection too.

**Automatic login.** With [IBC](https://github.com/IbcAlpha/IBC) unpacked at
`C:\IBC` and `IbLoginId` / `IbPassword` filled in at
`%USERPROFILE%\Documents\IBC\config.ini` (outside the repo, plaintext), Connect
starts Gateway through IBC, which fills in the login form; you only approve the
2FA prompt on your phone. A missed prompt is re-sent. The dashboard never reads
the password. IBC logs to `C:\IBC\Logs`. With `IbLoginId` blank, Connect falls
back to opening Gateway for a manual login.

**Gateway needs a desktop.** Its login window only opens in a Windows session a
user is logged on to. A dashboard started over SSH (the tmux session) runs in
session 0, which has none, so Connect starts Gateway through the scheduled task
`IB Gateway` instead, which runs `C:\IBC\launch_gateway.cmd` (written by Connect,
no credentials) on the logged-on desktop. Someone must be logged on to Windows
on that machine; a disconnected Remote Desktop session counts, a signed-out one
does not. Create the task once, in PowerShell:

```powershell
Register-ScheduledTask -TaskName 'IB Gateway' `
  -Action (New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/c C:\IBC\launch_gateway.cmd') `
  -Principal (New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive) `
  -Settings (New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero))
```

The connector uses `ib_async`. **Refresh IBKR** reuses the open connection, or
without one opens a read-only connection just for that refresh. It requests
balances, positions, all open orders, recent completed orders and executions,
filtered to one account. Refresh has a 45-second overall deadline. A failed refresh preserves the previous snapshot with its
original timestamp. This is a snapshot viewer, not a streaming order monitor.
TWS order/execution history is limited; this is not a full historical broker
reconciliation or strategy PnL service. Account summaries retain currencies;
the UI does not add balances across currencies or infer market values.

**Flag for review** saves a local pending review linked to the exact promoted
signal/variant, strategy module and latest signal run (if any). Duplicate flags
for the same signal/run are ignored. A flag is not an order preview or approval;
instrument expiry, legs and DV01 sizing still need an explicit ticket workflow.
The stable `order_ref` is reserved for future broker submissions. Orders and
fills are attributed only when their reference matches a local intent; net
broker positions remain unattributed. Submission stays disabled.

Snapshots and flags live in `store/execution.sqlite3` (or under
`VIEWS_STORE_DIR`), separate from the research ledger. Treat this local database
as private account data. Each snapshot preserves the statuses returned by IBKR;
pending review flags have a separate local status and do not become broker orders.

API references: [ib_async](https://ib-api-reloaded.github.io/ib_async/api.html),
[IBKR positions](https://www.interactivebrokers.com/docs/tws-api/doc/accounts/positions).

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
3. Install the optional `ibkr` extra described above.
4. A future implementation can wire submission into `IBKRExecutor.submit()` where the
   `NotImplementedError` stub is. Keep the env-var gate.

## Intended integration

Per `notes/PLAN.md`: signals → ledger (timestamped JSON) → model portfolio →
**then** execution. The adapter's job is only the last hop: take a
`TargetPosition` produced from a logged signal, preview it, and (eventually)
submit + record fills for broker reconciliation. Sizing should come from
`backtest.sizing` DV01 logic, not ad-hoc contract counts.
