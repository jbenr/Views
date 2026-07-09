"""IBKR adapter safety: dry-run by default, live path blocked without opt-in."""

import pytest

from execution import IBKRConfig, IBKRExecutor, TargetPosition
from execution.ibkr import target_to_preview


def make_target(**kw):
    defaults = dict(strategy="curve", instrument="10y", direction=1, risk_units=1.0)
    defaults.update(kw)
    return TargetPosition(**defaults)


def test_target_to_preview():
    p = target_to_preview(make_target())
    assert p.contract.symbol == "ZN"
    assert p.action == "BUY"
    assert p.quantity == 1
    assert "BUY 1 ZN" in p.describe()

    p = target_to_preview(make_target(direction=-1, risk_units=3.0))
    assert p.action == "SELL"
    assert p.quantity == 3


def test_unmapped_instrument_raises():
    with pytest.raises(KeyError, match="CONTRACT_MAP"):
        target_to_preview(make_target(instrument="gilt10"))


def test_bad_direction_raises():
    with pytest.raises(ValueError):
        target_to_preview(make_target(direction=0))


def test_dry_run_is_default_and_sends_nothing():
    executor = IBKRExecutor()
    assert executor.dry_run is True
    record = executor.submit(executor.preview([make_target()])[0])
    assert record["status"] == "dry_run"


def test_live_blocked_without_env_optin(monkeypatch):
    monkeypatch.delenv("VIEWS_IB_ALLOW_LIVE", raising=False)
    executor = IBKRExecutor(dry_run=False)
    with pytest.raises(PermissionError, match="VIEWS_IB_ALLOW_LIVE"):
        executor.submit(executor.preview([make_target()])[0])


def test_live_blocked_without_account(monkeypatch):
    monkeypatch.setenv("VIEWS_IB_ALLOW_LIVE", "1")
    executor = IBKRExecutor(config=IBKRConfig(account=None), dry_run=False)
    with pytest.raises(PermissionError, match="account"):
        executor.submit(executor.preview([make_target()])[0])


def test_live_submission_stubbed_even_with_full_optin(monkeypatch):
    monkeypatch.setenv("VIEWS_IB_ALLOW_LIVE", "1")
    executor = IBKRExecutor(config=IBKRConfig(account="DU000000"), dry_run=False)
    with pytest.raises(NotImplementedError):
        executor.submit(executor.preview([make_target()])[0])
