"""Tests for hypernix.t1api.billing — balances, transactions, and payment
tokens. This is an internal ledger, not a payment-processor integration —
see the module docstring in t1api/billing.py."""
from __future__ import annotations

import time

import pytest

from hypernix.t1api.billing import BillingLedger, TransactionKind
from hypernix.t1api.db import SQLiteBackend
from hypernix.t1api.errors import T1APIError, T1ErrorCode


@pytest.fixture
def ledger(tmp_path) -> BillingLedger:
    return BillingLedger(SQLiteBackend(tmp_path / "t1api.sqlite3"))


class TestBalances:
    def test_fresh_account_has_zero_balance(self, ledger: BillingLedger):
        assert ledger.get_balance("user", "u1") == 0.0

    def test_add_balance_increases_balance(self, ledger: BillingLedger):
        ledger.add_balance("user", "u1", 100.0, created_by="admin1")
        assert ledger.get_balance("user", "u1") == 100.0

    def test_add_balance_rejects_non_positive(self, ledger: BillingLedger):
        with pytest.raises(T1APIError) as exc_info:
            ledger.add_balance("user", "u1", -5, created_by="admin1")
        assert exc_info.value.code == T1ErrorCode.VALIDATION_ERROR

    def test_add_balance_rejects_zero(self, ledger: BillingLedger):
        with pytest.raises(T1APIError):
            ledger.add_balance("user", "u1", 0, created_by="admin1")

    def test_charge_decreases_balance(self, ledger: BillingLedger):
        ledger.add_balance("user", "u1", 100.0, created_by="admin1")
        ledger.charge("user", "u1", 30.0, created_by="system")
        assert ledger.get_balance("user", "u1") == 70.0

    def test_charge_beyond_balance_raises_insufficient_balance(self, ledger: BillingLedger):
        ledger.add_balance("user", "u1", 10.0, created_by="admin1")
        with pytest.raises(T1APIError) as exc_info:
            ledger.charge("user", "u1", 100.0, created_by="system")
        assert exc_info.value.code == T1ErrorCode.INSUFFICIENT_BALANCE

    def test_failed_charge_does_not_change_balance(self, ledger: BillingLedger):
        ledger.add_balance("user", "u1", 10.0, created_by="admin1")
        with pytest.raises(T1APIError):
            ledger.charge("user", "u1", 100.0, created_by="system")
        assert ledger.get_balance("user", "u1") == 10.0

    def test_accounts_are_independent(self, ledger: BillingLedger):
        ledger.add_balance("user", "u1", 100.0, created_by="admin1")
        assert ledger.get_balance("user", "u2") == 0.0

    def test_account_type_distinguishes_accounts_with_same_id(self, ledger: BillingLedger):
        """A server and a user could share the same literal id string —
        (account_type, account_id) together are the key, not id alone."""
        ledger.add_balance("user", "shared-id", 50.0, created_by="admin1")
        assert ledger.get_balance("server", "shared-id") == 0.0


class TestTransactions:
    def test_transaction_id_is_masked_in_to_dict(self, ledger: BillingLedger):
        txn = ledger.add_balance("user", "u1", 100.0, created_by="admin1")
        assert txn.to_dict()["transaction_id"].startswith("txn_")

    def test_transactions_lists_history_for_account(self, ledger: BillingLedger):
        ledger.add_balance("user", "u1", 100.0, created_by="admin1")
        ledger.charge("user", "u1", 20.0, created_by="system")
        txns = ledger.transactions("user", "u1")
        assert len(txns) == 2
        assert txns[0].kind == TransactionKind.CHARGE  # most recent first

    def test_transactions_scoped_to_account(self, ledger: BillingLedger):
        ledger.add_balance("user", "u1", 100.0, created_by="admin1")
        ledger.add_balance("user", "u2", 100.0, created_by="admin1")
        assert len(ledger.transactions("user", "u1")) == 1


class TestPaymentTokens:
    def test_mint_returns_raw_token_once(self, ledger: BillingLedger):
        raw_token, record = ledger.mint_payment_token(50.0, created_by="admin1")
        assert raw_token.startswith("T1PAY_")
        assert record.redeemed is False

    def test_raw_token_never_appears_in_masked_dict(self, ledger: BillingLedger):
        raw_token, record = ledger.mint_payment_token(50.0, created_by="admin1")
        assert raw_token not in str(record.to_dict())

    def test_mint_rejects_non_positive_amount(self, ledger: BillingLedger):
        with pytest.raises(T1APIError) as exc_info:
            ledger.mint_payment_token(0, created_by="admin1")
        assert exc_info.value.code == T1ErrorCode.VALIDATION_ERROR

    def test_redeem_credits_target_account(self, ledger: BillingLedger):
        raw_token, _ = ledger.mint_payment_token(50.0, created_by="admin1")
        ledger.redeem(raw_token, account_type="user", account_id="u2", redeemed_by="u2")
        assert ledger.get_balance("user", "u2") == 50.0

    def test_redeem_returns_redeem_kind_transaction(self, ledger: BillingLedger):
        raw_token, _ = ledger.mint_payment_token(50.0, created_by="admin1")
        txn = ledger.redeem(raw_token, account_type="user", account_id="u2", redeemed_by="u2")
        assert txn.kind == TransactionKind.REDEEM

    def test_double_redeem_rejected(self, ledger: BillingLedger):
        raw_token, _ = ledger.mint_payment_token(50.0, created_by="admin1")
        ledger.redeem(raw_token, account_type="user", account_id="u2", redeemed_by="u2")
        with pytest.raises(T1APIError) as exc_info:
            ledger.redeem(raw_token, account_type="user", account_id="u3", redeemed_by="u3")
        assert exc_info.value.code == T1ErrorCode.PAYMENT_TOKEN_ALREADY_REDEEMED

    def test_double_redeem_does_not_credit_second_account(self, ledger: BillingLedger):
        raw_token, _ = ledger.mint_payment_token(50.0, created_by="admin1")
        ledger.redeem(raw_token, account_type="user", account_id="u2", redeemed_by="u2")
        with pytest.raises(T1APIError):
            ledger.redeem(raw_token, account_type="user", account_id="u3", redeemed_by="u3")
        assert ledger.get_balance("user", "u3") == 0.0

    def test_invalid_token_rejected(self, ledger: BillingLedger):
        with pytest.raises(T1APIError) as exc_info:
            ledger.redeem("T1PAY_not_a_real_token", account_type="user", account_id="u1", redeemed_by="u1")
        assert exc_info.value.code == T1ErrorCode.PAYMENT_TOKEN_INVALID

    def test_expired_token_rejected(self, ledger: BillingLedger):
        raw_token, _ = ledger.mint_payment_token(50.0, created_by="admin1", expires_at=time.time() - 10)
        with pytest.raises(T1APIError) as exc_info:
            ledger.redeem(raw_token, account_type="user", account_id="u1", redeemed_by="u1")
        assert exc_info.value.code == T1ErrorCode.PAYMENT_TOKEN_INVALID

    def test_non_expired_token_with_future_expiry_works(self, ledger: BillingLedger):
        raw_token, _ = ledger.mint_payment_token(50.0, created_by="admin1", expires_at=time.time() + 3600)
        ledger.redeem(raw_token, account_type="user", account_id="u1", redeemed_by="u1")
        assert ledger.get_balance("user", "u1") == 50.0

    def test_get_payment_token_record_by_id(self, ledger: BillingLedger):
        raw_token, record = ledger.mint_payment_token(50.0, created_by="admin1")
        fetched = ledger.get_payment_token_record(record.payment_token_id)
        assert fetched is not None
        assert fetched.amount == 50.0

    def test_get_payment_token_record_unknown_returns_none(self, ledger: BillingLedger):
        assert ledger.get_payment_token_record("nope") is None
