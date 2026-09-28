"""t1api.billing — balances, transactions, and payment tokens.

Scope note up front: this is an **internal ledger**, not a payment
processor integration. There's no Stripe/PayPal/card-network call here —
"payment token" in the spec means an internal, admin-minted credit code
(think: a gift-card code) that an account redeems for ledger balance. A
real money-in path (charging an actual card) is a different, much bigger
integration that isn't part of this spec and isn't implemented.

Security properties, matching the spec's billing section directly:

* "Payment tokens MUST be treated as secrets... Never log or expose raw
  payment tokens" — :meth:`BillingLedger.mint_payment_token` returns the
  raw token exactly once, to the caller, and stores only its SHA-256 hash.
  There is no method anywhere in this class that returns a previously-
  minted token's raw value.
* "Use transaction IDs and masked identifiers for payment operations" —
  :meth:`Transaction.to_dict` and :meth:`PaymentTokenRecord.to_dict` both
  return masked forms (``txn_abcd1234…`` / ``T1PAY_abcd1234…``); the full
  underlying values exist in SQLite but are never serialized back out
  through the public dict methods.

Balances are keyed by ``(account_type, account_id)`` — matching "adding
balance to: user accounts, servers, organizations" — rather than assuming
every account is a T1 key.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .db import SQLiteBackend
from .errors import T1APIError, T1ErrorCode

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS billing_balances (
    account_type TEXT NOT NULL,
    account_id TEXT NOT NULL,
    balance REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    PRIMARY KEY (account_type, account_id)
);
CREATE TABLE IF NOT EXISTS billing_transactions (
    transaction_id TEXT PRIMARY KEY,
    account_type TEXT NOT NULL,
    account_id TEXT NOT NULL,
    amount REAL NOT NULL,
    kind TEXT NOT NULL,
    balance_after REAL NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_txn_account ON billing_transactions(account_type, account_id);
CREATE TABLE IF NOT EXISTS billing_payment_tokens (
    payment_token_id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    prefix TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL,
    redeemed INTEGER NOT NULL DEFAULT 0,
    redeemed_by_type TEXT,
    redeemed_by_id TEXT,
    redeemed_at REAL,
    created_by TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL
);
"""

_TOKEN_PREFIX = "T1PAY"


class TransactionKind(StrEnum):
    ADD_BALANCE = "add_balance"  # admin direct credit
    REDEEM = "redeem"  # payment-token redemption
    CHARGE = "charge"  # a debit (e.g. usage cost — not auto-wired in Beta 2)


@dataclass
class Transaction:
    transaction_id: str
    account_type: str
    account_id: str
    amount: float
    kind: TransactionKind
    balance_after: float
    note: str
    created_by: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": f"txn_{self.transaction_id[:8]}",
            "account_type": self.account_type,
            "account_id": self.account_id,
            "amount": self.amount,
            "kind": self.kind.value,
            "balance_after": self.balance_after,
            "note": self.note,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def _from_row(cls, row: Any) -> Transaction:
        return cls(
            transaction_id=row["transaction_id"],
            account_type=row["account_type"],
            account_id=row["account_id"],
            amount=row["amount"],
            kind=TransactionKind(row["kind"]),
            balance_after=row["balance_after"],
            note=row["note"],
            created_by=row["created_by"],
            created_at=row["created_at"],
        )


@dataclass
class PaymentTokenRecord:
    payment_token_id: str
    prefix: str
    amount: float
    currency: str
    redeemed: bool
    redeemed_by: tuple[str, str] | None  # (account_type, account_id)
    redeemed_at: float | None
    created_by: str
    created_at: float
    expires_at: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "payment_token_id": f"pt_{self.payment_token_id[:8]}",
            "masked_token": f"{_TOKEN_PREFIX}_{self.prefix}…",
            "amount": self.amount,
            "currency": self.currency,
            "redeemed": self.redeemed,
            "redeemed_by": (
                {"account_type": self.redeemed_by[0], "account_id": self.redeemed_by[1]}
                if self.redeemed_by
                else None
            ),
            "redeemed_at": self.redeemed_at,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


class BillingLedger:
    """Thread-safe, SQLite-backed balance/transaction/payment-token ledger."""

    def __init__(self, backend: SQLiteBackend | None = None) -> None:
        self.backend = backend or SQLiteBackend()
        self._lock = threading.Lock()
        self.backend.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Balances
    # ------------------------------------------------------------------

    def get_balance(self, account_type: str, account_id: str) -> float:
        with self._lock, self.backend.connect() as conn:
            row = conn.execute(
                "SELECT balance FROM billing_balances WHERE account_type=? AND account_id=?",
                (account_type, account_id),
            ).fetchone()
        return row["balance"] if row else 0.0

    def _apply_delta(
        self,
        *,
        account_type: str,
        account_id: str,
        amount: float,
        kind: TransactionKind,
        created_by: str,
        note: str = "",
    ) -> Transaction:
        now = time.time()
        with self._lock, self.backend.connect() as conn:
            row = conn.execute(
                "SELECT balance FROM billing_balances WHERE account_type=? AND account_id=?",
                (account_type, account_id),
            ).fetchone()
            current = row["balance"] if row else 0.0
            new_balance = current + amount
            if new_balance < 0:
                raise T1APIError(
                    T1ErrorCode.INSUFFICIENT_BALANCE,
                    f"Balance {current} insufficient for a charge of {-amount}.",
                    details={"account_type": account_type, "account_id": account_id, "balance": current},
                    http_status=402,
                )
            conn.execute(
                """INSERT INTO billing_balances (account_type, account_id, balance, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(account_type, account_id) DO UPDATE SET balance=excluded.balance, updated_at=excluded.updated_at""",
                (account_type, account_id, new_balance, now),
            )
            txn = Transaction(
                transaction_id=uuid.uuid4().hex,
                account_type=account_type,
                account_id=account_id,
                amount=amount,
                kind=kind,
                balance_after=new_balance,
                note=note,
                created_by=created_by,
                created_at=now,
            )
            conn.execute(
                """INSERT INTO billing_transactions
                   (transaction_id, account_type, account_id, amount, kind, balance_after,
                    note, created_by, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    txn.transaction_id, txn.account_type, txn.account_id, txn.amount,
                    txn.kind.value, txn.balance_after, txn.note, txn.created_by, txn.created_at,
                ),
            )
        logger.info(
            "t1api.billing: %s %+.2f for %s:%s -> balance=%.2f",
            kind.value, amount, account_type, account_id, new_balance,
        )
        return txn

    def add_balance(
        self, account_type: str, account_id: str, amount: float, *, created_by: str, note: str = ""
    ) -> Transaction:
        """Admin-only at the router layer — this call itself trusts the
        caller. POST /billing/add-balance."""
        if amount <= 0:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR, "add_balance amount must be positive.", http_status=422
            )
        return self._apply_delta(
            account_type=account_type, account_id=account_id, amount=amount,
            kind=TransactionKind.ADD_BALANCE, created_by=created_by, note=note,
        )

    def charge(
        self, account_type: str, account_id: str, amount: float, *, created_by: str, note: str = ""
    ) -> Transaction:
        """Debit *amount*. Raises ``INSUFFICIENT_BALANCE`` rather than
        allowing a negative balance. Not automatically wired to usage
        metering in Beta 2 — full cost-per-request billing integration is
        Beta 3 scope (the spec's "actual cost" / "estimated cost" usage
        endpoints)."""
        if amount <= 0:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "charge amount must be positive.", http_status=422)
        return self._apply_delta(
            account_type=account_type, account_id=account_id, amount=-amount,
            kind=TransactionKind.CHARGE, created_by=created_by, note=note,
        )

    def transactions(self, account_type: str, account_id: str, *, limit: int = 50) -> list[Transaction]:
        with self._lock, self.backend.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM billing_transactions WHERE account_type=? AND account_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (account_type, account_id, limit),
            ).fetchall()
        return [Transaction._from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Payment tokens
    # ------------------------------------------------------------------

    def mint_payment_token(
        self, amount: float, *, created_by: str, currency: str = "USD", expires_at: float | None = None
    ) -> tuple[str, PaymentTokenRecord]:
        """Admin-only at the router layer. Returns ``(raw_token, record)``
        — *raw_token* is shown to the caller exactly this once; only its
        hash is persisted, so there is no way to recover it later, by
        design (matches "Never log or expose raw payment tokens")."""
        if amount <= 0:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "Payment token amount must be positive.", http_status=422)
        raw_token = f"{_TOKEN_PREFIX}_{secrets.token_urlsafe(24)}"
        token_hash = _hash_token(raw_token)
        prefix = raw_token[len(_TOKEN_PREFIX) + 1 : len(_TOKEN_PREFIX) + 9]
        now = time.time()
        record = PaymentTokenRecord(
            payment_token_id=uuid.uuid4().hex,
            prefix=prefix,
            amount=amount,
            currency=currency,
            redeemed=False,
            redeemed_by=None,
            redeemed_at=None,
            created_by=created_by,
            created_at=now,
            expires_at=expires_at,
        )
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                """INSERT INTO billing_payment_tokens
                   (payment_token_id, token_hash, prefix, amount, currency, redeemed,
                    redeemed_by_type, redeemed_by_id, redeemed_at, created_by, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, 0, NULL, NULL, NULL, ?, ?, ?)""",
                (
                    record.payment_token_id, token_hash, prefix, amount, currency,
                    created_by, now, expires_at,
                ),
            )
        logger.info(
            "t1api.billing: minted payment token %s… (%s %s) by %s",
            prefix, amount, currency, created_by[:8] if created_by else "?",
        )
        return raw_token, record

    def redeem(
        self, raw_token: str, *, account_type: str, account_id: str, redeemed_by: str
    ) -> Transaction:
        """POST /billing/redeem. Validates the token by hash (the raw
        value is never stored, so this is the only way to "look it up"),
        checks it hasn't already been redeemed or expired, then credits
        the target account."""
        token_hash = _hash_token(raw_token)
        with self._lock, self.backend.connect() as conn:
            row = conn.execute(
                "SELECT * FROM billing_payment_tokens WHERE token_hash = ?", (token_hash,)
            ).fetchone()
            if row is None:
                raise T1APIError(
                    T1ErrorCode.PAYMENT_TOKEN_INVALID, "Payment token is invalid.", http_status=400
                )
            if row["redeemed"]:
                raise T1APIError(
                    T1ErrorCode.PAYMENT_TOKEN_ALREADY_REDEEMED,
                    "Payment token has already been redeemed.",
                    http_status=409,
                )
            if row["expires_at"] is not None and row["expires_at"] < time.time():
                raise T1APIError(
                    T1ErrorCode.PAYMENT_TOKEN_INVALID, "Payment token has expired.", http_status=400
                )
            now = time.time()
            # Conditional on redeemed=0 and checked via rowcount, rather
            # than trusting the SELECT above: the SELECT and this UPDATE
            # are two statements, and self._lock only excludes other
            # threads in *this* process — two processes (the production
            # target is multi-worker PostgreSQL) can both pass the SELECT
            # before either commits. The UPDATE itself is atomic, so
            # gating the flip on it, not the earlier read, is what
            # actually makes redemption single-use.
            cur = conn.execute(
                """UPDATE billing_payment_tokens SET redeemed=1, redeemed_by_type=?,
                   redeemed_by_id=?, redeemed_at=? WHERE payment_token_id=? AND redeemed=0""",
                (account_type, account_id, now, row["payment_token_id"]),
            )
            if cur.rowcount != 1:
                raise T1APIError(
                    T1ErrorCode.PAYMENT_TOKEN_ALREADY_REDEEMED,
                    "Payment token has already been redeemed.",
                    http_status=409,
                )
            amount = row["amount"]
        # add_balance re-enters the lock internally via _apply_delta — fine,
        # threading.Lock is not held across this call (the `with` block above
        # already exited).
        try:
            return self._apply_delta(
                account_type=account_type,
                account_id=account_id,
                amount=amount,
                kind=TransactionKind.REDEEM,
                created_by=redeemed_by,
                note="payment token redemption",
            )
        except Exception:
            # The token is already committed as redeemed. If crediting the
            # balance fails, un-burn it rather than leaving a token that is
            # spent but paid out nothing — the caller can retry redemption.
            with self._lock, self.backend.connect() as conn:
                conn.execute(
                    """UPDATE billing_payment_tokens SET redeemed=0, redeemed_by_type=NULL,
                       redeemed_by_id=NULL, redeemed_at=NULL
                       WHERE payment_token_id=?""",
                    (row["payment_token_id"],),
                )
            raise

    def get_payment_token_record(self, payment_token_id: str) -> PaymentTokenRecord | None:
        with self._lock, self.backend.connect() as conn:
            row = conn.execute(
                "SELECT * FROM billing_payment_tokens WHERE payment_token_id = ?", (payment_token_id,)
            ).fetchone()
        if row is None:
            return None
        return PaymentTokenRecord(
            payment_token_id=row["payment_token_id"],
            prefix=row["prefix"],
            amount=row["amount"],
            currency=row["currency"],
            redeemed=bool(row["redeemed"]),
            redeemed_by=(
                (row["redeemed_by_type"], row["redeemed_by_id"]) if row["redeemed_by_type"] else None
            ),
            redeemed_at=row["redeemed_at"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )


__all__ = ["TransactionKind", "Transaction", "PaymentTokenRecord", "BillingLedger"]
