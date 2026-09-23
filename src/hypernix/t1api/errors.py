"""t1api.errors — stable, machine-readable error codes for the T1 API.

Every failure mode the T1 API can produce is represented by a
:class:`T1ErrorCode` member and raised as a :class:`T1APIError`. The HTTP
layer (``t1api.app``) converts these into a stable JSON envelope::

    {
      "error": {"code": "MODEL_NOT_SUPPORTED", "message": "...", "details": {}},
      "request_id": "..."
    }

The codes and envelope shape are a contract used by the waiter TUI and any
other T1 client — do not rename existing members. Add new ones instead of
repurposing old ones, and add new members to the end of the enum so any code
that persists the ordinal (none should) stays stable.

This module has zero third-party dependencies on purpose: both the pure
``t1api.registry`` / ``t1api.usage`` core and the FastAPI HTTP layer import
it, and the core must stay importable in environments where FastAPI/Pydantic
aren't installed (e.g. this is exactly what ``hypernix.keymaster`` /
``hypernix.gatekeeper`` already do — no framework dependency baked into the
enforcement logic).
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any


class T1ErrorCode(StrEnum):
    """Stable error codes returned by the T1 API.

    See ``wiki/T1-API.md`` for the full contract, including HTTP status
    mapping (``t1api.app._STATUS_FOR_CODE``).
    """

    # --- Model registry / routing -----------------------------------
    MODEL_NOT_SUPPORTED = "MODEL_NOT_SUPPORTED"
    MODEL_QUOTA_EXHAUSTED = "MODEL_QUOTA_EXHAUSTED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"

    # --- Auth ----------------------------------------------------------
    AUTH_MISSING_CREDENTIALS = "AUTH_MISSING_CREDENTIALS"
    AUTH_INVALID_KEY = "AUTH_INVALID_KEY"
    AUTH_EXPIRED_KEY = "AUTH_EXPIRED_KEY"
    AUTH_REVOKED_KEY = "AUTH_REVOKED_KEY"
    AUTH_INVALID_TOKEN = "AUTH_INVALID_TOKEN"
    AUTH_EXPIRED_TOKEN = "AUTH_EXPIRED_TOKEN"
    AUTH_INSUFFICIENT_SCOPE = "AUTH_INSUFFICIENT_SCOPE"
    AUTH_ADMIN_REQUIRED = "AUTH_ADMIN_REQUIRED"

    # --- Quota / rate limiting -----------------------------------------
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"

    # --- Generic ---------------------------------------------------------
    NOT_SUPPORTED = "NOT_SUPPORTED"
    NOT_FOUND = "NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFLICT = "CONFLICT"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    # --- Beta 2: modules / servers / jobs -------------------------------
    MODULE_NOT_FOUND = "MODULE_NOT_FOUND"
    MODULE_ALREADY_EXISTS = "MODULE_ALREADY_EXISTS"
    MODULE_UPLOAD_REJECTED = "MODULE_UPLOAD_REJECTED"
    SERVER_NOT_FOUND = "SERVER_NOT_FOUND"
    SERVER_UNTRUSTED = "SERVER_UNTRUSTED"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_NOT_CANCELLABLE = "JOB_NOT_CANCELLABLE"
    PATH_TRAVERSAL_REJECTED = "PATH_TRAVERSAL_REJECTED"
    SSRF_BLOCKED = "SSRF_BLOCKED"

    # --- Beta 2: billing -------------------------------------------------
    PAYMENT_TOKEN_INVALID = "PAYMENT_TOKEN_INVALID"
    BILLING_PAYMENT_REQUIRED = "BILLING_PAYMENT_REQUIRED"
    BILLING_KEY_REFUSED = "BILLING_KEY_REFUSED"
    PAYMENT_TOKEN_ALREADY_REDEEMED = "PAYMENT_TOKEN_ALREADY_REDEEMED"
    INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"

    # --- Beta 2: routing ---------------------------------------------------
    ROUTING_EXHAUSTED = "ROUTING_EXHAUSTED"  # every model in the cascade is exhausted

    # --- Beta 3: network policy / transport / production hardening --------
    IP_BLOCKED = "IP_BLOCKED"  # client address is on the server's blocklist
    IP_NOT_ALLOWLISTED = "IP_NOT_ALLOWLISTED"  # server accepts allowlisted clients only
    MTLS_REQUIRED = "MTLS_REQUIRED"  # no verified client certificate presented
    MTLS_INVALID = "MTLS_INVALID"  # client certificate present but not acceptable
    TRANSPORT_FAILED = "TRANSPORT_FAILED"  # remote push/fetch failed at the network layer
    CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"  # transferred bytes don't match the expected digest
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"  # upload/fetch exceeded the configured size cap
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"  # destructive op needs explicit confirmation
    CONFIG_INVALID = "CONFIG_INVALID"  # production configuration validation failed


def hx_code_for(code: T1ErrorCode) -> str:
    """The ``L#-NNNNN.kS`` code for a T1 error member.

    Imported lazily because the catalogue registers codes at import and
    this module must stay importable from the zero-dependency core.
    """
    from ..system.errorcatalogue import T1_CODES

    found = T1_CODES.get(code.name)
    # INTERNAL_ERROR's code rather than an empty string: a member added
    # without a mapping is our fault, and saying so beats a blank field.
    return (found or T1_CODES["INTERNAL_ERROR"]).code


class T1APIError(Exception):
    """Raised anywhere in the T1 API stack; carries a stable error code.

    Args:
        code: One of :class:`T1ErrorCode`.
        message: Human-readable detail. Never include secrets (raw keys,
            tokens, payment tokens) — see ``wiki/T1-API.md#security``.
        details: Optional structured extra context (e.g. ``{"model_id": ...}``).
        http_status: Overrides the default status mapping for this code
            when set.
    """

    def __init__(
        self,
        code: T1ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        self.http_status = http_status
        self.hx_code = hx_code_for(code)
        super().__init__(f"[{code.value}] {message}")


__all__ = ["T1ErrorCode", "T1APIError"]
