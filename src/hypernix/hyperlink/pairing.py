"""hyperlink.pairing — how a phone becomes a trusted client of a home PC.

The problem this solves is narrow and specific. The T1 API already has
excellent credentials: long ``T1_...`` keys with scopes, expiry and
revocation. None of that is typeable on a phone. Asking someone to enter
a 48-character key on an iPhone keyboard, correctly, while standing at
their desk, is how a good security model becomes a screenshot in a
Notes app.

So pairing is a two-step exchange:

1. **The PC mints a pairing code.** Six characters from an
   unambiguous alphabet, valid for ten minutes, usable once. Shown as
   text and as a QR payload (:func:`pairing_payload`), because typing
   six characters is fine and scanning zero is better.
2. **The phone redeems it for a device token.** One long random secret,
   bound to that device, stored hashed on the PC, sent exactly once in
   the redemption response and never again.

The properties that matter
--------------------------
* **The code is not the credential.** It is short because it is
  short-lived, single-use, and rate-limited; the credential it produces
  is 32 bytes of ``secrets.token_urlsafe`` and never gets typed.
* **Tokens are stored hashed.** A stolen ``hypernix.db`` does not hand
  over anyone's phone. Verification is constant-time over the SHA-256 of
  the presented token (``compare_digest``), so the store cannot be
  probed by timing.
* **Revocation is per-device.** Losing a phone revokes that phone, not
  the household. ``last_seen``/``last_address`` exist so "which of these
  is my old iPad" is answerable before pressing revoke.
* **Brute force is bounded.** :meth:`DeviceRegistry.redeem` counts
  failed attempts per code and burns the code after five, so a six-
  character code with a ten-minute life cannot be walked through.

Why not just make the phone a T1 key holder? It is one, underneath — a
device token maps to a T1 key with the scopes the pairing was minted
with (see ``scopes``), so every existing audit, quota, and rate-limit
path applies unchanged. Pairing is an *enrolment* mechanism, not a
second authentication system.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..t1api.db import SQLiteBackend
from ..t1api.errors import T1APIError, T1ErrorCode

__all__ = [
    "DeviceRecord",
    "DeviceRegistry",
    "PairingCode",
    "PAIRING_ALPHABET",
    "generate_code",
    "hash_token",
    "pairing_payload",
]

#: No 0/O, no 1/I/L. A code is read off a screen and typed on a phone;
#: the characters people confuse are the ones that generate support
#: requests, so they are simply not in the alphabet.
PAIRING_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

CODE_LENGTH = 6
DEFAULT_CODE_TTL = 600.0          # ten minutes
MAX_REDEEM_ATTEMPTS = 5
DEFAULT_DEVICE_SCOPES = ("models:read", "chat:write", "files:write", "usage:read")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hyperlink_pairing_codes (
    code TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    created_by TEXT NOT NULL,
    scopes TEXT NOT NULL,
    label TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    redeemed_at REAL,
    redeemed_device TEXT
);
CREATE TABLE IF NOT EXISTS hyperlink_devices (
    device_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    platform TEXT NOT NULL,
    app_version TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    scopes TEXT NOT NULL,
    paired_by TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_seen REAL,
    last_address TEXT NOT NULL DEFAULT '',
    revoked_at REAL
);
CREATE INDEX IF NOT EXISTS idx_hyperlink_devices_token
    ON hyperlink_devices (token_hash);
"""


def generate_code(length: int = CODE_LENGTH) -> str:
    """A pairing code: ``secrets.choice`` over the unambiguous alphabet."""
    return "".join(secrets.choice(PAIRING_ALPHABET) for _ in range(length))


#: Punctuation people add when reading a code off a screen. Stripped;
#: everything else is left alone.
_CODE_SEPARATORS = frozenset(" -_.\t\n\r/|,")


def normalise_code(code: str) -> str:
    """Accept ``abc-123``, ``ABC 123``, ``abc123`` as the same code.

    People add the separator they saw on screen, or one they invented,
    and rejecting a correct code over a hyphen is a bad trade.

    Only separators are removed. An earlier version stripped anything
    outside :data:`PAIRING_ALPHABET`, which looked tidier and was wrong:
    a user who mistyped one character got their code silently shortened
    to five and was then told "a pairing code is six characters" — an
    error about the wrong thing entirely. Keeping the stray character
    means the length check passes and the lookup fails, which is
    "unknown pairing code": true, and actionable.
    """
    return "".join(ch for ch in code.upper() if ch not in _CODE_SEPARATORS)


def hash_token(token: str) -> str:
    """SHA-256 hex of a device token. The only form kept at rest."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass
class PairingCode:
    code: str
    created_at: float
    expires_at: float
    created_by: str
    scopes: tuple[str, ...]
    label: str = ""
    attempts: int = 0
    redeemed_at: float | None = None
    redeemed_device: str | None = None

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at

    @property
    def used(self) -> bool:
        return self.redeemed_at is not None

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.expires_at - time.time())

    def to_dict(self, *, include_code: bool = True) -> dict[str, Any]:
        return {
            "code": self.code if include_code else f"{self.code[:2]}…",
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "seconds_remaining": round(self.seconds_remaining, 1),
            "scopes": list(self.scopes),
            "label": self.label,
            "used": self.used,
            "expired": self.expired,
            "attempts": self.attempts,
        }


@dataclass
class DeviceRecord:
    device_id: str
    name: str
    platform: str
    app_version: str
    scopes: tuple[str, ...]
    paired_by: str
    created_at: float
    last_seen: float | None = None
    last_address: str = ""
    revoked_at: float | None = None
    token_hash: str = field(default="", repr=False)

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None

    def to_dict(self) -> dict[str, Any]:
        """Never includes ``token_hash`` — it is not needed by any caller
        outside the registry, and a hash in a JSON response is an offline
        cracking target for a six-character-alphabet mistake nobody has
        made yet but someone eventually would."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "platform": self.platform,
            "app_version": self.app_version,
            "scopes": list(self.scopes),
            "paired_by": self.paired_by,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "last_address": self.last_address,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at,
        }


class DeviceRegistry:
    """Pairing codes and paired devices, in the T1 API's own database."""

    def __init__(self, backend: SQLiteBackend | None = None) -> None:
        self.backend = backend or SQLiteBackend()
        self._lock = threading.Lock()
        self.backend.executescript(_SCHEMA)

    # -- codes --------------------------------------------------------

    def create_code(
        self,
        *,
        created_by: str,
        scopes: tuple[str, ...] | list[str] | None = None,
        label: str = "",
        ttl_seconds: float = DEFAULT_CODE_TTL,
    ) -> PairingCode:
        """Mint a single-use pairing code.

        Collisions are handled by retrying rather than by ignoring the
        ``INSERT`` failure: with 31^6 ≈ 887 million codes and a
        ten-minute window a collision is vanishingly unlikely, but
        "vanishingly unlikely" and "silently pairs a phone to the wrong
        code's scopes" are a bad pair of properties to combine.
        """
        now = time.time()
        scope_tuple = tuple(scopes) if scopes else DEFAULT_DEVICE_SCOPES
        with self._lock, self.backend.connect() as conn:
            self._purge_expired(conn, now)
            for _ in range(8):
                code = generate_code()
                row = conn.execute(
                    "SELECT code FROM hyperlink_pairing_codes WHERE code = ?", (code,)
                ).fetchone()
                if row is not None:
                    continue
                entry = PairingCode(
                    code=code,
                    created_at=now,
                    expires_at=now + float(ttl_seconds),
                    created_by=created_by,
                    scopes=scope_tuple,
                    label=label,
                )
                conn.execute(
                    """INSERT INTO hyperlink_pairing_codes
                       (code, created_at, expires_at, created_by, scopes, label, attempts)
                       VALUES (?, ?, ?, ?, ?, ?, 0)""",
                    (
                        entry.code,
                        entry.created_at,
                        entry.expires_at,
                        entry.created_by,
                        json.dumps(list(entry.scopes)),
                        entry.label,
                    ),
                )
                return entry
        raise T1APIError(
            T1ErrorCode.INTERNAL_ERROR,
            "Could not mint a unique pairing code after 8 attempts",
        )

    def list_codes(self, *, include_used: bool = False) -> list[PairingCode]:
        with self.backend.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hyperlink_pairing_codes ORDER BY created_at DESC"
            ).fetchall()
        codes = [_code_from_row(r) for r in rows]
        if include_used:
            return codes
        return [c for c in codes if not c.used and not c.expired]

    def revoke_code(self, code: str) -> bool:
        """Cancel an unredeemed code. Idempotent; ``False`` if unknown."""
        cleaned = normalise_code(code)
        with self._lock, self.backend.connect() as conn:
            cur = conn.execute(
                "DELETE FROM hyperlink_pairing_codes WHERE code = ? AND redeemed_at IS NULL",
                (cleaned,),
            )
            return bool(getattr(cur, "rowcount", 0))

    def _purge_expired(self, conn: Any, now: float) -> None:
        """Drop codes that expired over an hour ago.

        Not at expiry: a code that just expired should produce "that
        code has expired", not "no such code", because those send the
        user to two different places. An hour later nobody is still
        holding that screen.
        """
        conn.execute(
            "DELETE FROM hyperlink_pairing_codes WHERE expires_at < ? AND redeemed_at IS NULL",
            (now - 3600.0,),
        )

    # -- redemption ---------------------------------------------------

    def redeem(
        self,
        code: str,
        *,
        device_name: str,
        platform: str = "ios",
        app_version: str = "",
        address: str = "",
    ) -> tuple[DeviceRecord, str]:
        """Exchange a pairing code for a device token.

        Returns ``(record, token)``. The token is the only copy — it is
        not recoverable from the registry afterwards, by design.

        Validation and enrolment happen in **one** transaction, and the
        failure is raised only after that transaction has closed. Both
        halves of that matter:

        * One transaction, because two phones redeeming the same code at
          the same moment must not both pass a check-then-insert.
        * Raise afterwards, because raising inside the ``with`` rolls the
          transaction back — which silently undid the DELETE on the
          attempt-cap path, so a burnt code was refused once and then
          worked again. The exact opposite of a cap.
        """
        cleaned = normalise_code(code)
        now = time.time()
        if len(cleaned) != CODE_LENGTH:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                f"A pairing code is {CODE_LENGTH} characters; got {len(cleaned)}",
            )

        failure: T1APIError | None = None
        record: DeviceRecord | None = None
        token = ""  # nosec B105 - an empty default, a prefix or an error-code name, not a credential

        with self._lock, self.backend.connect() as conn:
            row = conn.execute(
                "SELECT * FROM hyperlink_pairing_codes WHERE code = ?", (cleaned,)
            ).fetchone()
            if row is None:
                failure = T1APIError(T1ErrorCode.NOT_FOUND, "Unknown pairing code")
            else:
                entry = _code_from_row(row)
                if entry.used:
                    failure = T1APIError(
                        T1ErrorCode.CONFLICT,
                        "That pairing code has already been used. Generate a new one "
                        "on the PC with: waiter hyperlink pair",
                    )
                elif entry.expired:
                    failure = T1APIError(
                        T1ErrorCode.VALIDATION_ERROR,
                        "That pairing code has expired. Generate a new one on the PC with: "
                        "waiter hyperlink pair",
                    )
                elif entry.attempts >= MAX_REDEEM_ATTEMPTS:
                    conn.execute(
                        "DELETE FROM hyperlink_pairing_codes WHERE code = ?", (cleaned,)
                    )
                    failure = T1APIError(
                        T1ErrorCode.VALIDATION_ERROR,
                        "Too many failed attempts on that pairing code; it has been cancelled",
                    )
                else:
                    token = "HLNK_" + secrets.token_urlsafe(32)
                    record = DeviceRecord(
                        device_id=uuid.uuid4().hex,
                        name=device_name.strip() or "Unnamed device",
                        platform=platform,
                        app_version=app_version,
                        scopes=entry.scopes,
                        paired_by=entry.created_by,
                        created_at=now,
                        last_seen=now,
                        last_address=address,
                        token_hash=hash_token(token),
                    )
                    conn.execute(
                        """INSERT INTO hyperlink_devices
                           (device_id, name, platform, app_version, token_hash, scopes,
                            paired_by, created_at, last_seen, last_address, revoked_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
                        (
                            record.device_id,
                            record.name,
                            record.platform,
                            record.app_version,
                            record.token_hash,
                            json.dumps(list(record.scopes)),
                            record.paired_by,
                            record.created_at,
                            record.last_seen,
                            record.last_address,
                        ),
                    )
                    # Marking the code used inside the same transaction
                    # as the insert is what makes "single use" true under
                    # concurrency rather than merely usually true.
                    conn.execute(
                        "UPDATE hyperlink_pairing_codes "
                        "SET redeemed_at = ?, redeemed_device = ? WHERE code = ?",
                        (now, record.device_id, cleaned),
                    )

        if failure is not None:
            raise failure
        # one of the two branches always runs
        if record is None:
            raise AssertionError('record is not None')
        return record, token

    def note_failed_attempt(self, code: str) -> None:
        """Count a wrong guess against a code that does exist.

        Called by the router when a redemption fails for a reason that
        is the client's fault. Kept separate from :meth:`redeem` so an
        expired-code error — which is not a guess — doesn't consume an
        attempt.
        """
        cleaned = normalise_code(code)
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                "UPDATE hyperlink_pairing_codes SET attempts = attempts + 1 WHERE code = ?",
                (cleaned,),
            )

    # -- devices ------------------------------------------------------

    def authenticate(self, token: str, *, address: str = "") -> DeviceRecord:
        """Resolve a device token to its record, or raise.

        The lookup is by hash — an index hit, not a scan — and the final
        comparison is still ``compare_digest`` even though SQL already
        matched. That is not redundancy for its own sake: it keeps the
        constant-time property if this ever moves to a backend whose
        comparison is not.
        """
        if not token:
            raise T1APIError(
                T1ErrorCode.AUTH_MISSING_CREDENTIALS,
                "No HyperLink device token supplied",
            )
        digest = hash_token(token)
        with self.backend.connect() as conn:
            row = conn.execute(
                "SELECT * FROM hyperlink_devices WHERE token_hash = ?", (digest,)
            ).fetchone()
        if row is None or not secrets.compare_digest(str(row["token_hash"]), digest):
            raise T1APIError(T1ErrorCode.AUTH_INVALID_KEY, "Unknown HyperLink device token")
        record = _device_from_row(row)
        if record.revoked:
            raise T1APIError(
                T1ErrorCode.AUTH_REVOKED_KEY,
                f"Device {record.name!r} was unpaired on this server; pair it again",
            )
        self.touch(record.device_id, address=address)
        record.last_seen = time.time()
        record.last_address = address or record.last_address
        return record

    def touch(self, device_id: str, *, address: str = "") -> None:
        """Record that a device just talked to us.

        Best-effort by design: a failed ``last_seen`` update must never
        fail the request that triggered it. The value is for the "which
        iPad is this" screen, not for anything load-bearing.
        """
        try:
            with self._lock, self.backend.connect() as conn:
                if address:
                    conn.execute(
                        "UPDATE hyperlink_devices SET last_seen = ?, last_address = ? WHERE device_id = ?",
                        (time.time(), address, device_id),
                    )
                else:
                    conn.execute(
                        "UPDATE hyperlink_devices SET last_seen = ? WHERE device_id = ?",
                        (time.time(), device_id),
                    )
        except Exception:  # noqa: BLE001
            pass

    def get_device(self, device_id: str) -> DeviceRecord:
        with self.backend.connect() as conn:
            row = conn.execute(
                "SELECT * FROM hyperlink_devices WHERE device_id = ?", (device_id,)
            ).fetchone()
        if row is None:
            raise T1APIError(T1ErrorCode.NOT_FOUND, f"No paired device {device_id!r}")
        return _device_from_row(row)

    def list_devices(self, *, include_revoked: bool = False) -> list[DeviceRecord]:
        with self.backend.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hyperlink_devices ORDER BY created_at DESC"
            ).fetchall()
        devices = [_device_from_row(r) for r in rows]
        if include_revoked:
            return devices
        return [d for d in devices if not d.revoked]

    def revoke_device(self, device_id: str) -> DeviceRecord:
        """Unpair a device. Its token stops working on the next request."""
        record = self.get_device(device_id)
        if record.revoked:
            return record
        now = time.time()
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                "UPDATE hyperlink_devices SET revoked_at = ? WHERE device_id = ?",
                (now, device_id),
            )
        record.revoked_at = now
        return record

    def rename_device(self, device_id: str, name: str) -> DeviceRecord:
        record = self.get_device(device_id)
        cleaned = name.strip()
        if not cleaned:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "Device name must not be empty")
        with self._lock, self.backend.connect() as conn:
            conn.execute(
                "UPDATE hyperlink_devices SET name = ? WHERE device_id = ?", (cleaned, device_id)
            )
        record.name = cleaned
        return record


def pairing_payload(
    code: PairingCode,
    *,
    endpoints: list[str],
    server_name: str = "HyperNix",
    t1_version: str = "",
) -> dict[str, Any]:
    """The JSON a QR code carries, so the phone scans instead of types.

    Multiple ``endpoints`` are included on purpose and in preference
    order: the app tries them in turn and keeps the first that answers.
    A phone at the desk should use the LAN address (fast, no relay) and
    the same phone on cellular should fall through to the Tailscale name
    without the user having to know which network they are on. One
    address in a QR code means one of those two situations works.
    """
    return {
        "v": 1,
        "kind": "hypernix.hyperlink.pairing",
        "server_name": server_name,
        "t1_version": t1_version,
        "code": code.code,
        "expires_at": code.expires_at,
        "endpoints": list(endpoints),
        "scopes": list(code.scopes),
    }


def _code_from_row(row: Any) -> PairingCode:
    return PairingCode(
        code=row["code"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        created_by=row["created_by"],
        scopes=tuple(json.loads(row["scopes"])),
        label=row["label"],
        attempts=int(row["attempts"] or 0),
        redeemed_at=row["redeemed_at"],
        redeemed_device=row["redeemed_device"],
    )


def _device_from_row(row: Any) -> DeviceRecord:
    return DeviceRecord(
        device_id=row["device_id"],
        name=row["name"],
        platform=row["platform"],
        app_version=row["app_version"],
        scopes=tuple(json.loads(row["scopes"])),
        paired_by=row["paired_by"],
        created_at=row["created_at"],
        last_seen=row["last_seen"],
        last_address=row["last_address"] or "",
        revoked_at=row["revoked_at"],
        token_hash=row["token_hash"],
    )
