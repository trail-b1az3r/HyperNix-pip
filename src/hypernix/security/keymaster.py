"""keymaster — API key generation, lifecycle management, and secure storage.

Provides the full lifecycle for T1-format API keys:

* **Generate** — cryptographically random T1 keys with enforced structure.
* **Revoke** — mark a key as inactive; it is archived, not deleted.
* **Rotate** — atomically replace a key with a fresh one, archiving the old.
* **Auto-rotate** — background thread checks for keys nearing expiry
  (within ``rotation_window`` hours) and auto-regenerates them.
* **Import / Export** — JSON round-trip for backup and migration.
* **Encrypt** — key secrets are stored Fernet-encrypted when the
  ``cryptography`` package is installed; plain-JSON fallback otherwise
  (with a one-time warning).

T1 Key Format
-------------
::

    T1_<body><suffix>

Where ``suffix`` is the last 8 characters with fixed structure::

    Position  Allowed
    -8        lowercase letter [a-z]
    -7        lowercase letter [a-z]
    -6        special char  (!@#$%^&*()-_=+[]{};:',.<>?/|~`)
    -5        special char
    -4        special char
    -3        special char
    -2        special char
    -1 & -0   slash-or-backslash + digit 1-9  (e.g. /4  or \\7)

The body may contain [A-Za-z0-9] only (minimum 16 chars).

Server ID Format
----------------
::

    NNNNN-X#

* ``NNNNN`` — 1–5 decimal digits (e.g. ``00001`` .. ``99999``)
* ``X`` — uppercase A–Z cycling letter
* ``#`` — generation counter (increments every time Z is reached)

Sequence: 00001-A1 → 99999-A1 → 00001-B1 → … → 99999-Z1 → 00001-A2 → …

Usage::

    from hypernix.keymaster import Keymaster, KeyType, KeyScope

    km = Keymaster()
    meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE})
    print(meta.key)       # T1_…
    print(meta.key_id)    # uuid

    km.revoke(meta.key_id)
    km.rotate(meta.key_id)  # → new KeyMeta
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import string
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional encryption backend
# ---------------------------------------------------------------------------

try:
    from cryptography.fernet import Fernet as _Fernet  # type: ignore

    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False
    _Fernet = None  # type: ignore

# ---------------------------------------------------------------------------
# Key taxonomy
# ---------------------------------------------------------------------------


class KeyType(StrEnum):
    """Supported API key types."""

    DEVELOPMENT = "development"
    USER = "user"
    SERVICE = "service"
    SESSION = "session"
    ADMIN = "admin"


class KeyScope(StrEnum):
    """Permission scopes a key may be granted."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    PLUGIN = "plugin"
    SERVICE = "service"


# ---------------------------------------------------------------------------
# T1 Key format constants
# ---------------------------------------------------------------------------

#: Characters allowed in the key body (everything before the suffix).
_BODY_CHARS: str = string.ascii_letters + string.digits

#: Special characters allowed in positions -6 .. -2 of the suffix.
_SPECIAL_CHARS: str = r"!@#$%^&*()-_=+[]{};:',.<>?|~`"

#: Minimum body length (characters before the suffix).
_MIN_BODY_LEN: int = 16

#: Compiled regex for full T1 key validation (suffix only).
#  Group 1 = body, group 2 = two lowercase, group 3 = five specials,
#  group 4 = slash-or-backslash, group 5 = digit 1-9.
_SPECIAL_RE = re.escape(_SPECIAL_CHARS)
_T1_PATTERN: re.Pattern[str] = re.compile(
    r"^T1_"
    r"(?P<body>[A-Za-z0-9]{16,})"
    r"(?P<ll>[a-z]{2})"
    r"(?P<sp>[" + _SPECIAL_RE + r"]{5})"
    r"(?P<slash>[/\\])"
    r"(?P<digit>[1-9])$"
)


def _key_digest(key: str) -> str:
    """SHA-256 of a raw key, for indexing without storing or comparing it.

    Authentication looks a key up by this rather than by scanning for a
    string match — see :meth:`Keymaster.get_by_key`. A digest is the
    right index because it is uniformly distributed (so the dict probe
    tells an observer nothing about the key) and because a plain hash is
    exactly the right strength here: the input is 32+ characters of
    CSPRNG output, not a human-chosen password, so there is no dictionary
    to attack and nothing for a slow KDF to buy.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Server ID management
# ---------------------------------------------------------------------------


def _parse_server_id(server_id: str) -> tuple[int, str, int]:
    """Parse ``NNNNN-X#`` into ``(seq, letter, gen)``."""
    m = re.fullmatch(r"(\d{1,5})-([A-Z])(\d+)", server_id)
    if not m:
        raise ValueError(f"Invalid server_id: {server_id!r}")
    return int(m.group(1)), m.group(2), int(m.group(3))


def _format_server_id(seq: int, letter: str, gen: int) -> str:
    return f"{seq:05d}-{letter}{gen}"


def _server_id_order(server_id: str) -> tuple[int, int, int]:
    """Sort key for a server ID, in the order it actually advances.

    ``_next_server_id`` bumps the sequence, rolls it into the letter, and
    rolls the letter into the generation — so generation is the most
    significant part and sequence the least. Sorting the raw strings would
    put ``00002-A1`` after ``00001-B1``, which is backwards.
    """
    seq, letter, gen = _parse_server_id(server_id)
    return (gen, ord(letter), seq)


def _next_server_id(current: str) -> str:
    """Increment a server ID, cycling through A–Z then bumping the generation."""
    seq, letter, gen = _parse_server_id(current)
    seq += 1
    if seq > 99999:
        seq = 1
        next_ord = ord(letter) + 1
        if next_ord > ord("Z"):
            letter = "A"
            gen += 1
        else:
            letter = chr(next_ord)
    return _format_server_id(seq, letter, gen)


# ---------------------------------------------------------------------------
# T1 Key generator
# ---------------------------------------------------------------------------


class T1KeyGenerator:
    """Generates and validates T1-format API keys.

    All generated keys pass :meth:`validate` immediately. The suffix is
    deterministic in structure but cryptographically random in content.
    """

    @staticmethod
    def generate(body_length: int = 24) -> str:
        """Return a fresh T1 key string.

        Args:
            body_length: Number of alphanumeric characters before the suffix.
                         Must be >= 16. Total key length = 4 (prefix) +
                         body_length + 8 (suffix).
        """
        if body_length < _MIN_BODY_LEN:
            raise ValueError(f"body_length must be >= {_MIN_BODY_LEN}, got {body_length}")

        body = "".join(secrets.choice(_BODY_CHARS) for _ in range(body_length))
        ll = "".join(secrets.choice(string.ascii_lowercase) for _ in range(2))
        sp = "".join(secrets.choice(_SPECIAL_CHARS) for _ in range(5))
        slash = secrets.choice(r"/\\")
        digit = str(secrets.randbelow(9) + 1)  # 1-9

        return f"T1_{body}{ll}{sp}{slash}{digit}"

    @staticmethod
    def validate(key: str) -> bool:
        """Return True if *key* matches the T1 format exactly."""
        return bool(_T1_PATTERN.fullmatch(key))

    @staticmethod
    def deconstruct(key: str) -> dict[str, str]:
        """Parse a T1 key into its structural parts.

        Returns a dict with keys: prefix, body, lowercase_pair,
        special_chars, slash, digit, suffix.

        Raises ValueError if the key is invalid.
        """
        m = _T1_PATTERN.fullmatch(key)
        if not m:
            raise ValueError(f"Not a valid T1 key: {key!r}")
        suffix = m.group("ll") + m.group("sp") + m.group("slash") + m.group("digit")
        return {
            "prefix": "T1_",
            "body": m.group("body"),
            "lowercase_pair": m.group("ll"),
            "special_chars": m.group("sp"),
            "slash": m.group("slash"),
            "digit": m.group("digit"),
            "suffix": suffix,
        }


# ---------------------------------------------------------------------------
# Key metadata dataclass
# ---------------------------------------------------------------------------


@dataclass
class KeyMeta:
    """Complete metadata for a single managed key.

    The ``key`` field contains the raw T1 key string.  It is stored
    encrypted-at-rest when ``cryptography`` is available.
    """

    key_id: str
    key: str
    key_type: KeyType
    scopes: set[KeyScope]
    created_at: float  # POSIX timestamp
    expires_at: float | None  # POSIX timestamp or None = never
    usage_cap: int | None  # max lifetime token consumption, or None
    request_limit: int | None  # max lifetime requests, or None
    prefix: str  # user-visible prefix label (e.g. "myapp")
    tags: dict[str, str]  # arbitrary metadata
    server_id: str  # e.g. "00001-A1"
    active: bool = True
    rotation_window: int = 24  # hours before expiry to auto-rotate
    usage_count: int = 0  # lifetime token usage
    request_count: int = 0  # lifetime request count
    rotated_from: str | None = None  # predecessor key_id, if rotated
    revoked_at: float | None = None
    rotated_at: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "key": self.key,
            "key_type": self.key_type.value,
            "scopes": [s.value for s in sorted(self.scopes, key=lambda x: x.value)],
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "usage_cap": self.usage_cap,
            "request_limit": self.request_limit,
            "prefix": self.prefix,
            "tags": self.tags,
            "server_id": self.server_id,
            "active": self.active,
            "rotation_window": self.rotation_window,
            "usage_count": self.usage_count,
            "request_count": self.request_count,
            "rotated_from": self.rotated_from,
            "revoked_at": self.revoked_at,
            "rotated_at": self.rotated_at,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeyMeta:
        return cls(
            key_id=data["key_id"],
            key=data["key"],
            key_type=KeyType(data["key_type"]),
            scopes={KeyScope(s) for s in data.get("scopes", [])},
            created_at=data["created_at"],
            expires_at=data.get("expires_at"),
            usage_cap=data.get("usage_cap"),
            request_limit=data.get("request_limit"),
            prefix=data.get("prefix", ""),
            tags=data.get("tags", {}),
            server_id=data.get("server_id", "00001-A1"),
            active=data.get("active", True),
            rotation_window=data.get("rotation_window", 24),
            usage_count=data.get("usage_count", 0),
            request_count=data.get("request_count", 0),
            rotated_from=data.get("rotated_from"),
            revoked_at=data.get("revoked_at"),
            rotated_at=data.get("rotated_at"),
            note=data.get("note", ""),
        )

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def expires_soon(self) -> bool:
        """True if the key expires within its rotation_window hours."""
        if self.expires_at is None:
            return False
        horizon = self.expires_at - self.rotation_window * 3600
        return time.time() >= horizon

    @property
    def is_valid(self) -> bool:
        return self.active and not self.is_expired

    def display(self) -> str:
        """Human-readable summary line."""
        exp = (
            datetime.fromtimestamp(self.expires_at, tz=UTC).strftime("%Y-%m-%d")
            if self.expires_at
            else "never"
        )
        scopes_str = ",".join(s.value for s in sorted(self.scopes, key=lambda x: x.value))
        status = "active" if self.active else ("expired" if self.is_expired else "revoked")
        return (
            f"{self.key_id[:8]}…  {self.key_type.value:<10}  "
            f"scopes=[{scopes_str}]  expires={exp}  status={status}"
        )


# ---------------------------------------------------------------------------
# Encryption helpers
# ---------------------------------------------------------------------------


def _make_fernet(secret: bytes) -> Any:
    """Return a Fernet instance keyed from *secret* (must be 32 raw bytes)."""
    import base64

    key = base64.urlsafe_b64encode(secret[:32].ljust(32, b"\x00"))
    return _Fernet(key)


def _get_or_create_master_key(store_dir: Path) -> bytes:
    """Load or generate the per-user master encryption key (32 bytes)."""
    key_file = store_dir / ".master.key"
    if key_file.exists():
        return bytes.fromhex(key_file.read_text(encoding="ascii").strip())
    raw = secrets.token_bytes(32)
    store_dir.mkdir(parents=True, exist_ok=True)
    key_file.write_text(raw.hex(), encoding="ascii")
    # Restrict permissions on POSIX; Windows has no chmod equivalent
    try:
        os.chmod(key_file, 0o600)
    except OSError:
        pass
    return raw


# ---------------------------------------------------------------------------
# Keymaster
# ---------------------------------------------------------------------------

_DEFAULT_STORE: Path = Path.home() / ".hypernix" / "keymaster"
_ARCHIVE_SUBDIR = "archive"


class Keymaster:
    """Full-lifecycle API key manager.

    All operations are thread-safe. Key secrets are stored
    Fernet-encrypted (``cryptography`` optional extra) or as plain JSON
    with a one-time warning.

    Args:
        store_dir: Directory for persisting key records.  Defaults to
            ``~/.hypernix/keymaster/``.
        auto_rotate: If *True* (default), a background thread polls
            every ``poll_interval`` seconds and rotates keys that are
            within their ``rotation_window``.
        poll_interval: Seconds between auto-rotation poll cycles.
        server_id: Starting server-ID string.  Defaults to ``"00001-A1"``.
    """

    def __init__(
        self,
        store_dir: Path | None = None,
        auto_rotate: bool = True,
        poll_interval: float = 3600.0,
        server_id: str = "00001-A1",
    ) -> None:
        self._store = Path(store_dir) if store_dir else _DEFAULT_STORE
        self._store.mkdir(parents=True, exist_ok=True)
        (self._store / _ARCHIVE_SUBDIR).mkdir(exist_ok=True)

        self._lock = threading.RLock()
        self._keys: dict[str, KeyMeta] = {}
        # SHA-256 of each raw key -> its metadata. See :meth:`get_by_key`
        # for why authentication goes through a digest rather than
        # comparing key strings. Rebuilt by :meth:`_reindex` after every
        # mutation of ``_keys``; rebuilding is O(n) and minting or
        # revoking a key is rare, while authenticating is the hot path.
        self._by_digest: dict[str, KeyMeta] = {}
        self._server_id = server_id
        self._cipher: Any = None
        self._revoke_observers: list[Callable[[str, str], None]] = []
        self._rotate_observers: list[Callable[[str, str], None]] = []

        # Encryption setup
        if _CRYPTO_AVAILABLE:
            master = _get_or_create_master_key(self._store)
            self._cipher = _make_fernet(master)
        else:
            warnings.warn(
                "hypernix.keymaster: 'cryptography' package not installed. "
                "Key secrets will be stored as plain JSON. "
                "Install with: pip install hypernix[security]",
                UserWarning,
                stacklevel=2,
            )

        self._load_all()
        self._resume_server_id(server_id)

        self._stop_event = threading.Event()
        self._bg_thread: threading.Thread | None = None
        if auto_rotate:
            self._bg_thread = threading.Thread(
                target=self._auto_rotate_loop,
                args=(poll_interval,),
                daemon=True,
                name="keymaster-autorotate",
            )
            self._bg_thread.start()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _key_path(self, key_id: str) -> Path:
        return self._store / f"{key_id}.json"

    def _archive_path(self, key_id: str) -> Path:
        return self._store / _ARCHIVE_SUBDIR / f"{key_id}.json"

    def _encrypt(self, text: str) -> str:
        if self._cipher is None:
            return text
        return self._cipher.encrypt(text.encode()).decode()

    def _decrypt(self, text: str) -> str:
        if self._cipher is None:
            return text
        try:
            return self._cipher.decrypt(text.encode()).decode()
        except Exception:
            # Fallback: might be a plain-text record written before encryption
            return text

    def _save(self, meta: KeyMeta, *, archive: bool = False) -> None:
        d = meta.to_dict()
        d["key"] = self._encrypt(d["key"])
        path = self._archive_path(meta.key_id) if archive else self._key_path(meta.key_id)
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def _load_file(self, path: Path) -> KeyMeta | None:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            d["key"] = self._decrypt(d["key"])
            return KeyMeta.from_dict(d)
        except Exception as exc:
            logger.warning("keymaster: could not load %s: %s", path, exc)
            return None

    def _load_all(self) -> None:
        with self._lock:
            for p in self._store.glob("*.json"):
                if p.name.startswith("."):
                    continue
                meta = self._load_file(p)
                if meta:
                    self._keys[meta.key_id] = meta
            self._reindex()

    def _resume_server_id(self, requested_start: str) -> None:
        """Continue the server-ID sequence from what the store already holds.

        The counter only ever lived in memory, so every process started
        again at ``00001-A1`` — and since each ``gkey`` invocation is its
        own process, every key ever minted from the CLI carried the same
        server ID. The field was there, documented, and constant.

        Archived keys are scanned too. ``_load_all`` reads only the active
        store, and revoking a key moves it into ``archive/``; resuming
        from the active keys alone would hand a revoked key's server ID to
        a new key the moment the highest-numbered one was revoked. A
        server ID that comes back around is worse than one that never
        moves, because audit trails stop being able to tell the two keys
        apart.

        An explicitly requested start still wins when it is ahead of the
        store, so a caller can move a fresh sequence forward on purpose.
        """
        highest: tuple[int, int, int] | None = None
        highest_id = ""
        paths = list(self._store.glob("*.json"))
        paths.extend((self._store / _ARCHIVE_SUBDIR).glob("*.json"))
        for path in paths:
            if path.name.startswith("."):
                continue
            meta = self._load_file(path)
            if meta is None or not meta.server_id:
                continue
            try:
                order = _server_id_order(meta.server_id)
            except ValueError:
                # A hand-edited or truncated record must not stop the
                # store from opening; it just does not advance anything.
                logger.warning(
                    "keymaster: ignoring unparseable server_id %r on key %s",
                    meta.server_id,
                    meta.key_id,
                )
                continue
            if highest is None or order > highest:
                highest, highest_id = order, meta.server_id

        if highest is None:
            return
        resumed = _next_server_id(highest_id)
        with self._lock:
            try:
                requested_order = _server_id_order(requested_start)
            except ValueError:
                requested_order = None
            if requested_order is None or _server_id_order(resumed) > requested_order:
                self._server_id = resumed

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        key_type: KeyType = KeyType.USER,
        scopes: set[KeyScope] | None = None,
        expires_at: float | None = None,
        usage_cap: int | None = None,
        request_limit: int | None = None,
        prefix: str = "",
        tags: dict[str, str] | None = None,
        rotation_window: int = 24,
        note: str = "",
        body_length: int = 24,
    ) -> KeyMeta:
        """Create and persist a new T1 API key.

        Returns the :class:`KeyMeta` containing the raw key string.
        """
        key_str = T1KeyGenerator.generate(body_length=body_length)
        key_id = str(uuid.uuid4())
        with self._lock:
            server_id = self._server_id
            self._server_id = _next_server_id(self._server_id)
        meta = KeyMeta(
            key_id=key_id,
            key=key_str,
            key_type=key_type,
            scopes=scopes or {KeyScope.READ},
            created_at=time.time(),
            expires_at=expires_at,
            usage_cap=usage_cap,
            request_limit=request_limit,
            prefix=prefix,
            tags=tags or {},
            server_id=server_id,
            rotation_window=rotation_window,
            note=note,
        )
        with self._lock:
            self._keys[key_id] = meta
            self._reindex()
        self._save(meta)
        logger.info("keymaster: created key %s (type=%s)", key_id[:8], key_type.value)
        return meta

    def get(self, key_id: str) -> KeyMeta | None:
        """Return a key by its ID, or *None* if not found."""
        with self._lock:
            return self._keys.get(key_id)

    def _reindex(self) -> None:
        """Rebuild the digest index. Caller must hold ``self._lock``.

        Rebuilt wholesale rather than patched per mutation: an index that
        is updated at six call sites is an index that will be missed at a
        seventh, and a missed one here does not corrupt anything — it
        makes a valid key stop authenticating, which is the kind of bug
        that gets "fixed" by going back to the string compare.
        """
        self._by_digest = {
            _key_digest(meta.key): meta
            for meta in self._keys.values()
            if meta.key
        }

    def get_by_key(self, key_str: str) -> KeyMeta | None:
        """Return the KeyMeta for a given raw T1 key string.

        Constant-time in two senses, both of which matter because this is
        the T1 API's primary authentication path —
        :meth:`hypernix.security.gatekeeper.Gatekeeper.authenticate`
        calls it with whatever an unauthenticated caller sent.

        It used to be ``meta.key == key_str`` inside a loop over every
        key. ``==`` on ``str`` short-circuits at the first differing
        byte, so the time it takes is a function of how long a prefix the
        attacker guessed right — recover one byte, try 256 values for the
        next, and a key falls out in a few thousand requests rather than
        in the heat death of the universe. That is the classic timing
        oracle and a key check is the classic place to find one.

        The loop was the second half of it: the *number of iterations*
        depended on where in the dictionary the matching key sat, so even
        a constant-time compare inside the loop would still have leaked
        which key was being presented.

        Both are fixed by comparing against a digest instead. Every key's
        SHA-256 is indexed once at load; a lookup is one hash of the
        candidate and one dict probe, and the dict probe is over a
        uniformly-distributed 32-byte digest rather than over a secret.
        :func:`hmac.compare_digest` guards the final confirmation, which
        is belt-and-braces over a dict hit but costs nothing and means no
        ``==`` on a secret survives anywhere in this path.
        """
        if not key_str:
            return None
        digest = _key_digest(key_str)
        with self._lock:
            meta = self._by_digest.get(digest)
            if meta is None:
                return None
            # A dict hit on a SHA-256 means the keys are equal unless
            # somebody has broken SHA-256. Confirmed anyway, in constant
            # time, so that nothing here depends on that assumption.
            if hmac.compare_digest(meta.key, key_str):
                return meta
        return None

    # ------------------------------------------------------------------
    # Reversal primitives
    # ------------------------------------------------------------------
    #
    # The T1 API's auth undo/redo (POST /t1/auth/undo) reverses an
    # operation by putting a key back the way it was. Each of these is one
    # such inverse. They existed only as ``hasattr`` checks in the undo
    # handler, which meant every undo answered 501 "this server's
    # Keymaster cannot …" — a feature that could not work on any
    # deployment, because no Keymaster had the methods.
    #
    # They are deliberately narrow. None of them creates or destroys a
    # key, none changes a key's identity, and each writes exactly one
    # field: an undo must land on a state the system could have reached
    # forwards.

    def restore_key(self, key_id: str, key_str: str) -> KeyMeta:
        """Put previous key material back on an existing key.

        Used to reverse a rotation. The key ID does not change — this is
        the same record with its secret restored — so anything that
        references the key by ID keeps working.
        """
        if not T1KeyGenerator.validate(key_str):
            raise ValueError("Not a valid T1 key")
        with self._lock:
            meta = self._keys.get(key_id)
            if meta is None:
                raise KeyError(f"Key not found: {key_id!r}")
            meta.key = key_str
            # The only place a key's *material* changes without the
            # dictionary changing, which makes it the one mutation the
            # digest index cannot notice on its own. Missing it left the
            # index holding the pre-restore digest, so `POST
            # /t1/auth/undo` reported success and the restored key then
            # failed to authenticate -- an undo that undid nothing.
            self._reindex()
        self._save(meta)
        logger.info("keymaster: restored key material on %s", key_id[:8])
        return meta

    def set_key_type(self, key_id: str, key_type: KeyType | str) -> KeyMeta:
        """Change a key's type. Reverses a promotion."""
        key_type = KeyType(key_type)
        with self._lock:
            meta = self._keys.get(key_id)
            if meta is None:
                raise KeyError(f"Key not found: {key_id!r}")
            meta.key_type = key_type
        self._save(meta)
        logger.info("keymaster: set type of %s to %s", key_id[:8], key_type.value)
        return meta

    def set_scopes(self, key_id: str, scopes: Iterable[KeyScope | str]) -> KeyMeta:
        """Replace a key's scope set. Reverses a scope change."""
        resolved = {KeyScope(s) for s in scopes}
        if not resolved:
            raise ValueError("A key must keep at least one scope")
        with self._lock:
            meta = self._keys.get(key_id)
            if meta is None:
                raise KeyError(f"Key not found: {key_id!r}")
            meta.scopes = resolved
        self._save(meta)
        logger.info("keymaster: set scopes of %s", key_id[:8])
        return meta

    def set_server_id(self, key_id: str, server_id: str) -> KeyMeta:
        """Move a key to a different V1 Server ID.

        Validated, not trusted: the identifier comes from a configuration
        source that may be somewhere else on the network, and an
        unparseable one written into the store would break
        ``_resume_server_id`` for every key minted afterwards — the
        counter reads existing keys to work out where to continue from.
        """
        try:
            _parse_server_id(server_id)
        except Exception as exc:
            raise ValueError(
                f"{server_id!r} is not a V1 Server ID (expected NNNNN-LN, "
                "e.g. 00042-C1)"
            ) from exc
        with self._lock:
            meta = self._keys.get(key_id)
            if meta is None:
                raise KeyError(f"Key not found: {key_id!r}")
            meta.server_id = server_id
        self._save(meta)
        logger.info("keymaster: set server_id of %s to %s", key_id[:8], server_id)
        return meta

    def set_revoked(self, key_id: str, revoked: bool) -> KeyMeta:
        """Revoke or un-revoke a key.

        Un-revoking has to move the record back out of the archive, since
        :meth:`revoke` archives rather than deletes — which is exactly what
        makes the operation reversible at all.
        """
        if revoked:
            with self._lock:
                if key_id in self._keys:
                    self.revoke(key_id)
                    return self._load_file(self._archive_path(key_id))  # type: ignore[return-value]
                archived = self._load_file(self._archive_path(key_id))
            if archived is None:
                raise KeyError(f"Key not found: {key_id!r}")
            return archived

        with self._lock:
            if key_id in self._keys:
                return self._keys[key_id]
        meta = self._load_file(self._archive_path(key_id))
        if meta is None:
            raise KeyError(f"Key not found in the archive: {key_id!r}")
        meta.active = True
        meta.revoked_at = None
        with self._lock:
            self._keys[key_id] = meta
            self._reindex()
        self._save(meta)
        try:
            self._archive_path(key_id).unlink()
        except FileNotFoundError:
            pass
        logger.info("keymaster: un-revoked key %s", key_id[:8])
        return meta

    @property
    def store_dir(self) -> Path:
        """Where this Keymaster's keys live.

        Public because anything that indexes these keys has to live beside
        them — the SSPKID registry most of all, since a registry pointing
        at a different directory indexes keys that are not there.
        """
        return self._store

    def on_revoke(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback to run when a key is revoked or rotated away.

        For state that hangs off a key and must not outlive it — a billing
        binding is the motivating case, since a binding left behind is a
        spend cap and a recorded spend waiting to be inherited by whatever
        holds that ID next.

        Observers are advisory. Revoking a key is the safety-critical
        operation and an observer that raises must not be able to prevent
        it, so exceptions are logged and swallowed. Anything that *must*
        happen for correctness belongs in :meth:`revoke` itself, not here.
        """
        self._revoke_observers.append(callback)

    def on_rotate(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback for ``(old_key_id, new_key_id)`` on rotation.

        Rotation is not revocation: the credential changed, the
        relationship did not. State that hangs off a key should **follow**
        it here rather than being dropped — a rotated key that silently
        lost its spend cap would be worse than one that kept it.
        """
        self._rotate_observers.append(callback)

    def _notify(
        self,
        observers: list[Callable[[str, str], None]],
        what: str,
        first: str,
        second: str,
    ) -> None:
        for callback in list(observers):
            try:
                callback(first, second)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "keymaster: %s observer failed for %s", what, first[:8]
                )

    def _notify_revoked(self, key_id: str, reason: str) -> None:
        self._notify(self._revoke_observers, "revocation", key_id, reason)

    def revoke(self, key_id: str, reason: str = "") -> None:
        """Revoke a key. It is archived but not deleted."""
        with self._lock:
            meta = self._keys.get(key_id)
            if meta is None:
                raise KeyError(f"Key not found: {key_id!r}")
            meta.active = False
            meta.revoked_at = time.time()
            if reason:
                meta.note = (meta.note + f" [revoked: {reason}]").strip()
            del self._keys[key_id]
            self._reindex()
        # Save to archive, remove active record
        self._save(meta, archive=True)
        active_path = self._key_path(key_id)
        try:
            active_path.unlink()
        except FileNotFoundError:
            pass
        logger.info("keymaster: revoked key %s", key_id[:8])
        self._notify_revoked(key_id, reason)

    def rotate(self, key_id: str) -> KeyMeta:
        """Replace *key_id* with a fresh key, archiving the old one.

        Returns the new :class:`KeyMeta`.  The old key is immediately
        revoked and archived.
        """
        with self._lock:
            old = self._keys.get(key_id)
            if old is None:
                raise KeyError(f"Key not found: {key_id!r}")
            # Preserve settings from old key
            new_meta = self.create(
                key_type=old.key_type,
                scopes=set(old.scopes),
                expires_at=old.expires_at,
                usage_cap=old.usage_cap,
                request_limit=old.request_limit,
                prefix=old.prefix,
                tags=dict(old.tags),
                rotation_window=old.rotation_window,
                note=old.note,
            )
            new_meta.rotated_from = key_id
            new_meta.rotated_at = time.time()
            self._save(new_meta)

        # Archive the old key
        old.active = False
        old.rotated_at = time.time()
        old.note = (old.note + f" [rotated → {new_meta.key_id[:8]}]").strip()
        self._save(old, archive=True)
        active_path = self._key_path(key_id)
        try:
            active_path.unlink()
        except FileNotFoundError:
            pass
        with self._lock:
            self._keys.pop(key_id, None)
            self._reindex()

        logger.info("keymaster: rotated key %s → %s", key_id[:8], new_meta.key_id[:8])
        self._notify(self._rotate_observers, "rotation", key_id, new_meta.key_id)
        return new_meta

    # ------------------------------------------------------------------
    # Listing / querying
    # ------------------------------------------------------------------

    def list(
        self,
        key_type: KeyType | None = None,
        scope: KeyScope | None = None,
        active_only: bool = True,
        include_expired: bool = False,
    ) -> list[KeyMeta]:
        """Return keys matching the given filters."""
        with self._lock:
            results = list(self._keys.values())

        if active_only:
            results = [m for m in results if m.active]
        if not include_expired:
            results = [m for m in results if not m.is_expired]
        if key_type is not None:
            results = [m for m in results if m.key_type == key_type]
        if scope is not None:
            results = [m for m in results if scope in m.scopes]
        results.sort(key=lambda m: m.created_at, reverse=True)
        return results

    def list_archived(self) -> list[KeyMeta]:
        """Return all archived (revoked/rotated) keys."""
        archive_dir = self._store / _ARCHIVE_SUBDIR
        results = []
        for p in archive_dir.glob("*.json"):
            meta = self._load_file(p)
            if meta:
                results.append(meta)
        results.sort(key=lambda m: m.created_at, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Usage recording (called by Gatekeeper)
    # ------------------------------------------------------------------

    def record_usage(self, key_id: str, tokens: int = 0, requests: int = 1) -> None:
        """Increment usage counters for *key_id*."""
        with self._lock:
            meta = self._keys.get(key_id)
            if meta is None:
                return
            meta.usage_count += tokens
            meta.request_count += requests
        self._save(meta)

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export(
        self,
        path: str | Path | None = None,
        key_id: str | None = None,
    ) -> dict[str, Any]:
        """Export key(s) as a JSON-serialisable dict.

        If *key_id* is given, exports only that key; otherwise exports all
        active keys.  If *path* is given, also writes to disk.
        """
        with self._lock:
            if key_id:
                meta = self._keys.get(key_id)
                if meta is None:
                    raise KeyError(f"Key not found: {key_id!r}")
                records = [meta.to_dict()]
            else:
                records = [m.to_dict() for m in self._keys.values()]

        payload: dict[str, Any] = {
            "version": "1",
            "exported_at": datetime.now(tz=UTC).isoformat(),
            "keys": records,
        }
        if path:
            p = Path(path)
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def import_keys(self, source: str | Path | dict[str, Any]) -> list[str]:
        """Import keys from a JSON file path, file object, or dict.

        Returns a list of imported key_id strings.  Duplicate IDs are
        skipped (existing key takes precedence).
        """
        if isinstance(source, dict):
            payload = source
        else:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))

        imported: list[str] = []
        for record in payload.get("keys", []):
            meta = KeyMeta.from_dict(record)
            with self._lock:
                if meta.key_id in self._keys:
                    logger.warning(
                        "keymaster import: skipping duplicate key_id %s", meta.key_id[:8]
                    )
                    continue
                self._keys[meta.key_id] = meta
                self._reindex()
            self._save(meta)
            imported.append(meta.key_id)
        logger.info("keymaster: imported %d key(s)", len(imported))
        return imported

    # ------------------------------------------------------------------
    # Auto-rotation background thread
    # ------------------------------------------------------------------

    def _auto_rotate_loop(self, poll_interval: float) -> None:
        while not self._stop_event.wait(poll_interval):
            self._check_auto_rotate()

    def _check_auto_rotate(self) -> None:
        """Rotate any key that is expired or within its rotation_window."""
        with self._lock:
            candidates = [
                m.key_id
                for m in self._keys.values()
                if m.active and (m.is_expired or m.expires_soon)
            ]
        for key_id in candidates:
            try:
                new_meta = self.rotate(key_id)
                logger.info(
                    "keymaster: auto-rotated %s → %s", key_id[:8], new_meta.key_id[:8]
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("keymaster: auto-rotate failed for %s: %s", key_id[:8], exc)

    def stop(self) -> None:
        """Stop the background auto-rotation thread."""
        self._stop_event.set()

    def __repr__(self) -> str:
        with self._lock:
            n = len(self._keys)
        return f"Keymaster(store={self._store!r}, keys={n}, encrypted={_CRYPTO_AVAILABLE})"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def generate_t1_key(body_length: int = 24) -> str:
    """Shortcut: generate a single T1 key string without creating a Keymaster."""
    return T1KeyGenerator.generate(body_length=body_length)


def validate_t1_key(key: str) -> bool:
    """Shortcut: validate a T1 key string."""
    return T1KeyGenerator.validate(key)


__all__ = [
    "KeyMeta",
    "KeyScope",
    "KeyType",
    "Keymaster",
    "T1KeyGenerator",
    "generate_t1_key",
    "validate_t1_key",
]
