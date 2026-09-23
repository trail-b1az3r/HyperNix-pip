"""waiter.local_config — the waiter TUI's local config file.

Backs the ``-F/--config-file``, ``-E/--encrypt``, ``-s/--save`` flags of
``waiter serv``. Encryption mirrors ``hypernix.keymaster``'s own pattern
exactly (same Fernet-from-master-key approach, same graceful plain-JSON
fallback with a one-time warning when ``cryptography`` isn't installed) —
see ``hypernix/keymaster.py::_make_fernet`` / ``_get_or_create_master_key``.
Waiter keeps its own master key rather than sharing Keymaster's, since the
two encrypt different things for different lifetimes.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from cryptography.fernet import Fernet as _Fernet  # type: ignore

    _CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only w/o the security extra
    _Fernet = None  # type: ignore
    _CRYPTO_AVAILABLE = False

_DEFAULT_CONFIG_DIR = Path.home() / ".hypernix" / "waiter"
_DEFAULT_CONFIG_PATH = _DEFAULT_CONFIG_DIR / "waiter.config.jsonl"
_MASTER_KEY_FILE = _DEFAULT_CONFIG_DIR / ".master.key"


def _make_fernet(secret: bytes) -> Any:
    key = base64.urlsafe_b64encode(secret[:32].ljust(32, b"\x00"))
    return _Fernet(key)


def _get_or_create_master_key(config_dir: Path) -> bytes:
    key_file = config_dir / ".master.key"
    if key_file.exists():
        return bytes.fromhex(key_file.read_text(encoding="ascii").strip())
    raw = secrets.token_bytes(32)
    config_dir.mkdir(parents=True, exist_ok=True)
    key_file.write_text(raw.hex(), encoding="ascii")
    try:
        os.chmod(key_file, 0o600)
    except OSError:
        pass
    return raw


@dataclass
class WaiterLocalConfig:
    """Everything ``waiter serv -A`` resolves and persists.

    Field names track the spec's ``serv`` flags directly (``server`` <- -I,
    ``key`` <- -K, etc.) so the CLI layer stays a thin translation from
    argparse Namespace to this dataclass.
    """

    server: str | None = None  # -I
    key: str | None = None  # -K (raw T1 key OR scoped token)
    local_only: bool = False  # -L
    port: int | None = None  # -P
    home_url: str | None = None  # -H
    blacklist: list[str] = field(default_factory=list)  # -B
    whitelist: list[str] = field(default_factory=list)  # -W
    extra_config: dict[str, str] = field(default_factory=dict)  # -C
    # -r, stored as the raw "SUBJECT=COUNT/WINDOW" strings the operator
    # typed. Keeping the source form rather than the parsed dict means a
    # re-run of `waiter serv -A` can re-apply them verbatim against a
    # rebuilt server, and `waiter config` shows what was actually asked
    # for instead of a normalized shape nobody typed.
    forced_limits: list[str] = field(default_factory=list)
    # 0.72.6: -c asked the server to conceal this key. Kept so `waiter
    # config` shows it and a rebuilt server can be told again.
    concealed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WaiterLocalConfig:
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


#: What a password-locked config file (``waiter serv -e``) starts with.
LOCK_PREFIX = "RVLOCK1:"
_LOCK_CONTEXT = "waiter-config"


class WaiterConfigLocked(Exception):
    """The config is locked and no (or the wrong) password was given."""


class WaiterConfigStore:
    """Load/save a :class:`WaiterLocalConfig` to a ``.jsonl``-style file
    (one JSON object per line is overkill for a single config record, but
    matches the ``-s`` flag's ".jsonl" naming from the spec — we write
    exactly one line so it's trivially `tail -f`-able if you're debugging
    a live rewrite).

    Encryption is auto-detected on read: :meth:`load` tries plain JSON
    first, and falls back to Fernet decryption (using this machine's local
    waiter master key) if that fails. This means a caller reading back a
    config — e.g. ``waiter config -F path``, run without ``-E`` — doesn't
    need to remember whether the file was originally written with
    ``-E``; only :meth:`save` needs the flag, matching how the ``-E``
    flag itself is described in the spec as a write-time concern ("encrypt
    configuration/secrets").
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        encrypt: bool = False,
        password: str | None = None,
        password_provider: Any = None,
    ) -> None:
        self.path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
        self.encrypt = encrypt
        #: Set, the file is written locked with it (Rotorvault + scrypt).
        #: Loading a locked file sets it, so a later save stays locked.
        self.password = password
        self.password_provider = password_provider
        self._write_cipher: Any = None
        if encrypt:
            if _CRYPTO_AVAILABLE:
                master = _get_or_create_master_key(self.path.parent or _DEFAULT_CONFIG_DIR)
                self._write_cipher = _make_fernet(master)
            else:
                warnings.warn(
                    "hypernix.waiter: -E/--encrypt requested but 'cryptography' is not "
                    "installed. Config will be saved as plain JSON. "
                    "Install with: pip install hypernix[security]",
                    UserWarning,
                    stacklevel=2,
                )

    def _read_cipher(self) -> Any:
        """Cipher used for decryption on load — independent of whether
        *this* WaiterConfigStore instance was constructed with encrypt=True,
        since the master key (and therefore the cipher) only depends on the
        config directory, not on this call's intent."""
        if not _CRYPTO_AVAILABLE:
            return None
        master = _get_or_create_master_key(self.path.parent or _DEFAULT_CONFIG_DIR)
        return _make_fernet(master)

    def is_locked(self) -> bool:
        try:
            with self.path.open(encoding="utf-8") as handle:
                return handle.read(len(LOCK_PREFIX)) == LOCK_PREFIX
        except OSError:
            return False

    def save(self, config: WaiterLocalConfig) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(config.to_dict(), separators=(",", ":"))
        if self.password:
            from ..security.rotorvault import seal_with_password

            payload = LOCK_PREFIX + seal_with_password(
                self.password, payload.encode("utf-8"), context=_LOCK_CONTEXT)
        elif self._write_cipher is not None:
            payload = self._write_cipher.encrypt(payload.encode("utf-8")).decode("ascii")
        self.path.write_text(payload + "\n", encoding="utf-8")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return self.path

    def load(self) -> WaiterLocalConfig | None:
        if not self.path.exists():
            return None
        raw = self.path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        if raw.startswith(LOCK_PREFIX):
            return self._unlock(raw[len(LOCK_PREFIX):])
        try:
            return WaiterLocalConfig.from_dict(json.loads(raw))
        except json.JSONDecodeError:
            pass  # not plain JSON — try decrypting it below
        cipher = self._read_cipher()
        if cipher is None:
            warnings.warn(
                f"hypernix.waiter: {self.path} doesn't look like plain JSON and "
                "'cryptography' isn't installed to try decrypting it. "
                "Install with: pip install hypernix[security]",
                UserWarning,
                stacklevel=2,
            )
            return None
        try:
            decrypted = cipher.decrypt(raw.encode("ascii")).decode("utf-8")
        except Exception as exc:  # noqa: BLE001 - genuinely unreadable file
            warnings.warn(
                f"hypernix.waiter: could not read {self.path} as plain JSON or "
                f"decrypt it with the local waiter master key ({exc}). If this "
                "config was created on a different machine, its encrypted config "
                "can't be read here — the master key is per-machine by design.",
                UserWarning,
                stacklevel=2,
            )
            return None
        return WaiterLocalConfig.from_dict(json.loads(decrypted))


    def _unlock(self, token: str) -> WaiterLocalConfig:
        from ..security.rotorvault import RotorvaultError, open_with_password

        password = self.password
        if not password and self.password_provider is not None:
            password = self.password_provider()
        if not password:
            raise WaiterConfigLocked(
                f"{self.path} is locked (waiter serv -e). Give its password: run waiter "
                "in a terminal to be asked, or set HNX_WAITER_PASSWORD."
            )
        try:
            plain = open_with_password(password, token, context=_LOCK_CONTEXT)
        except RotorvaultError as exc:
            raise WaiterConfigLocked(f"wrong password for {self.path}") from exc
        self.password = password
        return WaiterLocalConfig.from_dict(json.loads(plain.decode("utf-8")))


__all__ = ["LOCK_PREFIX", "WaiterConfigLocked", "WaiterLocalConfig", "WaiterConfigStore"]
