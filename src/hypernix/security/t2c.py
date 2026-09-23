"""hypernix.security.t2c — v2.1 keys: T2C, sealed with Rotorvault.

A T2C key is a T2 key that is never written down in the clear. What the
client keeps is a **kit**; what it sends is that day's **key**, and the
two are different strings::

    kit   T2CK_<base64 json>            kept by waiter, never sent
    key   T2C_<device>.<seal>-<level>   sent, and different every day

How a key is built
------------------
1. **Inner** (once, stable): the underlying T2 key sealed for the
   server's RSA public key — :func:`rotorvault.seal_for`: RSA-OAEP wraps
   a content key, then Blowfish, Twofish, the rotor stage, AES-256-GCM,
   inversion and base64. Only the server can open it, and the client can
   make it without the server's help, because it needs only the public
   key.
2. **Outer** (daily): the inner sealed again under the device's key for
   the day (:func:`rotorvault.daily_key`, HMAC of the device's secret and
   the UTC date). So the string that crosses the network changes every
   day, and a copied one stops working after its day and the grace day
   either side.

How the server reads one
------------------------
Device ID -> that device's secret -> the day's key (today, then
yesterday and tomorrow for clocks that disagree) -> the inner -> the
server's RSA private key -> the T2 key -> its T1 form, which is looked up
in the key store exactly like every other T2 spelling. A device is bound
to the key it was registered with, so one device's secret cannot vouch
for a different key.

What it protects against, and what it does not
----------------------------------------------
A leaked config file or shell history no longer holds a key that works
forever; a key read off the wire works for at most three calendar days.
It is still a bearer credential for that window — anyone holding it can
use it until it expires or the device is revoked
(``DELETE /auth/t2c/devices/{id}``). And the kit is what makes new keys:
protect it like the key it replaces (``waiter serv -e`` locks it with a
password).

The server's RSA private key and each device's secret are kept in
``<keymaster dir>/t2c/`` with owner-only permissions. They are not
encrypted at rest: a key to decrypt them would have to live on the same
disk, which would protect nothing.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import string
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from . import rotorvault as rv

__all__ = [
    "T2C_PREFIX",
    "KIT_PREFIX",
    "GRACE_DAYS",
    "T2CKit",
    "T2CAuthority",
    "T2CError",
    "looks_like_t2c",
    "looks_like_kit",
    "split_key",
    "compose_key",
    "build_inner",
    "wrap_device_secret",
    "today_utc",
]

T2C_PREFIX = "T2C_"
KIT_PREFIX = "T2CK_"

#: Days either side of the server's UTC date a key is still accepted.
GRACE_DAYS = 1

_DEVICE_ALPHABET = string.ascii_lowercase + string.digits
_DEVICE_ID_LENGTH = 12
_SECRET_BYTES = 32


class T2CError(ValueError):
    """A T2C key or kit that cannot be used, with the reason."""


def today_utc() -> date:
    return datetime.now(UTC).date()


def looks_like_t2c(value: str) -> bool:
    return isinstance(value, str) and value.startswith(T2C_PREFIX)


def looks_like_kit(value: str) -> bool:
    return isinstance(value, str) and value.startswith(KIT_PREFIX)


def _outer_context(device_id: str, day: date) -> str:
    return f"t2c/outer|{device_id}|{day.isoformat()}"


_INNER_CONTEXT = "t2c/inner"


def build_inner(server_public_key, t2_raw: str) -> str:
    """The stable half: the T2 key, sealed for the server's public key."""
    return rv.seal_for(server_public_key, t2_raw.encode("utf-8"), context=_INNER_CONTEXT)


def compose_key(device_id: str, secret: bytes, inner: str, level: int, day: date) -> str:
    """That day's key for a device."""
    outer = rv.seal(rv.daily_key(secret, device_id, day), inner.encode("ascii"),
                    context=_outer_context(device_id, day))
    return f"{T2C_PREFIX}{device_id}.{outer}-{int(level)}"


def split_key(key: str) -> tuple[str, str, int]:
    """``(device_id, sealed, level)`` of a T2C key, or :class:`T2CError`."""
    if not looks_like_t2c(key):
        raise T2CError("not a T2C key (they start T2C_)")
    body, dash, level_text = key[len(T2C_PREFIX):].rpartition("-")
    if not dash or not level_text.isdigit() or not 1 <= int(level_text) <= 9:
        raise T2CError("a T2C key ends in -<access level 1-9>")
    device_id, dot, sealed = body.partition(".")
    if not dot or len(device_id) != _DEVICE_ID_LENGTH or not all(c in _DEVICE_ALPHABET for c in device_id):
        raise T2CError("a T2C key starts T2C_<12-character device id>.")
    if not sealed:
        raise T2CError("this T2C key has no sealed part")
    return device_id, sealed, int(level_text)


def wrap_device_secret(server_public_key, secret: bytes) -> str:
    """A device secret encrypted for the server (RSA-OAEP-SHA256), base64.

    How a client registers a device without the secret ever crossing the
    network in the clear.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    if isinstance(server_public_key, (str, bytes)):
        server_public_key = rv.load_public_key(server_public_key)
    wrapped = server_public_key.encrypt(secret, padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    return base64.b64encode(wrapped).decode("ascii")


# ---------------------------------------------------------------------------
# The client's kit
# ---------------------------------------------------------------------------


@dataclass
class T2CKit:
    """What a client keeps in place of a key. Makes each day's key."""

    device_id: str
    secret: bytes
    inner: str
    access_level: int
    server_fingerprint: str = ""
    label: str = ""

    def key_for(self, day: date | None = None) -> str:
        return compose_key(self.device_id, self.secret, self.inner, self.access_level,
                           day or today_utc())

    def to_text(self) -> str:
        payload = {
            "v": 1,
            "scheme": rv.SCHEME,
            "device": self.device_id,
            "secret": base64.urlsafe_b64encode(self.secret).decode("ascii"),
            "inner": self.inner,
            "level": self.access_level,
            "server": self.server_fingerprint,
            "label": self.label,
        }
        blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return KIT_PREFIX + base64.urlsafe_b64encode(blob).decode("ascii").rstrip("=")

    @classmethod
    def from_text(cls, text: str) -> T2CKit:
        text = (text or "").strip()
        if not looks_like_kit(text):
            raise T2CError("not a T2C kit (they start T2CK_)")
        body = text[len(KIT_PREFIX):]
        try:
            payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
            secret = base64.urlsafe_b64decode(payload["secret"])
            kit = cls(
                device_id=str(payload["device"]),
                secret=secret,
                inner=str(payload["inner"]),
                access_level=int(payload["level"]),
                server_fingerprint=str(payload.get("server", "")),
                label=str(payload.get("label", "")),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise T2CError("this T2C kit is damaged — copy it again from where it was issued") from exc
        if len(kit.secret) < 16 or not 1 <= kit.access_level <= 9:
            raise T2CError("this T2C kit is damaged — copy it again from where it was issued")
        return kit

    def describe(self) -> dict[str, Any]:
        """Everything but the secret parts."""
        return {"device_id": self.device_id, "access_level": self.access_level,
                "server_fingerprint": self.server_fingerprint, "label": self.label,
                "scheme": rv.SCHEME}


# ---------------------------------------------------------------------------
# The server's authority
# ---------------------------------------------------------------------------


def _binding(t1_key: str) -> str:
    return hashlib.sha256(t1_key.encode("utf-8")).hexdigest()


class T2CAuthority:
    """The server side: its RSA key, and the devices it has registered.

    Nothing is created until something needs it, so a server that never
    sees a T2C key never writes a private key to disk.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self._private_key = None

    # -- the RSA key -------------------------------------------------

    @property
    def _key_path(self) -> Path:
        return self.directory / "server_rsa.pem"

    @property
    def _devices_path(self) -> Path:
        return self.directory / "devices.json"

    def _write_private(self, path: Path, data: bytes) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.directory, 0o700)
        except OSError:
            pass
        fd, temp = tempfile.mkstemp(dir=str(self.directory), prefix=".t2c-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.chmod(temp, 0o600)
            os.replace(temp, path)
        except BaseException:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise

    def private_key(self):
        with self._lock:
            if self._private_key is None:
                if self._key_path.exists():
                    self._private_key = rv.load_private_key(self._key_path.read_bytes())
                else:
                    from cryptography.hazmat.primitives import serialization

                    key = rv.generate_rsa_private_key()
                    pem = key.private_bytes(
                        serialization.Encoding.PEM,
                        serialization.PrivateFormat.PKCS8,
                        serialization.NoEncryption(),
                    )
                    self._write_private(self._key_path, pem)
                    self._private_key = key
            return self._private_key

    def public_key_pem(self) -> str:
        return rv.public_key_pem(self.private_key())

    def fingerprint(self) -> str:
        return rv.public_key_fingerprint(self.private_key())

    # -- devices -----------------------------------------------------

    def _load_devices(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self._devices_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            return {}

    def _save_devices(self, devices: dict[str, dict[str, Any]]) -> None:
        self._write_private(self._devices_path, json.dumps(devices, indent=2).encode("utf-8"))

    def register_device(self, *, bound_key: str, secret: bytes | None = None,
                        label: str = "", access_level: int = 1) -> tuple[str, bytes]:
        """Register a device for the T1 key *bound_key*. Returns (id, secret)."""
        secret = secret if secret is not None else secrets.token_bytes(_SECRET_BYTES)
        if len(secret) < 16:
            raise T2CError("a device secret is at least 16 bytes")
        with self._lock:
            devices = self._load_devices()
            device_id = ""
            while not device_id or device_id in devices:
                device_id = "".join(secrets.choice(_DEVICE_ALPHABET) for _ in range(_DEVICE_ID_LENGTH))
            devices[device_id] = {
                "secret": base64.b64encode(secret).decode("ascii"),
                "binding": _binding(bound_key),
                "label": label[:80],
                "access_level": int(access_level),
                "created_at": time.time(),
                "revoked": False,
            }
            self._save_devices(devices)
        return device_id, secret

    def register_wrapped(self, *, bound_key: str, wrapped_secret: str, label: str = "",
                         access_level: int = 1) -> str:
        """Register a device whose secret the client encrypted for us."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        try:
            secret = self.private_key().decrypt(
                base64.b64decode(wrapped_secret, validate=True),
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                             algorithm=hashes.SHA256(), label=None),
            )
        except (ValueError, TypeError) as exc:
            raise T2CError("the device secret was not encrypted for this server's key") from exc
        device_id, _ = self.register_device(bound_key=bound_key, secret=secret, label=label,
                                            access_level=access_level)
        return device_id

    def devices_for(self, bound_key: str) -> list[dict[str, Any]]:
        binding = _binding(bound_key)
        return [
            {"device_id": device_id, "label": record.get("label", ""),
             "access_level": record.get("access_level", 1),
             "created_at": record.get("created_at"), "revoked": bool(record.get("revoked"))}
            for device_id, record in sorted(self._load_devices().items())
            if record.get("binding") == binding
        ]

    def revoke(self, device_id: str, *, bound_key: str | None = None) -> bool:
        """Revoke a device. With *bound_key*, only one bound to that key."""
        with self._lock:
            devices = self._load_devices()
            record = devices.get(device_id)
            if record is None or (bound_key is not None and record.get("binding") != _binding(bound_key)):
                return False
            record["revoked"] = True
            record["revoked_at"] = time.time()
            self._save_devices(devices)
            return True

    # -- issuing and opening -----------------------------------------

    def issue_kit(self, t2_raw: str, *, bound_key: str, access_level: int, label: str = "") -> T2CKit:
        """Everything a client needs, made on the server (``gkey create -v v2.1``)."""
        device_id, secret = self.register_device(bound_key=bound_key, label=label,
                                                 access_level=access_level)
        inner = build_inner(self.private_key().public_key(), t2_raw)
        return T2CKit(device_id=device_id, secret=secret, inner=inner,
                      access_level=access_level, server_fingerprint=self.fingerprint(),
                      label=label)

    def open(self, key: str, *, today: date | None = None) -> tuple[str, str]:
        """The T2 key inside *key*, and the device it came from.

        Raises :class:`T2CError` naming the reason: an unknown or revoked
        device, a key from outside the accepted days, or one sealed for a
        different server.
        """
        device_id, sealed, level = split_key(key)
        record = self._load_devices().get(device_id)
        if record is None:
            raise T2CError(f"device {device_id} is not registered on this server")
        if record.get("revoked"):
            raise T2CError(f"device {device_id} has been revoked")
        secret = base64.b64decode(record["secret"])
        day = today or today_utc()
        inner: bytes | None = None
        offsets = [0]
        for distance in range(1, GRACE_DAYS + 1):
            offsets += [-distance, distance]
        for offset in offsets:
            candidate = day + timedelta(days=offset)
            try:
                inner = rv.open_sealed(rv.daily_key(secret, device_id, candidate), sealed,
                                       context=_outer_context(device_id, candidate))
                break
            except rv.RotorvaultError:
                continue
        if inner is None:
            raise T2CError(
                "this T2C key is not valid today — it was made for another day, or altered. "
                "waiter makes a fresh one from its kit each day."
            )
        try:
            t2_raw = rv.open_sealed_for(self.private_key(), inner.decode("ascii"),
                                        context=_INNER_CONTEXT).decode("utf-8")
        except (rv.RotorvaultError, UnicodeDecodeError) as exc:
            raise T2CError("this T2C key was sealed for a different server") from exc
        from .t2keys import T2KeyGenerator

        parsed = T2KeyGenerator.parse(t2_raw)
        if parsed.access_level != level:
            raise T2CError("this T2C key's access level does not match the key inside it")
        if _binding(T2KeyGenerator.to_t1(parsed)) != record.get("binding"):
            raise T2CError(f"device {device_id} is registered to a different key")
        return t2_raw, device_id
