"""The first credential a new server has.

A freshly started T1 API has a key store with nothing in it, and every
route that could put something there is admin-only. That is a closed
loop: you cannot mint the first key over the API, because minting needs a
key. The documented way out is to run ``gkey`` on the server box — which
works, and is one more thing to know at exactly the moment someone is
trying to find out whether any of this works at all.

So a new server issues itself one credential, with three limits that make
it safe to print on a terminal:

* **Loopback only.** The key is bound to the machine it was created on,
  enforced on every request against the address the network policy
  already resolved. A key that leaks off the box authenticates nowhere.
* **Three days.** Long enough to set a server up over a weekend, short
  enough that forgetting about it is not a standing risk. Expiry is the
  key store's own, so it needs nothing here to enforce it.
* **Once.** It is minted when the store has no bootstrap key, not on
  every start, so restarting a server does not litter it with admin
  credentials.

It is a T2 admin key rather than a T1 one because the operator is going
to type it into ``waiter`` — and because admin on a T2 key is carried by
the password component, which makes "this is the powerful one" visible in
the credential itself.
"""
from __future__ import annotations

import ipaddress
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "BOOTSTRAP_TAG",
    "BOOTSTRAP_TTL_SECONDS",
    "BootstrapKey",
    "is_loopback",
    "ensure_bootstrap_key",
    "bootstrap_banner",
]

#: Marks the key in the store. Tags survive a restart, which is what lets
#: "have I already made one?" be answered without extra state.
BOOTSTRAP_TAG = "bootstrap"

#: Three days, as specified.
BOOTSTRAP_TTL_SECONDS = 3 * 24 * 60 * 60


@dataclass(frozen=True)
class BootstrapKey:
    """A freshly minted bootstrap credential, or a note that one exists."""

    key_id: str
    #: The T2 spelling, to be shown once. Empty when the key already
    #: existed — the store keeps no plaintext to show a second time.
    key: str
    password: str
    expires_at: float
    created: bool

    @property
    def expires_in_hours(self) -> float:
        return max(0.0, (self.expires_at - time.time()) / 3600.0)


def is_loopback(address: str) -> bool:
    """Is *address* this machine talking to itself?

    Anything unparseable is not loopback. An address the server cannot
    make sense of must not be given the benefit of the doubt by the one
    check standing between a printed admin key and the network.
    """
    text = (address or "").strip()
    if not text:
        return False
    # A bracketed or zone-suffixed v6 address reaches here from some proxies.
    text = text.strip("[]").split("%", 1)[0]
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def _existing(keymaster: Any) -> Any | None:
    """The live bootstrap key in the store, if there is one.

    Expired keys do not count: the point of a three-day key is that on
    day four the server is willing to issue another one rather than
    leaving the operator locked out.
    """
    try:
        keys = keymaster.list()
    except Exception:  # noqa: BLE001
        return None
    for meta in keys:
        if meta.tags.get(BOOTSTRAP_TAG) != "1":
            continue
        if getattr(meta, "is_expired", False):
            continue
        return meta
    return None


def ensure_bootstrap_key(keymaster: Any, *, ttl_seconds: int = BOOTSTRAP_TTL_SECONDS) -> BootstrapKey | None:
    """Mint the local admin key if this server has not got a live one.

    Returns None when the key store cannot be used at all — a server that
    cannot mint its bootstrap key still has to start, because the
    operator may have a perfectly good key already.
    """
    from ..security.keymaster import KeyScope, KeyType
    from ..security.t2keys import T2KeyGenerator

    existing = _existing(keymaster)
    if existing is not None:
        return BootstrapKey(
            key_id=existing.key_id,
            key="",
            password="",  # nosec B106 - an empty default, not a credential
            expires_at=existing.expires_at or 0.0,
            created=False,
        )

    expires_at = time.time() + ttl_seconds
    try:
        meta = keymaster.create(
            key_type=KeyType.ADMIN,
            scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE},
            expires_at=expires_at,
            prefix="bootstrap",
            tags={BOOTSTRAP_TAG: "1", "local_only": "1"},
            note="Local bootstrap key, created on first start. Loopback only.",
        )
        wrapped = T2KeyGenerator.from_t1_admin(meta.key, access_level=9)
    except Exception:  # noqa: BLE001
        logger.warning("t1api.bootstrap: could not mint the bootstrap key", exc_info=True)
        return None

    logger.info(
        "t1api.bootstrap: minted a loopback-only admin key %s, valid for %d hours",
        meta.key_id[:8],
        ttl_seconds // 3600,
    )
    return BootstrapKey(
        key_id=meta.key_id,
        key=wrapped.raw,
        password=wrapped.password,
        expires_at=expires_at,
        created=True,
    )


def is_bootstrap_key(meta: Any) -> bool:
    """Does this key carry the loopback restriction?"""
    tags = getattr(meta, "tags", None) or {}
    return tags.get(BOOTSTRAP_TAG) == "1"


def bootstrap_banner(key: BootstrapKey, *, base_url: str = "") -> str:
    """What the operator sees on the terminal, once.

    Printed rather than written to a file on purpose: this is key
    material, and a file is a copy that outlives the three days.

    *base_url* is whatever the deployment has actually been told about
    itself, and is often nothing: uvicorn owns the bind address, so a
    server created as a factory does not know its own port. When it is
    unknown the command carries a visible ``<port>`` placeholder rather
    than a plausible guess — a copy-pasteable line that connects to the
    wrong place is worse than one that obviously needs filling in.
    """
    if not key.created:
        return ""
    target = base_url or "http://127.0.0.1:<port>"
    return (
        "\n"
        "  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        "  ┃  First start — here is a key to set this server up with.     ┃\n"
        "  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
        "\n"
        f"    {key.key}\n"
        "\n"
        f"    Works only from this machine, and only for {key.expires_in_hours:.0f} more hours.\n"
        "    Shown once — it is not stored in plaintext anywhere.\n"
        "\n"
        "    Point waiter at it:\n"
        f"      waiter serv -A -I {target} -K '{key.key}' -L\n"
        "\n"
        "    Then mint yourself a real key and stop relying on this one:\n"
        "      waiter keys --create --type admin --scopes admin,read,write\n"
    )
