"""Which machine this is, said in a way a name cannot fake.

A phone that has found a server at some address needs to answer one
question before it sends anything: *is this the machine I paired with?*
The obvious answer — compare the server name — is the wrong one. Names
are chosen by whoever set the machine up, they are advertised in the
clear, and on a tailnet or a LAN anything can call itself ``desktop``.
Authenticating on a human-readable name means the first machine to claim
the name wins.

So each installation gets a fingerprint: a hash of a random seed
generated once and kept on disk. Properties that matter:

* **Stable.** Same across restarts, upgrades, address changes and key
  rotations, which is what makes pinning it useful at all. A fingerprint
  that changed when the token secret was rotated would train people to
  click through the warning.
* **Unguessable.** Derived from 32 random bytes, not from the hostname,
  the MAC address or anything else an attacker can look up. Claiming
  another machine's fingerprint requires its seed file.
* **Not a secret.** It is a one-way hash of the seed and reveals nothing
  about it, so it is safe to hand to any authenticated caller.

What it is *not* is proof on its own. Anyone who can read the
fingerprint can repeat it; it identifies a machine to a client that
already knows what to expect, in the same way a TLS certificate
fingerprint does. The credential is still the credential. What this adds
is the ability to notice that the address you reached is answering for a
*different* machine than last time — which is exactly the case a name
comparison cannot see.
"""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["fingerprint", "seed_path", "FINGERPRINT_LENGTH"]

#: Domain separation. If the seed is ever reused for anything else, the
#: two derivations must not be able to collide.
_DOMAIN = b"hypernix.hyperlink.server-identity.v1"

#: 32 hex characters — 128 bits. Long enough that a collision is not a
#: thing that happens, short enough to read out over a phone call when
#: someone is comparing what their phone shows against what the server
#: printed.
FINGERPRINT_LENGTH = 32

#: In-process cache. The seed is read on every ``/hyperlink/endpoints``
#: call otherwise, which is a file read on a hot path for a value that
#: cannot change while the process is up.
_CACHE: dict[str, str] = {}


def _config_root(config_dir: str | Path | None = None) -> Path:
    if config_dir:
        return Path(config_dir)
    configured = os.environ.get("T1_CONFIG_DIR", "")
    return Path(configured) if configured else Path.home() / ".hypernix" / "t1api"


def seed_path(config_dir: str | Path | None = None) -> Path:
    return _config_root(config_dir) / "hyperlink" / "server-identity"


#: Below this, a seed file is treated as damage rather than as key
#: material. A truncated write leaves something that *looks* like a
#: seed, and deriving an identity from five known bytes is worse than
#: having no identity at all -- so a short file is replaced, not used.
MIN_SEED_BYTES = 16


def _existing_seed(path: Path) -> bytes | None:
    try:
        seed = path.read_bytes()
    except OSError:
        return None
    return seed if len(seed) >= MIN_SEED_BYTES else None


def _read_or_create_seed(path: Path) -> bytes:
    existing = _existing_seed(path)
    if existing is not None:
        return existing

    seed = secrets.token_bytes(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            # It exists and _existing_seed rejected it, so it is short.
            # Removed before the exclusive create rather than truncated
            # in place, so a concurrent reader never sees a zero-length
            # file where a seed used to be.
            path.unlink()
        # Written 0600 from the start rather than chmod-ed afterwards:
        # between the two there is a window where the seed is readable
        # by every user on the box, and the whole value of the seed is
        # that only this machine has it.
        handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(handle, seed)
        finally:
            os.close(handle)
    except FileExistsError:
        # Two workers started at once and the other won. Its seed is as
        # good as ours; take it, so every worker in the process group
        # reports the same fingerprint. If what it wrote is *also* too
        # short, keep ours: a per-process identity is bad, an identity
        # derived from a known truncated file is worse.
        return _existing_seed(path) or seed
    except OSError as exc:
        # A read-only config directory is not a reason to fail a request.
        # The fingerprint is then per-process rather than per-machine,
        # which is worse but still better than no answer -- and the log
        # line says which case you are in.
        logger.warning(
            "hyperlink.identity: could not persist the server identity seed "
            "at %s (%s); this process will report a fingerprint that changes "
            "on restart.", path, exc,
        )
    return seed


def fingerprint(config_dir: str | Path | None = None) -> str:
    """This installation's stable identifier, as 32 hex characters."""
    path = seed_path(config_dir)
    key = str(path)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    digest = hashlib.sha256(_DOMAIN + _read_or_create_seed(path)).hexdigest()
    value = digest[:FINGERPRINT_LENGTH]
    _CACHE[key] = value
    return value
