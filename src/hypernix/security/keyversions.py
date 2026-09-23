"""Key format versions — the registry behind ``gkey create -v`` and ``gkey version``.

Four formats are issuable (0.72.6):

===========  ========  ==========================================================
``v1``       ``T1_``   The long-standing key. No access level, no password slot.
``v2``       ``T2_``   Adds an access level (1-9), an optional admin password in
                       the prefix, and an SSPKID. Converts to and from ``v1``.
``v2short``  ``T2S_``  HyperLink's key: a body of exactly 26 characters so it can
                       be typed on a phone, and never an administrator.
``v2.1``     ``T2C_``  A v2 key sealed with Rotorvault. The client keeps a kit
                       (``T2CK_``) and sends a key that changes every day — see
                       :mod:`hypernix.security.t2c`.
===========  ========  ==========================================================

:data:`RESERVED_KEY_VERSIONS` stays, empty, so the next named-but-not-yet
format has a place to be refused with a reason rather than as a typo.

One thing worth stating plainly, because it drives the whole design of
``gkey create -v``: **a T2 key is a spelling of a T1 key, not a separate
credential.** Authentication converts it back to its T1 form and looks
*that* up in the key store (see ``T1AuthService._validate_t2``). So a T2
key that was never minted into the store authenticates as nothing at all.
Issuing one means minting the T1 key and then presenting it — never
generating a T2 key on its own.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "KeyVersion",
    "KEY_VERSIONS",
    "RESERVED_KEY_VERSIONS",
    "LATEST_KEY_VERSION",
    "DEFAULT_KEY_VERSION",
    "resolve_key_version",
    "key_version_names",
]


@dataclass(frozen=True)
class KeyVersion:
    """One issuable key format."""

    name: str
    family: str
    prefix: str
    summary: str
    #: Alternate spellings accepted on the command line. People type
    #: "v2s" and "v2-short" and mean the same thing; refusing those is
    #: pedantry with no upside.
    aliases: tuple[str, ...] = ()
    #: Some formats pin the body length. ``None`` means "caller's choice".
    body_length: int | None = None
    #: Whether the format itself can carry administrative authority. T2S
    #: cannot: it is short enough to type, which is exactly why.
    supports_admin: bool = True
    #: Whether the format has an access-level suffix.
    supports_access_level: bool = False
    #: False for a format that is named but not issuable.
    issuable: bool = True
    #: Why it is not issuable, when it is not.
    unavailable_reason: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} ({self.prefix}…)"


V1 = KeyVersion(
    name="v1",
    family="T1",
    prefix="T1_",
    summary="The long-standing key. Accepted everywhere.",
    aliases=("1", "t1"),
    supports_admin=True,
    supports_access_level=False,
)

V2 = KeyVersion(
    name="v2",
    family="T2",
    prefix="T2_",
    summary="Access level 1-9, optional admin password, SSPKID. Converts to v1.",
    aliases=("2", "t2"),
    supports_admin=True,
    supports_access_level=True,
)

V2_SHORT = KeyVersion(
    name="v2short",
    family="T2S",
    prefix="T2S_",
    summary="HyperLink's key: a 26-character body, never an administrator.",
    aliases=("2short", "v2s", "t2s", "v2-short", "short"),
    body_length=26,
    supports_admin=False,
    supports_access_level=True,
)

V2_1 = KeyVersion(
    name="v2.1",
    family="T2C",
    prefix="T2C_",
    summary="A v2 key sealed with Rotorvault; the key sent changes daily.",
    aliases=("2.1", "t2c", "v2c"),
    supports_admin=True,
    supports_access_level=True,
)

#: Issuable today, in the order ``gkey version`` lists them.
KEY_VERSIONS: tuple[KeyVersion, ...] = (V1, V2, V2_SHORT, V2_1)

#: Named so a request for one is refused with a reason rather than
#: treated as a typo. Empty since v2.1 shipped in 0.72.6.
RESERVED_KEY_VERSIONS: tuple[KeyVersion, ...] = ()

#: The newest issuable format. v2short is a *variant* of v2 for a
#: constrained client, not a later version, so it is not "latest".
LATEST_KEY_VERSION = V2_1

#: What ``gkey create`` mints when told nothing. Stays v1: changing the
#: default output format of a key-minting command is the kind of surprise
#: that ends with someone pasting a key their server refuses.
DEFAULT_KEY_VERSION = V1

_BY_NAME: dict[str, KeyVersion] = {}
for _version in (*KEY_VERSIONS, *RESERVED_KEY_VERSIONS):
    _BY_NAME[_version.name.lower()] = _version
    for _alias in _version.aliases:
        _BY_NAME[_alias.lower()] = _version


def key_version_names() -> list[str]:
    """Canonical names of the issuable formats, for help text and choices."""
    return [version.name for version in KEY_VERSIONS]


def resolve_key_version(name: str) -> KeyVersion:
    """Look up a key version by name or alias.

    Raises ``ValueError`` naming the actual problem: a reserved version
    says why it is reserved, and an unknown one lists what exists.
    """
    key = (name or "").strip().lower()
    version = _BY_NAME.get(key)
    if version is None:
        raise ValueError(
            f"Unknown key version {name!r}. Available: "
            + ", ".join(key_version_names())
        )
    if not version.issuable:
        raise ValueError(version.unavailable_reason)
    return version
