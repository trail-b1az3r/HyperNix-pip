"""hypernix.security.t2keys — the T2 key family, SSPKIDs, and T2 to T1 conversion.

T2 is not a replacement for T1. It is T1's structure plus three things
T1 has no room for:

1. **An access level** (1-9), carried in the suffix, so a key says what
   tier it is without a database lookup.
2. **An admin password component** in the prefix, so an admin key cannot
   be forged by pattern alone — the body being random is not enough when
   the format itself is public.
3. **An SSPKID**, so one V1 Server ID can own several keys and each is
   still individually addressable.

The key shapes
--------------
Four types, three of which exist today::

    T2   — the general key.        T2_<prefix>_<body><t1suffix>-<level>
    T2S  — the HyperLink key.      Exactly 26 body characters.
    T2C  — the sealed key.         See hypernix.security.t2c (0.72.6).

``T2S`` exists because HyperLink is a phone talking to a home PC, and the
general T2 key is too long to be a fallback when a QR scan fails. Its body
is fixed at 26 characters — short enough to type once, long enough that
26 characters of ``[A-Za-z0-9]`` is ~155 bits. Outside HyperLink a T2S key
is deliberately weak: read, and non-admin write. Nothing else. That is
what makes it safe to type into a phone in a coffee shop.

``T2C`` does not use the pattern below and :func:`generate` does not
mint one. A T2C key is a T2 key *sealed*: the underlying T2 key encrypted
for the server's RSA key and again under a per-device key that changes
daily (Rotorvault). It was held back until it had a real key-agreement
step — the first specification derived it from the holder's public IP,
which is no secret. :mod:`hypernix.security.t2c` builds and opens them;
this module only recognises the prefix.

Release gating
--------------
The **T2 API** does not ship until HyperNix 1.x. What ships now is the T2
*key system* and T1's ability to recognise, validate, and convert T2 keys
(``T1 v1.0.26.8.1.0``). :data:`T2_API_RELEASE_VERSION` records where the
API itself lands, and :func:`t2_api_available` is the single check —
scattered version comparisons are how a feature half-ships.

Compatibility, which is the whole point
---------------------------------------
Every existing T1 key keeps working, unchanged. Every existing V1 Server
ID stays valid. A T2 key converts to a T1 key that a T1-only server
accepts, losing exactly the information T1 cannot express (the access
level and the SSPKID) and nothing else. :func:`to_t1` is that conversion,
and it is lossy *on purpose and in one direction* — see its docstring.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import string
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Sentinel for "the caller said nothing", so that an explicit
#: ``store_dir=None`` can mean "ephemeral" and be told apart from the
#: default. A plain None default cannot express both.
_UNSET: Any = object()


def default_sspkid_store_dir() -> Path:
    """Where SSPKID assignments live when nobody says otherwise.

    The Keymaster's directory, and by the same reasoning: ``gkey`` and
    the server both read ``T1_KEYMASTER_DIR``, and two halves of one tool
    disagreeing about where identity lives is a failure in which both
    halves appear to work.
    """
    configured = os.environ.get("T1_KEYMASTER_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".hypernix" / "keymaster"

__all__ = [
    "T2Type",
    "T2Key",
    "T2KeyGenerator",
    "SSPKID",
    "default_sspkid_store_dir",
    "ServerKeyRegistry",
    "SSPKIDCollision",
    "encode_sspkid_index",
    "decode_sspkid_index",
    "generate_admin_password",
    "validate_admin_password",
    "generate_host_id",
    "validate_host_id",
    "T2_API_RELEASE_VERSION",
    "t2_api_available",
    "T2C_UNAVAILABLE_REASON",
    "ACCESS_LEVELS",
    "T2S_BODY_LENGTH",
    "HOST_ID_LENGTH",
]

# ---------------------------------------------------------------------------
# Release gating
# ---------------------------------------------------------------------------

#: The T2 *API* ships in the 1.x line. T1 v1.0.26.8.1.0 recognises T2
#: keys; it does not serve a T2 API. One constant, one check — see
#: :func:`t2_api_available`.
T2_API_RELEASE_VERSION = "1.0.0"

#: What generate() and parse() say to a T2C request: the family exists
#: (0.72.6) but is sealed, not spelled, so this module is the wrong door.
T2C_UNAVAILABLE_REASON = (
    "A T2C (v2.1) key is sealed, not spelled like other T2 keys: mint one with "
    "'gkey create -v v2.1', or let waiter seal an existing key with 'waiter serv -E'. "
    "hypernix.security.t2c opens and builds them."
)


def looks_like_t2(key: str) -> bool:
    """Is this credential in the T2 family, whichever member?

    Derived from :class:`T2Type` rather than from a list of prefixes.
    Two call sites tested ``key[:3] in ("T2_", "T2S")``, which silently
    stopped covering the family the moment T2P was added: a T2P key fell
    through to the T1 path and was rejected as malformed, several layers
    before the code that had an opinion about it.
    """
    if not isinstance(key, str) or not key.startswith("T2"):
        return False
    return any(key.startswith(f"{member.value}_") for member in T2Type)


def t2_api_available(hypernix_version: str | None = None) -> bool:
    """Is the T2 *API* (as opposed to T2 keys) released in this build?

    Deliberately the only place that answers this. A feature gated by
    version comparisons scattered across a codebase ships halfway.
    """
    if hypernix_version is None:
        try:
            import hypernix

            hypernix_version = getattr(hypernix, "__version__", "0")
        except ImportError:  # pragma: no cover - hypernix importing itself
            hypernix_version = "0"
    return _version_tuple(hypernix_version) >= _version_tuple(T2_API_RELEASE_VERSION)


def _version_tuple(text: str) -> tuple[int, ...]:
    """``"0.72.1rc2"`` -> ``(0, 72, 1)``. Non-numeric tails are ignored.

    Not a PEP 440 parser: this compares release lines, and every suffix
    that matters here (rc, b, .post) sorts inside its own release, which
    truncation already gets right.
    """
    parts: list[int] = []
    for chunk in text.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# ---------------------------------------------------------------------------
# Key shape constants
# ---------------------------------------------------------------------------


class T2Type(StrEnum):
    """Which member of the T2 family a key is."""

    T2 = "T2"        # general purpose
    T2S = "T2S"      # HyperLink; 26-character body, restricted outside HyperLink
    T2P = "T2P"      # carries a billing binding — see hypernix.t1api.billingkeys
    T2C = "T2C"      # sealed with Rotorvault — see hypernix.security.t2c


#: Valid access levels. Level is carried in the suffix and validated on
#: creation, conversion, and every authentication.
ACCESS_LEVELS: tuple[int, ...] = tuple(range(1, 10))

#: What a payment-bearing key is *not*.
#:
#: A T2P key names a billing binding held by the server. It is a
#: credential for spending money, so it is never an administrator: the
#: two authorities have no reason to travel together, and a key that can
#: both pay and reconfigure the server is a worse thing to lose than
#: either alone.
#:
#: The binding itself is never in the key. A key is pasted into terminals
#: and config files and shows up in shell history; a payment token that
#: lives there is a payment token that leaks. The key carries only the
#: fact that a binding exists, and the server looks the rest up.
T2P_IS_NEVER_ADMIN = True

#: T2S bodies are exactly this long — "26 chars, not counting prefix and
#: suffix". Fixed rather than minimum: a variable-length key that must be
#: typed is a key people get wrong.
T2S_BODY_LENGTH = 26

#: Host IDs are 54 characters: 53 alphanumerics plus one special-character
#: suffix, so a Host ID can never be mistaken for a V1 Server ID (which
#: contains a dash and is at most 8 characters) or an SSPKID (which
#: contains a ``#``).
HOST_ID_LENGTH = 54

_BODY_CHARS = string.ascii_letters + string.digits
_MIN_BODY_LEN = 16

#: The password alphabet, exactly as specified: A-Z, a-z, and 1-9.
#: Zero is excluded (it is not in 1-9); nothing else is. An earlier
#: version also dropped O/I/o/l for readability, which was a nice idea
#: that quietly made half the six-letter word list unrepresentable — a
#: validator and a generator disagreeing about the alphabet is worse
#: than a password containing a letter that looks like a digit.
_PASSWORD_CHARS = string.ascii_uppercase + string.ascii_lowercase + "123456789"
_PASSWORD_MIN, _PASSWORD_MAX = 7, 13

#: The optional six-letter word an admin password may contain. Kept short
#: and unremarkable on purpose: the word is a memorability aid inside a
#: random string, not entropy, and the entropy check below never counts it.
_SIX_LETTER_WORDS = (
    "anchor", "basalt", "cinder", "damper", "ember", "fathom", "girder", "hammer",
    "indigo", "jigsaw", "kelvin", "lumber", "marrow", "nickel", "onyxes",
    "piston", "quartz", "rivets", "shovel", "timber", "umbral", "vessel",
    "welder", "yonder", "zenith",
)

#: **The same special-character set as T1**, ``-`` included.
#:
#: Excluding ``-`` looked safer — it is the suffix separator — and was
#: wrong. The suffix is anchored at the end of the string and the special
#: block is a fixed five characters at a fixed offset, so a ``-`` inside
#: the block is never ambiguous. What excluding it actually bought was a
#: conversion that was not a round trip: any T1 key whose specials
#: contained ``-`` came back as a *different* T1 key, which then failed
#: authentication. Sharing the alphabet makes T1 -> T2 -> T1 exact, which
#: is the property the whole compatibility story rests on.
from .keymaster import _SPECIAL_CHARS as _T1_SPECIAL_CHARS  # noqa: E402

_T2_SPECIAL_CHARS = _T1_SPECIAL_CHARS
_T2_SPECIAL_RE = re.escape(_T2_SPECIAL_CHARS)

#: Host IDs must not contain ``#`` or ``-`` in their suffix, because those
#: characters appear in SSPKIDs and V1 Server IDs respectively. This is
#: the full T1/T2 special set minus those two.
_HOST_ID_SPECIAL_CHARS = _T2_SPECIAL_CHARS.replace("#", "").replace("-", "")

# T2_<pw?>_<body><ll><sp5><slash><digit>-<level>
#
# The prefix is three components exactly as specified: the family tag
# ("T2"/"T2S"/"T2C"), the optional admin password, and the separator that
# makes the boundary unambiguous. Without the separator a body starting
# with alphanumerics is indistinguishable from a password, and a parser
# that has to guess is a parser an attacker gets to steer.
_T2_PATTERN: re.Pattern[str] = re.compile(
    r"^(?P<family>T2S|T2P|T2C|T2)_"
    r"(?:(?P<password>[A-Za-z1-9]{7,13})_)?"
    r"(?P<body>[A-Za-z0-9]{16,})"
    r"(?P<ll>[a-z]{2})"
    r"(?P<sp>[" + _T2_SPECIAL_RE + r"]{5})"
    r"(?P<slash>[/\\])"
    r"(?P<digit>[1-9])"
    r"-(?P<level>[1-9])$"
)

_HOST_ID_SPECIAL_RE = re.escape(_HOST_ID_SPECIAL_CHARS)

_HOST_ID_PATTERN = re.compile(
    r"^(?P<body>[A-Za-z0-9]{53})(?P<special>[" + _HOST_ID_SPECIAL_RE + r"])$"
)


# ---------------------------------------------------------------------------
# Admin passwords
# ---------------------------------------------------------------------------


def generate_admin_password(
    length: int | None = None, *, include_word: bool = False
) -> str:
    """A 7-13 character admin password from ``[A-Za-z1-9]``.

    Every character comes from :func:`secrets.choice`, so there is no
    sequence to predict. ``include_word`` embeds one six-letter word at a
    random offset — the spec allows it and people ask for it, but the word
    is a memorability aid, not entropy, and
    :func:`validate_admin_password` never credits it as such.
    """
    if length is None:
        length = secrets.choice(range(_PASSWORD_MIN, _PASSWORD_MAX + 1))
    if not _PASSWORD_MIN <= length <= _PASSWORD_MAX:
        raise ValueError(
            f"Admin password length must be {_PASSWORD_MIN}-{_PASSWORD_MAX}, got {length}"
        )
    if include_word and length < 7:
        raise ValueError(
            f"A password containing a six-letter word needs at least 7 characters; got {length}"
        )

    # Generate and check, rather than generate and hope. Uniform random
    # choice over a 61-character alphabet produces a three-character run
    # ("abc", "789") roughly 0.6% of the time at these lengths — so a
    # generator that did not re-roll would occasionally hand back a
    # password its own validator rejects, and the caller would see a
    # confusing failure from T2KeyGenerator.generate() one time in two
    # hundred. Rejection sampling is the standard fix and the loop is
    # bounded so a future validator change cannot spin here forever.
    for _ in range(200):
        if include_word:
            word = secrets.choice(_SIX_LETTER_WORDS)
            filler_len = length - len(word)
            offset = secrets.randbelow(filler_len + 1)
            filler = "".join(secrets.choice(_PASSWORD_CHARS) for _ in range(filler_len))
            candidate = filler[:offset] + word + filler[offset:]
        else:
            candidate = "".join(secrets.choice(_PASSWORD_CHARS) for _ in range(length))
        if validate_admin_password(candidate)[0]:
            return candidate
    raise RuntimeError(
        "Could not generate an admin password satisfying the strength rules in 200 attempts. "
        "This means the generator and validate_admin_password() disagree — a bug, not bad luck."
    )


def validate_admin_password(password: str) -> tuple[bool, str]:
    """``(ok, reason)``. Rejects the predictable shapes explicitly.

    Length and alphabet are the spec. The three extra checks — a run of
    consecutive characters, a repeated character, and a single-class
    password — are there because "securely generated and must not use
    predictable sequences" is a property of the *value*, and a validator
    that only measures length lets ``abc1234`` through.
    """
    if not password:
        return False, "Admin password is required for an admin T2 key"
    if not _PASSWORD_MIN <= len(password) <= _PASSWORD_MAX:
        return False, (
            f"Admin password must be {_PASSWORD_MIN}-{_PASSWORD_MAX} characters, "
            f"got {len(password)}"
        )
    bad = sorted({c for c in password if c not in _PASSWORD_CHARS})
    if bad:
        return False, (
            f"Admin password may only contain A-Z, a-z and 1-9; found {''.join(bad)!r}"
        )

    stripped = _strip_known_word(password)
    if _has_run(stripped, 3):
        return False, "Admin password contains a run of 3+ consecutive characters (e.g. abc, 123)"
    if _max_repeat(stripped) >= 3:
        return False, "Admin password repeats one character 3+ times in a row"
    classes = sum(
        (
            any(c.isupper() for c in password),
            any(c.islower() for c in password),
            any(c.isdigit() for c in password),
        )
    )
    if classes < 2:
        return False, "Admin password must mix at least two of: uppercase, lowercase, digits"
    return True, ""


def _strip_known_word(password: str) -> str:
    """Remove one embedded six-letter word before the sequence checks.

    The word is permitted by the spec, and its own letters would trip the
    run detector for no reason — ``anchor`` has no run, but a word that
    did would be rejected for being itself.
    """
    lowered = password.lower()
    for word in _SIX_LETTER_WORDS:
        index = lowered.find(word)
        if index >= 0:
            return password[:index] + password[index + len(word):]
    return password


def _has_run(text: str, length: int) -> bool:
    """True if *text* contains *length* consecutive codepoints (abc / 321)."""
    if len(text) < length:
        return False
    for i in range(len(text) - length + 1):
        window = text[i : i + length]
        deltas = {ord(window[j + 1]) - ord(window[j]) for j in range(length - 1)}
        if deltas in ({1}, {-1}):
            return True
    return False


def _max_repeat(text: str) -> int:
    best = run = 1
    for i in range(1, len(text)):
        run = run + 1 if text[i] == text[i - 1] else 1
        best = max(best, run)
    return best


# ---------------------------------------------------------------------------
# SSPKID — Specific Server Key ID
# ---------------------------------------------------------------------------

#: Value -> character, largest first. Greedy decomposition over this table
#: is the whole codec (see :func:`encode_sspkid_index`).
_SSPKID_SYMBOLS: tuple[tuple[int, str], ...] = (
    (100, "$"),
    (75, "€"),   # €
    (40, "^"),
    (25, "*"),
    (15, "•"),   # •
    (10, "?"),
    (5, "!"),
)
_SSPKID_VALUES: dict[str, int] = {char: value for value, char in _SSPKID_SYMBOLS}


class SSPKIDCollision(ValueError):
    """An SSPKID is already assigned to a different key.

    A distinct type because the caller's response is distinct: not "your
    input was malformed" but "pick another index, this one is taken".
    """


def encode_sspkid_index(index: int) -> str:
    """Encode a server-key index as its canonical identifier string.

    1-4 are plain digits. From 5 upward the index is decomposed greedily
    over :data:`_SSPKID_SYMBOLS`, largest symbol first, with any remainder
    below 5 written as a trailing digit::

        1   -> "1"
        4   -> "4"
        5   -> "!"
        7   -> "!2"
        10  -> "?"
        20  -> "•!"
        100 -> "$"
        123 -> "$?•3"

    Three properties this must have, and does:

    * **Deterministic.** Greedy over a fixed table with a fixed order.
    * **Injective.** Decoding is a plain sum, and greedy encoding is the
      unique canonical form, so ``decode(encode(n)) == n`` for every
      ``n >= 1``. :func:`decode_sspkid_index` is the inverse, and the
      round trip is tested exhaustively over a wide range.
    * **Total.** Every positive index has a representation; there is no
      ceiling at which the scheme runs out.
    """
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(f"SSPKID index must be an int, got {type(index).__name__}")
    if index < 1:
        raise ValueError(f"SSPKID index must be >= 1, got {index}")

    remaining = index
    out: list[str] = []
    for value, char in _SSPKID_SYMBOLS:
        count, remaining = divmod(remaining, value)
        out.append(char * count)
    if remaining:
        out.append(str(remaining))
    return "".join(out)


def decode_sspkid_index(encoded: str) -> int:
    """Inverse of :func:`encode_sspkid_index`. Raises on anything else.

    Strict rather than forgiving: this parses an identifier that decides
    which key a request is for, and a lenient parser that quietly reads
    ``!!`` as 10 would make two identifiers resolve to one key.
    """
    if not encoded:
        raise ValueError("Empty SSPKID index")
    total = 0
    seen_digit = False
    for position, char in enumerate(encoded):
        if char.isdigit():
            if seen_digit:
                raise ValueError(f"SSPKID index {encoded!r} has more than one digit")
            if encoded.isdigit() and len(encoded) > 1:
                # "#20" reads like twenty and is the single most likely
                # thing for a human to type. Only 1-4 are written as
                # digits, so say how twenty is actually spelled — this
                # check comes before the positional one because
                # "digit before a symbol" is true but unhelpful here.
                raise ValueError(
                    f"SSPKID index {encoded!r}: only 1-4 are written as digits. "
                    f"{int(encoded)} is written {encode_sspkid_index(int(encoded))!r}"
                )
            if position != len(encoded) - 1:
                raise ValueError(f"SSPKID index {encoded!r} has a digit before a symbol")
            digit = int(char)
            if not 1 <= digit <= 4:
                raise ValueError(
                    f"SSPKID index {encoded!r} ends in {digit}, but a trailing digit is 1-4 "
                    "(5 and above are symbols)"
                )
            total += digit
            seen_digit = True
        elif char in _SSPKID_VALUES:
            total += _SSPKID_VALUES[char]
        else:
            raise ValueError(f"SSPKID index {encoded!r} contains {char!r}, which is not a symbol")

    if encode_sspkid_index(total) != encoded:
        # A non-canonical spelling such as "!!" (=10, canonically "?").
        # Accepting it would give one key two identifiers.
        raise ValueError(
            f"SSPKID index {encoded!r} is not canonical; {total} is written "
            f"{encode_sspkid_index(total)!r}"
        )
    return total


_V1_SERVER_ID = re.compile(r"^(?P<seq>\d{1,5})-(?P<letter>[A-Z])(?P<gen>\d+)$")


@dataclass(frozen=True)
class SSPKID:
    """One key's identity within one server.

    A **V1 Server ID** identifies the server (``00042-C1``). An **SSPKID**
    identifies one key on it (``00042-C1#3``). A server may hold many
    keys; a given SSPKID belongs to exactly one. That asymmetry is the
    reason the two are separate types rather than one string with a
    convention, and it is enforced by :class:`ServerKeyRegistry`.
    """

    server_id: str
    index: int

    def __post_init__(self) -> None:
        if not _V1_SERVER_ID.fullmatch(self.server_id):
            raise ValueError(
                f"{self.server_id!r} is not a V1 Server ID (expected NNNNN-Xg, e.g. 00042-C1)"
            )
        if self.index < 1:
            raise ValueError(f"SSPKID index must be >= 1, got {self.index}")

    @property
    def encoded_index(self) -> str:
        return encode_sspkid_index(self.index)

    def __str__(self) -> str:
        return f"{self.server_id}#{self.encoded_index}"

    @classmethod
    def parse(cls, text: str) -> SSPKID:
        """Parse ``00042-C1#3``. A bare V1 Server ID is refused.

        Refused rather than defaulted to ``#1``: the caller either means a
        server or means a key, and guessing turns "which key?" into a
        silent choice.
        """
        if "#" not in text:
            raise ValueError(
                f"{text!r} is a V1 Server ID, not an SSPKID. An SSPKID names one key: "
                f"{text}#1"
            )
        server_id, _, encoded = text.partition("#")
        return cls(server_id=server_id, index=decode_sspkid_index(encoded))

    @staticmethod
    def is_sspkid(text: str) -> bool:
        """True for an SSPKID, False for a bare V1 Server ID or anything else."""
        try:
            SSPKID.parse(text)
        except ValueError:
            return False
        return True


class ServerKeyRegistry:
    """Which SSPKID belongs to which key, and the guarantee that it is one.

    The rule it enforces — many keys per V1 Server ID, one key per
    SSPKID — is what makes an SSPKID an identifier rather than a label.

    **This used to be memory-only.** That was the same defect that made
    every key `gkey` minted carry server ID ``00001-A1``: each `gkey`
    invocation is its own process, and the server is another, so an
    assignment made in one was invisible to the other and gone entirely
    on the next restart. An identifier that does not survive the process
    that issued it cannot identify anything, and worse, it comes back
    around — a fresh registry hands out ``#1`` again to a different key
    while an audit trail still names the first one.

    So it persists. ``store_dir`` defaults to the same directory the
    Keymaster uses (``T1_KEYMASTER_DIR``, else ``~/.hypernix/keymaster``)
    for exactly the reason that variable exists: two halves of one tool
    disagreeing about where identity lives is a failure where both halves
    appear to work.

    Pass ``store_dir=None`` explicitly for an ephemeral registry — the
    rule can still be exercised without a filesystem, which is what the
    unit tests want.
    """

    #: Where the registry lives inside the store directory.
    #:
    #: A subdirectory, not a bare file, because the Keymaster globs
    #: ``*.json`` in its store to load keys — a registry sitting there
    #: gets read as a malformed key on every start, and the CI teardown
    #: that counts ``*.json`` to prove no keys leaked would count it as a
    #: leaked key. Neither is a failure this file should be able to cause.
    SUBDIR = "sspkid"
    FILENAME = "registry.json"

    def __init__(self, store_dir: Path | str | None = _UNSET) -> None:
        self._by_sspkid: dict[str, str] = {}          # str(SSPKID) -> key_id
        self._by_key: dict[str, SSPKID] = {}          # key_id -> SSPKID
        self._by_server: dict[str, set[str]] = {}     # server_id -> {key_id}
        self._lock = threading.RLock()

        if store_dir is _UNSET:
            store_dir = default_sspkid_store_dir()
        self._path: Path | None = (
            Path(store_dir) / self.SUBDIR / self.FILENAME
            if store_dir is not None
            else None
        )
        self._load()

    # -- persistence ---------------------------------------------------

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt registry must not stop a server from starting.
            # Losing the assignments is bad; refusing to boot is worse,
            # and the collision check still protects what gets written
            # from here on.
            logger.warning("t2keys: could not read %s; starting empty", self._path)
            return
        for text, key_id in (raw.get("assignments") or {}).items():
            try:
                sspkid = SSPKID.parse(text)
            except ValueError:
                continue
            self._by_sspkid[text] = key_id
            self._by_key[key_id] = sspkid
            self._by_server.setdefault(sspkid.server_id, set()).add(key_id)

    def _save(self) -> None:
        if self._path is None:
            return
        payload = {"version": 1, "assignments": dict(self._by_sspkid)}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Written beside the target and moved, so a crash mid-write
            # leaves the previous registry rather than a truncated one.
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
            os.chmod(self._path, 0o600)
        except OSError:
            logger.warning("t2keys: could not write %s", self._path)

    @property
    def path(self) -> Path | None:
        """Where assignments are kept, or None for an ephemeral registry."""
        return self._path

    def assign(self, key_id: str, sspkid: SSPKID) -> SSPKID:
        """Bind *key_id* to *sspkid*, or raise :class:`SSPKIDCollision`."""
        text = str(sspkid)
        existing = self._by_sspkid.get(text)
        if existing is not None and existing != key_id:
            raise SSPKIDCollision(
                f"{text} is already assigned to key {existing}; an SSPKID identifies "
                "exactly one key"
            )
        previous = self._by_key.get(key_id)
        if previous is not None and str(previous) != text:
            # Re-homing a key is allowed, but the old identifier must be
            # released or it would resolve to a key that no longer has it.
            self._by_sspkid.pop(str(previous), None)
            self._by_server.get(previous.server_id, set()).discard(key_id)
        self._by_sspkid[text] = key_id
        self._by_key[key_id] = sspkid
        self._by_server.setdefault(sspkid.server_id, set()).add(key_id)
        self._save()
        return sspkid

    def allocate(self, key_id: str, server_id: str) -> SSPKID:
        """Assign the lowest free index on *server_id*.

        Lowest free rather than next-highest so that a server which has
        had keys revoked does not climb into symbol territory for no
        reason — an operator reading ``00042-C1#2`` should not have to
        wonder what happened to the first one.
        """
        taken = {
            self._by_key[k].index for k in self._by_server.get(server_id, set()) if k in self._by_key
        }
        index = 1
        while index in taken:
            index += 1
        return self.assign(key_id, SSPKID(server_id=server_id, index=index))

    def resolve(self, sspkid: SSPKID | str) -> str | None:
        """SSPKID -> key_id, or ``None``."""
        text = str(sspkid)
        return self._by_sspkid.get(text)

    def sspkid_for(self, key_id: str) -> SSPKID | None:
        return self._by_key.get(key_id)

    def keys_on(self, server_id: str) -> list[str]:
        """Every key_id on one V1 Server ID — the many side of the relation."""
        return sorted(self._by_server.get(server_id, set()))

    def release(self, key_id: str) -> bool:
        """Forget a key. Its SSPKID becomes free for reassignment."""
        sspkid = self._by_key.pop(key_id, None)
        if sspkid is None:
            return False
        self._by_sspkid.pop(str(sspkid), None)
        self._by_server.get(sspkid.server_id, set()).discard(key_id)
        self._save()
        return True

    def __len__(self) -> int:
        return len(self._by_key)


# ---------------------------------------------------------------------------
# Host IDs
# ---------------------------------------------------------------------------


def generate_host_id() -> str:
    """A 54-character Host ID: 53 alphanumerics plus a special character.

    Distinct from both a V1 Server ID and an SSPKID by construction — it
    has no dash and no ``#`` — so ``waiter -F`` can tell what it has been
    given without being told.
    """
    body = "".join(secrets.choice(_BODY_CHARS) for _ in range(HOST_ID_LENGTH - 1))
    return body + secrets.choice(_HOST_ID_SPECIAL_CHARS)


def validate_host_id(host_id: str) -> bool:
    return bool(_HOST_ID_PATTERN.fullmatch(host_id))


# ---------------------------------------------------------------------------
# T2 keys
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class T2Key:
    """A parsed T2 key. ``raw`` is the only field that is a secret."""

    raw: str = field(repr=False)
    family: T2Type
    body: str = field(repr=False)
    password: str = field(default="", repr=False)
    access_level: int = 1
    lowercase_pair: str = ""
    special_chars: str = field(default="", repr=False)
    slash: str = ""
    digit: str = ""
    sspkid: SSPKID | None = None

    @property
    def is_admin(self) -> bool:
        """Admin is carried by the password component, not by the level.

        A level-9 key without a password is a very privileged *user* key.
        Conflating "highest tier" with "administrator" is how a tier bump
        silently grants key management.
        """
        return bool(self.password)

    @property
    def is_hyperlink_key(self) -> bool:
        return self.family is T2Type.T2S

    @property
    def key_id(self) -> str:
        """A stable, non-secret identifier: SHA-256 of the raw key, truncated.

        Derived rather than stored so two processes agree on it without
        coordinating, and truncated to 24 hex characters because it goes
        in logs and audit records where the full digest is noise.
        """
        return "t2_" + hashlib.sha256(self.raw.encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        """Everything except the secret parts. Safe to log or return."""
        return {
            "key_id": self.key_id,
            "family": self.family.value,
            "access_level": self.access_level,
            "is_admin": self.is_admin,
            "is_hyperlink_key": self.is_hyperlink_key,
            "body_length": len(self.body),
            "sspkid": str(self.sspkid) if self.sspkid else "",
            "server_id": self.sspkid.server_id if self.sspkid else "",
        }

    def permits(self, scope: str, *, in_hyperlink: bool = False) -> bool:
        """Can this key do *scope*?

        The T2S rule lives here rather than at each call site: outside
        HyperLink a T2S key gets read and non-admin write, and nothing
        else, however high its access level. That is the trade that makes
        a 26-character typeable key acceptable — it is short *because* it
        is limited.
        """
        scope = scope.lower()
        if self.family is T2Type.T2S and not in_hyperlink:
            return scope in ("read", "write")
        if scope == "admin":
            return self.is_admin
        return True


class T2KeyGenerator:
    """Mint, validate and convert T2 keys."""

    # -- generation ---------------------------------------------------

    @staticmethod
    def generate(
        *,
        family: T2Type | str = T2Type.T2,
        access_level: int = 1,
        admin: bool = False,
        password: str | None = None,
        body_length: int = 24,
        include_word: bool = False,
        sspkid: SSPKID | None = None,
    ) -> T2Key:
        """Mint a T2 key.

        ``admin=True`` adds the password component to the prefix,
        generating one when it is not supplied. A supplied password is
        validated, not trusted: an operator pasting a memorable string is
        exactly the case the strength checks exist for.
        """
        family = T2Type(family)
        if family is T2Type.T2C:
            raise NotImplementedError(T2C_UNAVAILABLE_REASON)
        if access_level not in ACCESS_LEVELS:
            raise ValueError(
                f"Access level must be 1-9, got {access_level!r}"
            )

        if family is T2Type.T2S:
            if body_length not in (24, T2S_BODY_LENGTH):
                raise ValueError(
                    f"A T2S key's body is fixed at {T2S_BODY_LENGTH} characters; "
                    f"got body_length={body_length}"
                )
            body_length = T2S_BODY_LENGTH
        elif body_length < _MIN_BODY_LEN:
            raise ValueError(f"body_length must be >= {_MIN_BODY_LEN}, got {body_length}")

        if admin:
            if family is T2Type.T2S:
                raise ValueError(
                    "A T2S key cannot be an admin key. It is short enough to type, which "
                    "is precisely why it must not carry administrative authority."
                )
            if family is T2Type.T2P:
                raise ValueError(
                    "A T2P key cannot be an admin key. It authorises spending; "
                    "administering the server is a separate authority and there is no "
                    "reason for the two to travel on one credential."
                )
            password = password or generate_admin_password(include_word=include_word)
            ok, reason = validate_admin_password(password)
            if not ok:
                raise ValueError(reason)
        elif password:
            raise ValueError(
                "A password component marks a key as admin; pass admin=True as well, or "
                "drop the password."
            )
        else:
            password = ""

        body = "".join(secrets.choice(_BODY_CHARS) for _ in range(body_length))
        ll = "".join(secrets.choice(string.ascii_lowercase) for _ in range(2))
        sp = "".join(secrets.choice(_T2_SPECIAL_CHARS) for _ in range(5))
        slash = secrets.choice("/\\")
        digit = str(secrets.randbelow(9) + 1)

        prefix = f"{family.value}_" + (f"{password}_" if password else "")
        raw = f"{prefix}{body}{ll}{sp}{slash}{digit}-{access_level}"
        return T2Key(
            raw=raw,
            family=family,
            body=body,
            password=password,
            access_level=access_level,
            lowercase_pair=ll,
            special_chars=sp,
            slash=slash,
            digit=digit,
            sspkid=sspkid,
        )

    # -- validation ---------------------------------------------------

    @staticmethod
    def validate(key: str) -> bool:
        try:
            T2KeyGenerator.parse(key)
        except ValueError:
            return False
        return True

    @staticmethod
    def parse(key: str, *, sspkid: SSPKID | None = None) -> T2Key:
        """Parse a T2 key, or raise ``ValueError`` saying what is wrong.

        Every failure names the actual problem. "Invalid key" with no
        detail turns a typo into a support ticket, and these keys are
        typed by hand.
        """
        if not isinstance(key, str) or not key:
            raise ValueError("T2 key must be a non-empty string")
        if key.startswith("T1_"):
            raise ValueError(
                "That is a T1 key. T1 keys stay valid — use the T1 path, or convert "
                "with T2KeyGenerator.from_t1()."
            )
        match = _T2_PATTERN.fullmatch(key)
        if not match:
            raise ValueError(
                "Not a valid T2 key. Expected "
                "T2[S]_[<password>_]<body><2 lowercase><5 specials><slash><digit>-<level 1-9>"
            )

        family = T2Type(match.group("family"))
        if family is T2Type.T2C:
            raise ValueError(T2C_UNAVAILABLE_REASON)

        password = match.group("password") or ""
        body = match.group("body")
        level = int(match.group("level"))

        if family is T2Type.T2S:
            if len(body) != T2S_BODY_LENGTH:
                raise ValueError(
                    f"A T2S key's body is exactly {T2S_BODY_LENGTH} characters; "
                    f"this one has {len(body)}"
                )
            if password:
                raise ValueError("A T2S key cannot carry an admin password")
        if password:
            ok, reason = validate_admin_password(password)
            if not ok:
                raise ValueError(f"T2 admin password rejected: {reason}")

        return T2Key(
            raw=key,
            family=family,
            body=body,
            password=password,
            access_level=level,
            lowercase_pair=match.group("ll"),
            special_chars=match.group("sp"),
            slash=match.group("slash"),
            digit=match.group("digit"),
            sspkid=sspkid,
        )

    # -- conversion ---------------------------------------------------

    @staticmethod
    def to_t1(key: str | T2Key) -> str:
        """Convert a T2 key into a valid T1 key.

        Lossy in exactly one direction, and only where T1 has no field:

        * The **body** is carried across unchanged — it is the entropy,
          and preserving it is what makes the conversion meaningful
          rather than a re-roll.
        * The **T1 suffix** (two lowercase, five specials, slash, digit)
          is carried across unchanged. T2 shares T1's special-character
          alphabet exactly, so this is a copy rather than a remap and
          ``to_t1(from_t1(k)) == k`` holds for every T1 key — which is
          what lets an existing key be presented as T2 and still
          authenticate against the store that already holds it.
        * The **access level** and the **admin password** are dropped.
          T1 has nowhere to put them, and inventing a place would produce
          a "T1" key that only a T2-aware server could read.
        * The **SSPKID** is dropped for the same reason. A T1 key is
          addressed by its V1 Server ID, which is unchanged.

        Callers that need the dropped fields should keep the T2 key and
        use :meth:`parse`; the conversion is for talking to T1-only
        servers, not for storage.
        """
        parsed = key if isinstance(key, T2Key) else T2KeyGenerator.parse(key)
        body = parsed.body
        if len(body) < _MIN_BODY_LEN:
            # T2S bodies are 26, T2 bodies are >= 16, so this cannot fire
            # today. Kept because a future shorter family would otherwise
            # produce an invalid T1 key rather than an error.
            raise ValueError(
                f"Cannot convert: T1 requires a body of at least {_MIN_BODY_LEN} characters, "
                f"this key has {len(body)}"
            )
        # The alphabets are identical, so this is an exact carry-over
        # rather than a remap: T1 -> T2 -> T1 returns the original key.
        return (
            f"T1_{body}{parsed.lowercase_pair}{parsed.special_chars}"
            f"{parsed.slash}{parsed.digit}"
        )

    @staticmethod
    def from_t1(
        t1_key: str,
        *,
        access_level: int = 1,
        family: T2Type | str = T2Type.T2,
        sspkid: SSPKID | None = None,
    ) -> T2Key:
        """Wrap an existing T1 key as a T2 key at a stated access level.

        The reverse of :meth:`to_t1`, and necessarily not an inverse: the
        access level cannot be recovered from a T1 key because it was
        never in one, so the caller states it. Never produces an admin
        key — promoting a key during a format conversion would be a
        privilege escalation with no audit trail.
        """
        from .keymaster import T1KeyGenerator

        family = T2Type(family)
        if not T1KeyGenerator.validate(t1_key):
            raise ValueError("Not a valid T1 key")
        if access_level not in ACCESS_LEVELS:
            raise ValueError(f"Access level must be 1-9, got {access_level!r}")
        parts = T1KeyGenerator.deconstruct(t1_key)
        body = parts["body"]
        if family is T2Type.T2S and len(body) != T2S_BODY_LENGTH:
            raise ValueError(
                f"Cannot present this T1 key as T2S: a T2S body is exactly "
                f"{T2S_BODY_LENGTH} characters and this key's is {len(body)}"
            )
        raw = (
            f"{family.value}_{body}{parts['lowercase_pair']}{parts['special_chars']}"
            f"{parts['slash']}{parts['digit']}-{access_level}"
        )
        return T2KeyGenerator.parse(raw, sspkid=sspkid)

    @staticmethod
    def from_t1_admin(
        t1_key: str,
        *,
        password: str | None = None,
        access_level: int = 9,
        sspkid: SSPKID | None = None,
    ) -> T2Key:
        """Present an *already administrative* T1 key as an admin T2 key.

        :meth:`from_t1` never produces an admin key, because converting an
        arbitrary T1 key must not grant administrative authority. This is
        the deliberate exception, and it is a separate name rather than a
        flag so that every place it happens can be found by grepping for
        one word.

        It does not itself grant anything: T2's ``is_admin`` is the
        presence of the password component, while the authority a request
        actually gets comes from the key store — ``KeyType.ADMIN`` plus
        the ADMIN scope on the record behind this key (see
        ``AuthContext.is_admin``). Handing this an ordinary user key
        produces a key that *looks* administrative in its prefix and is
        refused by every admin endpoint, which is a worse outcome than
        refusing here. **Callers must verify the key's store record says
        administrator before calling.** ``gkey create`` does: it minted
        the key moments earlier and checks the returned metadata.

        Always T2 — a T2S key cannot be an administrator at all.
        """
        wrapped = T2KeyGenerator.from_t1(
            t1_key, access_level=access_level, family=T2Type.T2, sspkid=sspkid
        )
        # Always T2 — the signature pins the family, but saying so here
        # means a later edit that parameterises it cannot quietly produce
        # an admin T2S or T2P.
        assert wrapped.family is T2Type.T2
        password = password or generate_admin_password()
        ok, reason = validate_admin_password(password)
        if not ok:
            raise ValueError(reason)
        raw = (
            f"{T2Type.T2.value}_{password}_{wrapped.body}{wrapped.lowercase_pair}"
            f"{wrapped.special_chars}{wrapped.slash}{wrapped.digit}-{access_level}"
        )
        return T2KeyGenerator.parse(raw, sspkid=sspkid)
