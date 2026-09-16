"""t1api.version — the T1 API's own six-part version scheme.

The T1 API stopped tracking the ``hypernix`` package version at
**T1 v1.0.26.8.0.1**. The two ship together but they answer different
questions: the package version says "which pip release is this", the T1
version says "which API contract is this, and how old is it". A client
pinned to a contract needs the second one, and deriving it from
``0.71.5rc2`` was never possible.

The scheme
----------
Six components, most significant first::

    1   .   0    .   2026  .  8    .   0     .  1
    │       │        │        │        │        │
    │       │        │        │        │        └── bug fix + assorted minor features
    │       │        │        │        └─────────── new feature
    │       │        │        └──────────────────── month of the release
    │       │        └───────────────────────────── year of the release
    │       └────────────────────────────────────── major update
    └────────────────────────────────────────────── T1 API generation

Two spellings of the same version, and this module is the only place
that knows how to move between them:

* **long** — ``1.0.2026.8.0.1``, the four-digit year. What a changelog,
  a release note, or a human reading a date wants.
* **short** — ``1.0.26.8.0.1``, the two-digit year. What
  ``__t1api_version__``, ``GET /status``, ``waiter --version`` and every
  wire response carry, because it is the form people type.

Both parse. :meth:`T1Version.parse` accepts either, with or without a
``v``/``t1 v``/``T1 v`` prefix, so a config file written by hand does not
have to guess::

    >>> T1Version.parse("t1 v1.0.2026.8.0.1").short
    '1.0.26.8.0.1'
    >>> T1Version.parse("1.0.26.8.0.1").long
    '1.0.2026.8.0.1'

Why the year is ambiguous, and why that is fine
-----------------------------------------------
``26`` could be the year 26 or the year 2026. The parser resolves a
two-digit year into the 2000s (``26`` → ``2026``) and leaves a
four-digit year alone, which is well-defined for every release this
scheme will ever carry — the T1 API did not exist in the year 26, and a
release in 2126 will spell its year out. What the parser must *not* do
is silently accept a three-digit year, so it doesn't: ``1.0.202.8.0.1``
is a typo for ``2026``, and a typo that parses is worse than one that
raises.

Ordering
--------
:class:`T1Version` is a totally ordered value type, comparing on the
normalised (four-digit-year) tuple, so ``1.0.26.8.0.1 <
1.0.26.9.0.0`` and the short/long spellings of one version compare
equal. That is what lets a server say "this client is too old" with a
comparison instead of a string match.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "T1Version",
    "T1_VERSION",
    "T1_VERSION_SHORT",
    "T1_VERSION_LONG",
    "MIN_CLIENT_VERSION",
    "T2_AWARE_VERSION",
    "INFERENCE_VERSION",
    "parse_version",
]

# ``t1 v1.0.2026.8.0.1`` / ``T1 v1.0.26.8.0.1`` / ``v1.0.26.8.0.1`` / bare.
_PREFIX = re.compile(r"^\s*(?:t1\s*)?v?\s*", re.IGNORECASE)


@dataclass(frozen=True, order=False)
class T1Version:
    """One T1 API version, in its six-part form.

    ``year`` is always stored normalised (four digits). The two-digit
    spelling is a rendering choice, not a different value — see
    :attr:`short` and :attr:`long`.
    """

    api: int
    major: int
    year: int
    month: int
    feature: int
    fix: int

    # -- construction -------------------------------------------------

    @classmethod
    def parse(cls, text: str) -> T1Version:
        """Parse either spelling, with or without a ``t1 v`` prefix.

        Raises ``ValueError`` with the offending text included — this is
        called on config values and on client-supplied headers, and
        "invalid version" with no sample is not a debuggable error.
        """
        if not isinstance(text, str):
            raise ValueError(f"T1 version must be a string, got {type(text).__name__}")
        cleaned = _PREFIX.sub("", text).strip()
        parts = cleaned.split(".")
        if len(parts) != 6:
            raise ValueError(
                f"T1 version must have 6 components "
                f"(api.major.year.month.feature.fix), got {len(parts)} in {text!r}"
            )
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            raise ValueError(f"T1 version components must all be integers: {text!r}") from None
        if any(n < 0 for n in nums):
            raise ValueError(f"T1 version components must not be negative: {text!r}")

        api, major, year, month, feature, fix = nums
        year = _normalise_year(year, text)
        if not 1 <= month <= 12:
            raise ValueError(f"T1 version month must be 1-12, got {month} in {text!r}")
        return cls(api=api, major=major, year=year, month=month, feature=feature, fix=fix)

    # -- rendering ----------------------------------------------------

    @property
    def short(self) -> str:
        """``1.0.26.8.0.1`` — two-digit year. The canonical wire form."""
        return f"{self.api}.{self.major}.{self.year % 100:02d}.{self.month}.{self.feature}.{self.fix}"

    @property
    def long(self) -> str:
        """``1.0.2026.8.0.1`` — four-digit year. The changelog form."""
        return f"{self.api}.{self.major}.{self.year}.{self.month}.{self.feature}.{self.fix}"

    @property
    def display(self) -> str:
        """``t1 v1.0.26.8.0.1`` — what a CLI banner prints."""
        return f"t1 v{self.short}"

    @property
    def generation(self) -> str:
        """``1.0`` — the part a client pins against for compatibility."""
        return f"{self.api}.{self.major}"

    @property
    def release(self) -> str:
        """``2026-08`` — the release month, for humans and changelogs."""
        return f"{self.year:04d}-{self.month:02d}"

    def __str__(self) -> str:
        return self.short

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"T1Version({self.long!r})"

    # -- comparison ---------------------------------------------------

    @property
    def key(self) -> tuple[int, int, int, int, int, int]:
        """The normalised ordering tuple. Long and short spellings share one."""
        return (self.api, self.major, self.year, self.month, self.feature, self.fix)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, T1Version):
            return self.key == other.key
        if isinstance(other, str):
            try:
                return self.key == T1Version.parse(other).key
            except ValueError:
                return NotImplemented
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.key)

    def __lt__(self, other: T1Version | str) -> bool:
        return self.key < _coerce(other).key

    def __le__(self, other: T1Version | str) -> bool:
        return self.key <= _coerce(other).key

    def __gt__(self, other: T1Version | str) -> bool:
        return self.key > _coerce(other).key

    def __ge__(self, other: T1Version | str) -> bool:
        return self.key >= _coerce(other).key

    # -- compatibility ------------------------------------------------

    def compatible_with(self, other: T1Version | str) -> bool:
        """True when both sides are the same ``api.major`` generation.

        Within a generation the API only adds; across one it may remove.
        That is the whole promise, and it is why ``generation`` exists as
        a separate property rather than being re-derived by callers.
        """
        return self.generation == _coerce(other).generation

    def bump(
        self,
        *,
        api: int | None = None,
        major: int | None = None,
        year: int | None = None,
        month: int | None = None,
        feature: int | None = None,
        fix: int | None = None,
    ) -> T1Version:
        """Return a new version with the named components replaced.

        Bumping a component does **not** zero the ones below it — the
        year and month components are dates, not counters, so a fix
        release in the same month keeps its feature number, and a
        feature release in a new month is spelled out in full by the
        caller. Making this implicit is how a release script ends up
        publishing ``1.0.26.9.0.0`` when it meant ``1.0.26.9.1.0``.
        """
        return T1Version(
            api=self.api if api is None else api,
            major=self.major if major is None else major,
            year=_normalise_year(self.year if year is None else year, "bump()"),
            month=self.month if month is None else month,
            feature=self.feature if feature is None else feature,
            fix=self.fix if fix is None else fix,
        )

    def to_dict(self) -> dict[str, Any]:
        """The shape ``GET /status`` and the SDK exchange."""
        return {
            "short": self.short,
            "long": self.long,
            "display": self.display,
            "generation": self.generation,
            "release": self.release,
            "api": self.api,
            "major": self.major,
            "year": self.year,
            "month": self.month,
            "feature": self.feature,
            "fix": self.fix,
        }


def _normalise_year(year: int, source: str) -> int:
    """Two-digit years mean the 2000s; four-digit years are taken as-is."""
    if 0 <= year <= 99:
        return 2000 + year
    if 1000 <= year <= 9999:
        return year
    raise ValueError(
        f"T1 version year must be 2 digits (26) or 4 digits (2026), got {year} in {source!r}"
    )


def _coerce(value: T1Version | str) -> T1Version:
    return value if isinstance(value, T1Version) else T1Version.parse(value)


def parse_version(text: str) -> T1Version:
    """Module-level alias for :meth:`T1Version.parse`."""
    return T1Version.parse(text)


#: The version this build of the T1 API implements.
#:
#: 1.0.2026.9.2.1 moves the month to September and takes feature 2: the
#: governed inference surface (``/inference/*``), which puts generation
#: behind the registry, the plan cascade, the quota and the cost ledger
#: that ``/bridge/lmstudio/*`` passes straight through. Also in this
#: release: the SSPKID stops being carried in memory alongside the key,
#: and ``gkey create -Con`` takes identity from a configuration source.
T1_VERSION = T1Version(api=1, major=0, year=2026, month=9, feature=2, fix=3)
T1_VERSION_SHORT = T1_VERSION.short   # "1.0.26.9.2.3"
T1_VERSION_LONG = T1_VERSION.long     # "1.0.2026.9.2.3"

#: Oldest client this server still answers without a compatibility
#: warning. Same generation, first release of it — everything from the
#: 1.0 line is accepted, and a 0.71.x client is told to upgrade rather
#: than being left to fail on a missing field.
MIN_CLIENT_VERSION = T1Version(api=1, major=0, year=2026, month=8, feature=0, fix=0)

#: The release that added T2 key recognition, /t1/auth/undo, /t1/auth/redo,
#: /backup/list and /backup/restore. A client that needs any of those can
#: check for it by comparison rather than by probing an endpoint and
#: catching the 404.
T2_AWARE_VERSION = T1Version(api=1, major=0, year=2026, month=8, feature=1, fix=0)

#: The release that added ``/inference/*``. A client that wants the
#: governed path rather than the raw bridge can check for it by
#: comparison instead of probing and catching the 404.
INFERENCE_VERSION = T1Version(api=1, major=0, year=2026, month=9, feature=2, fix=0)
