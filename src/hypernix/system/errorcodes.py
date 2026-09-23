"""hypernix.system.errorcodes — one shape for every error this package raises.

The format
----------
``L#-NNNNN.kS`` — for example ``M4-01015.c3``.

===========  ====================================================
``L``        domain: which part of HyperNix (``M`` models, ``T``
             the T1 API, ``D`` data, ``S`` system, ``I``
             interfaces, ``Q`` quantisation, ``E`` elements,
             ``R`` the runtime/runner, ``X`` training)
``#``        tier, ``0``–``9``: how deep the thing sits. ``1`` is a
             user-facing surface, ``9`` is a kernel.
``NNNNN``    five digits, unique inside the domain
``k``        kind, ``a``–``f`` (see :class:`Kind`)
``S``        severity, ``1``–``5`` (see :class:`Severity`)
===========  ====================================================

Why a format rather than a name
-------------------------------
A message is what somebody reads; a code is what they can search for.
``BackendUnavailable`` is a good name and a bad search term — it appears
in four modules with four meanings, and the one they hit is not the one
the first result describes.

The two halves at the end are the useful part. **Kind** says whose
problem it is, which decides who should be reading: ``a`` is the
caller's, ``d`` is something outside HyperNix, ``f`` is ours. **Severity**
says whether to act now, and is ordered, so a log filter is a comparison
rather than a list of strings to keep in sync.

The registry is not optional
----------------------------
Every code is declared here, with its explanation, before anything can
raise it. A scheme where the raise site invents the code is not a scheme
— it is a string with dots in it, and within a release you have six
spellings of the same failure and no way to find out. :func:`raise_for`
refuses a code that is not registered, and the registry refuses two
codes with the same number in one domain at import time.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any

__all__ = [
    "Domain",
    "Kind",
    "Severity",
    "ErrorCode",
    "HyperNixError",
    "CODES",
    "parse",
    "lookup",
    "register",
    "raise_for",
    "explain",
    "codes_for",
]

_PATTERN = re.compile(r"^([A-Z])([0-9])-([0-9]{5})\.([a-f])([1-5])$")


class Domain(StrEnum):
    """The first letter. Which part of HyperNix owns the failure."""

    MODELS = "M"
    T1API = "T"
    DATA = "D"
    SYSTEM = "S"
    INTERFACES = "I"
    QUANT = "Q"
    ELEMENTS = "E"
    RUNTIME = "R"
    TRAINING = "X"


class Kind(StrEnum):
    """The letter after the dot. Whose problem it is.

    This is the field that decides who should be reading the line. A
    log full of ``d`` is an integration that needs looking at; a log
    full of ``f`` is a bug report.
    """

    #: The caller asked for something that cannot be done.
    USAGE = "a"
    #: Configuration or environment is wrong or missing.
    CONFIG = "b"
    #: Not enough of something — memory, disk, cores, quota.
    RESOURCE = "c"
    #: Something outside HyperNix failed or changed under us.
    EXTERNAL = "d"
    #: Data is present and wrong — corrupt, inconsistent, truncated.
    INTEGRITY = "e"
    #: HyperNix's own fault. Should not happen; if it does, it is a bug.
    INTERNAL = "f"


class Severity(IntEnum):
    """The digit at the end. Ordered, so filters can compare.

    Strings would need every consumer to keep the same list in the same
    order; an integer lets a handler say ``>= Severity.ERROR`` and be
    right for codes added after it was written.
    """

    NOTICE = 1      # worth saying, nothing is wrong
    WARNING = 2     # degraded, still working
    ERROR = 3       # this operation failed
    CRITICAL = 4    # this subsystem is down
    FATAL = 5       # the process cannot continue


@dataclass(frozen=True)
class ErrorCode:
    """One declared code, its explanation and its remedy."""

    domain: Domain
    tier: int
    number: int
    kind: Kind
    severity: Severity
    #: One line. What happened, in the words somebody searching would use.
    explanation: str
    #: What to do about it. Empty when there is nothing useful to say —
    #: which is honest, and better than inventing advice.
    remedy: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.tier <= 9:
            raise ValueError(f"tier {self.tier} is not 0-9")
        if not 0 <= self.number <= 99_999:
            raise ValueError(f"number {self.number} does not fit in five digits")

    @property
    def code(self) -> str:
        return (f"{self.domain.value}{self.tier}-{self.number:05d}"
                f".{self.kind.value}{int(self.severity)}")

    def __str__(self) -> str:
        return self.code

    def message(self, detail: str = "") -> str:
        """``CODE: explanation`` plus whatever the raise site knows.

        The code comes first because that is what gets pasted into a
        search box, and a line that buries it behind a sentence of prose
        gets truncated in exactly the place that mattered.
        """
        parts = [f"{self.code}: {self.explanation}"]
        if detail:
            parts.append(detail)
        if self.remedy:
            parts.append(self.remedy)
        return " — ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "domain": self.domain.value,
            "tier": self.tier,
            "number": self.number,
            "kind": self.kind.value,
            "severity": int(self.severity),
            "severity_name": self.severity.name.lower(),
            "explanation": self.explanation,
            "remedy": self.remedy,
        }


class HyperNixError(Exception):
    """An error that carries a registered code.

    Subclass it where a module wants its own type; the code travels
    either way, and ``except HyperNixError`` catches the lot.
    """

    def __init__(self, code: ErrorCode | str, detail: str = "", **context: Any):
        self.code = code if isinstance(code, ErrorCode) else lookup(str(code))
        self.detail = detail
        self.context = context
        super().__init__(self.code.message(detail))

    @property
    def severity(self) -> Severity:
        return self.code.severity

    @property
    def kind(self) -> Kind:
        return self.code.kind

    def to_dict(self) -> dict[str, Any]:
        return {**self.code.to_dict(), "detail": self.detail,
                "context": dict(self.context)}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

CODES: dict[str, ErrorCode] = {}
#: ``(domain, number)`` already taken, so a collision is caught at import
#: rather than by two modules quietly sharing a code.
_TAKEN: dict[tuple[str, int], str] = {}


def register(
    domain: Domain | str,
    tier: int,
    number: int,
    kind: Kind | str,
    severity: Severity | int,
    explanation: str,
    remedy: str = "",
) -> ErrorCode:
    """Declare a code. Returns it, so a module can keep the object."""
    entry = ErrorCode(
        domain=Domain(domain), tier=tier, number=number,
        kind=Kind(kind), severity=Severity(severity),
        explanation=explanation, remedy=remedy,
    )
    key = (entry.domain.value, entry.number)
    held = CODES.get(_TAKEN.get(key, ""))
    if held is not None and held != entry:
        # Different code on the number, or the same code with different
        # words behind it: two meanings either way. Only an identical
        # re-declaration — a module imported twice — is let through.
        raise ValueError(
            f"{entry.code} reuses number {entry.number:05d} in domain "
            f"{entry.domain.value}, already held by {held.code} "
            f"({held.explanation!r}). Numbers are the searchable part; two "
            f"meanings behind one makes the search useless."
        )
    if not explanation.strip():
        # A code with no explanation is a number, and the whole point of
        # the registry is that the number means something everywhere.
        raise ValueError(f"{entry.code} has no explanation")
    _TAKEN[key] = entry.code
    CODES[entry.code] = entry
    return entry


def parse(code: str) -> ErrorCode:
    """Read a code string into its parts. Does **not** require registration.

    For a code that arrived from somewhere else — a log line, an older
    release, another machine — where the question is "what shape of
    failure is this" rather than "is this one of mine".
    """
    match = _PATTERN.match((code or "").strip())
    if not match:
        raise ValueError(
            f"{code!r} is not a HyperNix error code. The shape is "
            f"`L#-NNNNN.kS` — a domain letter, a tier digit, five digits, "
            f"a dot, a kind letter a-f and a severity 1-5. For example "
            f"`M4-01015.c3`."
        )
    letter, tier, number, kind, severity = match.groups()
    return ErrorCode(
        domain=Domain(letter), tier=int(tier), number=int(number),
        kind=Kind(kind), severity=Severity(int(severity)),
        explanation=CODES[code].explanation if code in CODES else "",
        remedy=CODES[code].remedy if code in CODES else "",
    )


def lookup(code: str) -> ErrorCode:
    """A *registered* code, or a refusal naming the nearest ones.

    The nearest ones matter: a mistyped digit is the common way to get
    here, and a bare "unknown code" sends somebody to grep.
    """
    found = CODES.get((code or "").strip())
    if found is not None:
        return found
    shape = parse(code)          # raises first if it is not even a code
    near = sorted(
        c for c in CODES
        if c.startswith(f"{shape.domain.value}{shape.tier}-")
    )
    hint = f" Codes in {shape.domain.value}{shape.tier}: {', '.join(near[:6])}" if near else ""
    raise KeyError(
        f"{code} is a well-formed code but nothing registers it.{hint}"
    )


def raise_for(code: str | ErrorCode, detail: str = "", **context: Any):
    """Raise :class:`HyperNixError` for a registered code.

    Refuses an unregistered one, which is the rule that keeps this a
    scheme rather than a convention nobody follows.
    """
    raise HyperNixError(code, detail, **context)


def explain(code: str) -> str:
    """One line for a person who has a code and nothing else."""
    try:
        return lookup(code).message()
    except KeyError as exc:
        return str(exc)


def codes_for(domain: Domain | str) -> Iterator[ErrorCode]:
    """Every registered code in one domain, lowest number first."""
    letter = Domain(domain).value
    yield from sorted(
        (c for c in CODES.values() if c.domain.value == letter),
        key=lambda c: c.number,
    )
