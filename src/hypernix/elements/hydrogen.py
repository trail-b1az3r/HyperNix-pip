"""hydrogen — the framework element modules are built on.

Element 1, because everything else is made of it.

What an element is
------------------
A class with a :class:`ElementSpec` and a few optional hooks. The spec
says which element it is, which hydrogen generation it was written
against, and what it is allowed to touch. The hooks are how it does
anything: :meth:`Element.activate` and :meth:`Element.deactivate` for
its own lifetime, and the ``transform_*`` hooks for an oven it has been
attached to by :mod:`.natural_gas`.

Three rules, each one a thing that went wrong in a plugin system
somewhere and is refused here rather than documented:

**The symbol is the real one.** An element called Magnesium is ``Mg``,
number 12. Registering it as ``Mn`` — manganese — is refused, because a
name that points at a different element than its symbol is worse than
no name at all: whoever searches for it finds the wrong thing.

**Periods 6 and 7 are experimental**, decided from the atomic number.
Nothing there activates unless the caller has said experimental
elements are allowed.

**Permissions are enforced, not declared.** An element lists what it
needs — processes, network, filesystem, the oven, a shell — and the
:class:`ElementContext` it is handed refuses anything else at the moment
of use. A permission list that is only read by humans is a comment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ..system import errorcatalogue as codes
from ..system.errorcodes import HyperNixError

logger = logging.getLogger(__name__)

__all__ = [
    "GENERATION",
    "PERMISSIONS",
    "SYMBOLS",
    "Element",
    "ElementSpec",
    "ElementContext",
    "Registry",
    "atomic_number",
    "period_of",
    "is_experimental",
    "name_of",
    "default_registry",
]

#: Bumped when the Element interface changes in a way an element written
#: against the old one cannot survive. Elements declare the generation
#: they need; a mismatch is refused at registration, not at first call.
GENERATION = 1

#: Everything an element can ask for. A request outside this set is a
#: typo, and a typo in a permission list is how an element ends up
#: silently unable to do its job.
PERMISSIONS: frozenset[str] = frozenset({
    "processes",   # read and change other processes (magnesium)
    "network",     # outbound connections
    "filesystem",  # write outside the element's own data directory
    "oven",        # transform an oven's prompts and outputs
    "shell",       # run commands
})

#: Symbol by atomic number; index 0 is unused so ``SYMBOLS[12] == "Mg"``.
SYMBOLS: tuple[str, ...] = (
    "",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)

_NAMES: dict[str, str] = {
    "H": "hydrogen", "He": "helium", "Li": "lithium", "Be": "beryllium",
    "B": "boron", "C": "carbon", "N": "nitrogen", "O": "oxygen",
    "F": "fluorine", "Ne": "neon", "Na": "sodium", "Mg": "magnesium",
    "Al": "aluminium", "Si": "silicon", "P": "phosphorus", "S": "sulfur",
    "Cl": "chlorine", "Ar": "argon", "K": "potassium", "Ca": "calcium",
    "Fe": "iron", "Cu": "copper", "Zn": "zinc", "Ag": "silver",
    "Sn": "tin", "Au": "gold", "Hg": "mercury", "Pb": "lead",
    "U": "uranium", "Pu": "plutonium", "Og": "oganesson",
}

_BY_SYMBOL: dict[str, int] = {s: n for n, s in enumerate(SYMBOLS) if s}
_PERIOD_ENDS = (2, 10, 18, 36, 54, 86, 118)


def atomic_number(symbol: str) -> int:
    """``"Mg"`` -> 12. Case matters, as it does on the table: ``Co`` is
    cobalt and ``CO`` is carbon monoxide."""
    try:
        return _BY_SYMBOL[symbol]
    except KeyError:
        raise HyperNixError(
            codes.ELEMENT_NOT_FOUND,
            f"{symbol!r} is not a symbol on the periodic table",
        ) from None


def period_of(number: int) -> int:
    """The row. 1 to 7."""
    if not 1 <= number <= 118:
        raise HyperNixError(codes.ELEMENT_NOT_FOUND,
                            f"there is no element number {number}")
    for period, end in enumerate(_PERIOD_ENDS, start=1):
        if number <= end:
            return period
    raise AssertionError("unreachable")  # pragma: no cover


def is_experimental(number: int) -> bool:
    """Periods 6 and 7: caesium (55) onward."""
    return period_of(number) >= 6


def name_of(symbol: str) -> str:
    return _NAMES.get(symbol, symbol.lower())


@dataclass(frozen=True)
class ElementSpec:
    """What an element is and what it may do."""

    symbol: str
    summary: str
    version: str = "0.1.0"
    generation: int = GENERATION
    permissions: frozenset[str] = frozenset()
    #: Written by somebody other than HyperNix. User elements may not
    #: take a symbol a built-in already holds.
    user: bool = False

    def __post_init__(self) -> None:
        atomic_number(self.symbol)  # raises for a symbol that is not one
        unknown = set(self.permissions) - PERMISSIONS
        if unknown:
            raise HyperNixError(
                codes.ELEMENT_PERMISSION,
                f"{self.symbol} asks for {sorted(unknown)}, which are not "
                f"permissions. There are: {sorted(PERMISSIONS)}",
            )

    @property
    def number(self) -> int:
        return atomic_number(self.symbol)

    @property
    def name(self) -> str:
        return name_of(self.symbol)

    @property
    def period(self) -> int:
        return period_of(self.number)

    @property
    def experimental(self) -> bool:
        return is_experimental(self.number)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "name": self.name, "number": self.number,
            "period": self.period, "experimental": self.experimental,
            "summary": self.summary, "version": self.version,
            "generation": self.generation,
            "permissions": sorted(self.permissions), "user": self.user,
        }


@dataclass
class ElementContext:
    """What an element is handed. Its only way to reach anything."""

    spec: ElementSpec
    data_dir: Path
    config: dict[str, Any] = field(default_factory=dict)

    def require(self, permission: str) -> None:
        """Refuse unless the element declared *permission*.

        Called by the element before it does the thing, and by the
        framework on the element's behalf where the framework is the one
        doing it — natural gas asks for ``oven`` before attaching.
        """
        if permission not in self.spec.permissions:
            raise HyperNixError(
                codes.ELEMENT_PERMISSION,
                f"{self.spec.symbol} ({self.spec.name}) did not declare "
                f"{permission!r}",
                element=self.spec.symbol, permission=permission,
            )

    @property
    def log(self) -> logging.Logger:
        return logging.getLogger(f"hypernix.elements.{self.spec.name}")


class Element:
    """Base class. Subclass, set ``spec``, override what you need.

    Every hook has a no-op default, so an element implements only what
    it is for — an oven addon never writes ``activate``, and magnesium
    never writes ``transform_prompt``.
    """

    spec: ClassVar[ElementSpec]

    def __init__(self, context: ElementContext) -> None:
        self.context = context
        self.active = False

    # -- lifetime --------------------------------------------------------

    def activate(self) -> None:
        """Start doing whatever this element does."""

    def deactivate(self) -> None:
        """Undo it. Must be safe to call when ``activate`` never ran."""

    # -- oven hooks, used by natural gas ------------------------------------

    def transform_prompt(self, prompt: str) -> str:
        return prompt

    def transform_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return messages

    def transform_output(self, text: str) -> str:
        return text

    def status(self) -> dict[str, Any]:
        return {**self.spec.to_dict(), "active": self.active}


class Registry:
    """Which elements exist here. One per process is the normal case."""

    def __init__(self, *, allow_experimental: bool = False,
                 data_root: Path | None = None) -> None:
        self.allow_experimental = allow_experimental
        self.data_root = data_root or Path.home() / ".hypernix" / "elements" / "data"
        self._classes: dict[str, type[Element]] = {}
        self._live: dict[str, Element] = {}
        #: symbol -> why it failed to load, for `hypernix elements list`.
        self.failures: dict[str, str] = {}

    # -- registering -----------------------------------------------------

    def register(self, cls: type[Element]) -> type[Element]:
        """Add an element class. Usable as a decorator."""
        spec = getattr(cls, "spec", None)
        if not isinstance(spec, ElementSpec):
            raise HyperNixError(codes.ELEMENT_LOAD_FAILED,
                                f"{cls.__name__} has no ElementSpec")
        if spec.generation != GENERATION:
            raise HyperNixError(
                codes.ELEMENT_API_MISMATCH,
                f"{spec.symbol} was written for hydrogen generation "
                f"{spec.generation}; this is generation {GENERATION}",
            )
        held = self._classes.get(spec.symbol)
        if held is not None and held is not cls:
            if spec.user and not held.spec.user:
                raise HyperNixError(
                    codes.ELEMENT_CONFLICT,
                    f"{spec.symbol} is a built-in; a user element cannot "
                    f"take its symbol",
                )
            if not spec.user and held.spec.user:
                # A built-in arriving after a user element with the same
                # symbol: the built-in wins and the user one is reported,
                # rather than a user file shadowing HyperNix's own code.
                self.failures[spec.symbol] = (
                    f"user element displaced by the built-in {spec.name}"
                )
            else:
                raise HyperNixError(
                    codes.ELEMENT_CONFLICT,
                    f"{spec.symbol} is already registered by "
                    f"{held.__module__}.{held.__name__}",
                )
        self._classes[spec.symbol] = cls
        return cls

    def lookup(self, symbol_or_name: str) -> type[Element]:
        key = symbol_or_name.strip()
        if key in self._classes:
            return self._classes[key]
        for cls in self._classes.values():
            if cls.spec.name == key.lower():
                return cls
        raise HyperNixError(
            codes.ELEMENT_NOT_FOUND,
            f"{key!r}; registered: {', '.join(sorted(self._classes)) or 'none'}",
        )

    def specs(self) -> list[ElementSpec]:
        return sorted((c.spec for c in self._classes.values()),
                      key=lambda s: s.number)

    # -- running ---------------------------------------------------------

    def instance(self, symbol_or_name: str, *, config: dict | None = None) -> Element:
        """The live instance, created on first use. Gated here."""
        cls = self.lookup(symbol_or_name)
        spec = cls.spec
        if spec.experimental and not self.allow_experimental:
            raise HyperNixError(
                codes.ELEMENT_EXPERIMENTAL,
                f"{spec.symbol} ({spec.name}) is element {spec.number}, "
                f"period {spec.period}",
            )
        live = self._live.get(spec.symbol)
        if live is None:
            data_dir = self.data_root / spec.name
            live = cls(ElementContext(spec, data_dir, dict(config or {})))
            self._live[spec.symbol] = live
        return live

    def activate(self, symbol_or_name: str, *, config: dict | None = None) -> Element:
        element = self.instance(symbol_or_name, config=config)
        if not element.active:
            element.activate()
            element.active = True
        return element

    def deactivate(self, symbol_or_name: str) -> bool:
        cls = self.lookup(symbol_or_name)
        live = self._live.get(cls.spec.symbol)
        if live is None or not live.active:
            return False
        try:
            live.deactivate()
        finally:
            # Marked inactive even if deactivate raised: an element that
            # failed half way through undoing itself must not be reported
            # as still running, or nothing will try again.
            live.active = False
        return True

    def deactivate_all(self) -> list[str]:
        stopped = []
        # Highest number first, the reverse of activation order, so a
        # heavier element built on a lighter one goes before it.
        for symbol in sorted(self._live, key=atomic_number, reverse=True):
            try:
                if self.deactivate(symbol):
                    stopped.append(symbol)
            except Exception as exc:  # noqa: BLE001 - keep stopping the rest
                logger.warning("elements: %s failed to deactivate: %s", symbol, exc)
        return stopped

    def active(self) -> list[Element]:
        return sorted((e for e in self._live.values() if e.active),
                      key=lambda e: e.spec.number)


_DEFAULT: Registry | None = None


def default_registry(*, allow_experimental: bool | None = None) -> Registry:
    """The process-wide registry, with the built-in elements in it."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Registry()
        from . import builtins as _builtins

        _builtins.register_all(_DEFAULT)
    if allow_experimental is not None:
        _DEFAULT.allow_experimental = allow_experimental
    return _DEFAULT

