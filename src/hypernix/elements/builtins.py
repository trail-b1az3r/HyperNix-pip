"""The elements HyperNix ships. Registered into the default registry."""
from __future__ import annotations

from .carbon import Carbon
from .hydrogen import Element, ElementSpec, Registry
from .magnesium import Magnesium

__all__ = ["BUILTINS", "Hydrogen", "register_all"]


class Hydrogen(Element):
    """Element 1 is the framework itself; this entry is so it is listed."""

    spec = ElementSpec(
        symbol="H",
        summary="The framework every element is built on.",
        version="1.0.0",
    )


BUILTINS: tuple[type[Element], ...] = (Hydrogen, Carbon, Magnesium)


def register_all(registry: Registry) -> Registry:
    for cls in BUILTINS:
        registry.register(cls)
    return registry
