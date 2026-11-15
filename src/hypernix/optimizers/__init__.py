"""hypernix.optimizers — The Pressure Cooker optimizer family and optimizer plumbing.

Modules: ``optimizer_framework``, ``pressure_cooker``,
``pressure_cooker_v3``, ``pressure_cooker_v4``, ``pressure_cooker_v5``,
``pressure_cooker_v5s``, ``pressure_cooker_v6``,
``pressure_cooker_v6v``, ``pressure_cooker_v7``, ``pressure_cooker_v7v``,
``v7m_features``, ``optimizer_framework_v2``, ``benchmark`` and ``cli``
(the last two are ``hnx optimizer``).

Every module here is also reachable under its historical flat name
(``hypernix.optimizer_framework``); see the alias finder in ``hypernix/__init__.py``.
Submodules are imported lazily, so importing this package costs nothing
until you touch one of them.
"""
from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "optimizer_framework",
    "pressure_cooker",
    "pressure_cooker_v3",
    "pressure_cooker_v4",
    "pressure_cooker_v5",
    "pressure_cooker_v5s",
    "pressure_cooker_v6",
    "pressure_cooker_v6v",
    "pressure_cooker_v7",
    "pressure_cooker_v7v",
    "v7m_features",
    "optimizer_framework_v2",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
