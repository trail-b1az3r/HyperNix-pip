"""Marking an optimizer generation deprecated.

Warns on *construction*, never on import. V4 imports its helpers from
the V3 module; an import-time warning would tell every V4 user their V3
was deprecated, and HyperNix's own package loader imports these modules
for its flat-name aliases — so the warning would fire for people who
never touched a Pressure Cooker at all.

``FutureWarning`` rather than ``DeprecationWarning``: the latter is
hidden by default unless the call is made from ``__main__``, and most
training runs call the optimizer from a module. A deprecation nobody
sees has not been announced.
"""
from __future__ import annotations

import functools
import warnings

__all__ = ["deprecate", "DEPRECATED_GENERATIONS"]

#: Generation -> what to use instead. V2 is absent on purpose: no
#: `PressureCookerV2` has ever existed in this codebase (see
#: wiki/Optimizers.md), so there is nothing to deprecate.
DEPRECATED_GENERATIONS: dict[str, str] = {
    "v1": "PressureCookerV4 (kept), or V5/V6 for new training",
    "v3": "PressureCookerV4 (kept), or V5/V6 for new training",
}


def deprecate(cls: type, generation: str) -> type:
    """Wrap ``cls.__init__`` to warn once per construction."""
    original = cls.__init__
    replacement = DEPRECATED_GENERATIONS[generation]

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        # Only for the outermost call: a subclass whose __init__ reaches
        # this through super() is one construction, and two warnings for
        # it would read as two optimizers.
        if not getattr(self, "_hnx_deprecation_warned", False):
            object.__setattr__(self, "_hnx_deprecation_warned", True)
            warnings.warn(
                f"{type(self).__name__} is Pressure Cooker {generation.upper()}, "
                f"which is deprecated and will be removed. Use {replacement}.",
                FutureWarning,
                stacklevel=2,
            )
        original(self, *args, **kwargs)

    cls.__init__ = __init__
    cls.__hnx_deprecated__ = generation
    return cls
