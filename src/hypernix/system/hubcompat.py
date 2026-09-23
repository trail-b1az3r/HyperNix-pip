"""hypernix.hubcompat — one place that knows what huggingface_hub broke.

`huggingface-hub` 1.0 removed `direction` from `list_models` and
`list_datasets`. The 0.x calls that passed `direction=-1` do not
degrade under 1.x — they raise ``TypeError: got an unexpected keyword
argument 'direction'``, at the moment somebody searches, which is
usually the first thing they do.

Sorting is not lost. In 1.x the listing endpoints sort descending for
the sort keys people actually use (`downloads`, `likes`, `created_at`),
so dropping the argument gives the order the caller wanted anyway.

Why a module rather than a try/except at each call site
-------------------------------------------------------
There were three call sites and they would each have grown their own
version check, drifted, and been wrong in different ways. More to the
point: the next removal wants one file to change, not a grep. The
version is inspected once, here, by asking the function what it accepts
rather than by parsing `__version__` — a fork, a vendored copy, or a
backport all answer that question correctly and none of them answer a
string comparison correctly.
"""
from __future__ import annotations

import functools
import inspect
from collections.abc import Iterable
from typing import Any

__all__ = [
    "hub_version",
    "accepts",
    "list_models",
    "list_datasets",
    "supports_direction",
]


def hub_version() -> str:
    try:
        import huggingface_hub

        return str(getattr(huggingface_hub, "__version__", ""))
    except ImportError:
        return ""


@functools.lru_cache(maxsize=32)
def accepts(function_name: str, parameter: str) -> bool:
    """Does ``HfApi.<function_name>`` take *parameter*?

    Asked of the signature rather than of ``__version__``: a fork, a
    vendored copy or a backport all answer this correctly, and a string
    comparison answers none of them correctly.
    """
    try:
        from huggingface_hub import HfApi

        signature = inspect.signature(getattr(HfApi, function_name))
    except (ImportError, AttributeError, TypeError, ValueError):
        return False
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return True
    return parameter in parameters


def supports_direction() -> bool:
    """True on huggingface-hub 0.x, false from 1.0."""
    return accepts("list_datasets", "direction")


def _listing(method: str, api, **kwargs) -> Iterable[Any]:
    # `direction` only when it is accepted. On 1.x the endpoint already
    # sorts descending for downloads/likes/created_at, so dropping it
    # gives the order the caller asked for.
    if "direction" in kwargs and not accepts(method, "direction"):
        kwargs.pop("direction")
    return getattr(api, method)(**kwargs)


def list_datasets(api=None, **kwargs) -> Iterable[Any]:
    """``HfApi.list_datasets`` that works on 0.x and 1.x alike."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    return _listing("list_datasets", api, **kwargs)


def list_models(api=None, **kwargs) -> Iterable[Any]:
    """``HfApi.list_models`` that works on 0.x and 1.x alike."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    return _listing("list_models", api, **kwargs)
