"""strife.blasctl — how many threads BLAS uses, set at run time.

Strife's native kernels have their own threads. numpy's BLAS has its own
too, and after a multiply its idle workers keep spinning for a while,
which takes the cores the kernels need next: measured here, a native
multiply that takes 0.7 ms alone takes 8 ms right after a BLAS call. So
when the kernels do the heavy work, BLAS is held to one thread.

``threadpoolctl`` does this when it is installed. Without it, this module
finds the OpenBLAS that numpy loaded and calls its own thread setter --
the same thing, for the one library numpy's wheels ship. Anything else
(Accelerate, MKL without threadpoolctl) is reported as not controllable
rather than guessed at.
"""
from __future__ import annotations

import ctypes
import re
import sys
from typing import Any

__all__ = ["set_threads", "get_threads", "controllable"]

_SETTERS = ("scipy_openblas_set_num_threads64_", "scipy_openblas_set_num_threads",
            "openblas_set_num_threads64_", "openblas_set_num_threads")
_GETTERS = ("scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads",
            "openblas_get_num_threads64_", "openblas_get_num_threads")
_found: list[tuple[Any, Any]] | None = None


def _openblas() -> list[tuple[Any, Any]]:
    """``(set, get)`` for every OpenBLAS already loaded in this process."""
    global _found
    if _found is not None:
        return _found
    import numpy as np

    np.ones((2, 2), dtype=np.float32) @ np.ones((2, 2), dtype=np.float32)  # make sure it is loaded
    paths: set[str] = set()
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/maps", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) >= 6 and re.search(r"openblas", parts[-1], re.I):
                        paths.add(parts[-1])
        except OSError:
            pass
    found: list[tuple[Any, Any]] = []
    for path in sorted(paths):
        try:
            lib = ctypes.CDLL(path)  # already loaded: this only returns its handle
        except OSError:
            continue
        setter = next((getattr(lib, n) for n in _SETTERS if hasattr(lib, n)), None)
        getter = next((getattr(lib, n) for n in _GETTERS if hasattr(lib, n)), None)
        if setter is None:
            continue
        setter.argtypes, setter.restype = [ctypes.c_int], None
        if getter is not None:
            getter.argtypes, getter.restype = [], ctypes.c_int
        found.append((setter, getter))
    _found = found
    return found


def controllable() -> bool:
    try:
        import threadpoolctl  # noqa: F401

        return True
    except ImportError:
        return bool(_openblas())


def set_threads(count: int) -> bool:
    """Hold BLAS to *count* threads. ``False`` when nothing here can do that."""
    count = max(1, int(count))
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(count, user_api="blas")
        return True
    except ImportError:
        pass
    libs = _openblas()
    for setter, _getter in libs:
        setter(count)
    return bool(libs)


def get_threads() -> int | None:
    """The BLAS thread count, when it can be read."""
    try:
        from threadpoolctl import threadpool_info

        counts = [int(i["num_threads"]) for i in threadpool_info() if i.get("user_api") == "blas"]
        if counts:
            return max(counts)
    except ImportError:
        pass
    for _setter, getter in _openblas():
        if getter is not None:
            return int(getter())
    return None
