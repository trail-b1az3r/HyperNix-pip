"""One way to profile HyperNix on every supported Python (PEP 799).

Python 3.15 gathers its profilers into a ``profiling`` package:
``profiling.tracing`` is the deterministic profiler (what ``cProfile``
was, and ``cProfile`` stays as an alias of it), and ``profiling.sampling``
is new -- a statistical profiler that can attach to a running process
with almost no overhead. Python 3.12-3.14 have only ``cProfile``.

This module is the boundary. Callers ask for *deterministic* or
*sampling* profiling and get the best backend this interpreter has; on
3.12-3.14 nothing is removed or changed (``cProfile`` is what they used
before), and sampling says plainly that it needs 3.15 rather than
pretending.

    from hypernix._profiling import trace, profile_call, sampling_argv

    with trace() as run:                # deterministic, any version
        do_work()
    print(run.report(limit=15))

    result, stats = profile_call(do_work, arg)

    argv = sampling_argv(pid=1234, duration=10, output="run.pstats")  # 3.15+
"""
from __future__ import annotations

import io
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "BACKEND",
    "SAMPLING_AVAILABLE",
    "SAMPLING_FORMATS",
    "SamplingUnavailable",
    "TraceRun",
    "new_tracer",
    "profile_call",
    "sample",
    "sampling_argv",
    "trace",
]

#: True from 3.15, where the ``profiling`` package exists.
HAS_PROFILING_PACKAGE: bool = sys.version_info >= (3, 15)

#: Which deterministic profiler :func:`new_tracer` uses.
BACKEND: Literal["profiling.tracing", "cProfile"] = (
    "profiling.tracing" if HAS_PROFILING_PACKAGE else "cProfile"
)

#: Whether :func:`sample` can run here (``profiling.sampling``, 3.15+).
SAMPLING_AVAILABLE: bool = HAS_PROFILING_PACKAGE

#: Output formats ``profiling.sampling`` writes, by the flag that asks.
SAMPLING_FORMATS: dict[str, str] = {
    "pstats": "--pstats",
    "collapsed": "--collapsed",
    "flamegraph": "--flamegraph",
    "gecko": "--gecko",
    "jsonl": "--jsonl",
}


class SamplingUnavailable(RuntimeError):
    """Sampling was asked for on a Python that has no sampling profiler."""


def new_tracer() -> Any:
    """A fresh deterministic profiler from the best module this Python has.

    The object is ``profiling.tracing.Profile`` on 3.15 and
    ``cProfile.Profile`` before it -- the same class under two names, so
    ``enable()``, ``disable()``, ``runcall()`` and ``pstats`` all work
    the same either way.
    """
    if HAS_PROFILING_PACKAGE:
        import profiling.tracing as tracing

        return tracing.Profile()
    import cProfile

    return cProfile.Profile()


@dataclass
class TraceRun:
    """What :func:`trace` collected, usable once the block has exited."""

    backend: str = BACKEND
    profiler: Any = None
    _stats: Any = field(default=None, repr=False)

    @property
    def stats(self) -> Any:
        """The run as ``pstats.Stats``."""
        if self._stats is None:
            if self.profiler is None:
                raise RuntimeError("nothing was profiled")
            import pstats

            self._stats = pstats.Stats(self.profiler, stream=io.StringIO())
        return self._stats

    def report(self, *, sort: str = "cumulative", limit: int = 25) -> str:
        """The top *limit* rows, sorted by *sort*, as text."""
        import pstats

        out = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=out)
        stats.sort_stats(sort).print_stats(limit)
        return out.getvalue()

    def dump(self, path: str | os.PathLike[str]) -> None:
        """Write the run as a ``.pstats`` file (snakeviz, ``pstats``)."""
        self.stats.dump_stats(os.fspath(path))


@contextmanager
def trace() -> Iterator[TraceRun]:
    """Deterministically profile the ``with`` block."""
    run = TraceRun(profiler=new_tracer())
    run.profiler.enable()
    try:
        yield run
    finally:
        run.profiler.disable()


def profile_call[T](fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> tuple[T, Any]:
    """Call ``fn(*args, **kwargs)`` under the deterministic profiler.

    Returns its result and the run's ``pstats.Stats``.
    """
    with trace() as run:
        result = fn(*args, **kwargs)
    return result, run.stats


def sampling_argv(*, pid: int | None = None, script: str | os.PathLike[str] | None = None,
                  script_args: list[str] | None = None, duration: float | None = None,
                  rate: str | int | None = None, output: str | os.PathLike[str] | None = None,
                  fmt: str = "pstats", all_threads: bool = True,
                  python: str | None = None) -> list[str]:
    """The command line that samples a process (``attach``) or a script (``run``).

    Exactly one of *pid* and *script*. *fmt* is one of
    :data:`SAMPLING_FORMATS`. Raises :class:`SamplingUnavailable` before
    3.15, naming what to use instead.
    """
    if not SAMPLING_AVAILABLE:
        raise SamplingUnavailable(
            f"statistical sampling needs Python 3.15's profiling.sampling; this is "
            f"{sys.version_info.major}.{sys.version_info.minor}. Use trace() for a "
            f"deterministic profile, or an external sampler such as py-spy."
        )
    if (pid is None) == (script is None):
        raise ValueError("give exactly one of pid= or script=")
    if fmt not in SAMPLING_FORMATS:
        raise ValueError(f"fmt is one of {', '.join(SAMPLING_FORMATS)}; got {fmt!r}")
    argv = [python or sys.executable, "-m", "profiling.sampling",
            "attach" if pid is not None else "run", SAMPLING_FORMATS[fmt]]
    if all_threads:
        argv.append("--all-threads")
    if duration is not None:
        argv += ["--duration", str(duration)]
    if rate is not None:
        argv += ["--sampling-rate", str(rate)]
    if output is not None:
        argv += ["-o", os.fspath(output)]
    if pid is not None:
        argv.append(str(int(pid)))
    else:
        argv.append(os.fspath(script))  # type: ignore[arg-type]
        argv += list(script_args or [])
    return argv


def sample(*, timeout: float | None = None, **kwargs: Any) -> Any:
    """Run :func:`sampling_argv` and return the ``CompletedProcess``.

    Attaching to another process may need privileges the OS grants only
    to its owner (and, on Linux, a ptrace scope that allows it); the
    profiler's own message says which.
    """
    import subprocess

    argv = sampling_argv(**kwargs)
    return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",  # noqa: S603
                          timeout=timeout, check=False)
