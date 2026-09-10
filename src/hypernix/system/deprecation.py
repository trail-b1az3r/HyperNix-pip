"""hypernix.system.deprecation — one way to say a module is on its way out.

Five modules are deprecated. Four of them said so, since 0.71.5a2, like
this::

    from rich.console import Console

    Console().print("[bold red]WARNING: old_oven is deprecated. ...[/]")

That is visible, which is the hard part and the thing it got right. It
is also wrong in four ways that matter:

**It writes to stdout.** `rich.Console()` defaults there, so the notice
lands in whatever the caller was capturing. ``hnx ... > out.json``
gets a line of English in its JSON; a subprocess parsing stdout gets a
parse error rather than a hint. A diagnostic belongs on stderr, which is
what stderr is for.

**Tooling cannot see it.** No ``DeprecationWarning`` means ``-W error``
does not fail on it, ``pytest.warns`` cannot assert it, and no linter or
audit can find callers of the deprecated surface. A printed string is
invisible to every tool that exists for exactly this.

**It cannot be turned off.** A script that knowingly uses the old API
has no way to quiet the line, so the only options are noise forever or
patching the library.

**It pulls in rich** to print eleven words.

So: a real ``DeprecationWarning`` for tooling, *and* a guaranteed
one-line notice on stderr, because ``DeprecationWarning`` is hidden by
default — outside ``__main__`` Python shows nothing at all, which would
turn "prints immediately on import" into "prints for nobody".

Getting both without printing twice needs a way to know whether the
warning was actually shown. There is no public API for that, so the one
``warnings.warn`` call is made with :data:`warnings.showwarning` briefly
swapped for a spy: if the active filters let the warning through, the
spy sees it and the stderr line is skipped; if they suppressed it, the
stderr line is printed instead. Under ``-W error`` the warning is raised
and propagates, which is what that flag asks for.

``HYPERNIX_DEPRECATION_WARNINGS=0`` silences the stderr line only. It
deliberately does not touch the ``DeprecationWarning``: the environment
variable is there to control *noise*, and ``-W`` / ``filterwarnings`` is
the mechanism that already exists for controlling *semantics*. Making
the variable suppress both would let a stray export disarm
``-W error::DeprecationWarning`` for a whole CI run, which is the one
thing that must not happen quietly.

So to hear nothing at all, silence each with its own switch::

    PYTHONWARNINGS=ignore::DeprecationWarning \
    HYPERNIX_DEPRECATION_WARNINGS=0 \
        python your_script.py

Both are environment variables, so this needs no change to how the
program is launched.
"""
from __future__ import annotations

import os
import sys
import warnings

__all__ = ["deprecated_module", "notice_for", "reset_for_testing"]

ENV_VAR = "HYPERNIX_DEPRECATION_WARNINGS"

# Once per module per process. Module-level code already runs once, so
# this only matters for importlib.reload and for tests -- but a helper
# that warns twice when called twice is a helper nobody trusts.
_announced: set[str] = set()


def notice_for(
    name: str,
    *,
    instead: str,
    since: str | None = None,
    removed_in: str | None = None,
    extra: str | None = None,
) -> str:
    """The one-line message, built the same way for both channels."""
    parts = [f"{name} is deprecated"]
    if since:
        parts.append(f"since {since}")
    parts.append(f"— use {instead} instead")
    message = " ".join(parts) + "."
    if removed_in:
        message += f" It is scheduled for removal in {removed_in}."
    if extra:
        message += f" {extra}"
    return message


def _stderr_line(message: str) -> None:
    prefix = "hypernix: "
    # Colour the way bin/hypernix-t1 does -- only for a terminal, and
    # never when NO_COLOR is set. A notice that emits escape codes into
    # a log file has swapped one unreadable output for another.
    stream = sys.stderr
    try:
        tty = stream.isatty()
    except (AttributeError, ValueError):
        tty = False
    if tty and not os.environ.get("NO_COLOR"):
        body = f"\033[38;5;179m{prefix}{message}\033[0m"
    else:
        body = f"{prefix}{message}"
    print(body, file=stream, flush=True)


def deprecated_module(
    name: str,
    *,
    instead: str,
    since: str | None = None,
    removed_in: str | None = None,
    extra: str | None = None,
    stacklevel: int = 3,
) -> str:
    """Announce that the module named *name* is deprecated.

    Call it once, at module level, from the deprecated module itself.
    Returns the message so a caller can reuse or assert on it.

    *stacklevel* counts out from :func:`warnings.warn`: 1 is this
    function, 2 is this module, 3 is the deprecated module's own top
    level — which is the default, and the frame worth blaming.
    """
    message = notice_for(
        name, instead=instead, since=since, removed_in=removed_in, extra=extra
    )
    if name in _announced:
        return message
    _announced.add(name)

    shown = False
    original = warnings.showwarning

    def _spy(*args: object, **kwargs: object) -> None:
        nonlocal shown
        shown = True
        original(*args, **kwargs)  # type: ignore[arg-type]

    warnings.showwarning = _spy  # type: ignore[assignment]
    try:
        warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
    finally:
        warnings.showwarning = original

    if not shown and os.environ.get(ENV_VAR, "1") != "0":
        _stderr_line(message)
    return message


def reset_for_testing() -> None:
    """Forget which modules have announced themselves.

    Only for tests: a process normally imports a module once, so the
    once-per-process guard is invisible. A test that wants to observe
    the announcement twice needs to clear it.
    """
    _announced.clear()
