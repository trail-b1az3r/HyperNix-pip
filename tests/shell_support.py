"""Is there a bash here that actually runs a script?

``shutil.which("bash")`` is not the question. On a GitHub Windows runner
it finds ``C:\\Windows\\System32\\bash.exe`` — the WSL launcher stub,
which exists on every Windows install, exits non-zero, and prints
UTF-16LE text saying no distribution is installed. Tests gated on
``which`` therefore ran there and failed on output that was never a
shell's.

So this asks the only question that matters: does running a trivial
script through it produce the script's output?
"""
from __future__ import annotations

import shutil
import subprocess

_MARKER = "hypernix-bash-ok"


def _candidates() -> list[str]:
    """``bash`` on PATH, then -- on Windows, where that is usually the WSL
    stub in System32 -- the Git Bash that comes with Git for Windows."""
    import os

    found = [shutil.which("bash")]
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            # ...\Git\cmd\git.exe -> ...\Git\bin\bash.exe
            root = os.path.dirname(os.path.dirname(git))
            found.append(os.path.join(root, "bin", "bash.exe"))
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432")):
            if base:
                found.append(os.path.join(base, "Git", "bin", "bash.exe"))
    return [path for path in found if path and os.path.isfile(path)]


def _works(path: str) -> bool:
    try:
        result = subprocess.run(
            [path, "-c", f"echo {_MARKER}"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    # Bytes, not text: the WSL stub answers in UTF-16LE, and decoding that
    # as UTF-8 either raises or produces NUL-separated characters that no
    # substring check would match anyway.
    return result.returncode == 0 and _MARKER.encode() in result.stdout


def _probe() -> str | None:
    for path in _candidates():
        if _works(path):
            return path
    return None


#: The bash to run scripts with, or None when there is no usable one.
BASH: str | None = _probe()

#: Reason string for a skip marker, so every gated module says the same thing.
NO_BASH_REASON = "no working bash on this machine (a WSL stub does not count)"


def shell_path(path, *, style: str = "auto") -> str:
    """Spell *path* the way the shell in these tests resolves it.

    Three spellings are possible on Windows and only one of them works,
    which one depending on the bash. ``C:/x`` inside a colon-separated
    ``PATH`` is ambiguous -- the drive letter's colon is also the
    separator -- so Git Bash may split it into ``C`` and ``/x`` and find
    neither. ``/c/x`` is the mount Git Bash actually publishes. On POSIX
    all three are the same string.

    ``style="auto"`` asks :func:`python3_path_entry` which one this
    machine answered to.
    """
    import os
    from pathlib import Path

    text = Path(path).as_posix()
    if os.name != "nt":
        return text
    if style == "auto":
        style = python3_path_entry()[1]
    if style == "native":
        return str(Path(path))
    if style == "mount" and len(text) > 2 and text[1] == ":":
        return f"/{text[0].lower()}{text[2:]}"
    return text


def _write_shim(bindir, style: str) -> None:
    """A ``python3`` that is this interpreter, for a shell that has none."""
    import sys

    interpreter = shell_path(sys.executable, style=style)
    shim = bindir / "python3"
    shim.write_text(f'#!/bin/sh\nexec "{interpreter}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)


def _python3_works(entry: str) -> bool:
    marker = "hypernix-python3-ok"
    try:
        result = subprocess.run(
            [BASH, "-c", f'PATH="{entry}:/usr/bin:/bin"; exec python3 -c '
                         f'"print(\'{marker}\')"'],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return marker in (result.stdout or "")


def _probe_python3() -> tuple[str | None, str]:
    """``(PATH entry, spelling)`` for a directory whose python3 bash can run.

    Both halves are unknown on Windows and guessing at either produces a
    test that fails for a reason unrelated to what it is about: there is
    no ``python3`` on any PATH there -- the interpreter is ``python.exe``
    in the tool cache -- and which spelling of a Windows directory
    survives a colon-separated ``PATH`` depends on the bash. So the shim
    is built and then bash is *asked*, rather than told.
    """
    import atexit
    import shutil as _shutil
    import tempfile
    from pathlib import Path

    if BASH is None:
        return None, "posix"
    bindir = Path(tempfile.mkdtemp(prefix="hypernix-python3-shim-"))
    atexit.register(_shutil.rmtree, bindir, True)
    for style in ("posix", "mount", "native"):
        _write_shim(bindir, style)
        entry = shell_path(bindir, style=style)
        if _python3_works(entry):
            return entry, style
    return None, "posix"


_PYTHON3: tuple[str | None, str] | None = None


def python3_path_entry() -> tuple[str | None, str]:
    """Cached :func:`_probe_python3`. ``(None, ...)`` when nothing worked."""
    global _PYTHON3
    if _PYTHON3 is None:
        _PYTHON3 = _probe_python3()
    return _PYTHON3


#: Reason string for skipping a test that needs the scripts to run python3.
NO_PYTHON3_REASON = (
    "no python3 the test shell can run (and no shim spelling it resolves)"
)
