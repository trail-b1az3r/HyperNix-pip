#!/usr/bin/env python3
"""HyperNix multi-version launcher.

Finds an interpreter that has hypernix installed and runs the CLI there,
so `hnx` works on a machine with several Pythons and the package in only
some of them.

The interpreter that owns this script goes first
-----------------------------------------------
That was not always so, and the bug it caused is worth writing down.
This used to try python3.12, then 3.13, then 3.14, and run the first one
where hypernix imported -- *whichever* one that was. So on a machine
with an old hypernix on 3.12 and a fresh `pip install --upgrade` on
3.13, `hnx` ran the old one. Every command added since that 3.12 install
was missing, and the symptom was the usage table: the CLI does not
recognise the subcommand, so it prints what it does recognise. Nothing
said a different install was answering.

`pip` put this console script somewhere for a reason -- ``sys.executable``
here *is* the interpreter it was installed into, and that is the install
the person just upgraded. So it is tried first, and the version-priority
list is only a fallback for the case this was really meant to cover: the
script is on PATH but its own interpreter no longer has the package.

When the fallback does fire and lands on a *different* version of
hypernix, it says so on stderr. Silently running something other than
what you installed is the thing this got wrong once already.
"""
from __future__ import annotations

import subprocess
import sys
from typing import NamedTuple


class PythonVersion(NamedTuple):
    """Represents a Python version to check."""
    major: int
    minor: int
    
    @property
    def exe_name(self) -> str:
        return f"python{self.major}.{self.minor}"
    
    @property
    def version_tuple(self) -> tuple[int, int]:
        return (self.major, self.minor)


# Priority order: prefer 3.12, then 3.13, then 3.14
VERSION_PRIORITY = [
    PythonVersion(3, 12),
    PythonVersion(3, 13),
    PythonVersion(3, 14),
]


def check_hypernix_installed(py_exe: str) -> bool:
    """Check if hypernix is installed for the given Python executable."""
    try:
        result = subprocess.run(
            [py_exe, "-c", "import hypernix; print(hypernix.__version__)"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def installed_version(py_exe: str) -> str:
    """The hypernix version `py_exe` has, or "" if it has none."""
    try:
        result = subprocess.run(
            [py_exe, "-c", "import hypernix; print(hypernix.__version__)"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def find_best_python() -> str | None:
    """The interpreter to run the CLI on, or None if none has hypernix.

    This interpreter first. It is the one `pip` installed this console
    script into, so it is the one holding the version the person last
    installed -- see the module docstring for the bug that came of
    preferring a version number instead.
    """
    try:
        import hypernix  # noqa: F401

        return sys.executable
    except ImportError:
        pass

    # Only now the version list, for the case this was meant to cover:
    # the script is on PATH but its own interpreter lost the package.
    for version in VERSION_PRIORITY:
        if check_hypernix_installed(version.exe_name):
            return version.exe_name

        if sys.platform == "win32":
            win_exe = f"python{version.major}{version.minor}"
            if check_hypernix_installed(win_exe):
                return win_exe

    return None


def run_with_selected_python(args: list[str]) -> int:
    """Run the CLI on the selected interpreter."""
    selected = find_best_python()

    if selected is None:
        # No Python version with hypernix found, fall back to current
        from hypernix.interfaces.cli import main
        return main(args)

    if selected == sys.executable:
        # The common case now, and the cheap one: no subprocess, no
        # re-import, and the CLI runs in the interpreter that owns this
        # script.
        from hypernix.interfaces.cli import main
        return main(args)

    # Landing somewhere else means this interpreter lost the package.
    # Say which install is answering when it is a different version --
    # running an older hypernix than the one just installed, with no
    # indication, is the bug this whole ordering exists to prevent.
    try:
        import hypernix

        mine = hypernix.__version__
    except Exception:  # noqa: BLE001
        mine = ""
    theirs = installed_version(selected)
    if theirs and theirs != mine:
        print(
            f"hypernix: running {theirs} from {selected} "
            f"({sys.executable} has "
            f"{mine or 'no hypernix'}).",
            file=sys.stderr,
        )

    cmd = [selected, "-m", "hypernix"] + args
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        # Selected executable not found, fall back to current
        from hypernix.interfaces.cli import main
        return main(args)


def _maybe_setup_path() -> None:
    """Offer to fix PATH once, from the console-script entry point only.

    Guarded by an environment variable rather than a module-level flag
    because :func:`run_with_selected_python` re-execs ``python -m hypernix``
    in a child process — without the marker, both the parent and the child
    would run the check and the person would see the notice twice.

    Any failure here is swallowed: a convenience that could stop ``hypernix
    --help`` from running would be a much worse bug than an unfixed PATH.
    """
    import os

    if os.environ.get("HYPERNIX_PATH_SETUP_RAN"):
        return
    os.environ["HYPERNIX_PATH_SETUP_RAN"] = "1"
    try:
        from hypernix.system.pathfix import maybe_autoconfigure
        maybe_autoconfigure()
    except Exception:  # noqa: BLE001
        pass


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the multi-version launcher."""
    import os

    raw = list(sys.argv[1:] if argv is None else argv)

    _maybe_setup_path()
    
    # Check if we should use the launcher logic
    # Skip if HYPERNOX_NO_VERSION_CHECK is set (for development)
    if os.environ.get("HYPERNIX_NO_VERSION_CHECK"):
        from hypernix.interfaces.cli import main as cli_main
        return cli_main(raw)
    
    # If running as module directly, just use current Python
    if "__main__.py" in sys.argv[0]:
        from hypernix.interfaces.cli import main as cli_main
        return cli_main(raw)
    
    # For console script entry points, check and potentially re-exec
    return run_with_selected_python(raw)


if __name__ == "__main__":
    raise SystemExit(main())
