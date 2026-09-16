"""What this server is running, and the exact commands to update it.

The report was "the T1 installed thinks it is running an older
version". Part of that was ``install-t1.sh`` printing a hand-maintained
constant that had gone stale — fixed there. The other part is real: a
server *does* fall behind, and until now the only way to find out was to
walk over to the machine, and the only way to fix it was to remember
which of several interpreters it is running under.

That last bit is the whole point of this module. Advice like "run
``pip install -U hypernix``" is worse than useless on a machine with a
system Python, a pyenv, and the venv the service actually uses: it is
advice that succeeds loudly and changes nothing, because it upgraded a
different installation. The server knows ``sys.executable``. So the
commands it hands out name it.

What this does not do
---------------------
It does not decide that an update is *needed*. Comparing against PyPI
means a network call from somebody's home machine to answer a question
they did not ask, and :mod:`hypernix.system.release` already owns that
decision and its opt-outs. This reports what is installed and how to
change it; whether to is the reader's.
"""
from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: What a `pip install` would be told to fetch.
PACKAGE = "hypernix"


@dataclass(frozen=True)
class Installation:
    """Where this server's code lives, and how it got there."""

    #: The interpreter running the server. The one number that makes the
    #: commands below correct rather than plausible.
    executable: str = ""
    #: The environment prefix — a venv root, or the system prefix.
    prefix: str = ""
    #: True when the interpreter is inside a virtual environment.
    in_venv: bool = False
    #: True when hypernix is installed in editable/development mode, in
    #: which case pip will not update it and saying so beats a command
    #: that silently no-ops.
    editable: bool = False
    #: Where the package was imported from.
    location: str = ""
    python_version: str = ""
    package_version: str = ""
    t1_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "executable": self.executable,
            "prefix": self.prefix,
            "in_venv": self.in_venv,
            "editable": self.editable,
            "location": self.location,
            "python_version": self.python_version,
            "package_version": self.package_version,
            "t1_version": self.t1_version,
        }


@dataclass(frozen=True)
class Command:
    """One copyable line, and what it is for."""

    label: str
    command: str
    #: True for the one the reader most likely wants. Exactly one is.
    primary: bool = False
    #: Why this one and not another.
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "command": self.command,
            "primary": self.primary,
            "note": self.note,
        }


@dataclass(frozen=True)
class UpgradePlan:
    installation: Installation
    commands: list[Command] = field(default_factory=list)
    #: Anything that makes the commands above not enough on their own.
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "installation": self.installation.to_dict(),
            "commands": [c.to_dict() for c in self.commands],
            "warnings": list(self.warnings),
        }


def _editable_install() -> tuple[bool, str]:
    """``(is_editable, location)`` for the imported hypernix.

    Checked through the distribution metadata rather than by looking for
    a ``.pth`` file: a modern editable install (PEP 660) writes a
    ``__editable__`` finder module, an old one writes an egg-link, and
    ``direct_url.json`` describes both.
    """
    import json

    location = ""
    try:
        import hypernix

        location = str(Path(hypernix.__file__).resolve().parent)
    except Exception:  # noqa: BLE001 - reporting must not fail the server
        pass

    try:
        from importlib.metadata import distribution

        direct = distribution(PACKAGE).read_text("direct_url.json")
        if direct:
            parsed = json.loads(direct)
            if parsed.get("dir_info", {}).get("editable"):
                return True, location
    except Exception:  # noqa: BLE001
        pass
    return False, location


def describe() -> Installation:
    """This server's installation, as it actually is."""
    import hypernix

    from .version import T1_VERSION

    editable, location = _editable_install()
    return Installation(
        executable=sys.executable or "",
        prefix=sys.prefix or "",
        # `base_prefix != prefix` is the venv test that works for venv,
        # virtualenv and `python -m venv --system-site-packages` alike.
        in_venv=sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        editable=editable,
        location=location,
        python_version=".".join(str(part) for part in sys.version_info[:3]),
        package_version=getattr(hypernix, "__version__", ""),
        t1_version=T1_VERSION.short,
    )


def _service_hint() -> str:
    """How this server was probably started, when that can be told.

    A pip upgrade does not restart a running process, and an update that
    appears to succeed and changes nothing is the single most confusing
    outcome here. This is best-effort: an environment variable set by
    systemd, or nothing.
    """
    if os.environ.get("INVOCATION_ID") or os.environ.get("JOURNAL_STREAM"):
        # Set by systemd for any unit it starts.
        return "systemctl --user restart hypernix-t1"
    return ""


def plan(installation: Installation | None = None) -> UpgradePlan:
    """The commands that would update *installation*.

    Every command names the interpreter explicitly. ``pip install -U
    hypernix`` on a machine with three Pythons upgrades whichever one is
    first on the path, reports success, and leaves the server running
    the version it was — which is precisely the confusion this exists to
    end.
    """
    found = installation or describe()
    python = found.executable or sys.executable or "python3"
    quoted = shlex.quote(python)

    commands: list[Command] = []
    warnings: list[str] = []

    if found.editable:
        warnings.append(
            "This is a development install: the code is being imported from "
            f"{found.location or 'a checkout'} and pip will not replace it. "
            "Update it with git instead."
        )
        commands.append(Command(
            label="Update the checkout",
            command=f"cd {shlex.quote(found.location or '.')} && git pull",
            primary=True,
            note="An editable install runs the checkout directly, so git is "
                 "what changes the code.",
        ))
    else:
        commands.append(Command(
            label="Update HyperNix",
            command=f"{quoted} -m pip install --upgrade {PACKAGE}",
            primary=True,
            note="Names the interpreter this server is running under, so it "
                 "cannot upgrade a different Python by mistake.",
        ))
        commands.append(Command(
            label="Update to a specific version",
            command=f"{quoted} -m pip install --upgrade '{PACKAGE}==X.Y.Z'",
            note="Replace X.Y.Z. Useful for going back after an update.",
        ))

    commands.append(Command(
        label="Check what is installed now",
        command=f"{quoted} -m pip show {PACKAGE}",
        note="Confirms the upgrade landed in the environment the server uses.",
    ))

    restart = _service_hint()
    if restart:
        commands.append(Command(
            label="Restart the service",
            command=restart,
            note="A running server keeps the old code in memory until it is "
                 "restarted.",
        ))
        warnings.append(
            "Updating the package does not restart this server. Until it is "
            "restarted it keeps serving the version it started with."
        )
    else:
        warnings.append(
            "Updating the package does not restart this server — stop and "
            "start it however you started it, or the running process keeps "
            "the old code."
        )

    if not found.in_venv:
        warnings.append(
            "This server is not running inside a virtual environment, so the "
            "upgrade touches a system-wide Python. On some distributions pip "
            "refuses that without --break-system-packages."
        )

    return UpgradePlan(installation=found, commands=commands, warnings=warnings)
