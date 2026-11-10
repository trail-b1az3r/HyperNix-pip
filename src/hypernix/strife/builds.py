"""strife.builds — Strife builds a user made, registered for the T1 runner.

The ``strife`` runner backend serves with the Strife that ships in this
package. A user who builds their own (a source checkout with their
changes, or HyperNix installed in another interpreter) registers it once::

    hnx strife build --source ~/src/HyperNix-pip --name mine
    hnx strife build --python ~/venvs/strife/bin/python --name fast

Registration *proves* the build before recording it: the build's own
interpreter imports its own ``hypernix.strife``, runs its server's
``--selftest`` (a generated model through load, prefill and decode), and
reports its version. Only a build that passes is written to
``~/.hypernix/strife/builds.json``.

The T1 API lists registered builds (``GET /runner/status`` ->
``strife_builds``) and loads with one by name
(``POST /runner/load {"backend": "strife", "strife_build": "mine"}``).
Only names are accepted over the API -- never a path or an interpreter --
so the API can start what the machine's owner registered and nothing
else. The registry file is written owner-only and refused if anyone else
can write it.
"""
from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import StrifeError

__all__ = ["StrifeBuild", "registry_path", "list_builds", "get_build", "register",
           "remove", "default_build", "launch_argv", "launch_env", "BUILTIN"]

#: The name of the Strife in this package; always available, never stored.
BUILTIN = "builtin"
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,47}$")
_SELFTEST_TIMEOUT = 300.0


@dataclass
class StrifeBuild:
    name: str
    python: str
    #: A source tree put first on PYTHONPATH (its ``src`` when it has one);
    #: empty when the interpreter has HyperNix installed itself.
    source: str = ""
    version: str = ""
    registered_at: float = 0.0
    selftest: dict[str, Any] = field(default_factory=dict)
    default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def public(self) -> dict[str, Any]:
        """What the T1 API shows: the facts, not the selftest's full output."""
        return {"name": self.name, "version": self.version, "python": self.python,
                "source": self.source, "registered_at": self.registered_at,
                "default": self.default, "selftest_ok": bool(self.selftest.get("ok"))}


def registry_path() -> Path:
    base = os.environ.get("HYPERNIX_HOME") or str(Path.home() / ".hypernix")
    return Path(base) / "strife" / "builds.json"


def _check_owner(path: Path) -> None:
    """Refuse a registry another user could have written: it names programs to run."""
    if os.name != "posix" or not path.exists():
        return
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise StrifeError("strife.build.unsafe_registry",
                          f"{path} is writable by someone other than its owner, so the builds "
                          "it names cannot be trusted",
                          remedy=f"chmod 600 {path} (and check what it lists), or delete it.")


def _read() -> list[StrifeBuild]:
    path = registry_path()
    if not path.is_file():
        return []
    _check_owner(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StrifeError("strife.build.bad_registry", f"{path} could not be read: {exc}") from exc
    builds = []
    for item in data.get("builds", []) if isinstance(data, dict) else []:
        if isinstance(item, dict) and _NAME.match(str(item.get("name", ""))):
            known = {k: item[k] for k in StrifeBuild.__dataclass_fields__ if k in item}
            builds.append(StrifeBuild(**known))
    return builds


def _write(builds: list[StrifeBuild]) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"version": 1, "builds": [b.to_dict() for b in builds]}, handle, indent=2)
    os.replace(tmp, path)


def list_builds() -> list[StrifeBuild]:
    return _read()


def get_build(name: str) -> StrifeBuild | None:
    if name in ("", BUILTIN):
        return None
    for build in _read():
        if build.name == name:
            return build
    raise StrifeError("strife.build.unknown", f"no Strife build is registered as {name!r}",
                      remedy="`hnx strife build --list` shows the registered builds.")


def default_build() -> StrifeBuild | None:
    """The build the runner uses when a load names none (``None``: the built-in one)."""
    wanted = os.environ.get("HNX_STRIFE_BUILD", "").strip()
    if wanted:
        return get_build(wanted)
    return next((b for b in _read() if b.default), None)


def launch_env(build: StrifeBuild | None, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    if build is not None and build.source:
        env["PYTHONPATH"] = build.source + (os.pathsep + env["PYTHONPATH"]
                                            if env.get("PYTHONPATH") else "")
    return env


def launch_argv(build: StrifeBuild | None) -> list[str]:
    """``python -m hypernix.strife.server`` for *build* (or this package's)."""
    python = build.python if build is not None else sys.executable
    return [python, "-m", "hypernix.strife.server"]


def _source_root(source: Path) -> Path:
    """A checkout's import root: ``src`` when the package lives there."""
    for root in (source / "src", source):
        if (root / "hypernix" / "strife" / "server.py").is_file():
            return root
    raise StrifeError("strife.build.no_source",
                      f"{source} has no hypernix/strife/server.py (looked in it and in src/)",
                      remedy="Point --source at a HyperNix checkout.")


def _selftest(build: StrifeBuild) -> dict[str, Any]:
    argv = [*launch_argv(build), "--selftest"]
    started = time.monotonic()
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=_SELFTEST_TIMEOUT,
                              env=launch_env(build), check=False)
    except FileNotFoundError as exc:
        raise StrifeError("strife.build.no_python", f"{build.python} could not be run: {exc}",
                          remedy="Pass --python the path to the interpreter.") from exc
    except subprocess.TimeoutExpired as exc:
        raise StrifeError("strife.build.selftest_failed",
                          f"the build's selftest did not finish in {_SELFTEST_TIMEOUT:.0f}s") from exc
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    try:
        report = json.loads(lines[-1]) if lines else {}
    except ValueError:
        report = {}
    if done.returncode != 0 or not report.get("ok"):
        tail = (done.stderr or done.stdout)[-1500:]
        raise StrifeError("strife.build.selftest_failed",
                          f"the build failed its selftest (exit {done.returncode}):\n{tail}",
                          remedy="Fix the build and register it again; nothing was recorded.")
    report["seconds"] = round(time.monotonic() - started, 2)
    return report


def register(name: str, *, source: str | Path | None = None, python: str | None = None,
             make_default: bool = False) -> StrifeBuild:
    """Selftest a build and record it. Raises :class:`StrifeError` and records nothing on failure."""
    name = name.strip().lower()
    if not _NAME.match(name) or name == BUILTIN:
        raise StrifeError("strife.build.bad_name",
                          f"{name!r} is not a build name: lower-case letters, digits, '.', '_' and "
                          f"'-', up to 48 characters, and not {BUILTIN!r}")
    if source is None and python is None:
        raise StrifeError("strife.build.nothing", "give --source (a checkout) or --python "
                          "(an interpreter with HyperNix installed), or both")
    root = str(_source_root(Path(source).expanduser().resolve())) if source else ""
    interpreter = str(Path(python).expanduser()) if python else sys.executable
    build = StrifeBuild(name=name, python=interpreter, source=root)
    report = _selftest(build)
    build.version = str(report.get("version", ""))
    build.selftest = report
    build.registered_at = time.time()
    builds = [b for b in _read() if b.name != name]
    if make_default:
        for other in builds:
            other.default = False
    build.default = make_default
    builds.append(build)
    _write(builds)
    return build


def remove(name: str) -> bool:
    builds = _read()
    kept = [b for b in builds if b.name != name]
    if len(kept) == len(builds):
        return False
    _write(kept)
    return True
