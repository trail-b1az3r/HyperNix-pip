"""``hnx sandbox`` (``hnx sb``) — run a command shut inside one directory.

::

    hnx sb ./project                         # a shell, with the default settings
    hnx sb ./project -- python train.py      # one command
    hnx sb ./project -s no-web,no-write      # change the settings (kept for next time)
    hnx sb ./project -s                      # change them interactively
    hnx sb ./project --show                  # what is set, and how this machine enforces it

Settings
--------
========== =================== ==================================================
setting    values (default)    meaning
========== =================== ==================================================
web        **off** | on        off: no network at all (no web access)
write      **dir** | none | all dir: create and edit files only inside the
                               directory; none: no file editing or creation
                               anywhere; all: no limit
tmp        **on** | off        a private, empty /tmp that disappears afterwards
home       **hide** | show     hide: your home folder looks empty (the
                               directory itself stays visible)
env        **clean** | keep    clean: variables that look like secrets (KEY,
                               TOKEN, SECRET, PASSWORD, ...) are not passed in
timeout    seconds, 0 = none   kill the command after this long
memory     MB, 0 = none        most memory the command may allocate
cpu        seconds, 0 = none   most CPU time it may use
filesize   MB, 0 = none        largest file it may write
========== =================== ==================================================

``-s`` takes ``name=value`` pairs separated by commas or spaces, and the
short forms ``no-web``, ``web``, ``no-write`` (= ``write=none``),
``write-dir``, ``no-tmp``, ``show-home``, ``keep-env``. What ``-s`` sets
is saved for that directory in ``~/.config/hypernix/sandbox.json`` --
outside the sandbox, so the command inside cannot loosen its own
settings -- unless ``--no-save`` is given.

How it is enforced
------------------
* **Linux**: ``unshare`` (util-linux) and unprivileged user namespaces.
  A network namespace has no interfaces (``web=off``); a mount namespace
  remounts everything read-only except the directory (``write``), and
  mounts empty tmpfs over ``/tmp`` and the home folder. The command runs
  as you, not as root. See :mod:`hypernix.interfaces.sandbox_launch`.
* **macOS**: ``sandbox-exec`` with a generated profile that denies
  network access, writes and reads of the home folder as set.
* **Anywhere else**, or where these are missing, the settings that need
  them are **refused**, never quietly skipped: a sandbox you think has no
  web access must not have it. ``env``, ``timeout`` and the limits work
  everywhere.

It is a boundary for programs that are careless, not a VM: the command
still runs as your user, and can read what you can read outside the
home folder.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

__all__ = ["SandboxSettings", "SettingsError", "parse_settings", "plan_for", "cli_main"]

SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSW|PASSPHRASE|CREDENTIAL|COOKIE|SESSION|AUTH)",
                         re.IGNORECASE)
#: Kept even with ``env=clean``: a shell needs them and none is a secret.
_ALWAYS_KEPT = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "LC_ALL",
                "COLORTERM", "TZ", "TMPDIR", "PWD"}


class SettingsError(ValueError):
    pass


def _choice(*allowed: str) -> Callable[[str], str]:
    def parse(value: str) -> str:
        value = value.strip().lower()
        if value not in allowed:
            raise SettingsError(f"must be one of {' | '.join(allowed)}, not {value!r}")
        return value
    return parse


def _amount(high: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise SettingsError(f"must be a whole number, not {value!r}") from None
        if not 0 <= number <= high:
            raise SettingsError(f"must be between 0 and {high}")
        return number
    return parse


_PARSERS: dict[str, Callable[[str], Any]] = {
    "web": _choice("off", "on"),
    "write": _choice("dir", "none", "all"),
    "tmp": _choice("on", "off"),
    "home": _choice("hide", "show"),
    "env": _choice("clean", "keep"),
    "timeout": _amount(7 * 24 * 3600),
    "memory": _amount(1024 * 1024),
    "cpu": _amount(7 * 24 * 3600),
    "filesize": _amount(1024 * 1024),
}

_HELP = {
    "web": "network access (off = no web access)",
    "write": "dir = edit/create only in the directory, none = no file editing or creation, all = no limit",
    "tmp": "a private /tmp that disappears afterwards",
    "home": "hide = your home folder looks empty",
    "env": "clean = secret-looking variables are not passed in",
    "timeout": "seconds before the command is killed (0 = none)",
    "memory": "MB the command may allocate (0 = none)",
    "cpu": "CPU seconds (0 = none)",
    "filesize": "largest file in MB (0 = none)",
}

_SHORT = {
    "no-web": ("web", "off"), "noweb": ("web", "off"), "offline": ("web", "off"),
    "web": ("web", "on"), "online": ("web", "on"),
    "no-write": ("write", "none"), "nowrite": ("write", "none"), "read-only": ("write", "none"),
    "readonly": ("write", "none"), "ro": ("write", "none"), "no-edit": ("write", "none"),
    "write-dir": ("write", "dir"), "write-all": ("write", "all"),
    "no-tmp": ("tmp", "off"), "tmp": ("tmp", "on"),
    "show-home": ("home", "show"), "hide-home": ("home", "hide"),
    "keep-env": ("env", "keep"), "clean-env": ("env", "clean"),
}


@dataclass(frozen=True)
class SandboxSettings:
    web: str = "off"
    write: str = "dir"
    tmp: str = "on"
    home: str = "hide"
    env: str = "clean"
    timeout: int = 0
    memory: int = 0
    cpu: int = 0
    filesize: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SandboxSettings:
        known = {f.name for f in fields(cls)}
        clean = {}
        for key, value in data.items():
            if key in known:
                try:
                    clean[key] = _PARSERS[key](str(value))
                except SettingsError:
                    continue  # a hand-edited bad value keeps the default
        return cls(**clean)

    def needs_isolation(self) -> list[str]:
        """The settings that only an OS sandbox can enforce."""
        out = []
        if self.web == "off":
            out.append("web=off")
        if self.write != "all":
            out.append(f"write={self.write}")
        if self.tmp == "on":
            out.append("tmp=on")
        if self.home == "hide":
            out.append("home=hide")
        return out


def parse_settings(text: str, base: SandboxSettings | None = None) -> SandboxSettings:
    """``"no-web, write=none timeout=60"`` applied on top of *base*."""
    changes: dict[str, Any] = {}
    for token in re.split(r"[,\s]+", text.strip()):
        if not token:
            continue
        if "=" in token:
            name, _, value = token.partition("=")
            name = name.strip().lower()
            if name not in _PARSERS:
                raise SettingsError(f"unknown setting {name!r}; the settings are: "
                                    + ", ".join(_PARSERS))
            try:
                changes[name] = _PARSERS[name](value)
            except SettingsError as exc:
                raise SettingsError(f"{name}: {exc}") from None
        elif token.lower() in _SHORT:
            name, value = _SHORT[token.lower()]
            changes[name] = value
        else:
            raise SettingsError(f"unknown setting {token!r}; use name=value ("
                                + ", ".join(_PARSERS) + ") or one of " + ", ".join(sorted(_SHORT)))
    return replace(base or SandboxSettings(), **changes)


# -- saved settings ----------------------------------------------------------

def _store_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "hypernix" / "sandbox.json"


def _load_all() -> dict[str, Any]:
    try:
        data = json.loads(_store_path().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_settings(directory: Path) -> SandboxSettings:
    saved = _load_all().get(str(directory))
    return SandboxSettings.from_dict(saved) if isinstance(saved, dict) else SandboxSettings()


def save_settings(directory: Path, settings: SandboxSettings | None) -> None:
    data = _load_all()
    if settings is None:
        data.pop(str(directory), None)
    else:
        data[str(directory)] = settings.to_dict()
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


# -- what this machine can do --------------------------------------------------

def backend() -> str:
    """``"linux"``, ``"macos"`` or ``""`` (nothing to enforce isolation with)."""
    if sys.platform.startswith("linux"):
        unshare = shutil.which("unshare")
        if unshare:
            try:
                ok = subprocess.run([unshare, "-r", "-m", "-n", "true"], capture_output=True,
                                    timeout=10).returncode == 0
            except (OSError, subprocess.SubprocessError):
                ok = False
            if ok:
                return "linux"
        return ""
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return "macos"
    return ""


def _environment(settings: SandboxSettings) -> dict[str, str]:
    env = dict(os.environ)
    if settings.env == "clean":
        env = {k: v for k, v in env.items() if k in _ALWAYS_KEPT or not SECRET_NAME.search(k)}
    env["HNX_SANDBOX"] = "1"
    if settings.tmp == "on" and sys.platform.startswith("linux"):
        env["TMPDIR"] = "/tmp"
    return env


def _sbpl_string(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def seatbelt_profile(directory: Path, settings: SandboxSettings, home: str) -> str:
    """The ``sandbox-exec`` profile for these settings (macOS)."""
    d = _sbpl_string(str(directory))
    rules = ["(version 1)", "(allow default)"]
    if settings.web == "off":
        rules += ["(deny network*)", '(allow network* (remote unix-socket))',
                  '(allow network* (local unix-socket))']
    if settings.write != "all":
        rules += ["(deny file-write*)",
                  '(allow file-write* (literal "/dev/null") (literal "/dev/zero")'
                  ' (regex #"^/dev/tty") (regex #"^/dev/fd/") (literal "/dev/dtracehelper"))']
        if settings.write == "dir":
            rules.append(f"(allow file-write* (subpath {d}))")
        if settings.tmp == "on":
            rules.append('(allow file-write* (subpath "/private/tmp") (subpath "/private/var/folders"))')
    if settings.home == "hide" and home:
        rules += [f"(deny file-read* (subpath {_sbpl_string(home)}))",
                  f"(allow file-read* (literal {_sbpl_string(home)}))",
                  f"(allow file-read* (subpath {d}))"]
    return "\n".join(rules) + "\n"


def plan_for(directory: Path, settings: SandboxSettings, argv: list[str], *,
             how: str) -> list[str]:
    """The argv that runs *argv* in the sandbox with *how* (a :func:`backend`)."""
    home = os.path.realpath(os.path.expanduser("~"))
    if how == "linux":
        plan = {
            "dir": str(directory), "write": settings.write, "tmp": settings.tmp == "on",
            "hide_home": settings.home == "hide", "home_path": home,
            "uid": os.getuid(), "gid": os.getgid(), "argv": argv,
            "env": _environment(settings), "memory": settings.memory, "cpu": settings.cpu,
            "filesize": settings.filesize,
        }
        launcher = Path(__file__).with_name("sandbox_launch.py")
        command = [shutil.which("unshare") or "unshare", "-r", "-m"]
        if settings.web == "off":
            command.append("-n")
        return [*command, "--propagation", "private", "--", sys.executable, "-I", str(launcher),
                base64.b64encode(json.dumps(plan).encode()).decode()]
    if how == "macos":
        return ["sandbox-exec", "-p", seatbelt_profile(directory, settings, home), *argv]
    return list(argv)


def describe(settings: SandboxSettings, how: str) -> list[tuple[str, str, str]]:
    """``(setting, value, how it is enforced here)`` for ``--show``."""
    by = {"linux": "Linux namespaces (unshare)", "macos": "macOS sandbox-exec", "": ""}[how]
    rows = []
    for name in _PARSERS:
        value = getattr(settings, name)
        if name in ("web", "write", "tmp", "home"):
            loose = {"web": "on", "write": "all", "tmp": "off", "home": "show"}[name]
            if value == loose:
                enforced = "nothing to enforce"
            else:
                enforced = by or "REFUSED here: no unshare/sandbox-exec"
        elif name == "env":
            enforced = "always"
        elif name == "timeout":
            enforced = "always" if value else "no limit"
        else:
            enforced = ("always" if os.name == "posix" else "not on this OS") if value else "no limit"
        rows.append((name, str(value), enforced))
    return rows


# -- the command -------------------------------------------------------------

def _interactive(settings: SandboxSettings) -> SandboxSettings:
    if not sys.stdin.isatty():
        raise SettingsError("-s with no settings asks for them, which needs a terminal; "
                            "give them instead, e.g. -s no-web,no-write")
    print("Sandbox settings: press Enter to keep a value.\n")
    changes: dict[str, Any] = {}
    for name, parse in _PARSERS.items():
        current = getattr(settings, name)
        while True:
            answer = input(f"  {name:<9} [{current}]  {_HELP[name]}: ").strip()
            if not answer:
                break
            try:
                changes[name] = parse(answer)
                break
            except SettingsError as exc:
                print(f"    {exc}")
    return replace(settings, **changes)


def _print_settings(directory: Path, settings: SandboxSettings, how: str) -> None:
    print(f"hnx sandbox {directory}")
    for name, value, enforced in describe(settings, how):
        print(f"  {name:<9} {value:<6} {enforced}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hnx sandbox",
        description="Run a command (or a shell) shut inside one directory. Also: hnx sb.",
        epilog="settings: " + "; ".join(f"{k}: {v}" for k, v in _HELP.items())
        + ". Short forms: no-web, no-write, no-tmp, show-home, keep-env.")
    parser.add_argument("directory", help="the directory to work in (created if missing)")
    parser.add_argument("-s", "--settings", nargs="?", const="", default=None, metavar="SETTINGS",
                        help="change settings, e.g. -s no-web,no-write,timeout=600; "
                             "-s alone asks for each one")
    parser.add_argument("--no-save", action="store_true",
                        help="use -s for this run only, without saving it for the directory")
    parser.add_argument("--show", action="store_true",
                        help="print the settings and how this machine enforces them, then stop")
    parser.add_argument("--reset", action="store_true",
                        help="forget the directory's saved settings (back to the defaults)")
    parser.add_argument("command", nargs="*",
                        help="the command to run; put it after -- when it has options of "
                             "its own. Default: your shell")
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    tail: list[str] = []
    if "--" in argv:  # everything after -- is the command, options and all
        cut = argv.index("--")
        argv, tail = argv[:cut], argv[cut + 1:]
    args = _parser().parse_args(argv)
    args.command = [*args.command, *tail]
    directory = Path(args.directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve()

    if args.reset:
        save_settings(directory, None)
        print(f"hnx sandbox: {directory} is back to the default settings.")
        if not args.command and args.settings is None:
            return 0
    settings = load_settings(directory)
    try:
        if args.settings is not None:
            settings = (_interactive(settings) if args.settings == ""
                        else parse_settings(args.settings, settings))
            if not args.no_save:
                save_settings(directory, settings)
    except SettingsError as exc:
        print(f"hnx sandbox: {exc}", file=sys.stderr)
        return 2

    how = backend() if settings.needs_isolation() else ""
    if args.show:
        _print_settings(directory, settings, how)
        return 0

    needed = settings.needs_isolation()
    if needed and not how:
        print("hnx sandbox: this machine has nothing to enforce "
              + ", ".join(needed) + " with (Linux needs `unshare` and unprivileged user "
              "namespaces; macOS needs sandbox-exec). Refusing to run without them rather "
              "than pretend.\nTo run with them loosened: hnx sb "
              f"{directory} -s web=on,write=all,tmp=off,home=show", file=sys.stderr)
        return 3

    command = list(args.command)
    if not command:
        command = [os.environ.get("SHELL") or "/bin/sh"]
    if settings.write == "dir" and directory == Path(os.path.realpath(os.path.expanduser("~"))):
        print("hnx sandbox: the directory is your home folder; write=dir would let the "
              "command change everything in it. Use a folder inside it.", file=sys.stderr)
        return 2

    full = plan_for(directory, settings, command, how=how)
    # On Linux the launcher sets the limits inside the namespaces; elsewhere
    # they are set in the child here, before it execs.
    limits_here = how != "linux" and os.name == "posix" and bool(
        settings.memory or settings.cpu or settings.filesize)

    def _limits() -> None:
        import resource

        for value, which, scale in ((settings.memory, "RLIMIT_DATA", 1024 * 1024),
                                    (settings.cpu, "RLIMIT_CPU", 1),
                                    (settings.filesize, "RLIMIT_FSIZE", 1024 * 1024)):
            if value and hasattr(resource, which):
                resource.setrlimit(getattr(resource, which), (value * scale, value * scale))

    try:
        proc = subprocess.Popen(  # noqa: S603 - an argv list, never a shell
            full, cwd=str(directory) if how != "linux" else "/",
            env=None if how == "linux" else _environment(settings),
            start_new_session=bool(settings.timeout),
            preexec_fn=_limits if limits_here else None)
    except FileNotFoundError as exc:
        print(f"hnx sandbox: {exc.filename}: not found", file=sys.stderr)
        return 127
    try:
        return proc.wait(timeout=settings.timeout or None)
    except subprocess.TimeoutExpired:
        print(f"hnx sandbox: stopped after {settings.timeout} s (timeout)", file=sys.stderr)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, AttributeError):
            proc.kill()
        proc.wait()
        return 124
    except KeyboardInterrupt:
        proc.wait()
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
