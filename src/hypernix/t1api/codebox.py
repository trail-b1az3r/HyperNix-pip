"""``/code`` — sandboxes that run code, and the permissions they run under.

The endpoints, from the top of the tree down:

* ``GET /code`` — whether this server runs code, and how to ask it to
* ``POST /code/create`` — run a snippet once, in a sandbox that is gone
  when the answer comes back
* ``POST /code/create/sandbox`` — a sandbox that stays: put files in it,
  run them, read what they wrote
* ``GET /code/create/sandbox/perms`` and
  ``/code/create/sandbox/perms/<directives>`` — read and change what
  every sandbox may do, in the same grammar as ``/web/v1/config``

This module is the part with no FastAPI in it, as
:mod:`hypernix.t1api.websearch` is for ``/web/v1``.

The permissions grammar
-----------------------
The web settings' grammar, word for word, so there is one to learn::

    /code/create/sandbox/perms/s2?:=on|s3?:=120|s1?=k

* ``?`` marks a setting, ``=k`` keeps it, ``?:=`` sets it, ``|`` divides.
* ``;`` divides as well, and a clause may follow ``perms`` directly:
  ``perms|s1?=k;s2?:=on``.
* ``s1=?`` asks for a setting, the same as ``s1?=k``; so does ``s1=``,
  which is what ``s1=?`` becomes when it ends the URL and HTTP drops
  the ``?``.
* ``s2=on`` -- an ``=`` with no ``?`` and no colon -- is refused, for the
  reason the web grammar refuses ``s2?=google``: it reads like a change,
  and a grammar that quietly treated it as one or as nothing would leave
  somebody certain a sandbox had no network when it had.

Settings are numbered, and each has a name that works as well::

    s1 execute    on | off         whether code runs at all
    s2 network    off | on         off: no network inside the sandbox
    s3 timeout    1..600           seconds before a run is killed
    s4 languages  python,bash,...  which interpreters may be used
    s5 disk       1..4096          MB a sandbox may hold
    s6 memory     32..65536        MB a run may allocate
    s7 ttl        1..1440          minutes an idle sandbox is kept

The boundary
------------
Running code sent over an API is running code on this machine, so:

* Nothing runs unless the operator set ``T1_CODE_SANDBOX=1``. The
  permissions narrow what that allows; they cannot turn it on.
* A run never goes through a shell. The code is written to a file and
  its interpreter is started with an argv list.
* Every file path is resolved inside the sandbox, symlinks included,
  by noodle's :class:`~hypernix.interfaces.noodle.tools.ToolContext`.
* The environment is minimal: no API keys, no tokens, ``HOME`` is the
  sandbox.
* Limits are set in the child before the interpreter starts (a small
  launcher, not ``preexec_fn``, which is unsafe in a threaded server):
  memory by ``RLIMIT_DATA`` (``RLIMIT_AS`` refuses Node outright, which
  reserves gigabytes of address space it never uses), file size by
  ``RLIMIT_FSIZE``, and CPU time by ``RLIMIT_CPU``. A limit the kernel
  refuses (macOS takes no memory limit) is listed under ``not_enforced``
  in the run's ``limits``, never claimed. The wall-clock timeout kills
  the whole process group.
* ``network off`` is enforced with a new network namespace
  (``unshare -rn``), which has no interface but a downed loopback. Where
  that is not available -- macOS, Windows, a kernel without user
  namespaces -- a run with the network off is **refused**, not run with
  the network on. ``s2?:=on`` is how an operator accepts that.

It is a boundary for code that is wrong, not for code that is hostile.
It is not a VM or a container; code in a sandbox runs as the server's
user, and can read what that user can read outside it. That is why it
is off by default and needs an operator to create sandboxes.
"""
from __future__ import annotations

import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .websearch import GrammarError, parse_directives

logger = logging.getLogger(__name__)

__all__ = [
    "LANGUAGES",
    "SETTINGS",
    "CodePerms",
    "GrammarError",
    "RunResult",
    "Sandbox",
    "SandboxError",
    "SandboxStore",
    "apply_perms",
    "available_languages",
    "network_isolation",
]

#: ``name -> (interpreter, file suffix)``. Python is this server's own
#: interpreter, so a sandbox has the packages the server has.
LANGUAGES: dict[str, tuple[str, str]] = {
    "python": (sys.executable or "python3", ".py"),
    "bash": ("bash", ".sh"),
    "sh": ("sh", ".sh"),
    "node": ("node", ".js"),
    "fish": ("fish", ".fish"),
}

#: Per stream. A build log's tail, not a way to move files.
MAX_OUTPUT = 200_000
#: Largest file read back through the API.
MAX_READ = 1_000_000
#: Longest snippet accepted by ``/code/create``.
MAX_CODE = 500_000
#: Sandboxes one caller may hold at once.
MAX_PER_OWNER = 8

_ID = re.compile(r"^sbx_[0-9a-f]{16}$")
_ON = {"on", "true", "yes", "1", "allow", "allowed"}
_OFF = {"off", "false", "no", "0", "deny", "denied"}


class SandboxError(ValueError):
    """A request the sandbox refused. ``code`` says which kind."""

    def __init__(self, message: str, *, code: str = "invalid", status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodePerms:
    """What every sandbox may do. The defaults are the cautious ones."""

    execute: bool = True
    network: bool = False
    timeout: int = 30
    languages: tuple[str, ...] = ("python", "bash")
    disk_mb: int = 64
    memory_mb: int = 512
    ttl_minutes: int = 60

    def to_dict(self) -> dict[str, Any]:
        return {
            f"s{number}": {"name": name, "value": _show(getattr(self, attr)), "means": means}
            for number, (name, attr, _parse, means) in SETTINGS.items()
        }


def _show(value: Any) -> Any:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, tuple):
        return ",".join(value)
    return value


def _switch(value: str) -> bool:
    word = value.strip().lower()
    if word in _ON:
        return True
    if word in _OFF:
        return False
    raise GrammarError(f"{value!r} is not on or off")


def _between(low: int, high: int):
    def parse(value: str) -> int:
        try:
            number = int(value.strip())
        except ValueError:
            raise GrammarError(f"{value!r} is not a whole number") from None
        if not low <= number <= high:
            raise GrammarError(f"{number} is outside {low}..{high}")
        return number
    return parse


def _languages(value: str) -> tuple[str, ...]:
    names = [n.strip().lower() for n in re.split(r"[,+ ]", value) if n.strip()]
    if names == ["all"]:
        return tuple(LANGUAGES)
    unknown = [n for n in names if n not in LANGUAGES]
    if unknown:
        raise GrammarError(f"unknown language {', '.join(unknown)}; "
                           f"known: {', '.join(LANGUAGES)}, or all")
    if not names:
        raise GrammarError("no languages given; to stop code running, set s1?:=off")
    return tuple(dict.fromkeys(names))


#: ``number -> (name, CodePerms field, parser, what it means)``.
SETTINGS: dict[int, tuple[str, str, Any, str]] = {
    1: ("execute", "execute", _switch, "whether code runs at all"),
    2: ("network", "network", _switch, "off: no network inside the sandbox"),
    3: ("timeout", "timeout", _between(1, 600), "seconds before a run is killed"),
    4: ("languages", "languages", _languages, "interpreters that may be used"),
    5: ("disk", "disk_mb", _between(1, 4096), "MB of files one sandbox may hold"),
    6: ("memory", "memory_mb", _between(32, 65536), "MB one run may allocate"),
    7: ("ttl", "ttl_minutes", _between(1, 1440), "minutes an idle sandbox is kept"),
}

_BY_NAME: dict[str, int] = {}
for _number, (_name, _attr, _p, _m) in SETTINGS.items():
    _BY_NAME[f"s{_number}"] = _number
    _BY_NAME[_name] = _number
    _BY_NAME[_attr] = _number

GRAMMAR = (
    "s1?=k|s2?:=on|s3?:=120  -- `?` marks a setting, `=k` keeps it, `?:=` sets it, "
    "`|` or `;` divides; `s1=?` or `s1=` reads one. s1 execute, s2 network, s3 timeout, "
    "s4 languages, s5 disk, s6 memory, s7 ttl"
)


def _normalise(raw: str) -> str:
    """The user-facing spellings, rewritten into the web grammar."""
    text = (raw or "").strip().lstrip("/|;").strip()
    clauses = []
    for clause in re.split(r"[|;]", text):
        clause = clause.strip()
        if not clause:
            continue
        # `s1=?` reads. So does `s1=`: a `?` that ends a URL is dropped
        # by HTTP with the empty query after it, and `perms|s2=?` arrives
        # as `perms|s2=` -- the two cannot be told apart on the server.
        asked = re.fullmatch(r"([A-Za-z0-9_]+)\s*=\s*\??", clause)
        if asked:
            clause = f"{asked.group(1)}?=k"
        elif "?" not in clause and "=" in clause:
            name, _, value = clause.partition("=")
            raise GrammarError(
                f"{clause!r} is not a change: a setting is changed with `?:=`. "
                f"Write `{name.strip()}?:={value.strip()}` to set it, or "
                f"`{name.strip()}=?` to read it."
            )
        clauses.append(clause)
    return "|".join(clauses)


def apply_perms(perms: CodePerms, raw: str) -> tuple[CodePerms, list[str], list[str]]:
    """Apply a directive string. Returns ``(new perms, changed, asked)``.

    All or nothing: a clause that is refused leaves every setting as it
    was, including the ones before it.
    """
    directives = parse_directives(_normalise(raw), separator="|")
    changes: dict[str, Any] = {}
    changed: list[str] = []
    asked: list[str] = []
    for directive in directives:
        number = _BY_NAME.get(directive.name)
        if number is None:
            raise GrammarError(
                f"there is no setting {directive.name!r}; they are s1..s{len(SETTINGS)} "
                f"({', '.join(name for name, *_ in SETTINGS.values())})"
            )
        name, attr, parse, _means = SETTINGS[number]
        label = f"s{number}"
        if directive.action == "keep":
            asked.append(label)
            continue
        try:
            value = parse(directive.value)
        except GrammarError as exc:
            raise GrammarError(f"s{number} ({name}): {exc}") from None
        changes[attr] = value
        if value != getattr(perms, attr):
            changed.append(label)
    return replace(perms, **changes), changed, asked


# ---------------------------------------------------------------------------
# What the machine can do
# ---------------------------------------------------------------------------

_NETNS: list[str] | None | bool = False   # False: not probed yet
_NETNS_LOCK = threading.Lock()


def network_isolation() -> list[str] | None:
    """The argv prefix that runs a command with no network, or ``None``.

    Probed once, by running ``true`` in a new network namespace: whether
    unprivileged user namespaces work is a kernel setting, and finding
    ``unshare`` on PATH does not answer it.
    """
    global _NETNS
    with _NETNS_LOCK:
        if _NETNS is False:
            _NETNS = None
            binary = shutil.which("unshare") if sys.platform.startswith("linux") else None
            if binary:
                try:
                    ok = subprocess.run(  # noqa: S603 - fixed argv
                        [binary, "-rn", "true"], capture_output=True, timeout=5, check=False,
                    ).returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    ok = False
                if ok:
                    _NETNS = [binary, "-rn"]
        return _NETNS  # type: ignore[return-value]


def available_languages() -> dict[str, str]:
    """``name -> interpreter path`` for the languages installed here."""
    found = {}
    for name, (interpreter, _suffix) in LANGUAGES.items():
        path = interpreter if os.path.isabs(interpreter) else shutil.which(interpreter)
        if path and os.path.exists(path):
            found[name] = path
    return found


# Sets the limits, then becomes the interpreter. Run as a separate
# process rather than through preexec_fn, which may deadlock in a
# threaded program -- and the T1 server is one. A limit the kernel
# refuses (macOS takes no data-segment limit: EINVAL) is written to the
# report file and the run goes on, so the result can say which limits
# held instead of every run failing.
_LAUNCHER = """\
import os, sys
refused = []
try:
    import resource
    mem, fsize, cpu = (int(v) for v in sys.argv[1:4])
    data = getattr(resource, "RLIMIT_DATA", resource.RLIMIT_AS)
    for name, which, value in (("memory_bytes", data, mem), ("file_bytes", resource.RLIMIT_FSIZE, fsize),
                               ("cpu_seconds", resource.RLIMIT_CPU, cpu)):
        if value > 0:
            try:
                resource.setrlimit(which, (value, value))
            except (ValueError, OSError):
                refused.append(name)
except ImportError:
    refused = ["memory_bytes", "file_bytes", "cpu_seconds"]
if refused:
    with open(sys.argv[4], "w") as report:
        report.write(" ".join(refused))
os.execvp(sys.argv[5], sys.argv[5:])
"""


# ---------------------------------------------------------------------------
# Sandboxes
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    language: str
    exit_code: int | None
    stdout: str
    stderr: str
    elapsed: float
    timed_out: bool = False
    truncated: bool = False
    network: bool = False
    limits: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed_seconds": round(self.elapsed, 3),
            "timed_out": self.timed_out,
            "truncated": self.truncated,
            "network": "on" if self.network else "off",
            "limits": self.limits,
        }


@dataclass
class Sandbox:
    id: str
    owner: str
    root: Path
    name: str = ""
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def files(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                rel = path.relative_to(self.root).as_posix()
                if not rel.startswith(".hnx/"):
                    out.append({"path": rel, "bytes": path.stat().st_size})
        return out

    def bytes_used(self) -> int:
        total = 0
        for dirpath, _dirs, names in os.walk(self.root):   # never follows links
            for name in names:
                try:
                    total += os.lstat(os.path.join(dirpath, name)).st_size
                except OSError:
                    pass
        return total

    def to_dict(self, perms: CodePerms) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created": self.created,
            "expires_at": self.last_used + perms.ttl_minutes * 60,
            "files": self.files(),
            "bytes_used": self.bytes_used(),
            "disk_budget_bytes": perms.disk_mb * 2**20,
        }


class SandboxStore:
    """Every sandbox on this server, each a directory under one base.

    Only directories named like a sandbox id are ever deleted, so a
    ``T1_CODE_SANDBOX_DIR`` pointed somewhere unwise loses nothing but
    sandboxes.
    """

    def __init__(self, base: str | Path | None = None) -> None:
        self.base = Path(base) if base else Path(tempfile.gettempdir()) / f"hypernix-code-{os.getuid() if hasattr(os, 'getuid') else 'user'}"
        self.base.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._boxes: dict[str, Sandbox] = {}
        self._lock = threading.Lock()
        # Left by a previous server: nothing refers to them any more.
        for stale in self.base.iterdir():
            if stale.is_dir() and _ID.match(stale.name):
                shutil.rmtree(stale, ignore_errors=True)

    # -- lifecycle ------------------------------------------------------

    def create(self, owner: str, perms: CodePerms, *, name: str = "",
               files: dict[str, str] | None = None) -> Sandbox:
        self.expire(perms)
        with self._lock:
            held = sum(1 for box in self._boxes.values() if box.owner == owner)
            if held >= MAX_PER_OWNER:
                raise SandboxError(
                    f"you already have {held} sandboxes; delete one first "
                    f"(DELETE /code/sandbox/<id>)", code="too_many", status=409)
            box_id = f"sbx_{secrets.token_hex(8)}"
            root = self.base / box_id
            root.mkdir(mode=0o700)
            box = Sandbox(id=box_id, owner=owner, root=root.resolve(), name=name[:80])
            self._boxes[box_id] = box
        try:
            for path, content in (files or {}).items():
                self.write(box, path, content, perms)
        except SandboxError:
            self.delete(box)
            raise
        return box

    def get(self, box_id: str, owner: str, *, is_admin: bool = False) -> Sandbox:
        with self._lock:
            box = self._boxes.get(box_id) if _ID.match(box_id or "") else None
        # Somebody else's sandbox is "not found", not "forbidden": its
        # existence is not the caller's business.
        if box is None or (box.owner != owner and not is_admin):
            raise SandboxError(f"no sandbox {box_id!r}", code="not_found", status=404)
        box.last_used = time.time()
        return box

    def list(self, owner: str, *, is_admin: bool = False) -> list[Sandbox]:
        with self._lock:
            return [b for b in self._boxes.values() if is_admin or b.owner == owner]

    def delete(self, box: Sandbox) -> None:
        with self._lock:
            self._boxes.pop(box.id, None)
        if _ID.match(box.root.name):
            shutil.rmtree(box.root, ignore_errors=True)

    def expire(self, perms: CodePerms) -> list[str]:
        cutoff = time.time() - perms.ttl_minutes * 60
        with self._lock:
            old = [b for b in self._boxes.values() if b.last_used < cutoff]
        for box in old:
            self.delete(box)
        return [b.id for b in old]

    # -- files ----------------------------------------------------------

    def _resolve(self, box: Sandbox, relative: str) -> Path:
        from ..interfaces.noodle.tools import ToolContext, ToolError

        try:
            path = ToolContext(root=box.root, memory_path=box.root / ".hnx" / "m").resolve(relative)
        except ToolError as exc:
            raise SandboxError(str(exc), code=exc.code) from None
        if path == box.root:
            raise SandboxError("a file path is required", code="bad_path")
        if path.relative_to(box.root).parts[0] == ".hnx":
            raise SandboxError(".hnx is the sandbox's own folder", code="bad_path")
        return path

    def write(self, box: Sandbox, relative: str, content: str, perms: CodePerms) -> Path:
        path = self._resolve(box, relative)
        data = content.encode("utf-8")
        existing = path.stat().st_size if path.is_file() else 0
        budget = perms.disk_mb * 2**20
        if box.bytes_used() - existing + len(data) > budget:
            raise SandboxError(
                f"that would take the sandbox past its {perms.disk_mb} MB (s5)",
                code="disk_budget", status=413)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        box.last_used = time.time()
        return path

    def read(self, box: Sandbox, relative: str) -> tuple[str, bool]:
        path = self._resolve(box, relative)
        if not path.is_file():
            raise SandboxError(f"no file {relative!r}", code="not_found", status=404)
        with path.open("rb") as handle:
            data = handle.read(MAX_READ + 1)
        return data[:MAX_READ].decode("utf-8", errors="replace"), len(data) > MAX_READ

    def remove(self, box: Sandbox, relative: str) -> None:
        path = self._resolve(box, relative)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
        else:
            raise SandboxError(f"no file {relative!r}", code="not_found", status=404)

    # -- running --------------------------------------------------------

    def run(self, box: Sandbox, perms: CodePerms, *, language: str = "", code: str | None = None,
            path: str = "", stdin: str = "", args: list[str] | None = None) -> RunResult:
        if not perms.execute:
            raise SandboxError("running code is off (s1). An admin can turn it on with "
                               "/code/create/sandbox/perms/s1?:=on", code="execute_off", status=403)
        if code is not None:
            language = (language or "python").lower()
            if language not in LANGUAGES:
                raise SandboxError(f"unknown language {language!r}; known: {', '.join(LANGUAGES)}")
            if len(code) > MAX_CODE:
                raise SandboxError(f"the code is over {MAX_CODE} characters", status=413)
            target = self.write(box, f"main{LANGUAGES[language][1]}", code, perms)
        elif path:
            target = self._resolve(box, path)
            if not target.is_file():
                raise SandboxError(f"no file {path!r}", code="not_found", status=404)
            if not language:
                language = next((n for n, (_i, suffix) in LANGUAGES.items()
                                 if target.suffix == suffix), "")
                if not language:
                    raise SandboxError(f"say which language runs {path!r}")
        else:
            raise SandboxError("send `code` to run, or the `path` of a file in the sandbox")

        language = language.lower()
        if language not in perms.languages:
            raise SandboxError(f"{language} is not allowed here (s4: {', '.join(perms.languages)})",
                               code="language_off", status=403)
        installed = available_languages()
        if language not in installed:
            raise SandboxError(f"{language} is not installed on this server",
                               code="not_installed", status=501)
        if box.bytes_used() > perms.disk_mb * 2**20:
            raise SandboxError(f"the sandbox is over its {perms.disk_mb} MB (s5); delete files first",
                               code="disk_budget", status=413)

        prefix: list[str] = []
        if not perms.network:
            isolation = network_isolation()
            if isolation is None:
                raise SandboxError(
                    "this machine cannot run code without the network (no user namespaces), "
                    "and s2 says the network is off. It refuses rather than run with the "
                    "network on; an admin can accept that with /code/create/sandbox/perms/s2?:=on",
                    code="network_unenforceable", status=501)
            prefix = isolation

        limits = {"memory_bytes": perms.memory_mb * 2**20,
                  "file_bytes": perms.disk_mb * 2**20,
                  "cpu_seconds": perms.timeout + 1,
                  "wall_seconds": perms.timeout}
        scratch = box.root / ".hnx" / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        report = box.root / ".hnx" / "limits-refused"
        report.unlink(missing_ok=True)
        argv = [*prefix, sys.executable, "-c", _LAUNCHER,
                str(limits["memory_bytes"]), str(limits["file_bytes"]), str(limits["cpu_seconds"]),
                str(report), installed[language], str(target), *[str(a) for a in (args or [])]]
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(box.root),
               "LANG": os.environ.get("LANG", "C.UTF-8"), "TMPDIR": str(scratch),
               "PYTHONDONTWRITEBYTECODE": "1"}

        started = time.monotonic()
        timed_out = False
        # Output goes to files, not pipes: a program printing gigabytes
        # fills a file RLIMIT_FSIZE stops, not the server's memory.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            try:
                proc = subprocess.Popen(  # noqa: S603 - argv list, never a shell
                    argv, cwd=str(box.root), env=env, stdin=subprocess.PIPE,
                    stdout=out, stderr=err, start_new_session=True,
                )
            except OSError as exc:
                raise SandboxError(f"could not start {language}: {exc}", status=500) from exc
            try:
                proc.communicate(input=stdin.encode("utf-8"), timeout=perms.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc)
                proc.wait()
            elapsed = time.monotonic() - started
            stdout, cut_out = _tail(out)
            stderr, cut_err = _tail(err)
        box.last_used = time.time()
        try:
            refused = report.read_text(encoding="utf-8").split()
            report.unlink()
        except OSError:
            refused = []
        for name in refused:
            # Not enforced here: say so rather than report a limit that did not hold.
            limits[name] = None
        if refused:
            limits["not_enforced"] = refused
        return RunResult(language=language, exit_code=None if timed_out else proc.returncode,
                         stdout=stdout, stderr=stderr, elapsed=elapsed, timed_out=timed_out,
                         truncated=cut_out or cut_err, network=perms.network, limits=limits)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        import signal
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, AttributeError):
        proc.kill()


def _tail(handle) -> tuple[str, bool]:
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(max(0, size - MAX_OUTPUT))
    return handle.read().decode("utf-8", errors="replace"), size > MAX_OUTPUT
