"""hyperlink.shell — run a command on the server from the phone.

Off by default, and only the person running the server can turn it on
(``T1_HYPERLINK_SHELL=1``): a paired phone is otherwise never more than a
chat client, and this route makes it a terminal. When it is on:

* the command runs as the server's own user, in the server's home
  directory unless a working directory is given, through ``bash -lc``
  (``sh -c`` where there is no bash);
* it has a timeout, after which the whole process group is killed —
  a phone that loses signal must not leave ``yes > /dev/null`` running;
* output is capped, so ``cat`` of a large file cannot fill the phone's
  memory;
* every command is written to the server log with the device that sent
  it, before it runs;
* the working directory must be inside one root — the server user's
  home, or ``T1_HYPERLINK_SHELL_ROOT`` — so the phone chooses where in
  that tree a command starts, not where on the disk.

The model is never given this. Letting a person type a command on their
own server is their choice; letting a model do it on text it read is a
different one, and this module does not make it.
"""
from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["MAX_OUTPUT", "ShellResult", "run_command", "shell_binary", "shell_root", "working_directory"]

#: Per stream. Enough for a build log's tail; not a way to move files.
MAX_OUTPUT = 200_000

MAX_COMMAND = 8_000


@dataclass
class ShellResult:
    command: str
    cwd: str
    exit_code: int | None
    stdout: str
    stderr: str
    elapsed: float
    timed_out: bool = False
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed_seconds": round(self.elapsed, 3),
            "timed_out": self.timed_out,
            "truncated": self.truncated,
        }


def shell_binary() -> list[str]:
    if sys.platform == "win32":
        return ["cmd.exe", "/c"]
    bash = shutil.which("bash")
    return [bash, "-lc"] if bash else ["/bin/sh", "-c"]


def shell_root() -> str:
    """The directory every command's working directory must be inside."""
    configured = os.environ.get("T1_HYPERLINK_SHELL_ROOT", "").strip()
    return os.path.realpath(os.path.expanduser(configured) if configured else str(Path.home()))


def working_directory(cwd: str | None) -> str:
    """*cwd* resolved inside :func:`shell_root`, or ValueError.

    Relative paths are taken from the root. Symlinks are resolved before
    the check, so a link inside the root that points out of it is
    refused like the path it points to.
    """
    root = shell_root()
    wanted = os.path.expanduser(cwd) if cwd else root
    resolved = os.path.realpath(os.path.join(root, wanted))
    if resolved == root:
        if not os.path.isdir(root):
            raise ValueError(f"{root} is not a directory")
        return root
    # One plain prefix test, on its own: the form a path check has to
    # take for a reader (and for CodeQL) to see that nothing outside the
    # root reaches the filesystem calls below.
    if not resolved.startswith(root.rstrip(os.sep) + os.sep):
        raise ValueError(f"the working directory must be inside {root}")
    if not os.path.isdir(resolved):
        raise ValueError(f"{resolved} is not a directory")
    return resolved


def _clip(data: bytes) -> tuple[str, bool]:
    text = data.decode("utf-8", errors="replace")
    if len(text) <= MAX_OUTPUT:
        return text, False
    return text[-MAX_OUTPUT:], True


def run_command(command: str, *, cwd: str | None = None, timeout: float = 60.0, actor: str = "") -> ShellResult:
    command = (command or "").strip()
    if not command:
        raise ValueError("an empty command")
    if len(command) > MAX_COMMAND:
        raise ValueError(f"a command is at most {MAX_COMMAND} characters")
    directory = working_directory(cwd)

    logger.warning("hyperlink shell: %s runs %r in %s", actor or "a device", command, directory)
    started = time.monotonic()
    kwargs: dict = {}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    process = subprocess.Popen(  # noqa: S603 - the server operator enabled exactly this
        [*shell_binary(), command],
        cwd=directory,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    timed_out = False
    try:
        out, err = process.communicate(timeout=max(1.0, float(timeout)))
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill(process)
        out, err = process.communicate()
    stdout, cut_out = _clip(out or b"")
    stderr, cut_err = _clip(err or b"")
    return ShellResult(
        command=command,
        cwd=directory,
        exit_code=None if timed_out else process.returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed=time.monotonic() - started,
        timed_out=timed_out,
        truncated=cut_out or cut_err,
    )


def _kill(process: subprocess.Popen) -> None:
    """The whole group: `sleep 999 | cat` is two processes."""
    try:
        if sys.platform != "win32":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()
