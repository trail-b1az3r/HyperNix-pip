"""The Python side of tvtop-max.

tvtop-max's screen is an OpenTUI app (TypeScript on Bun, like
hyped-pro). Everything it shows is worked out here and sent as JSON,
one object per line each way, on the protocol hyped-pro's bridge uses::

    -> {"id": 3, "cmd": "frame"}
    <- {"id": 3, "ok": true, "data": {...}}
    <- {"id": 3, "ok": false, "code": "TVM-...", "error": "..."}

Commands:

``info``
    What the run is: the script and log in use, the process, and the
    script's analysis from :mod:`hypernix.monitoring.run_inspect`
    (modules, libraries, model architecture, Pressure Cooker, warnings).
    Slow the first time -- reading the architecture presets imports
    torch -- so it runs on its own thread and ``frame`` is not held up.
``frame``
    What the machine and the run are doing now: tvtop-pro's statistics,
    the busiest Python processes, the log's last lines and the warnings
    in it.
``rescan``
    Look for the script and log again, as at start.

The statistics come from the same source as tvtop-pro
(:class:`~hypernix.monitoring.tvtop_plus_plus.TVTopPlusPlus`), so the
two dashboards always agree about the numbers.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .run_inspect import analyze_script, arch_from_folder, findings_from_log, script_from_command

__all__ = ["Monitor", "serve"]

#: Lines of log sent with each frame. The logs panel scrolls through
#: these; the warnings panel reads further back (run_inspect.LOG_WINDOW_LINES).
LOG_LINES = 400

#: Seconds between process-table scans, which are slow on some systems.
PROCESS_REFRESH_SECONDS = 5.0


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _own_family() -> set[int]:
    """This process and its ancestors: the tvtop-max launcher, bun, the
    shell. None of them is the run being watched."""
    pids = {os.getpid()}
    try:
        import psutil

        pids.update(parent.pid for parent in psutil.Process().parents())
        return pids
    except Exception:  # noqa: BLE001 - psutil missing: walk /proc instead
        pass
    pid = os.getpid()
    for _ in range(32):
        try:
            with open(f"/proc/{pid}/stat", encoding="ascii") as fh:
                pid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if pid <= 1:
            break
        pids.add(pid)
    return pids


def _tail_lines(path: Path, *, limit_bytes: int = 256 * 1024) -> list[str]:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - limit_bytes))
            data = fh.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > limit_bytes and lines:
        lines = lines[1:]  # the first line was cut in half by the seek
    return lines


class Monitor:
    """The run being watched, and the answers to the bridge's commands."""

    def __init__(self, *, log: str | None = None, script: str | None = None,
                 pid: int | None = None, discover: bool = True) -> None:
        self.explicit_log = Path(log).expanduser() if log else None
        self.explicit_script = Path(script).expanduser() if script else None
        self.explicit_pid = pid
        self.discover = discover
        self.log: Path | None = None
        self.script: Path | None = None
        self.pid: int | None = None
        self.process: dict[str, Any] | None = None
        self.notes: list[str] = []
        self._stats = None
        self._processes: list[dict[str, Any]] = []
        self._processes_at = 0.0
        self._info: dict[str, Any] | None = None
        self._log_findings: tuple[int, float, list[dict[str, Any]]] | None = None
        self._lock = threading.Lock()
        self.rescan()

    # -- finding the run -------------------------------------------------

    def rescan(self) -> dict[str, Any]:
        from .stale_log import _process_cwd

        self.notes = []
        self.pid, self.process = self.explicit_pid, None
        candidate = None
        if self.discover and (self.explicit_pid is None):
            candidate = self._pick_process()
            if candidate is not None:
                self.pid = candidate.pid
                self.process = candidate.to_dict()
        cwd = _process_cwd(self.pid) if self.pid else None

        self.script = self.explicit_script
        if self.script is None and self.process:
            self.script = script_from_command(self.process.get("command", ""), cwd=cwd)
            if self.script is None:
                self.notes.append("the busiest Python process is not running a .py file "
                                  "(pass --script to name one)")
        if self.script is None and not self.process and self.discover:
            self.notes.append("no Python process is running; pass --script to read one")

        self.log = self.explicit_log
        if self.log is None and self.discover:
            self.log = self._find_log(cwd)
        with self._lock:
            self._stats = None
            self._info = None
            self._log_findings = None
        return self.describe()

    def _pick_process(self):
        """The process to watch.

        With a script named, only a process running that script: the
        busiest Python on the machine is somebody else's run as often as
        it is this one, and naming it in the header would be a guess
        presented as a fact.
        """
        from .stale_log import _process_cwd, rank_python_processes

        family = _own_family()
        try:
            ranked = [c for c in rank_python_processes(limit=20) if c.pid not in family]
        except Exception:  # noqa: BLE001 - discovery is best-effort
            return None
        if self.explicit_script is None:
            return ranked[0] if ranked else None
        wanted = _resolved(self.explicit_script)
        for candidate in ranked:
            # The script it *runs*, not any mention of it: the tvtop-max
            # launcher's own command line says `-S train.py` too.
            runs = script_from_command(candidate.command, cwd=_process_cwd(candidate.pid))
            if runs is not None and _resolved(runs) == wanted:
                return candidate
        return None

    def _find_log(self, cwd: Path | None) -> Path | None:
        from .tv import _autodetect_log

        for start in [p for p in (cwd, Path.cwd()) if p is not None]:
            try:
                found = _autodetect_log(start)
            except Exception:  # noqa: BLE001
                found = None
            if found is not None:
                return found
        if self.pid:
            from .stale_log import _open_logs

            logs = _open_logs(self.pid, self.notes)
            if logs:
                return logs[0]
        self.notes.append("no training log found (pass --log to name one)")
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "script": str(self.script) if self.script else "",
            "log": str(self.log) if self.log else "",
            "pid": self.pid,
            "process": self.process,
            "notes": list(self.notes),
        }

    # -- info: what the run is ---------------------------------------------

    def info(self) -> dict[str, Any]:
        with self._lock:
            if self._info is not None:
                return self._info
        from hypernix import __version__

        report = analyze_script(self.script).to_dict() if self.script else None
        arch = (report or {}).get("arch") or {}
        if arch.get("source", "none") in ("none", "") or arch.get("repo"):
            folder = self._checkpoint_folder()
            if folder is not None:
                found = dataclasses.asdict(arch_from_folder(folder))
                if found.get("source") != "none":
                    arch = found
        result = {
            **self.describe(),
            "hypernix_version": __version__,
            "report": report,
            "arch": arch,
        }
        with self._lock:
            self._info = result
        return result

    def _checkpoint_folder(self) -> Path | None:
        from .map import discover_model

        base = self.script.parent if self.script else None
        try:
            found = discover_model(base)
        except Exception:  # noqa: BLE001
            return None
        if found is None:
            return None
        return found if found.is_dir() else found.parent

    # -- frame: what it is doing -------------------------------------------

    def frame(self) -> dict[str, Any]:
        from .tvtop_plus_plus import TVTopPlusPlus

        with self._lock:
            if self._stats is None:
                self._stats = TVTopPlusPlus(log_path=self.log)
            stats = self._stats
        data = dataclasses.asdict(stats.latest_frame())
        data.pop("log_tail", None)  # the frame's own tail is 8 lines; the logs panel wants more
        lines = _tail_lines(self.log) if self.log else []
        data["log_lines"] = lines[-LOG_LINES:]
        data["log_findings"] = self._findings_for(lines)
        data["log_age_seconds"] = self._log_age()
        data["processes"] = self._process_table()
        data["time"] = time.time()
        return data

    def _log_age(self) -> float | None:
        try:
            return max(0.0, time.time() - self.log.stat().st_mtime) if self.log else None
        except OSError:
            return None

    def _findings_for(self, lines: list[str]) -> list[dict[str, Any]]:
        # Recomputed only when the log changed: a quiet log is the common case.
        try:
            stat = self.log.stat() if self.log else None
        except OSError:
            stat = None
        key = (stat.st_size, stat.st_mtime) if stat else (0, 0.0)
        cached = self._log_findings
        if cached is not None and (cached[0], cached[1]) == key:
            return cached[2]
        found = [dataclasses.asdict(f) for f in findings_from_log(lines)]
        self._log_findings = (key[0], key[1], found)
        return found

    def _process_table(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        if now - self._processes_at >= PROCESS_REFRESH_SECONDS or not self._processes_at:
            from .stale_log import rank_python_processes

            try:
                self._processes = [c.to_dict() for c in rank_python_processes(limit=6)]
            except Exception:  # noqa: BLE001
                self._processes = []
            self._processes_at = now
        return self._processes

    # -- the protocol ------------------------------------------------------

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        """One reply for one request. Never raises."""
        request_id = message.get("id")
        command = message.get("cmd")
        try:
            if command == "info":
                data = self.info()
            elif command == "frame":
                data = self.frame()
            elif command == "rescan":
                data = self.rescan()
            elif command == "ping":
                data = {"pong": True}
            else:
                return {"id": request_id, "ok": False, "code": "TVM-BRIDGE-001",
                        "error": f"unknown command {command!r}"}
        except Exception as exc:  # noqa: BLE001 - a bad frame must not end the bridge
            return {"id": request_id, "ok": False, "code": "TVM-BRIDGE-002",
                    "error": f"{type(exc).__name__}: {exc}"}
        return {"id": request_id, "ok": True, "data": data}


def serve(monitor: Monitor, stdin=None, stdout=None) -> int:
    """Answer requests from *stdin* until it closes. Each on its own thread."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    write_lock = threading.Lock()

    def answer(message: dict[str, Any]) -> None:
        reply = monitor.handle(message)
        text = json.dumps(reply, default=str)
        with write_lock:
            stdout.write(text + "\n")
            stdout.flush()

    threads: list[threading.Thread] = []
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if not isinstance(message, dict):
            continue
        thread = threading.Thread(target=answer, args=(message,), daemon=True)
        thread.start()
        threads.append(thread)
        threads = [t for t in threads if t.is_alive()]
    for thread in threads:
        thread.join(timeout=5)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m hypernix.monitoring.tvtop_max_bridge")
    parser.add_argument("serve", nargs="?")
    parser.add_argument("--log")
    parser.add_argument("--script")
    parser.add_argument("--pid", type=int)
    args = parser.parse_args(argv)
    # Anything imported code prints goes to stderr, never into the protocol.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    monitor = Monitor(log=args.log, script=args.script, pid=args.pid)
    return serve(monitor, stdout=real_stdout)


if __name__ == "__main__":
    sys.exit(main())
