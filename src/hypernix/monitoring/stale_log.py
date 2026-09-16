"""hypernix.monitoring.stale_log — find the run when the log stops moving.

The failure this exists for is specific and common. Somebody starts a
training run, opens tvtoppro, and it sits on a ``train.log`` that has not
been written to since last Tuesday — because the run they are actually
watching writes to a different file, in a different directory, started
from a different shell. The dashboard is not broken. It is faithfully
reporting a log nobody is writing to, and there is nothing on screen that
says so.

So: if the log has not changed in a while, stop trusting it, find the
Python process on this machine that is actually working, and report what
*it* is doing.

How the process is chosen
-------------------------
Highest system usage, where "usage" is a blend rather than a single
number. CPU alone picks a busy-waiting poller over a training run that is
GPU-bound and mostly blocked on the device; RSS alone picks whatever has
the largest heap. The score weights CPU and memory together and gives a
bonus to a process whose command line looks like training —
``train.py``, ``accelerate launch``, ``torchrun``, ``hypernix`` — because
that is the actual question being asked, and refusing to use the strongest
available signal for it would be pedantry.

What it reports
---------------
Whatever that process will tell us, in order of how much it is worth:

1. **Its own log files.** Every writable ``.log`` in its open file
   descriptors, newest first. This is the answer when it works, because
   it is the log the run is really writing, and tvtoppro can then tail it
   instead of the stale one.
2. **Its working directory**, so the human can go and look.
3. **Progress**, parsed from that log with the same
   ``step N/M loss=X`` reader the dashboard already uses.
4. **The process itself**: command, user, start time, CPU, RSS.

Every one of those is best-effort. Reading another process's file
descriptors needs permission that a non-root user has only for their own
processes, and ``/proc`` does not exist at all on macOS or Windows.
Everything here degrades to "could not tell" rather than raising, because
this is a diagnostic that runs on a monitoring dashboard's refresh tick
and it must never be the thing that breaks it.

It only reads
-------------
Nothing here signals, traces, or attaches to anything. It reads
``/proc``, ``psutil``'s read-only accessors, and files the running user
can already open. A monitoring tool that could stop your training run is
worse than no monitoring tool.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_STALE_SECONDS",
    "StaleReport",
    "ProcessCandidate",
    "log_age_seconds",
    "is_stale",
    "busiest_python",
    "rank_python_processes",
    "investigate",
]

#: A week. The number in the request, and a defensible one: an overnight
#: run leaves a log untouched for hours only if it has hung, but a log
#: from a run that finished last week is not *wrong*, it is just old, and
#: nagging about it the moment a run ends would make the warning noise.
DEFAULT_STALE_SECONDS = 7 * 24 * 60 * 60

#: Command-line fragments that mark a process as probably-training. Not
#: required and not sufficient -- a run started from a notebook matches
#: none of them and is still found on CPU and memory alone.
_TRAINING_HINTS = (
    "train", "torchrun", "accelerate", "deepspeed", "hypernix", "brewer",
    "pressure_cooker", "instant_pot", "finetune", "fine-tune", "sft",
    "transformers", "lightning", "axolotl", "unsloth",
)

#: Directories whose .log files are somebody else's. A process's open
#: descriptors include things it never wrote -- a journal it reads, a
#: library's debug sink -- and offering one of those as "the training
#: log" sends the user somewhere useless.
_BORING_LOG_ROOTS = ("/var/log", "/usr/", "/etc/", "/dev/")

#: The progress line every HyperNix trainer writes, and which
#: :mod:`hypernix.monitoring.tv` already parses. Duplicated here rather
#: than imported so that this module has no import-time dependency on the
#: dashboard it is a diagnostic for.
_STEP = re.compile(
    r"step\s+(\d+)\s*/\s*(\d+).*?loss\s*[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


@dataclass
class ProcessCandidate:
    """One running Python process, and why it scored where it did."""

    pid: int
    name: str = ""
    command: str = ""
    user: str = ""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    rss_bytes: int = 0
    started_at: float = 0.0
    #: True when the command line matched :data:`_TRAINING_HINTS`.
    looks_like_training: bool = False

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.started_at) if self.started_at else 0.0

    @property
    def score(self) -> float:
        """How likely this is the run the user meant.

        CPU carries most of the weight, memory a third of it, and a
        command line that looks like training adds a flat bonus rather
        than a multiplier -- a multiplier would let a 2%-CPU process with
        "train" in its name outrank a genuinely busy one, and the hint is
        supporting evidence, not proof.
        """
        return (
            self.cpu_percent
            + self.memory_percent * 0.33
            + (25.0 if self.looks_like_training else 0.0)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "name": self.name,
            "command": self.command,
            "user": self.user,
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "rss_bytes": self.rss_bytes,
            "age_seconds": round(self.age_seconds),
            "looks_like_training": self.looks_like_training,
            "score": round(self.score, 1),
        }


@dataclass
class StaleReport:
    """What was found when the log stopped moving."""

    stale: bool = False
    log_path: Path | None = None
    log_age_seconds: float | None = None
    threshold_seconds: float = DEFAULT_STALE_SECONDS
    process: ProcessCandidate | None = None
    #: Candidate log files the process has open, newest first.
    candidate_logs: list[Path] = field(default_factory=list)
    #: The best candidate, if one parsed as a training log.
    suggested_log: Path | None = None
    cwd: Path | None = None
    step: int | None = None
    total_steps: int | None = None
    loss: float | None = None
    last_line: str = ""
    #: Why something could not be read. Shown, not swallowed.
    notes: list[str] = field(default_factory=list)

    @property
    def progress(self) -> float:
        if self.step and self.total_steps:
            return min(1.0, self.step / self.total_steps)
        return 0.0

    @property
    def found_something(self) -> bool:
        return self.process is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stale": self.stale,
            "log_path": str(self.log_path) if self.log_path else None,
            "log_age_seconds": (
                round(self.log_age_seconds) if self.log_age_seconds is not None else None
            ),
            "threshold_seconds": self.threshold_seconds,
            "process": self.process.to_dict() if self.process else None,
            "candidate_logs": [str(p) for p in self.candidate_logs],
            "suggested_log": str(self.suggested_log) if self.suggested_log else None,
            "cwd": str(self.cwd) if self.cwd else None,
            "step": self.step,
            "total_steps": self.total_steps,
            "loss": self.loss,
            "progress": round(self.progress, 4),
            "last_line": self.last_line,
            "notes": list(self.notes),
        }

    def describe(self) -> str:
        if not self.stale:
            return "The training log is current."
        age = _fmt_age(self.log_age_seconds)
        lines = [
            f"{self.log_path} has not been written to for {age}.",
        ]
        if not self.process:
            lines.append(
                "No busy Python process found either, so nothing here is "
                "training right now."
            )
            lines.extend(f"  {note}" for note in self.notes)
            return "\n".join(lines)

        process = self.process
        lines.append(
            f"The busiest Python process is pid {process.pid} "
            f"({process.cpu_percent:.0f}% cpu, "
            f"{process.memory_percent:.0f}% mem, up {_fmt_age(process.age_seconds)}):"
        )
        lines.append(f"  {_shorten(process.command, 100)}")
        if self.cwd:
            lines.append(f"  cwd: {self.cwd}")
        if self.suggested_log:
            lines.append(f"  writing to: {self.suggested_log}")
            if self.step is not None:
                lines.append(
                    f"  step {self.step}/{self.total_steps or '?'}"
                    + (f"  loss {self.loss:.4f}" if self.loss is not None else "")
                )
            lines.append(f"  tvtoppro --log {self.suggested_log}")
        elif self.candidate_logs:
            lines.append("  has these logs open, none with progress lines yet:")
            lines.extend(f"    {path}" for path in self.candidate_logs[:4])
        for note in self.notes:
            lines.append(f"  {note}")
        return "\n".join(lines)


def _shorten(text: str, limit: int) -> str:
    """Trim a command line from the *middle*.

    A training command is an absolute interpreter path, then the script,
    then the arguments — and everything that tells you which run it is
    lives at the end. Cutting the tail leaves
    ``/very/long/venv/bin/python3 /home/…/scratch/`` which identifies
    nothing.
    """
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head - 3
    return f"{text[:head]}...{text[-tail:]}"


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "an unknown time"
    for unit, scale in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= scale:
            return f"{seconds / scale:.0f}{unit}"
    return f"{seconds:.0f}s"


def log_age_seconds(path: str | Path | None) -> float | None:
    """Seconds since *path* was last written, or ``None`` if unknowable."""
    if path is None:
        return None
    try:
        return max(0.0, time.time() - Path(path).stat().st_mtime)
    except OSError:
        return None


def is_stale(
    path: str | Path | None, *, threshold_seconds: float = DEFAULT_STALE_SECONDS
) -> bool:
    """Whether *path* has gone untouched for longer than *threshold_seconds*.

    A missing log is **not** stale. "There is no log" is a different
    problem with a different message, and the dashboard already says it.
    """
    age = log_age_seconds(path)
    return age is not None and age > threshold_seconds


# ---------------------------------------------------------------------------
# Finding the process
# ---------------------------------------------------------------------------


def _looks_like_python(name: str, command: str) -> bool:
    lowered = f"{name} {command}".lower()
    return "python" in lowered or lowered.startswith(("torchrun", "accelerate"))


def _looks_like_training(command: str) -> bool:
    lowered = command.lower()
    return any(hint in lowered for hint in _TRAINING_HINTS)


def rank_python_processes(limit: int = 10) -> list[ProcessCandidate]:
    """Every Python process on this machine, busiest first.

    Needs psutil. Returns an empty list without it rather than raising —
    the caller is a dashboard refresh, and "no candidates" degrades into
    a message where an ImportError would be a crash.
    """
    try:
        import psutil
    except ImportError:
        return []

    now = time.time()
    candidates: list[ProcessCandidate] = []
    fields = ["pid", "name", "cmdline", "username", "memory_percent",
              "memory_info", "create_time", "cpu_times"]
    for process in psutil.process_iter(fields):
        try:
            info = process.info
            command = " ".join(info.get("cmdline") or [])
            name = str(info.get("name") or "")
            if not _looks_like_python(name, command):
                continue
            if process.pid == os.getpid():
                # The dashboard is a Python process and is frequently the
                # busiest one on an idle box. Reporting itself as the
                # training run is the single most confusing thing it
                # could do.
                continue
            memory = info.get("memory_info")
            started = float(info.get("create_time") or 0.0)
            candidates.append(ProcessCandidate(
                pid=int(info.get("pid") or process.pid),
                name=name,
                command=command or name,
                user=str(info.get("username") or ""),
                cpu_percent=_lifetime_cpu(info.get("cpu_times"), started, now),
                memory_percent=float(info.get("memory_percent") or 0.0),
                rss_bytes=int(getattr(memory, "rss", 0) or 0),
                started_at=started,
                looks_like_training=_looks_like_training(command),
            ))
        except Exception:  # noqa: BLE001 - psutil raises a family of these
            continue
    candidates.sort(key=lambda candidate: -candidate.score)
    return candidates[:limit]


def _lifetime_cpu(times: Any, started_at: float, now: float) -> float:
    """CPU seconds burned over wall-clock lifetime, as a percentage.

    Not ``psutil.Process.cpu_percent()``. That measures between two calls
    on the same ``Process`` object and documents itself as returning 0.0
    on the first one -- which is every call this makes, because
    ``process_iter`` hands back a fresh view each time. Sampling it
    properly would mean calling it, sleeping, and calling it again, and
    this runs on a monitoring dashboard's refresh tick where a blocking
    sleep is not available.

    The lifetime average needs no second sample and answers the question
    being asked better anyway. "Which process has been working" is about
    the whole run, not about the last 200 ms -- a training step that
    happens to be in an all-reduce when the sample lands is not idle, and
    an instantaneous reading would say it was.

    Scaled like ``top``: 100% is one core saturated, so a process using
    eight threads reads 800%.
    """
    if times is None:
        return 0.0
    elapsed = max(now - started_at, 1e-3) if started_at else 0.0
    if elapsed <= 0:
        return 0.0
    try:
        burned = float(times.user) + float(times.system)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    return max(0.0, burned / elapsed * 100.0)


def busiest_python() -> ProcessCandidate | None:
    """The Python process most likely to be the run, or ``None``."""
    ranked = rank_python_processes(limit=1)
    return ranked[0] if ranked else None


# ---------------------------------------------------------------------------
# Reading what it is doing
# ---------------------------------------------------------------------------


def _open_logs(pid: int, notes: list[str]) -> list[Path]:
    """Log files the process has open, newest first.

    Two sources, tried in order: psutil's ``open_files``, which works on
    more platforms, and ``/proc/<pid>/fd``, which works when psutil is
    absent. Both need permission, and the error when it is missing is
    worth showing -- "run it as the user that owns the process" is
    actionable where a silent empty list is not.
    """
    paths: list[Path] = []
    try:
        import psutil

        try:
            process = psutil.Process(pid)
            paths = [Path(entry.path) for entry in process.open_files()]
        except psutil.AccessDenied:
            notes.append(
                f"cannot read pid {pid}'s open files (different user; "
                f"run tvtoppro as that user to see its log)"
            )
            return []
        except psutil.NoSuchProcess:
            notes.append(f"pid {pid} exited while being inspected")
            return []
        except Exception:  # noqa: BLE001
            paths = []
    except ImportError:
        paths = []

    if not paths:
        descriptors = Path(f"/proc/{pid}/fd")
        if not descriptors.exists():
            notes.append(
                "no /proc and no psutil, so the process's open files are "
                "unreadable here"
            )
            return []
        try:
            for entry in descriptors.iterdir():
                try:
                    paths.append(Path(os.readlink(entry)))
                except OSError:
                    continue
        except PermissionError:
            notes.append(f"cannot list pid {pid}'s file descriptors (permission)")
            return []
        except OSError:
            return []

    logs = []
    for path in paths:
        text = str(path)
        if not text.endswith((".log", ".txt", ".jsonl")):
            continue
        if any(text.startswith(root) for root in _BORING_LOG_ROOTS):
            continue
        try:
            if path.is_file():
                logs.append(path)
        except OSError:
            continue

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    # Deduplicated because a process can hold the same file on several
    # descriptors -- stdout and a handler pointed at the same path is the
    # usual way -- and the same filename three times reads as three logs.
    unique = {str(path): path for path in logs}
    return sorted(unique.values(), key=_mtime, reverse=True)


def _process_cwd(pid: int) -> Path | None:
    try:
        import psutil

        return Path(psutil.Process(pid).cwd())
    except Exception:  # noqa: BLE001 - psutil missing, denied, or gone
        pass
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return None


def _tail(path: Path, *, limit: int = 64 * 1024) -> str:
    """The last *limit* bytes of *path*, decoded leniently.

    Bounded because a training log can be hundreds of megabytes and this
    runs on a refresh tick. The progress line is always near the end.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            if size > limit:
                stream.seek(size - limit)
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _progress_from(path: Path) -> tuple[int | None, int | None, float | None, str]:
    """``(step, total, loss, last non-empty line)`` from a log's tail."""
    text = _tail(path)
    if not text:
        return None, None, None, ""
    lines = [line for line in text.splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    for line in reversed(lines):
        match = _STEP.search(line)
        if match:
            try:
                return int(match.group(1)), int(match.group(2)), float(match.group(3)), last
            except ValueError:
                continue
    return None, None, None, last


def investigate(
    log_path: str | Path | None,
    *,
    threshold_seconds: float = DEFAULT_STALE_SECONDS,
    force: bool = False,
) -> StaleReport:
    """Check *log_path*'s age and, if it is stale, find the real run.

    *force* runs the investigation whatever the log's age, which is what
    ``tvtoppro --find-run`` does: "show me what is actually training" is
    a reasonable thing to ask on purpose, not only as a consequence of a
    stale file.

    Never raises. Every failure lands in
    :attr:`StaleReport.notes` as a sentence, because this is called from
    a dashboard's refresh and an exception there takes the screen down.
    """
    path = Path(log_path) if log_path else None
    age = log_age_seconds(path)
    report = StaleReport(
        log_path=path,
        log_age_seconds=age,
        threshold_seconds=threshold_seconds,
        stale=force or (age is not None and age > threshold_seconds),
    )
    if not report.stale:
        return report

    try:
        candidate = busiest_python()
    except Exception as exc:  # noqa: BLE001 - defensive: this runs on a refresh tick
        logger.debug("stale_log: ranking processes failed", exc_info=True)
        report.notes.append(f"could not inspect processes: {type(exc).__name__}: {exc}")
        return report

    if candidate is None:
        try:
            import psutil  # noqa: F401
        except ImportError:
            report.notes.append(
                "psutil is not installed, so no process could be inspected. "
                "pip install psutil"
            )
        else:
            report.notes.append("no Python process is doing anything here.")
        return report

    report.process = candidate
    report.cwd = _process_cwd(candidate.pid)
    report.candidate_logs = _open_logs(candidate.pid, report.notes)

    for log in report.candidate_logs:
        # The same file we already decided was stale is not the answer,
        # however open the process has it -- offering it back would be a
        # loop the user cannot get out of.
        if path is not None and log.resolve() == path.resolve():
            continue
        step, total, loss, last = _progress_from(log)
        if step is not None:
            report.suggested_log = log
            report.step, report.total_steps, report.loss = step, total, loss
            report.last_line = last
            break
        if not report.last_line:
            report.last_line = last
    if report.suggested_log is None and report.candidate_logs:
        # Nothing parsed, but the newest open log is still the best guess
        # and is what the user should go and look at.
        report.suggested_log = next(
            (log for log in report.candidate_logs
             if path is None or log.resolve() != path.resolve()),
            None,
        )
    return report
