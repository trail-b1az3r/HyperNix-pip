"""Run a script so it outlives the terminal that started it.

The problem is ordinary and the usual answers do not solve it. Start a
training run over SSH, close the laptop, and the job dies — because the
shell that owned it received SIGHUP when the connection dropped and
passed it to its process group. ``&`` does not help: a backgrounded
process is still in that group and still attached to the tty.

What actually survives is a process in its **own session**, with no
controlling terminal and nothing of the connection left in its file
descriptors. Two ways to get there, in preference order:

``systemd-run --user``
    A real supervisor. systemd owns the process, records its exit
    status, restarts it if asked, and keeps it after the user logs out
    when lingering is enabled. Used whenever there is a user bus.

``setsid`` + a recording wrapper
    The portable fallback: a new session, stdio redirected to the log
    file, and a small shell wrapper that writes the exit status to a
    file when the command finishes. No supervisor, but the job survives
    and the outcome is still recorded, which is the part that matters.

Everything about a job lives on disk under the config directory, so
``--status`` and ``--logs`` work from a completely different SSH session,
after a reboot, and whether or not the T1 server happens to be running.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "Job",
    "JobStatus",
    "LaunchError",
    "Supervisor",
    "JobStore",
    "launch",
    "default_root",
    "available_supervisor",
]


class LaunchError(RuntimeError):
    """A job could not be started, stopped or found."""


class JobStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"
    #: Started, and the outcome can no longer be determined — the record
    #: outlived the process without an exit status being written.
    UNKNOWN = "unknown"


class Supervisor(StrEnum):
    SYSTEMD = "systemd"
    SETSID = "setsid"


def default_root() -> Path:
    """Where job records live."""
    configured = os.environ.get("T1_CONFIG_DIR", "")
    base = Path(configured) if configured else Path.home() / ".hypernix" / "t1api"
    return base / "jobs"


@dataclass
class Job:
    job_id: str
    name: str
    command: list[str]
    cwd: str
    log_path: str
    supervisor: str
    created_at: float
    unit: str = ""
    pid: int = 0
    status: str = JobStatus.RUNNING.value
    exit_status: int | None = None
    finished_at: float | None = None
    timeout: int = 0
    priority: int = 0
    gpu: str = ""
    cpu: str = ""
    #: Names only. The values are the caller's environment and several of
    #: them are credentials; a job record is read by anything that can
    #: read the config directory.
    env_keys: list[str] = field(default_factory=list)

    @property
    def exit_file(self) -> Path:
        return Path(self.log_path).with_suffix(".exit")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            JobStatus.SUCCEEDED.value, JobStatus.FAILED.value,
            JobStatus.STOPPED.value, JobStatus.UNKNOWN.value,
        )


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str) -> str:
    return _SAFE_NAME.sub("-", text).strip("-").lower() or "job"


class JobStore:
    """Job records as one JSON file each, under *root*."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    def _path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def save(self, job: Job) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(job.job_id)
        # Write-then-rename: a reader polling --status must never catch a
        # half-written record, and this is polled in a loop by design.
        scratch = target.with_suffix(".tmp")
        scratch.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
        scratch.replace(target)

    def load(self, job_id: str) -> Job | None:
        try:
            data = json.loads(self._path(job_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        known = {f for f in Job.__dataclass_fields__}
        return Job(**{k: v for k, v in data.items() if k in known})

    def list(self) -> list[Job]:
        jobs = []
        if not self.root.is_dir():
            return jobs
        for path in sorted(self.root.glob("*.json")):
            job = self.load(path.stem)
            if job is not None:
                jobs.append(job)
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def find(self, needle: str) -> Job | None:
        """By job id, or by name — whichever the caller had to hand."""
        exact = self.load(needle)
        if exact is not None:
            return exact
        matches = [j for j in self.list() if j.name == needle]
        if not matches:
            matches = [j for j in self.list() if j.job_id.startswith(needle)]
        return matches[0] if matches else None


def _user_bus_available() -> bool:
    """Whether systemd will accept a --user unit from here.

    Asked by trying, not by looking for the binary: a container can have
    systemd-run installed and no user bus to talk to, and the difference
    is only visible when something connects.
    """
    if shutil.which("systemd-run") is None or shutil.which("systemctl") is None:
        return False
    try:
        probe = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def available_supervisor() -> Supervisor:
    return Supervisor.SYSTEMD if _user_bus_available() else Supervisor.SETSID


def _interpreter_for(script: Path) -> list[str]:
    """How to run *script*: its shebang, or an interpreter by suffix.

    An executable file with a shebang runs itself. Anything else gets an
    interpreter chosen from the extension, because "permission denied"
    for a .py file someone forgot to chmod is a worse first experience
    than just running it with python.
    """
    if os.access(script, os.X_OK):
        try:
            with script.open("rb") as handle:
                if handle.read(2) == b"#!":
                    return [str(script)]
        except OSError:
            pass
    suffix = script.suffix.lower()
    if suffix == ".py":
        import sys

        return [sys.executable, str(script)]
    if suffix in (".sh", ".bash"):
        return [shutil.which("bash") or "/bin/sh", str(script)]
    if os.access(script, os.X_OK):
        return [str(script)]
    raise LaunchError(
        f"{script} is not executable and has no extension this knows how to "
        f"run (.py, .sh). Either chmod +x it with a #! line, or rename it."
    )


def launch(
    script: str | Path,
    *,
    args: list[str] | None = None,
    name: str = "",
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 0,
    log_file: str | Path | None = None,
    priority: int = 0,
    gpu: str = "",
    cpu: str = "",
    store: JobStore | None = None,
    supervisor: Supervisor | None = None,
) -> Job:
    """Start *script* detached, and return its :class:`Job` record."""
    path = Path(script).expanduser()
    if not path.exists():
        raise LaunchError(f"No such script: {path}")
    if path.is_dir():
        raise LaunchError(f"{path} is a directory, not a script.")

    store = store or JobStore()
    job_id = uuid.uuid4().hex[:12]
    label = _slug(name or path.stem)
    working = str(Path(cwd).expanduser().resolve()) if cwd else os.getcwd()
    if not Path(working).is_dir():
        raise LaunchError(f"No such working directory: {working}")

    store.root.mkdir(parents=True, exist_ok=True)
    log_path = (
        Path(log_file).expanduser().resolve()
        if log_file else store.root / f"{job_id}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = _interpreter_for(path) + list(args or [])
    child_env = dict(os.environ)
    child_env.update(env or {})
    if gpu:
        # Checked against what is actually present. Setting
        # CUDA_VISIBLE_DEVICES=3 on a two-card machine does not fail: the
        # job starts, sees no GPU, and either runs on the CPU at a
        # hundredth of the speed or dies deep in a framework — hours
        # later, in a log nobody is watching.
        from . import gpus as _gpus

        visible = _gpus.detect()
        if visible:
            try:
                _gpus.select(gpu, visible)
            except ValueError as exc:
                raise LaunchError(f"--gpu {gpu}: {exc}") from None
        else:
            # Nothing detected. That is *not* proof there is no GPU: a
            # container without nvidia-smi installed sees none while the
            # device is right there, and refusing here would block a job
            # that would have run. An index we cannot check is passed
            # through and left to the framework.
            logger.warning(
                "launcher: no GPU is visible from here, so --gpu %s could not "
                "be checked. If the job needs one, make sure the vendor tool "
                "(nvidia-smi / rocm-smi) is available in this environment.",
                gpu,
            )
        # The vendor-neutral pair. Setting both is right rather than
        # sloppy: which one a runtime reads depends on whether it ends up
        # on CUDA or ROCm, and the job should not have to know.
        child_env["CUDA_VISIBLE_DEVICES"] = gpu
        child_env["HIP_VISIBLE_DEVICES"] = gpu
    child_env["HNX_JOB_ID"] = job_id
    child_env["HNX_JOB_NAME"] = label
    # What a training run reports itself under. The id rather than the
    # label, so relaunching a job of the same name supersedes nothing —
    # the monitor still resolves the label, because it falls back to
    # matching on name when an id does not exist.
    child_env["HNX_RUN_ID"] = job_id
    child_env["HNX_LOG_PATH"] = str(log_path)

    chosen = supervisor or available_supervisor()
    job = Job(
        job_id=job_id, name=label, command=command, cwd=working,
        log_path=str(log_path), supervisor=chosen.value,
        created_at=time.time(), timeout=timeout, priority=priority,
        gpu=gpu, cpu=cpu, env_keys=sorted(env or {}),
    )

    if chosen is Supervisor.SYSTEMD:
        _launch_systemd(job, command, working, child_env, log_path, timeout, priority, cpu)
    else:
        _launch_setsid(job, command, working, child_env, log_path, timeout, priority)

    store.save(job)
    logger.info(
        "launcher: started %s (%s) under %s", job.name, job.job_id, job.supervisor
    )
    return job


def _launch_systemd(job, command, working, env, log_path, timeout, priority, cpu) -> None:
    unit = f"hnx-{job.name}-{job.job_id}"
    argv = [
        "systemd-run", "--user", f"--unit={unit}",
        "--collect",  # keep the unit around only until it is queried
        "--property=KillMode=mixed",
        f"--working-directory={working}",
        # Output to the same file the setsid path uses, so --logs does
        # not have to care which supervisor ran the job.
        f"--property=StandardOutput=append:{log_path}",
        f"--property=StandardError=append:{log_path}",
    ]
    if timeout > 0:
        argv.append(f"--property=RuntimeMaxSec={timeout}")
    if priority:
        argv.append(f"--nice={priority}")
    if cpu:
        argv.append(f"--property=CPUAffinity={cpu}")
    # Environment goes through --setenv rather than the command line, so
    # nothing here lands in `ps`.
    for key in ("HNX_JOB_ID", "HNX_JOB_NAME", "HNX_RUN_ID", "HNX_LOG_PATH",
                "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        if key in env:
            argv.append(f"--setenv={key}={env[key]}")
    for key in job.env_keys:
        if key in env:
            argv.append(f"--setenv={key}={env[key]}")
    argv.append("--")
    argv.extend(command)

    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise LaunchError(
            f"systemd-run refused to start the job: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    job.unit = unit


def _launch_setsid(job, command, working, env, log_path, timeout, priority) -> None:
    """A new session, stdio on the log, and the exit status recorded.

    ``setsid`` is what makes this survive: a new session has no
    controlling terminal, so the SIGHUP that follows the connection
    dropping is never delivered here. The wrapper exists because without
    a supervisor nothing else would record how the job ended.
    """
    exit_file = Path(job.log_path).with_suffix(".exit")
    inner = " ".join(shlex.quote(part) for part in command)
    if timeout > 0 and shutil.which("timeout"):
        inner = f"timeout {int(timeout)} {inner}"
    if priority and shutil.which("nice"):
        inner = f"nice -n {int(priority)} {inner}"
    script = (
        f"{inner}\n"
        f"printf '%s' \"$?\" > {shlex.quote(str(exit_file))}\n"
    )

    # No `setsid` binary here, deliberately. It forks when it is already
    # a process-group leader and the parent then exits, so the pid we
    # would record is a process that has already gone -- and --status
    # reports "unknown" for a job that is running perfectly well.
    # Popen's start_new_session already calls setsid(2) in the child,
    # which is the same new session with the pid we actually want.
    argv = ["/bin/sh", "-c", script]

    with open(log_path, "ab") as log:
        proc = subprocess.Popen(  # noqa: S603
            argv, cwd=working, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            # Its own process group even without setsid, so a Ctrl-C in
            # the launching shell is not delivered to the job.
            start_new_session=True,
        )
    job.pid = proc.pid


def refresh(job: Job, store: JobStore | None = None) -> Job:
    """Update *job*'s status from the system, and persist any change."""
    if job.is_terminal:
        return job

    previous = job.status
    if job.supervisor == Supervisor.SYSTEMD.value and job.unit:
        _refresh_systemd(job)
    else:
        _refresh_setsid(job)

    if job.status != previous and store is not None:
        store.save(job)
    return job


def _refresh_systemd(job: Job) -> None:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", job.unit,
             "--property=ActiveState,Result,ExecMainStatus,ExecMainExitTimestampMonotonic"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    state = values.get("ActiveState", "")
    if state in ("activating", "active", "reloading"):
        job.status = JobStatus.RUNNING.value
        return
    if not state or state == "inactive" and not values.get("Result"):
        # The unit is gone. --collect reaps it, so an exit status we
        # never observed is genuinely unknowable rather than a failure.
        job.status = JobStatus.UNKNOWN.value
        job.finished_at = job.finished_at or time.time()
        return
    code = values.get("ExecMainStatus", "")
    job.exit_status = int(code) if code.isdigit() else None
    job.finished_at = job.finished_at or time.time()
    if values.get("Result") == "success" or job.exit_status == 0:
        job.status = JobStatus.SUCCEEDED.value
    else:
        job.status = JobStatus.FAILED.value


def _refresh_setsid(job: Job) -> None:
    exit_file = Path(job.log_path).with_suffix(".exit")
    if exit_file.exists():
        try:
            job.exit_status = int(exit_file.read_text(encoding="utf-8").strip() or 1)
        except ValueError:
            job.exit_status = 1
        job.finished_at = job.finished_at or exit_file.stat().st_mtime
        job.status = (
            JobStatus.SUCCEEDED.value if job.exit_status == 0
            else JobStatus.FAILED.value
        )
        return
    if job.pid and _alive(job.pid):
        job.status = JobStatus.RUNNING.value
        return
    # No exit file and no process. The wrapper writes the status as its
    # last act, so this is a job that was killed outright.
    job.status = JobStatus.UNKNOWN.value
    job.finished_at = job.finished_at or time.time()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def stop(job: Job, store: JobStore | None = None, *, timeout: float = 10.0) -> Job:
    """Ask the job to stop, then insist."""
    if job.supervisor == Supervisor.SYSTEMD.value and job.unit:
        subprocess.run(
            ["systemctl", "--user", "stop", job.unit],
            capture_output=True, timeout=30, check=False,
        )
    elif job.pid:
        try:
            # The whole group: the wrapper shell and the command it ran.
            os.killpg(os.getpgid(job.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        deadline = time.time() + timeout
        while time.time() < deadline and _alive(job.pid):
            time.sleep(0.1)
        if _alive(job.pid):
            try:
                os.killpg(os.getpgid(job.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    job.status = JobStatus.STOPPED.value
    job.finished_at = job.finished_at or time.time()
    if store is not None:
        store.save(job)
    return job


def read_logs(job: Job, *, tail: int = 200) -> str:
    path = Path(job.log_path)
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    return "\n".join(lines[-tail:]) if tail > 0 else "\n".join(lines)
