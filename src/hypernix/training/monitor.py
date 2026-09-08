"""What training is doing, and the controls for it.

0.72.4 item 5. A training run is the longest-lived and least observable
thing this package starts: it goes for hours, it is usually launched over
a connection that will not survive it, and until now the only way to know
how it was going was to read a log.

Two halves, deliberately separated.

:class:`ProgressReporter` is written *by* the trainer. It appends nothing
and holds no lock — each update is one atomic rewrite of a small JSON
file — because the reader is a web request that may arrive at any point,
including halfway through an epoch, and a half-written status is worse
than a stale one.

:class:`TrainingMonitor` is read *by* everything else: the API, Waiter,
HyperLink. It merges three sources that each know a different part of
the truth — the reporter's own numbers, the launcher's process state,
and live GPU/CPU/RAM figures — because none of them alone can tell you
that a run is stuck.

On stopping and pausing
-----------------------
Pause is SIGSTOP to the job's process group and resume is SIGCONT, which
is the only way to hold a training loop without cooperation from it. It
does not release GPU memory: the process is frozen with its allocations
intact, which is the point (resuming is instant) and also the caveat
(pausing does not free the card for something else). Said plainly here
because it is the thing people assume the opposite of.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "RunState",
    "TrainingRun",
    "ProgressReporter",
    "TrainingMonitor",
    "TrainingError",
    "default_root",
]


class TrainingError(RuntimeError):
    """A training control could not be applied."""


class RunState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    FAILED = "failed"
    STOPPED = "stopped"
    #: The record exists and nothing is behind it any more.
    STALE = "stale"


def default_root() -> Path:
    configured = os.environ.get("T1_CONFIG_DIR", "")
    base = Path(configured) if configured else Path.home() / ".hypernix" / "t1api"
    return base / "training"


@dataclass
class TrainingRun:
    """One run, as the trainer last described itself."""

    run_id: str
    name: str = ""
    state: str = RunState.STARTING.value
    #: Where in the schedule.
    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    total_steps: int = 0
    #: Whatever the trainer chose to report. loss, lr, grad_norm, …
    metrics: dict[str, float] = field(default_factory=dict)
    #: Recent history, newest last, for a sparkline.
    loss_history: list[float] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    model: str = ""
    started_at: float = 0.0
    updated_at: float = 0.0
    finished_at: float | None = None
    error: str = ""
    #: The launcher job, when it was started through one.
    job_id: str = ""
    pid: int = 0
    log_path: str = ""

    @property
    def progress(self) -> float | None:
        """Fraction complete, or None when the schedule is unknown.

        None rather than 0.0: a run with no declared total has made
        unknown progress, and a progress bar showing 0% for a job that is
        two hours in is a lie a dashboard tells confidently.
        """
        if self.total_steps:
            return min(1.0, self.step / self.total_steps)
        if self.total_epochs:
            return min(1.0, self.epoch / self.total_epochs)
        return None

    @property
    def percent(self) -> float | None:
        fraction = self.progress
        return None if fraction is None else round(fraction * 100, 2)

    @property
    def eta_seconds(self) -> float | None:
        """Seconds remaining, from the rate achieved so far.

        Needs a start time, some progress, a known total, and a run
        that is still going. Any of those missing means no estimate — an
        ETA invented from part of that is a number people plan around.

        The "still going" part is not pedantry: a stale run keeps its
        last measured rate, and reporting "eta 6s" for a trainer that
        died an hour ago reads as *nearly finished*, which is the
        opposite of what happened.
        """
        fraction = self.progress
        if not self.is_active:
            return None
        if not fraction or not self.started_at or fraction >= 1.0:
            return None
        elapsed = max(0.0, (self.updated_at or time.time()) - self.started_at)
        if elapsed <= 0:
            return None
        return round(elapsed / fraction - elapsed, 1)

    @property
    def is_active(self) -> bool:
        return self.state in (
            RunState.STARTING.value, RunState.RUNNING.value, RunState.PAUSED.value
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["percent"] = self.percent
        payload["eta_seconds"] = self.eta_seconds
        payload["is_active"] = self.is_active
        return payload


class ProgressReporter:
    """Written by the trainer. One atomic rewrite per update.

    ::

        reporter = ProgressReporter("run-1", total_epochs=3, total_steps=9000)
        for step in ...:
            reporter.update(step=step, epoch=epoch, loss=loss.item())
        reporter.finished()

    Deliberately does nothing clever. A training loop calling this a few
    times a second must not be able to block on it, so there is no lock,
    no append, and no fsync — just a rename over the previous file, which
    is atomic on every filesystem this runs on and cannot be observed
    half-written.
    """

    #: How much loss history to keep. Enough for a sparkline, small
    #: enough that the file stays a single small write.
    HISTORY = 120

    def __init__(
        self,
        run_id: str,
        *,
        name: str = "",
        total_epochs: int = 0,
        total_steps: int = 0,
        model: str = "",
        root: str | Path | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run = TrainingRun(
            run_id=run_id, name=name or run_id,
            total_epochs=total_epochs, total_steps=total_steps, model=model,
            started_at=time.time(), updated_at=time.time(),
            job_id=os.environ.get("HNX_JOB_ID", ""), pid=os.getpid(),
            # Set by the launcher. Without it the API can report a run's
            # progress but not its output, which is exactly the half
            # people want when a run looks wrong.
            log_path=os.environ.get("HNX_LOG_PATH", ""),
        )
        self._write()

    @property
    def path(self) -> Path:
        return self.root / f"{self.run.run_id}.json"

    def _write(self) -> None:
        scratch = self.path.with_suffix(".tmp")
        try:
            scratch.write_text(
                json.dumps(self.run.to_dict(), indent=2), encoding="utf-8"
            )
            scratch.replace(self.path)
        except OSError as exc:
            # A training run must never die because a status file could
            # not be written. Losing the reporting is bad; losing six
            # hours of training to it would be absurd.
            logger.warning("training.monitor: could not write progress: %s", exc)

    def update(
        self, *, step: int | None = None, epoch: int | None = None,
        state: str = RunState.RUNNING.value, **metrics: float,
    ) -> None:
        if step is not None:
            self.run.step = int(step)
        if epoch is not None:
            self.run.epoch = int(epoch)
        self.run.state = state
        for key, value in metrics.items():
            try:
                self.run.metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        loss = self.run.metrics.get("loss")
        if loss is not None:
            self.run.loss_history.append(loss)
            del self.run.loss_history[: -self.HISTORY]
        self.run.updated_at = time.time()
        self._write()

    def checkpoint(self, path: str | Path) -> None:
        self.run.checkpoints.append(str(path))
        self.run.updated_at = time.time()
        self._write()

    def failed(self, error: str) -> None:
        """Shorthand, so a caller never has to spell the state itself."""
        self.finished(state=RunState.FAILED.value, error=error)

    def finished(self, *, state: str = RunState.FINISHED.value, error: str = "") -> None:
        self.run.state = state
        self.run.error = error
        self.run.finished_at = time.time()
        self.run.updated_at = self.run.finished_at
        self._write()


#: A run whose file has not moved in this long, with no live process
#: behind it, is reported stale rather than running. Ten minutes is
#: comfortably longer than a slow epoch on a large model and far shorter
#: than the time someone would otherwise waste watching a dead run.
STALE_AFTER = 600.0


class TrainingMonitor:
    """Reads runs, and applies the controls.

    Merges the reporter's numbers with the process state, because
    neither alone is enough: a crashed trainer leaves a file that still
    says "running", and a live process says nothing about which epoch it
    is on.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    def _load(self, path: Path) -> TrainingRun | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        known = {f for f in TrainingRun.__dataclass_fields__}
        return TrainingRun(**{k: v for k, v in data.items() if k in known})

    def runs(self) -> list[TrainingRun]:
        found: list[TrainingRun] = []
        if not self.root.is_dir():
            return found
        for path in sorted(self.root.glob("*.json")):
            run = self._load(path)
            if run is not None:
                found.append(self._reconcile(run))
        return sorted(found, key=lambda r: r.started_at, reverse=True)

    def get(self, run_id: str) -> TrainingRun | None:
        run = self._load(self.root / f"{run_id}.json")
        if run is None:
            matches = [r for r in self.runs() if r.name == run_id]
            return matches[0] if matches else None
        return self._reconcile(run)

    def _reconcile(self, run: TrainingRun) -> TrainingRun:
        """Correct what the file says against what is actually running."""
        if not run.is_active:
            return run
        alive = _alive(run.pid) if run.pid else False
        if alive:
            if run.state == RunState.PAUSED.value or _stopped(run.pid):
                run.state = RunState.PAUSED.value
            return run
        # No process. A file that still claims to be running is a
        # trainer that died without saying so -- a segfault, an OOM
        # kill, a machine that rebooted.
        idle = time.time() - (run.updated_at or 0)
        if idle > STALE_AFTER or run.pid:
            run.state = RunState.STALE.value
            run.error = run.error or (
                f"the trainer is gone and the last update was "
                f"{int(idle)}s ago; it did not finish cleanly."
            )
        return run

    # -- controls ---------------------------------------------------

    def _signal(self, run: TrainingRun, sig: int, what: str) -> None:
        if not run.is_active:
            raise TrainingError(
                f"{run.run_id} has already ended ({run.state}), so it cannot "
                f"be {what}."
            )
        if not run.pid:
            raise TrainingError(
                f"{run.run_id} has no recorded pid, so it cannot be {what}. "
                f"It was probably not started through hypernix."
            )
        if not _alive(run.pid):
            raise TrainingError(f"{run.run_id} is not running.")
        try:
            os.killpg(os.getpgid(run.pid), sig)
        except (ProcessLookupError, PermissionError, OSError) as exc:
            raise TrainingError(f"could not {what} {run.run_id}: {exc}") from exc

    def pause(self, run: TrainingRun) -> TrainingRun:
        """SIGSTOP the run's process group.

        Note this does **not** free VRAM: the process is frozen with its
        allocations intact. That is what makes resuming instant, and it
        is also why pausing does not make the card available to anything
        else.
        """
        self._signal(run, signal.SIGSTOP, "paused")
        run.state = RunState.PAUSED.value
        self._persist(run)
        return run

    def resume(self, run: TrainingRun) -> TrainingRun:
        self._signal(run, signal.SIGCONT, "resumed")
        run.state = RunState.RUNNING.value
        self._persist(run)
        return run

    def stop(self, run: TrainingRun) -> TrainingRun:
        """SIGTERM the run's process group, if there is still one.

        Refuses a run that has already ended. Overwriting a ``finished``
        record with ``stopped`` would erase how the run actually ended
        and leave the history saying an operator killed a job that in
        fact completed.
        """
        if not run.is_active:
            raise TrainingError(f"{run.run_id} has already ended ({run.state}).")
        if run.state == RunState.PAUSED.value and run.pid and _alive(run.pid):
            # A stopped process cannot handle SIGTERM. Wake it first, or
            # it sits frozen forever and "stop" silently did nothing.
            try:
                os.killpg(os.getpgid(run.pid), signal.SIGCONT)
            except OSError:
                pass
        if run.pid and _alive(run.pid):
            self._signal(run, signal.SIGTERM, "stopped")
        run.state = RunState.STOPPED.value
        run.finished_at = run.finished_at or time.time()
        self._persist(run)
        return run

    def _persist(self, run: TrainingRun) -> None:
        path = self.root / f"{run.run_id}.json"
        scratch = path.with_suffix(".tmp")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            scratch.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
            scratch.replace(path)
        except OSError as exc:
            logger.warning("training.monitor: could not persist %s: %s", run.run_id, exc)

    # -- resources --------------------------------------------------

    def resources(self) -> dict:
        """GPU, CPU and RAM, for the same panel as the run itself.

        A loss curve without utilisation cannot tell you why a run is
        slow, and that is the question people actually have.
        """
        from ..system import gpus as _gpus

        payload: dict = {"gpus": [card.to_dict() for card in _gpus.detect()]}
        try:
            import psutil

            memory = psutil.virtual_memory()
            payload["cpu_percent"] = psutil.cpu_percent(interval=None)
            payload["ram_total_mb"] = int(memory.total / 1024 / 1024)
            payload["ram_used_mb"] = int(memory.used / 1024 / 1024)
            payload["ram_percent"] = memory.percent
        except Exception:  # noqa: BLE001 - resources must never raise
            payload["cpu_percent"] = None
            payload["ram_total_mb"] = None
            payload["ram_used_mb"] = None
            payload["ram_percent"] = None
        return payload


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


def _stopped(pid: int) -> bool:
    """Whether *pid* is in state T — SIGSTOPped rather than merely idle."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] in ("T", "t") if ")" in stat else False
