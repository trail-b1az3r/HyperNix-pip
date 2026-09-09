"""``hypernix-t1 training`` — the command line over the training monitor.

Reads the same records the API serves, and applies the same controls,
without a server in between. That matters more than it looks: the
machine a run is on is usually reachable when the API is not — the
server crashed, the port moved, the run *is* what is eating the box —
and "what is my training doing" should not depend on the thing that is
struggling.

There is no key check here on purpose. The controls are SIGSTOP and
SIGTERM to processes this user already owns, and the records are files
this user can already read. Asking for a credential to do what ``kill``
does anyway would be theatre; the access control is the filesystem's,
and it is the same one that decided whether this command could run at
all. Over the network is where credentials belong, and that path is
``t1api.routers.training``, which is admin-gated.

Also reachable as ``python -m hypernix.t1api.training_cli`` so the shell
wrapper has something to call that does not depend on a console script
being on ``PATH``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..training.monitor import RunState, TrainingError, TrainingMonitor, default_root

__all__ = ["main", "cli_main"]

_EPILOG = """\
Runs report themselves; nothing polls them. A run started through
`hypernix-t1 launch-script` appears here on its own, because the
launcher tells it what to report as.

  hypernix-t1 training                     every run, newest first
  hypernix-t1 training --active            only the ones still going
  hypernix-t1 training --show qwen-sft     one run in full
  hypernix-t1 training --logs qwen-sft     the tail of its output
  hypernix-t1 training --pause qwen-sft    freeze it (keeps its VRAM)
  hypernix-t1 training --resume qwen-sft   thaw it
  hypernix-t1 training --stop qwen-sft     SIGTERM, so it can checkpoint
  hypernix-t1 training --resources         GPU, CPU and RAM right now

A run can be named by its id or by the job name it was launched under.
"""

_MARK = {
    RunState.RUNNING.value: "·",
    RunState.STARTING.value: "·",
    RunState.PAUSED.value: "‖",
    RunState.FINISHED.value: "✓",
    RunState.FAILED.value: "✗",
    RunState.STOPPED.value: "■",
    RunState.STALE.value: "?",
}


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _progress(run) -> str:
    if run.percent is None:
        # Unknown, said as unknown. A bar at 0% for a run two hours in
        # is the one output people act on and should not.
        return f"step {run.step}" if run.step else "—"
    filled = int(run.percent / 10)
    return f"[{'#' * filled}{'.' * (10 - filled)}] {run.percent:5.1f}%"


def _list(monitor: TrainingMonitor, runs, *, active_only: bool = False) -> None:
    if not runs:
        # "Nothing recorded" and "nothing running" are different answers
        # and the wrong one sends people looking for a lost record.
        if active_only:
            total = len(monitor.runs())
            print("No training is running.")
            if total:
                print(f"  {total} run(s) are recorded and have ended. "
                      f"Drop --active to see them.")
            return
        print(f"No training runs recorded in {monitor.root}.")
        print()
        print("  Runs appear here once one reports itself. Start one with:")
        print("    hypernix-t1 launch-script ./train.py --name my-run --detach")
        return
    for run in runs:
        mark = _MARK.get(run.state, "·")
        loss = run.metrics.get("loss")
        print(f"  {mark} {run.name or run.run_id}")
        print(
            f"      {run.state:9} {_progress(run):24} "
            f"loss {f'{loss:.4f}' if loss is not None else '—':>9}  "
            f"eta {_duration(run.eta_seconds)}"
        )
        if run.error:
            print(f"      {run.error}")


def _show(run) -> None:
    print(f"  {run.name or run.run_id}   ({run.run_id})")
    print(f"      state       {run.state}")
    print(f"      model       {run.model or '—'}")
    print(f"      progress    {_progress(run)}")
    print(f"      step        {run.step}"
          + (f" / {run.total_steps}" if run.total_steps else ""))
    print(f"      epoch       {run.epoch}"
          + (f" / {run.total_epochs}" if run.total_epochs else ""))
    print(f"      eta         {_duration(run.eta_seconds)}")
    for name, value in sorted(run.metrics.items()):
        print(f"      {name:11} {value:.6g}")
    if run.pid:
        print(f"      pid         {run.pid}")
    if run.job_id:
        print(f"      job         {run.job_id}")
    if run.log_path:
        print(f"      log         {run.log_path}")
    if run.checkpoints:
        print(f"      checkpoints {len(run.checkpoints)}")
        for path in run.checkpoints[-5:]:
            here = "" if Path(path).exists() else "   (gone)"
            print(f"                  {path}{here}")
    if run.error:
        print(f"      error       {run.error}")


def _resources(payload: dict) -> None:
    cards = payload.get("gpus") or []
    if not cards:
        print("  No GPU detected (nvidia-smi / rocm-smi found nothing).")
    for card in cards:
        used, total = card.get("memory_used_mb"), card.get("memory_total_mb")
        memory = f"{used}/{total} MB" if used is not None and total else "—"
        util = card.get("utilization_pct")
        print(f"  GPU {card.get('index', '?')}  {card.get('name', '?')}")
        print(f"      {card.get('vendor', '?'):8} {memory:>16}  "
              f"util {f'{util}%' if util is not None else '—'}")
    cpu = payload.get("cpu_percent")
    print(f"  CPU  {f'{cpu:.0f}%' if cpu is not None else '—'}")
    ram_used, ram_total = payload.get("ram_used_mb"), payload.get("ram_total_mb")
    if ram_total:
        print(f"  RAM  {ram_used}/{ram_total} MB ({payload.get('ram_percent')}%)")


def _require(monitor: TrainingMonitor, run_id: str):
    run = monitor.get(run_id)
    if run is None:
        print(f"hypernix-t1 training: no run named {run_id!r} in {monitor.root}",
              file=sys.stderr)
        known = [r.name or r.run_id for r in monitor.runs()]
        if known:
            print(f"  known runs: {', '.join(known)}", file=sys.stderr)
        return None
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hypernix-t1 training",
        description="What training is doing on this machine, and the controls.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default="",
                        help="Where run records live (default: the "
                             "server's training directory).")
    parser.add_argument("--active", action="store_true",
                        help="Only runs that have not ended.")
    parser.add_argument("--show", metavar="RUN", default="",
                        help="One run, in full.")
    parser.add_argument("--logs", metavar="RUN", default="",
                        help="The tail of a run's output.")
    parser.add_argument("--tail", type=int, default=50,
                        help="Lines of log to show (default 50).")
    parser.add_argument("--pause", metavar="RUN", default="",
                        help="Freeze a run. Keeps its GPU memory.")
    parser.add_argument("--resume", metavar="RUN", default="",
                        help="Thaw a paused run.")
    parser.add_argument("--stop", metavar="RUN", default="",
                        help="SIGTERM a run, so it can checkpoint on the "
                             "way out.")
    parser.add_argument("--resources", action="store_true",
                        help="GPU, CPU and RAM right now.")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Machine-readable output.")

    args = parser.parse_args(argv)
    monitor = TrainingMonitor(args.root or default_root())

    if args.resources:
        payload = monitor.resources()
        print(json.dumps(payload, indent=2)) if args.as_json else _resources(payload)
        return 0

    for action in ("pause", "resume", "stop"):
        target = getattr(args, action)
        if not target:
            continue
        run = _require(monitor, target)
        if run is None:
            return 1
        try:
            run = getattr(monitor, action)(run)
        except TrainingError as exc:
            print(f"hypernix-t1 training: {exc}", file=sys.stderr)
            return 1
        if args.as_json:
            print(json.dumps(run.to_dict(), indent=2))
            return 0
        print(f"  {run.name or run.run_id} is now {run.state}.")
        if action == "pause":
            # The thing everyone assumes the opposite of.
            print("  Its GPU memory is still allocated — pausing does not "
                  "free the card.")
        return 0

    if args.logs:
        run = _require(monitor, args.logs)
        if run is None:
            return 1
        if not run.log_path:
            print("  This run has no log file. It was started outside the "
                  "launcher, so its output went wherever it was started from.")
            return 0
        try:
            lines = Path(run.log_path).read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError as exc:
            print(f"hypernix-t1 training: could not read {run.log_path}: {exc}",
                  file=sys.stderr)
            return 1
        print("\n".join(lines[-args.tail:]))
        return 0

    if args.show:
        run = _require(monitor, args.show)
        if run is None:
            return 1
        print(json.dumps(run.to_dict(), indent=2)) if args.as_json else _show(run)
        return 0

    runs = monitor.runs()
    if args.active:
        runs = [run for run in runs if run.is_active]
    if args.as_json:
        print(json.dumps(
            {"runs": [r.to_dict() for r in runs], "count": len(runs),
             "root": str(monitor.root)},
            indent=2,
        ))
    else:
        _list(monitor, runs, active_only=args.active)
    return 0


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
