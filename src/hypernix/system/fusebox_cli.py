"""``hnx fusebox`` — the command line over :mod:`hypernix.system.fusebox`.

    hnx fusebox status                       what the cards are doing now
    hnx fusebox watch --target 78            manage temperature continuously
    hnx fusebox watch --underclock --yes     ...and lower power limits to do it
    hnx fusebox restore                      undo what a crashed run left
    hnx fusebox plan --target 78             what it would do, changing nothing

Two things shape this file.

**Nothing hardware-changing happens without being asked twice.**
``--underclock`` turns the feature on and ``--yes`` confirms it. Without
both, every subcommand is read-only: it reports what it *would* set and
sets nothing. A power limit outlives the process that changed it, so a
flag left in shell history should not be enough to change one.

**It is usable in a script.** ``--json`` on every subcommand, progress on
stderr, and exit codes that mean something: 0 fine, 1 could not start,
2 a card is over the trip temperature right now, 3 changes are still
applied from an earlier run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .fusebox import (
    DEFAULT_POLL_SECONDS,
    DEFAULT_RESET_C,
    DEFAULT_TARGET_C,
    DEFAULT_TRIP_C,
    TRIP,
    ClockControl,
    FuseBox,
    Policy,
    read_snapshot,
    restore_all,
    watch,
)

__all__ = ["main", "cli_main"]

_EPILOG = """\
what the numbers mean
  --target   where a card should sit. Above it, steps get paced.
  --trip     the fuse. Above it, the run stops until --reset is reached.
  --reset    how cool it must get before a tripped run resumes.

this is not a speedup, and does not claim to be
  Holding a temperature costs throughput. Measured against a thermal
  model (see tests/test_fusebox.py), running flat out and taking the
  driver's own throttling is faster than either lever here: a lowered
  power limit costs 2-7%, and pausing between steps costs 12-24%. A
  throttled card still does most of the work; a paused one does none.
  What you buy is the temperature you asked for, at a cost this prints
  in seconds and percent when the run ends. --underclock is the cheaper
  of the two levers by three to four times, which is why it takes over
  from pausing when you allow it.

exit status
  0  fine
  1  could not start (bad flags, no sensors)
  2  a card is at or above the trip temperature right now
  3  power limits from an earlier run are still applied

fusebox lowers power limits and puts them back. It will not raise one
above the card's default, with any flag. Underclocking needs both
--underclock and --yes, and it never escalates privileges: if a vendor
tool refuses, that is reported and the run is paced instead.
"""


def _policy_from(args: argparse.Namespace) -> Policy:
    policy = Policy(
        target_c=args.target,
        trip_c=args.trip,
        reset_c=args.reset,
        cpu_trip_c=args.cpu_trip,
        poll_seconds=args.interval,
        max_ease=args.max_ease,
        auto_underclock=bool(getattr(args, "underclock", False)),
        limit_cpu_threads=bool(getattr(args, "cpu_threads", False)),
    )
    policy.validate()
    return policy


def _add_policy_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET_C,
                        help=f"Where a card should sit, °C "
                             f"(default {DEFAULT_TARGET_C:.0f}).")
    parser.add_argument("--trip", type=float, default=DEFAULT_TRIP_C,
                        help=f"Stop the run above this, °C "
                             f"(default {DEFAULT_TRIP_C:.0f}).")
    parser.add_argument("--reset", type=float, default=DEFAULT_RESET_C,
                        help=f"Resume below this, °C "
                             f"(default {DEFAULT_RESET_C:.0f}).")
    parser.add_argument("--cpu-trip", type=float, default=None,
                        help="Also trip on CPU temperature, °C. Off by "
                             "default: a hot CPU during data loading is "
                             "normal and is not the GPU's problem.")
    parser.add_argument("--interval", type=float, default=DEFAULT_POLL_SECONDS,
                        help=f"Seconds between sensor reads "
                             f"(default {DEFAULT_POLL_SECONDS:.0f}).")
    parser.add_argument("--max-ease", type=float, default=0.5,
                        help="Most of a step's duration that may be spent "
                             "paused (default 0.5).")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="One machine-readable object on stdout.")


def _add_change_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--underclock", action="store_true",
                        help="Lower power limits under sustained heat. "
                             "Needs --yes as well.")
    parser.add_argument("--yes", action="store_true",
                        help="Confirm hardware changes. For scripts.")
    parser.add_argument("--cpu-threads", action="store_true",
                        help="Also halve this process's torch thread count "
                             "while easing. Job-scoped; needs no privileges.")


def _describe(snapshot, verdict) -> str:
    lines = []
    for card in snapshot.cards:
        temp = "  ?  " if card.temperature_c is None else f"{card.temperature_c:5.1f}"
        power = "" if card.power_w is None else f"  {card.power_w:5.0f}W"
        limit = "" if card.power_limit_w is None else f" / {card.power_limit_w:.0f}W"
        util = "" if card.utilization_pct is None else f"  {card.utilization_pct:3.0f}%"
        lines.append(
            f"  [{card.index}] {card.vendor:<7} {card.name[:32]:<32} "
            f"{temp}°C{power}{limit}{util}"
        )
    if not lines:
        lines.append("  no GPUs found")
    if snapshot.cpu_celsius is not None:
        lines.append(f"  cpu     {snapshot.cpu_celsius:5.1f}°C")
    lines.append(f"  -> {verdict.action}: {verdict.reason}")
    return "\n".join(lines)


def _exit_for(box: FuseBox) -> int:
    if box.verdict.action == TRIP:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_status(args: argparse.Namespace) -> int:
    policy = _policy_from(args)
    box = FuseBox(policy, apply_changes=False)
    box.poll(force=True)

    pending = ClockControl(dry_run=True)
    stale = pending.load_state()

    if args.as_json:
        payload = box.status()
        payload["stale_changes"] = pending.applied if stale else {}
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(_describe(box.snapshot, box.verdict))
        if stale:
            print("\n  power limits from an earlier run are still applied:")
            for index, watts in pending.applied.items():
                original = pending.original.get(index)
                back = f" (was {original:.0f}W)" if original else ""
                print(f"    card {index}: {watts:.0f}W{back}")
            print("  run `hnx fusebox restore` to put them back.")
    if stale:
        return 3
    return _exit_for(box)


def _cmd_plan(args: argparse.Namespace) -> int:
    """What it would do, having changed nothing.

    Separate from ``status`` because it answers a different question:
    status is "how hot is it", plan is "if I turn this on, what happens
    to my cards". A person about to hand a tool the power limits of an
    expensive GPU should be able to see the commands first.
    """
    args.underclock = True
    policy = _policy_from(args)
    control = ClockControl(dry_run=True)
    limits = control.limits()
    snapshot = read_snapshot()

    rows = []
    for lim in limits:
        baseline = lim.default_w or lim.current_w
        floor = None if baseline is None else baseline * policy.underclock_floor
        rows.append({
            "index": lim.index,
            "vendor": lim.vendor,
            "current_w": lim.current_w,
            "default_w": lim.default_w,
            "lowest_fusebox_would_go_w": None if floor is None else round(floor, 1),
            "step_w": None if baseline is None
                      else round(baseline * policy.underclock_step, 1),
        })

    if args.as_json:
        print(json.dumps({
            "policy": policy.to_dict(),
            "cards": rows,
            "tool_available": control.available(),
            "snapshot": snapshot.to_dict(),
        }, indent=2, default=str))
    else:
        if not control.available():
            print("  no vendor tool that can set a power limit is installed.")
            print("  fusebox would still pace the run, which needs none.")
        for row in rows:
            low = row["lowest_fusebox_would_go_w"]
            print(f"  [{row['index']}] {row['vendor']}: now "
                  f"{row['current_w']}W, default {row['default_w']}W, "
                  f"would go no lower than "
                  f"{'?' if low is None else f'{low:.0f}W'} "
                  f"in {row['step_w']}W steps")
        if not rows:
            print("  no cards report a power limit.")
        print("\n  nothing was changed. Add --underclock --yes to a `watch` "
              "run to allow it.")
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    policy = _policy_from(args)
    apply_changes = bool(args.underclock and args.yes)
    if args.underclock and not args.yes:
        print(
            "  --underclock changes a card's power limit, which outlives this "
            "process. Add --yes to confirm.\n"
            "  Without it this run will pace the training instead, which "
            "changes nothing.",
            file=sys.stderr,
        )

    ticks = args.count if args.count > 0 else None
    started = time.time()

    def report(box: FuseBox) -> None:
        if args.as_json:
            print(json.dumps(box.status(), default=str), flush=True)
        elif not args.quiet:
            stamp = time.strftime("%H:%M:%S")
            hot = box.snapshot.hottest_c
            hot_s = "  ?  " if hot is None else f"{hot:5.1f}"
            print(f"  {stamp}  {hot_s}°C  {box.verdict.action:<5} "
                  f"{box.verdict.reason}", file=sys.stderr)

    box = watch(policy, apply_changes=apply_changes,
                iterations=ticks, on_tick=report)

    elapsed = time.time() - started
    if args.as_json:
        payload = box.status()
        payload["elapsed_seconds"] = round(elapsed, 1)
        payload["summary"] = box.summary()
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"\n  {box.summary()}", file=sys.stderr)
    return _exit_for(box)


def _cmd_restore(args: argparse.Namespace) -> int:
    control = ClockControl(dry_run=True)
    if not control.load_state():
        if args.as_json:
            print(json.dumps({"restored": 0, "note": "nothing was applied"}))
        else:
            print("  nothing to restore: no power limits are recorded as "
                  "changed.")
        return 0

    if not args.yes:
        if args.as_json:
            print(json.dumps({
                "restored": 0,
                "would_restore": {str(k): v for k, v in control.original.items()},
                "note": "add --yes to apply",
            }, indent=2))
        else:
            print("  would restore:")
            for index, original in control.original.items():
                now = control.applied.get(index)
                print(f"    card {index}: {now:.0f}W -> {original:.0f}W"
                      if now else f"    card {index}: -> {original:.0f}W")
            print("  add --yes to apply.")
        return 3

    done = restore_all()
    if args.as_json:
        print(json.dumps({"restored": done}))
    else:
        print(f"  restored {done} card(s).")
    return 0 if done else 3


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hnx fusebox",
        description="Hold a GPU at a temperature you chose, and stop the "
                    "run if it goes past a limit anyway. Costs throughput; "
                    "says how much.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    status = subparsers.add_parser(
        "status", help="What the cards are doing now.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_EPILOG,
    )
    _add_policy_flags(status)
    status.set_defaults(handler=_cmd_status, underclock=False, cpu_threads=False)

    plan = subparsers.add_parser(
        "plan", help="What underclocking would do. Changes nothing.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_EPILOG,
    )
    _add_policy_flags(plan)
    plan.set_defaults(handler=_cmd_plan, cpu_threads=False)

    watch_parser = subparsers.add_parser(
        "watch", help="Manage temperature continuously.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_EPILOG,
    )
    _add_policy_flags(watch_parser)
    _add_change_flags(watch_parser)
    watch_parser.add_argument("-n", "--count", type=int, default=0,
                              help="Stop after this many reads (0: until "
                                   "interrupted).")
    watch_parser.add_argument("-q", "--quiet", action="store_true",
                              help="No per-tick line on stderr.")
    watch_parser.set_defaults(handler=_cmd_watch)

    restore = subparsers.add_parser(
        "restore", help="Undo power limits left applied by an earlier run.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_EPILOG,
    )
    restore.add_argument("--yes", action="store_true",
                         help="Confirm. Without it, this only reports.")
    restore.add_argument("--json", dest="as_json", action="store_true")
    restore.set_defaults(handler=_cmd_restore)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    # A bare `hnx fusebox` is the question people mean by it.
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["status", *argv]

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        args = parser.parse_args(["status"])

    try:
        return args.handler(args)
    except ValueError as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


def cli_main(argv: list[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
