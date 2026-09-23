"""``hypernix elements`` — list, inspect, scaffold, and run elements.

    hypernix elements list
    hypernix elements info Mg
    hypernix elements plan Mg              # what magnesium would change
    hypernix elements run Mg --yes         # change it; Ctrl-C puts it back
    hypernix elements new Na               # scaffold a user element

``run`` holds for as long as the element is active. An element that
changes other processes and then exits leaves nobody to change them
back, so "apply and quit" is not offered.
"""
from __future__ import annotations

import argparse
import json
import signal
import threading

from ..system.errorcodes import HyperNixError
from . import carbon
from .hydrogen import default_registry

__all__ = ["main", "cli_main"]


def _registry(args):
    reg = default_registry(allow_experimental=getattr(args, "experimental", False))
    carbon.load_user_elements(reg)
    return reg


def _list(args) -> int:
    reg = _registry(args)
    rows = [s.to_dict() for s in reg.specs()]
    if args.json:
        print(json.dumps({"elements": rows, "failures": reg.failures}, indent=2))
        return 0
    for row in rows:
        tags = []
        if row["experimental"]:
            tags.append("experimental")
        if row["user"]:
            tags.append("user")
        tag = f"  [{', '.join(tags)}]" if tags else ""
        print(f"{row['number']:>3} {row['symbol']:<3} {row['name']:<12} "
              f"{row['summary']}{tag}")
    for symbol, why in sorted(reg.failures.items()):
        print(f"  ! {symbol}: {why}")
    return 0


def _info(args) -> int:
    print(json.dumps(_registry(args).lookup(args.element).spec.to_dict(), indent=2))
    return 0


def _plan(args) -> int:
    element = _registry(args).instance(args.element)
    plan = getattr(element, "plan", None)
    if plan is None:
        print(f"{element.spec.symbol} has nothing to plan; it does not "
              f"change anything outside HyperNix.")
        return 0
    result = plan()
    for decision in ("limit", "protected", "not-ours"):
        found = result.of(decision)
        if not found:
            continue
        print(f"{decision} ({len(found)}):")
        for target in sorted(found, key=lambda t: t.name.lower())[: args.limit]:
            why = f"  — {target.reason}" if target.reason else ""
            print(f"  {target.pid:>7}  {target.name}{why}")
    print(result.summary())
    return 0


def _run(args) -> int:
    reg = _registry(args)
    element = reg.instance(args.element)
    if "processes" in element.spec.permissions and not args.yes:
        _plan(args)
        print("\nThis changes other programs. Re-run with --yes to go ahead; "
              "Ctrl-C puts everything back.")
        return 1
    reg.activate(args.element)
    print(f"{element.spec.symbol} ({element.spec.name}) active — "
          f"{element.status().get('limited', '')} Ctrl-C to stop.")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        stop.wait()
    except KeyboardInterrupt:
        pass
    finally:
        reg.deactivate(args.element)
        print(f"\n{element.spec.symbol} stopped; everything it changed is back.")
    return 0


def _new(args) -> int:
    path = carbon.scaffold(args.symbol, registry=_registry(args))
    print(f"wrote {path}\nedit it, then: hypernix elements list")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hypernix elements",
                                     description="Addons named after the periodic table.")
    parser.add_argument("--experimental", action="store_true",
                        help="allow periods 6 and 7 (element 55 onward)")
    sub = parser.add_subparsers(dest="command", required=True)

    ls = sub.add_parser("list", help="every element, built-in and yours")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=_list)

    info = sub.add_parser("info", help="one element's spec")
    info.add_argument("element")
    info.set_defaults(func=_info)

    plan = sub.add_parser("plan", help="what an element would change")
    plan.add_argument("element")
    plan.add_argument("--limit", type=int, default=40)
    plan.set_defaults(func=_plan)

    run = sub.add_parser("run", help="activate until Ctrl-C")
    run.add_argument("element")
    run.add_argument("--yes", action="store_true")
    run.add_argument("--limit", type=int, default=40)
    run.set_defaults(func=_run)

    new = sub.add_parser("new", help="scaffold a user element")
    new.add_argument("symbol")
    new.set_defaults(func=_new)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except HyperNixError as exc:
        print(str(exc))
        return 2


def cli_main(argv: list[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
