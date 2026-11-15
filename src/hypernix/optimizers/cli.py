"""hypernix.optimizers.cli — ``hnx optimizer``.

    hnx optimizer list                       # every optimizer and what it supports
    hnx optimizer select --params 120000000 --verified
    hnx optimizer select --objective stability --feature factored_state
    hnx optimizer benchmark                  # measure them against each other

``select`` explains what ``OptimizerFrameworkV2.auto_select`` would choose
and why, without building anything.
"""
from __future__ import annotations

import argparse
import json
import sys

__all__ = ["cli_main"]

_USAGE = """usage: hnx optimizer <command> [options]

commands:
  list        the optimizers, their memory and what each supports
  select      what auto-selection would choose, and why
  benchmark   measure the optimizers on the same models, data and steps

Run `hnx optimizer <command> --help` for a command's options."""


def _list(argv: list[str]) -> int:
    from hypernix.optimizers.optimizer_framework_v2 import OptimizerFrameworkV2

    parser = argparse.ArgumentParser(prog="hnx optimizer list")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--measured", metavar="FILE", default=None,
                        help="an optimizer-bench.json to show measured figures from")
    args = parser.parse_args(argv)
    measured = None
    if args.measured:
        with open(args.measured, encoding="utf-8") as handle:
            measured = json.load(handle).get("measured")
    rows = OptimizerFrameworkV2.table(measured)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for row in rows:
        print(f"{row['key']:<8} {row['optimizer']}")
        print(f"         speed {row['speed']}; memory {row['memory']}; stability "
              f"{row['stability']}; adaptivity {row['adaptivity']} (targets)")
        print(f"         best for: {row['best_use']}")
        if row.get("note"):
            print(f"         {row['note']}")
        if row.get("measured"):
            got = row["measured"]
            print(f"         measured: {got.get('ms_per_step')} ms/step, "
                  f"{got.get('state_bytes_per_parameter')} state bytes/parameter "
                  f"({got.get('device')})")
    if not measured:
        print("\nNo measured figures shown. Run `hnx optimizer benchmark --out-dir DIR` and "
              "pass `--measured DIR/optimizer-bench.json`.")
    return 0


def _select(argv: list[str]) -> int:
    from hypernix.optimizers.optimizer_framework_v2 import (
        FEATURES,
        OBJECTIVES,
        OptimizerFrameworkV2,
    )

    parser = argparse.ArgumentParser(
        prog="hnx optimizer select",
        description="Show what auto-selection would choose and why.")
    parser.add_argument("--params", type=int, default=None,
                        help="the model's trainable parameter count")
    parser.add_argument("--verified", action="store_true",
                        help="the count was counted from the model, not estimated")
    parser.add_argument("--objective", default="balanced", choices=sorted(OBJECTIVES))
    parser.add_argument("--feature", action="append", default=[], choices=sorted(FEATURES),
                        help="a feature the optimizer must support (repeatable)")
    parser.add_argument("--optimizer", default=None,
                        help="an explicit choice: honoured, or refused with the reason")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--measured", metavar="FILE", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    measured = None
    if args.measured:
        with open(args.measured, encoding="utf-8") as handle:
            measured = json.load(handle).get("measured")
    try:
        chosen = OptimizerFrameworkV2.auto_select(
            parameter_count=args.params, parameter_count_verified=args.verified,
            objective=args.objective, required_features=tuple(args.feature),
            requested=args.optimizer, device=args.device, measured=measured)
    except ValueError as exc:
        print(f"optimizer select: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(chosen.to_dict(), indent=2))
        return 0
    print(f"{chosen.key}: {chosen.label}")
    print(f"why: {chosen.reason} [{chosen.reason_code}]")
    for note in chosen.notes:
        print(f"note: {note}")
    for other in chosen.considered:
        print(f"passed over {other.get('key', '?')}: {other.get('why_not', '')}")
    return 0


def cli_main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return 0
    command, rest = argv[0], argv[1:]
    if command == "list":
        return _list(rest)
    if command == "select":
        return _select(rest)
    if command in ("benchmark", "bench"):
        from hypernix.optimizers.benchmark import main as _bench

        return _bench(rest)
    print(f"hnx optimizer: unknown command {command!r}\n\n{_USAGE}", file=sys.stderr)
    return 2
