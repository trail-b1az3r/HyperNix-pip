"""``hypernix-t1 index`` — the command line over :mod:`modelindex`.

Also reachable as ``python -m hypernix.t1api.modelindex_cli`` so the
shell wrapper has something to call that does not depend on a console
script being on ``PATH``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .modelindex import (
    DEFAULT_MODELS_DIR,
    IndexError_,
    build_entry,
    index_directory,
    write_registry,
)

__all__ = ["main", "cli_main"]

_EPILOG = """\
Reads every .gguf under the models directory and writes the T1 model
registry from what the files actually say: architecture, context length,
and a parameter count summed from the tensor table rather than guessed
from the filename.

Pricing, plan and priority are policy, not measurements, so they come
from the flags and default to free-and-counted.

An entry already in the registry is left alone. --refresh re-reads the
measured fields for those too, and still leaves pricing, plan, priority,
status and notes exactly as you set them.
"""


def _parameters(billions: float) -> str:
    """A parameter count a human reads, at any size.

    The registry stores billions because that is the field's unit, but
    "0.00B" for a 1.4-million-parameter model is a rounding artefact
    reported as a measurement -- and a draft or a test model lands
    exactly there.
    """
    if billions >= 1.0:
        return f"{billions:.2f}B"
    if billions >= 0.001:
        return f"{billions * 1000:.0f}M"
    return f"{billions * 1e9:,.0f}"


def _human(result: dict, rows: list, *, dry_run: bool) -> None:
    unreadable = [r for r in rows if not r.readable]
    usable = [r for r in rows if r.readable]

    if not rows:
        print("No .gguf files found.")
        return

    for row in usable:
        tier = row.tier or "upstream"
        mark = "hnxrun" if row.is_extension else "llama.cpp"
        print(f"  {row.model_id}")
        print(f"      {_parameters(row.parameters_b):>9}  {tier:12} {mark:10} "
              f"ctx {row.context_limit}  {row.architecture}")
        if row.assumed:
            print(f"      assumed: {', '.join(row.assumed)} "
                  f"(not in the file's metadata)")
    for row in unreadable:
        print(f"  {row.path.name}: unreadable — {row.error}")

    print()
    if dry_run:
        print(f"  would write {len(usable)} entr"
              f"{'y' if len(usable) == 1 else 'ies'} to {result['path']}")
        print("  nothing written (--dry-run)")
        return

    print(f"  {result['path']}")
    if result["added"]:
        print(f"    added     {len(result['added'])}: "
              f"{', '.join(result['added'])}")
    if result["updated"]:
        print(f"    refreshed {len(result['updated'])}: "
              f"{', '.join(result['updated'])}")
    if result["unchanged"]:
        count = len(result["unchanged"])
        noun = "entry" if count == 1 else "entries"
        # The hint is only true when --refresh was not passed. Printing
        # it after a --refresh run that simply found nothing to change
        # tells the reader to do what they just did.
        hint = "" if result.get("refresh") else " (pass --refresh to re-read them)"
        print(f"    left      {count} already-registered {noun} unchanged{hint}")
    print(f"    {result['total']} model(s) in the registry")
    print()
    if result.get("discoverable"):
        # No env var to set: the server looks here on its own. Telling
        # someone to configure a path they do not need is how a working
        # setup acquires a setting nobody can later explain.
        print("  The server reads this location on its own. Restart it:")
        print("    hypernix-t1 restart")
    else:
        print("  This is not somewhere the server looks. Point it there:")
        print(f"    T1_MODEL_REGISTRY_PATH={result['path']}")
        print("    hypernix-t1 restart")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hypernix-t1 index",
        description="Index a folder of GGUFs into the T1 model registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument("--dir", dest="directory", default=str(DEFAULT_MODELS_DIR),
                        help=f"Where the models are (default: {DEFAULT_MODELS_DIR}).")
    parser.add_argument("-o", "--output", default="",
                        help="Registry to write. Defaults to models.json beside "
                             "the config, or ./models.json.")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-read the measured fields of models already "
                             "registered. Pricing, plan, priority, status and "
                             "notes are still left as you set them.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be written and write nothing.")
    parser.add_argument("--plan", default="free",
                        help="Minimum plan for new entries (default: free).")
    parser.add_argument("--input-price", type=float, default=0.0,
                        help="Price per 1k input tokens for new entries.")
    parser.add_argument("--output-price", type=float, default=0.0,
                        help="Price per 1k output tokens for new entries.")
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--availability", default="public",
                        choices=("public", "private", "internal", "beta"))
    parser.add_argument("--priority", type=int, default=10,
                        help="Routing priority for new entries; lower wins.")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Machine-readable output.")

    args = parser.parse_args(argv)

    output = args.output
    if not output:
        from .registry import discover, registry_locations

        # Write where the server will look, and to the file it is
        # already reading if there is one. Indexing to a path the server
        # does not consult is the failure this whole command exists to
        # avoid -- it produces a correct registry and an empty model
        # list, with nothing connecting the two.
        existing = discover()
        output = str(existing) if existing else str(registry_locations()[0])

    try:
        rows = index_directory(args.directory)
    except IndexError_ as exc:
        print(f"hypernix-t1 index: {exc}", file=sys.stderr)
        return 1

    usable = [r for r in rows if r.readable]
    entries = [
        build_entry(
            row, plan=args.plan, input_price=args.input_price,
            output_price=args.output_price, currency=args.currency,
            availability=args.availability, routing_priority=args.priority,
        )
        for row in usable
    ]

    from .registry import registry_locations

    discoverable = Path(output).resolve() in {
        p.resolve() for p in registry_locations()
    }

    if args.dry_run:
        result = {"path": output, "added": [], "updated": [],
                  "unchanged": [], "total": len(entries),
                  "refresh": args.refresh}
    else:
        try:
            result = write_registry(entries, output, refresh=args.refresh)
        except IndexError_ as exc:
            print(f"hypernix-t1 index: {exc}", file=sys.stderr)
            return 1
        result["refresh"] = args.refresh
    result["discoverable"] = discoverable

    if args.as_json:
        print(json.dumps(
            {**result, "dry_run": args.dry_run,
             "models": [r.to_dict() for r in rows]},
            indent=2,
        ))
    else:
        _human(result, rows, dry_run=args.dry_run)

    # An unreadable file is worth a non-zero exit: the registry written
    # is missing a model the operator put there on purpose.
    return 2 if any(not r.readable for r in rows) else 0


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
