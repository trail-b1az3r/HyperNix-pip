"""``hnx-bundle`` — several quantisations of one model, in one GGUF.

    hnx-bundle build model.f16.gguf Q8_0 Q4_K_M IQ0.5_XXXL -o model.bundle.gguf
    hnx-bundle list model.bundle.gguf
    hnx-bundle extract model.bundle.gguf Q4_K_M -o model.q4km.gguf
    hnx-bundle strip model.bundle.gguf -o model.gguf

Also installed as ``multiquant``, and reachable as ``hyprslug --multi``.
See :mod:`hypernix.quant.multiquant` for what a bundle is and where the
speed-up actually comes from.
"""
from __future__ import annotations

import argparse
import json
import sys

from .gguf import GGUFError
from .multiquant import (
    MultiQuantError,
    bundle,
    extract,
    read_bundle_info,
    strip,
    variants,
)

__all__ = ["main", "cli_main"]


def _build(args) -> int:
    targets: list[str] = []
    for chunk in args.targets:
        targets.extend(part for part in chunk.replace(" ", ",").split(",") if part)
    output = args.output or f"{args.source.rsplit('.', 1)[0]}.bundle.gguf"

    def _progress(event: dict) -> None:
        if args.quiet or args.as_json:
            return
        if event.get("event") == "variant":
            print(f"  [{event['index']}/{event['total']}] {event['tier']}"
                  + ("  (default)" if event["default"] else ""), file=sys.stderr)

    report = bundle(
        args.source, output, targets,
        default=args.default,
        imatrix=args.imatrix,
        share=not args.no_share,
        progress=None if args.quiet else _progress,
    )
    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2))
    elif not args.quiet:
        print(report.describe())
        print(f"hnx-bundle: wrote {output}")
    return 0


def _list(args) -> int:
    found = variants(args.source)
    if args.as_json:
        print(json.dumps(read_bundle_info(args.source), indent=2))
        return 0
    if not found:
        print(f"{args.source} carries one quantisation, the ordinary way.")
        return 0
    print(f"{args.source}: {len(found)} variants")
    for variant in found:
        mark = "*" if variant.default else " "
        print(f"  {mark} {variant.tier:14} {variant.bits_per_weight:6.3f} bpw  "
              f"{variant.nbytes / 1e6:8.1f} MB  {variant.tensor_count} tensors")
    print()
    print("  * is the one a stock llama.cpp runs when it opens this file.")
    return 0


def _extract(args) -> int:
    output = args.output or f"{args.source.rsplit('.', 1)[0]}.{args.tier}.gguf"
    info = extract(args.source, output, args.tier)
    if args.as_json:
        print(json.dumps(info.to_dict(), indent=2))
    elif not args.quiet:
        print(f"{info.tier}: {info.tensor_count} tensors, {info.nbytes / 1e6:.1f} MB")
        print(f"hnx-bundle: wrote {output}")
    return 0


def _strip(args) -> int:
    output = args.output or f"{args.source.rsplit('.', 1)[0]}.plain.gguf"
    removed = strip(args.source, output)
    if args.as_json:
        print(json.dumps({"output": output, "variants_removed": removed}, indent=2))
    elif not args.quiet:
        print(f"hnx-bundle: wrote {output}, {removed} extra variant(s) dropped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hnx-bundle",
        description=(
            "Put several quantisations of one model into a single GGUF. The "
            "default variant keeps the ordinary tensor names, so a stock "
            "llama.cpp opens the file and runs it with nothing unusual "
            "happening."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "A bundle is roughly the size of its variants added together --\n"
            "only the tensors every variant left untouched are stored once.\n"
            "The point is not compression. It is one download instead of five,\n"
            "one page cache instead of five, and switching tiers on a served\n"
            "model without a cold mmap.\n"
        ),
    )
    # On the top-level parser *and* on every subcommand, via a parent.
    # argparse only accepts a top-level flag before the subcommand name,
    # and `hnx-bundle build model.gguf Q8_0 -q` is how everyone types it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", dest="as_json", action="store_true")
    common.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", parents=[common],
                           help="Quantise to several tiers, into one file")
    build.add_argument("source")
    build.add_argument("targets", nargs="+",
                       help="Tiers to include, space- or comma-separated")
    build.add_argument("-o", "--output")
    build.add_argument("--default", metavar="TIER",
                       help="The variant a stock loader runs (default: the first)")
    build.add_argument("--imatrix", help="Importance matrix, applied to every variant")
    build.add_argument("--no-share", action="store_true",
                       help="Do not store shared tensors once")
    build.set_defaults(func=_build)

    listing = sub.add_parser("list", parents=[common], help="What quantisations a file carries")
    listing.add_argument("source")
    listing.set_defaults(func=_list)

    pull = sub.add_parser("extract", parents=[common], help="One variant out, as an ordinary GGUF")
    pull.add_argument("source")
    pull.add_argument("tier")
    pull.add_argument("-o", "--output")
    pull.set_defaults(func=_extract)

    plain = sub.add_parser("strip", parents=[common], help="Keep only the default variant")
    plain.add_argument("source")
    plain.add_argument("-o", "--output")
    plain.set_defaults(func=_strip)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except MultiQuantError as exc:
        print(f"hnx-bundle: {exc}", file=sys.stderr)
        return 1
    except (GGUFError, OSError) as exc:
        # A missing or unreadable file is an ordinary thing to get wrong
        # at a command line. A traceback for it buries the one line that
        # says which path was not there.
        print(f"hnx-bundle: {exc}", file=sys.stderr)
        return 2


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
