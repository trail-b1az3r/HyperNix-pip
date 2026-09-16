"""``dflash1`` — derive a standalone speculative-decoding draft.

    dflash1 model.f16.gguf -o model.draft.gguf
    dflash1 model.f16.gguf --depth 0.4 --quant Q4_0
    dflash1 model.draft.gguf --info

Then::

    llama-server -m model.gguf --model-draft model.draft.gguf

The same derivation ``hyprslug --draft dflash1`` does, under the name
people look for. See :mod:`hypernix.quant.dflash1` for what the draft is
and — more to the point — what it is not.
"""
from __future__ import annotations

import argparse
import json
import sys

from .dflash1 import Dflash1Error, derive, read_draft_info

__all__ = ["main", "cli_main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dflash1",
        description=(
            "Derive a draft model from a GGUF and write it as its own GGUF, "
            "for a runtime that takes --model-draft. Nothing is trained: the "
            "draft is the base model with layers dropped and the rest "
            "quantised hard."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "A draft is only worth running if its proposals are accepted often\n"
            "enough. Below roughly 30% acceptance, speculative decoding makes\n"
            "generation *slower* -- the base model spends forward passes\n"
            "checking tokens it then throws away. `dflash2 speculate` measures\n"
            "the rate; measure before shipping.\n"
        ),
    )
    parser.add_argument("source", help="The GGUF to derive a draft from")
    parser.add_argument("-o", "--output", help="Where to write the draft")
    parser.add_argument(
        "--depth", type=float, default=0.25, metavar="FRACTION",
        help="Fraction of the base's layers to keep (default: 0.25). The "
             "first and last are always among them: a model missing its "
             "first block proposes tokens from a different distribution "
             "entirely, and every one of them is rejected.",
    )
    parser.add_argument(
        "--layers", metavar="I,J,K",
        help="Exact layer indices to keep, instead of --depth.",
    )
    parser.add_argument(
        "--quant", default="Q4_0", metavar="FORMAT",
        help="Block format for the draft's tensors (default: Q4_0).",
    )
    parser.add_argument(
        "--draft-tokens", type=int, default=4, metavar="N",
        help="Tokens proposed per round (default: 4).",
    )
    parser.add_argument(
        "--no-tokenizer-check", action="store_true",
        help="Write a draft from a base with no tokenizer metadata. For test "
             "fixtures only: a draft whose vocabulary disagrees with the base "
             "has every proposal rejected, and nothing errors -- generation "
             "just gets slower.",
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Print SOURCE's dflash1 metadata and stop.",
    )
    parser.add_argument("--json", dest="as_json", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.info:
        try:
            info = read_draft_info(args.source)
        except Dflash1Error as exc:
            print(f"dflash1: {exc}", file=sys.stderr)
            return 1
        if args.as_json:
            print(json.dumps(info, indent=2))
        elif not info:
            print(f"{args.source} is not a dflash1 draft.")
        else:
            for key, value in info.items():
                print(f"  {key:20} {value}")
        # Non-zero when it is not a draft, so a script can gate on it.
        return 0 if info else 1

    layers = None
    if args.layers:
        try:
            layers = [int(part) for part in args.layers.replace(" ", ",").split(",") if part]
        except ValueError:
            parser.error(f"--layers {args.layers!r} is not a list of integers.")

    output = args.output or f"{args.source.rsplit('.', 1)[0]}.draft.gguf"

    def _progress(event: dict) -> None:
        if args.quiet or args.as_json or event.get("event") != "tensor":
            return
        mark = "pack" if event["quantized"] else "copy"
        print(f"  [{event['index']:>4}/{event['total']}] {mark} {event['name']}",
              file=sys.stderr)

    try:
        report = derive(
            args.source, output,
            layers=layers,
            depth=args.depth,
            quant=args.quant,
            draft_tokens=args.draft_tokens,
            require_tokenizer=not args.no_tokenizer_check,
            progress=None if args.quiet else _progress,
        )
    except Dflash1Error as exc:
        print(f"dflash1: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2))
    elif not args.quiet:
        print(report.describe())
        print(f"dflash1: wrote {output}")
        print(f"  llama-server -m {args.source} --model-draft {output}")
    return 0


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
