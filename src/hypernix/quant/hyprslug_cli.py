"""``hyprslug`` — quantise a GGUF without llama.cpp.

Also installed as ``doomslug``, ``doomslugthedestroyer`` and ``dstd``.

    hyprslug model.f16.gguf Q4_K_M -o model.q4km.gguf
    hyprslug model.f16.gguf IQ0.5_XXXL -o model.iq05.gguf
    doomslug model.q8_0.gguf Q4_K_M --imatrix imatrix.json
    dstd --list-tiers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .hyprslug import (
    ALIASES,
    RECIPES,
    TIER_TYPES,
    HyprslugError,
    quantize_gguf,
)
from .lowbit import CODECS
from .subbit import PACKINGS

__all__ = ["main", "cli_main"]


def _describe_tier(tier: str, type_id: int, packing: str) -> dict:
    """One extension tier, whichever family its packing belongs to.

    There are two now: sign-and-scale packings from
    :mod:`hypernix.quant.subbit`, which are described by how many signs
    of each group survive, and fixed codebooks from
    :mod:`hypernix.quant.lowbit`, which have no dropped signs to report
    and are described by their levels instead. Reaching into ``PACKINGS``
    for both is what this replaced, and it raised ``KeyError: 'INT4'``
    from inside a ``--json`` branch -- an unhandled crash on a listing
    command, which is the one thing a listing command must not do.
    """
    from .ggufcheck import llama_cpp_can_load_type

    reachable = llama_cpp_can_load_type(type_id)
    common = {
        "name": tier,
        "ggml_type": type_id,
        "packing": packing,
        "upstream": False,
        # Whether a *patched* llama.cpp can open it. Stock refuses every
        # tier here; these two are refused by the patched build as well,
        # because the patch never registered them.
        "llama_cpp": reachable,
        "summary": (
            "HyperNix extension type; stock llama.cpp refuses it by name."
            if reachable
            else "HyperNix extension type; runs only under the hnx runtime, "
                 "as no llama.cpp build registers this id."
        ),
    }
    if packing in PACKINGS:
        spec = PACKINGS[packing]
        return {
            **common,
            "family": "sign-and-scale",
            "bits_per_weight": spec.bits_per_weight,
            "signs_kept": spec.kept,
            "group": spec.group,
            "shape": f"{spec.kept} of every {spec.group} signs kept",
        }
    codec = CODECS[packing]
    return {
        **common,
        "family": "fixed-codebook",
        "bits_per_weight": codec.bits_per_weight,
        "code_bits": codec.code_bits,
        "levels": list(codec.levels),
        "shape": (
            f"{codec.code_bits}-bit codes over "
            f"{len(codec.levels)} fixed levels"
        ),
    }



def _check_or_repair(args) -> int:
    """``--check`` and ``--repair-to``, which share their one argument.

    Separated from the quantise path because neither takes a tier: a
    file already has one, and asking for it again is how a repair gets
    invoked with the wrong one.
    """
    from .gguf import GGUFError
    from .ggufcheck import check_gguf, repair_gguf

    if not args.source:
        print("--check and --repair-to need a GGUF to look at.", file=sys.stderr)
        return 2
    source = Path(args.source)
    if not source.exists():
        print(f"No such file: {source}", file=sys.stderr)
        return 2

    try:
        report = check_gguf(source)
    except GGUFError as exc:
        print(f"{source}: {exc}", file=sys.stderr)
        return 1

    if not args.repair_to:
        if args.as_json:
            print(json.dumps(report.as_dict(), indent=2))
        elif not args.quiet:
            print(report.describe())
        # Non-zero when the file will not load, so a script can gate on
        # it without parsing anything.
        return 0 if report.loadable else 1

    if report.loadable:
        if not args.quiet:
            print(f"{source} already loads; nothing to repair.")
        return 0
    try:
        repaired = repair_gguf(source, args.repair_to)
    except GGUFError as exc:
        print(f"{source}: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps({
            "source": repaired.source,
            "output": repaired.output,
            "repaired": repaired.repaired,
            "copied": repaired.copied,
            "source_bytes": repaired.source_bytes,
            "output_bytes": repaired.output_bytes,
        }, indent=2))
    elif not args.quiet:
        print(repaired.describe())
    # The repair is only done if the result actually loads.
    return 0 if check_gguf(repaired.output).loadable else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hyprslug",
        description=(
            "Quantise a GGUF to a llama.cpp quant type or a HyperNix sub-bit "
            "tier. No llama.cpp binary is looked for, downloaded or built at "
            "any point, for either."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Also answers to: " + ", ".join(a for a in ALIASES if a != "hyprslug") + "\n\n"
            "The Q* targets are upstream llama.cpp types and produce a GGUF any\n"
            "llama.cpp reads. The IQ0.x tiers are HyperNix extension types and\n"
            "stock llama.cpp will refuse those by name — which is the point of the\n"
            "type ids being far above anything upstream has allocated. Below ~1.5\n"
            "bits a model stops being a worse version of itself; evaluate before\n"
            "shipping one.\n"
        ),
    )
    parser.add_argument("source", nargs="?",
                        help="Input GGUF (unquantised, or an existing quant "
                             "to requantise from)")
    parser.add_argument("tier", nargs="?",
                        help="Target, e.g. Q4_K_M or IQ0.5_XXXL")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--imatrix", help="Importance matrix as JSON: {tensor: [weights]}")
    parser.add_argument("--quantize-embeddings", action="store_true", default=None,
                        help="Include token embeddings (they dominate a small model)")
    parser.add_argument("--no-quantize-embeddings", dest="quantize_embeddings",
                        action="store_false", help="Leave token embeddings alone")
    parser.add_argument("--quantize-output", action="store_true", default=None,
                        help="Include the output head")
    parser.add_argument("--no-quantize-output", dest="quantize_output",
                        action="store_false", help="Leave the output head alone")
    parser.add_argument("--list-tiers", action="store_true")
    parser.add_argument(
        "--check", action="store_true",
        help="Report whether SOURCE is a file llama.cpp will load, and stop. "
             "Reads the tensor table only, so it is quick on a large model.",
    )
    parser.add_argument(
        "--repair-to", metavar="PATH",
        help="Rewrite SOURCE to PATH with every tensor llama.cpp would refuse "
             "widened back to F32. Gets a file made by a pre-0.72.4.post16 "
             "hyprslug to load without re-quantising; it cannot recover what "
             "the quantiser already discarded.",
    )
    parser.add_argument("--json", dest="as_json", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.check or args.repair_to:
        return _check_or_repair(args)

    if args.list_tiers:
        # --json applies here too; this branch used to return before ever
        # looking at it, so a script that asked for JSON got a human
        # table and a parse error.
        if args.as_json:
            print(json.dumps({
                "recipes": [
                    {
                        "name": name,
                        "base": recipe.base,
                        "bits_per_weight": round(recipe.bits_per_weight, 3),
                        "overrides": {frag: fmt for frag, fmt in recipe.overrides},
                        "output": recipe.output,
                        "upstream": True,
                        "summary": recipe.summary,
                    }
                    for name, recipe in sorted(
                        RECIPES.items(), key=lambda kv: -kv[1].bits_per_weight
                    )
                ],
                "sub_bit_tiers": [
                    _describe_tier(tier, type_id, packing)
                    for tier, (type_id, packing) in TIER_TYPES.items()
                ],
            }, indent=2))
            return 0
        print("llama.cpp quant types (any llama.cpp reads the result):")
        for name, recipe in sorted(
            RECIPES.items(), key=lambda kv: -kv[1].bits_per_weight
        ):
            widened = ", ".join(sorted({fmt for _, fmt in recipe.overrides}))
            note = f"  wider: {widened}" if widened else ""
            print(f"  {name:8} {recipe.bits_per_weight:5.2f} bits/weight  "
                  f"{recipe.summary}{note}")
        print()
        print("HyperNix extension tiers (stock llama.cpp refuses these by name):")
        for tier, (type_id, packing) in TIER_TYPES.items():
            described = _describe_tier(tier, type_id, packing)
            # Five of the seven are registered in the ggml patch; INT4 and
            # FP2 are not, and a listing that presented all seven the same
            # way is how somebody picks a tier no llama.cpp can open.
            reach = "" if described["llama_cpp"] else "  [hnx runtime only]"
            print(
                f"  {tier:12} {described['bits_per_weight']:5.3f} bits/weight  "
                f"type {type_id}  {described['shape']}{reach}"
            )
        if any(
            not _describe_tier(t_, i_, p_)["llama_cpp"]
            for t_, (i_, p_) in TIER_TYPES.items()
        ):
            print()
            print(
                "  [hnx runtime only] runs under `hnx generate` and `hnx chat`."
            )
            print(
                "  The ggml patch registers type ids 200-204 and pins"
            )
            print(
                "  GGML_TYPE_COUNT to 205, so these are past the end of both"
            )
            print(
                "  trait tables and gguf.cpp rejects them before reading a"
            )
            print(
                "  tensor. Pick another tier for llama.cpp or llama-server."
            )
        return 0

    if not args.source or not args.tier:
        parser.error("source and tier are required (or use --list-tiers)")

    output = args.output or f"{args.source.rsplit('.', 1)[0]}.{args.tier}.gguf"

    def _progress(event: dict) -> None:
        if args.quiet or args.as_json or event.get("event") != "tensor":
            return
        mark = "pack" if event["quantized"] else "copy"
        print(f"  [{event['index']:>4}/{event['total']}] {mark} {event['name']}",
              file=sys.stderr)

    try:
        report = quantize_gguf(
            args.source, output, args.tier,
            imatrix=args.imatrix,
            quantize_embeddings=args.quantize_embeddings,
            quantize_output=args.quantize_output,
            progress=None if args.quiet else _progress,
        )
    except HyprslugError as exc:
        print(f"hyprslug: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2))
    elif not args.quiet:
        print(report.describe())
        print(f"hyprslug: wrote {output}")
    return 0


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
