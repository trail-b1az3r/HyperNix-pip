"""``hypernix hyprslug-headers`` — the runtime, and the three ways out.

Reached as ``hypernix hyprslug-headers <command>``, ``hnx
hyprslug-headers <command>``, or the ``hyprslug-headers`` console script.

The epilog is the important part of this file. Someone arrives here
because a model would not open, and the first thing they need is to know
which of three different problems they have.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .._compat import flatten
from .hyprslug_headers import (
    FALLBACKS,
    HEADER_VERSION,
    HeaderError,
    install,
    install_model,
    read_header,
    scan,
    stamp,
    status,
    uninstall,
    wrap,
)

__all__ = ["main", "cli_main"]

_EPILOG = """\
The model will not open in LM Studio. Which of these do you want?

  keep the tier        serve   — the model stays sub-bit inside hnxrun and
                                 LM Studio talks to it over HTTP. Nothing
                                 is converted. Costs a running process.
  open it anywhere     wrap    — re-encode to a type stock llama.cpp has.
                                 The file loads everywhere, is several
                                 times larger, and is no longer sub-bit.
  make it explain      stamp   — write the block geometry into the file's
                                 own metadata so any loader can be taught
                                 to read it. Does not make it loadable.

No header makes a stock llama.cpp read a 0.5-bit tensor: the type id is
how it notices, but the missing dequantisation kernel is why it stops.
A header claiming a type llama.cpp knows would load and produce noise,
which is worse than the error you have.
"""


def _print(payload, as_json: bool, human) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        human(payload)


def _human_install(result: dict) -> None:
    print(f"runtime installed to {result['installed_to']}")
    print(f"  config   : {result['config']}")
    print(f"  endpoint : http://{result['host']}:{result['port']}/v1")
    roots = result["lmstudio_roots"]
    if not roots:
        print("  LM Studio: no model directory found "
              "(set LMSTUDIO_HOME if it is somewhere unusual)")
        return
    for root in roots:
        print(f"  LM Studio: {root}")
    needs = result["models_needing_the_runtime"]
    print(f"  scanned {result['models_seen']} model(s); "
          f"{len(needs)} need this runtime")
    for model in needs:
        note = ""
        if model.get("misnamed"):
            note = (f"   <- named {model['named_tier']}, "
                    f"tensors say {model['tier']}")
        print(f"    {model['tier']:12} {Path(model['path']).name}{note}")
    if any(m.get("misnamed") for m in needs):
        print()
        print("  A tier above is not the one in the filename. The tensor")
        print("  types are what the loader reads, so those are what is")
        print("  reported -- the name is just a label someone typed.")
    if needs:
        first = needs[0]["path"]
        print()
        print("  Those will not open in LM Studio directly. Either:")
        print(f"    hypernix hyprslug-headers serve {first}")
        print(f"    hypernix hyprslug-headers wrap  {first} -o compat.gguf")


def _human_status(result: dict) -> None:
    if not result.get("installed"):
        print(f"not installed (would live in {result['runtime_dir']})")
        if result.get("error"):
            print(f"  {result['error']}")
        return
    print(f"installed   : {result['runtime_dir']}")
    print(f"header ver  : {result['header_version']}")
    print(f"endpoint    : http://{result['host']}:{result['port']}/v1")
    print(f"decodes     : {', '.join(result['types'])}")
    for root in result["lmstudio_roots"]:
        print(f"LM Studio   : {root}")


def _human_scan(rows: list[dict]) -> None:
    if not rows:
        print("no .gguf files found")
        return
    for row in rows:
        if not row.get("readable", True):
            print(f"  {'unreadable':12} {row['path']}  ({row.get('error', '')})")
            continue
        mark = "needs hnxrun" if row["extension"] else "stock llama.cpp"
        print(f"  {row['tier']:12} {mark:16} "
              f"{row['bytes'] / 1e6:8.1f} MB  {row['path']}")
        if row.get("misnamed"):
            print(f"  {'':12} {'':16} {'':>8}     "
                  f"^ filename says {row['named_tier']}")


def _human_install_model(result: dict) -> None:
    print(f"installed  {result['installed_to']}")
    if result["converted"]:
        print(f"  {result['from_tier']} -> {result['to_type']}, "
              f"{result['source_bytes'] / 1e6:.1f} MB -> "
              f"{result['output_bytes'] / 1e6:.1f} MB ({result['growth']}x)")
    else:
        print(f"  copied unchanged ({result['output_bytes'] / 1e6:.1f} MB)")
    print()
    print(f"  {result['honest_warning']}")
    print()
    print("  LM Studio and Bionic should list it after a rescan, under")
    print(f"  {result['publisher']} / {result['model_name']}.")
    if result["converted"]:
        print("  The original is untouched. To run *that* at its own bitrate:")
        print(f"    hypernix hyprslug-headers serve {result['source']} "
              f"--device auto")


def _human_wrap(result: dict) -> None:
    print(f"{result['source']}")
    print(f"  -> {result['output']}")
    print(f"  {result['from_tier']} -> {result['to_type']}, "
          f"{result['source_bytes'] / 1e6:.1f} MB -> "
          f"{result['output_bytes'] / 1e6:.1f} MB ({result['growth']}x)")
    print()
    print(f"  {result['honest_warning']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hypernix hyprslug-headers",
        description=(
            "Self-describing headers for GGUFs carrying HyperNix extension "
            "types, and the runtime that executes them."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Machine-readable output on every subcommand.")
    subparsers = parser.add_subparsers(dest="command")

    p_install = subparsers.add_parser(
        "install", help="Set up the runtime and find models that need it.")
    p_install.add_argument("--host", default="127.0.0.1")
    p_install.add_argument("--port", type=int, default=1234)
    p_install.add_argument("--no-scan", dest="scan_lmstudio",
                           action="store_false", default=True,
                           help="Skip looking through LM Studio's models.")

    subparsers.add_parser("status", help="What is installed, if anything.")
    subparsers.add_parser("uninstall", help="Remove the runtime config.")

    p_scan = subparsers.add_parser(
        "scan", help="Classify every GGUF under a directory.")
    p_scan.add_argument("root", nargs="?", default=None,
                        help="Directory to walk. Defaults to LM Studio's models.")

    p_show = subparsers.add_parser(
        "show", help="The header a model carries, stamped or derived.")
    p_show.add_argument("model")

    p_stamp = subparsers.add_parser(
        "stamp", help="Write the header block into a model's metadata.")
    p_stamp.add_argument("model")
    p_stamp.add_argument("-o", "--output", default=None,
                         help="Write elsewhere instead of rewriting in place.")

    p_wrap = subparsers.add_parser(
        "wrap", help="Re-encode to a type stock llama.cpp reads.")
    p_wrap.add_argument("model")
    p_wrap.add_argument("-o", "--output", default=None)
    p_wrap.add_argument("--to", dest="target", default="",
                        help=f"Target type. Defaults per tier: {FALLBACKS}")

    p_model = subparsers.add_parser(
        "install-model",
        help="Put a loadable copy where LM Studio and Bionic look.")
    p_model.add_argument("model")
    p_model.add_argument("--to", dest="target", default="",
                         help="Target type for the conversion. Per-tier "
                              "default otherwise.")
    p_model.add_argument("--publisher", default="HyperNix",
                         help="Folder LM Studio groups it under.")
    p_model.add_argument("--name", default="",
                         help="Model name. Defaults to the filename.")
    p_model.add_argument("--root", default=None,
                         help="LM Studio models directory, if it is somewhere "
                              "unusual. LMSTUDIO_HOME works too.")

    p_serve = subparsers.add_parser(
        "serve", help="Serve the model over an OpenAI-compatible endpoint.")
    p_serve.add_argument("model")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=1234)
    p_serve.add_argument("--name", default="",
                         help="Model id clients see. Defaults to the filename.")
    p_serve.add_argument("--cache-bytes", dest="cache_bytes", type=int, default=0,
                         help="Memory to spend keeping weights decoded.")
    p_serve.add_argument("--device", default="auto",
                         help="auto, cpu, cuda, cuda:1, mps, xpu. On an "
                              "accelerator the packed bytes are uploaded once "
                              "and decoded there. `hypernix devices` lists "
                              "what is available and why not.")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    try:
        return _dispatch(args, parser)
    except HeaderError as exc:
        print(f"hyprslug-headers: {exc}", file=sys.stderr)
        return 1


def _dispatch(args, parser) -> int:
    if args.command == "install":
        result = install(host=args.host, port=args.port,
                         scan_lmstudio=args.scan_lmstudio)
        _print(result, args.as_json, _human_install)
        return 0

    if args.command == "status":
        result = status()
        _print(result, args.as_json, _human_status)
        return 0 if result.get("installed") else 1

    if args.command == "uninstall":
        result = uninstall()
        _print(result, args.as_json,
               lambda r: print(f"removed {len(r['removed'])} file(s) from "
                               f"{r['runtime_dir']}"))
        return 0

    if args.command == "scan":
        from .hyprslug_headers import lmstudio_roots

        roots = [Path(args.root)] if args.root else lmstudio_roots()
        if not roots:
            print("No LM Studio model directory found. Pass one explicitly, "
                  "or set LMSTUDIO_HOME.", file=sys.stderr)
            return 1
        rows: list[dict] = flatten(scan(root) for root in roots)
        _print(rows, args.as_json, _human_scan)
        return 0

    if args.command == "show":
        header = read_header(args.model)
        _print(header.to_dict(), args.as_json,
               lambda _h: print(header.describe()))
        return 0

    if args.command == "stamp":
        header = stamp(args.model, args.output)
        _print(
            {"stamped": args.output or args.model,
             "header_version": HEADER_VERSION, **header.to_dict()},
            args.as_json,
            lambda _r: print(f"stamped {args.output or args.model}\n"
                             f"  {header.describe()}"),
        )
        return 0

    if args.command == "wrap":
        source = Path(args.model)
        target = Path(args.output) if args.output else source.with_suffix(
            f".{(args.target or read_header(source).fallback or 'Q4_K_M')}.gguf"
        )
        result = wrap(source, target, to=args.target)
        _print(result, args.as_json, _human_wrap)
        return 0

    if args.command == "install-model":
        result = install_model(
            args.model, to=args.target, publisher=args.publisher,
            name=args.name, root=args.root,
        )
        _print(result, args.as_json, _human_install_model)
        return 0

    if args.command == "serve":
        from .hyprslug_server import ServerError, serve

        if args.host not in ("127.0.0.1", "localhost", "::1"):
            print(
                f"hyprslug-headers: serving on {args.host} publishes an "
                f"unauthenticated inference endpoint to the network.",
                file=sys.stderr,
            )
        print(f"hyprslug-headers: loading {args.model}", file=sys.stderr)

        def announce(model) -> None:
            # Printed from inside serve(), after the model is loaded and
            # the port is bound. It used to print before the load, which
            # meant a failed load left the endpoint on screen next to the
            # traceback that said it was never listening.
            where = getattr(model.model, "device", "cpu")
            print(f"hyprslug-headers: loaded on {where}", file=sys.stderr)
            print(f"hyprslug-headers: http://{args.host}:{args.port}/v1  "
                  f"(ctrl-c to stop)", file=sys.stderr)

        try:
            serve(args.model, host=args.host, port=args.port,
                  cache_bytes=args.cache_bytes, name=args.name,
                  device=args.device, on_ready=announce)
        except ServerError as exc:
            print(f"hyprslug-headers: {exc}", file=sys.stderr)
            return 1
        return 0

    parser.error(f"Unknown command {args.command!r}")
    return 2


def cli_main() -> None:
    """Console-script entry point.

    Without the ``__main__`` guard below, ``python -m`` on this module
    imports it, runs nothing and exits 0 — which looks exactly like a
    command that worked.
    """
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
