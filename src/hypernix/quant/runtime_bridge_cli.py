"""``hnx runtime`` — using the HyperNix llama.cpp from other applications.

    hnx runtime status                     what is built, detected, installed
    hnx runtime serve model.gguf           start the patched server
    hnx runtime path                       print the build's bin directory
    hnx runtime install --yes              put it into LM Studio
    hnx runtime restore --yes              put LM Studio back

`serve` is the one to reach for: it starts a server anything
OpenAI-compatible can talk to, and changes nothing on the machine.
`install` replaces the libraries LM Studio bundles, which is surgery on
somebody else's application -- it needs --yes, it backs up first, and
`restore` undoes it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from . import runtime_bridge as bridge

__all__ = ["main", "cli_main"]

_EPILOG = """\
two routes, very different risk

  serve     starts the patched llama-server. Point LM Studio, Jan, Open
            WebUI, Continue, Zed, Cursor or anything else that speaks the
            OpenAI API at the printed base URL. Nothing on the machine is
            modified; closing the process puts everything back. Start here.

  install   copies the patched libraries over the ones LM Studio bundles,
            so it loads a sub-bit model natively. Needs --yes. Backs up
            what it replaces and records it, so `restore` works even after
            this process is gone. LM Studio updates will overwrite it, and
            LM Studio does not support this.

exit status
  0  fine
  1  could not start (no build, bad arguments)
  2  nothing was applied because --yes was not given
"""


#: The subcommand names, so a bare invocation can be told from one that
#: merely starts with a global option.
_SUBCOMMANDS = frozenset({"status", "path", "serve", "install", "restore"})


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _cmd_status(args) -> int:
    payload = bridge.status(args.build)
    if args.as_json:
        _print_json(payload)
        return 0

    build = payload.get("build")
    if build is None:
        print("  no built llama.cpp found")
        print("  " + payload.get("build_error", "").replace("\n", "\n  "))
    else:
        mark = "patched" if build["patched"] else "NOT patched"
        print(f"  build      {build['bin_dir']}  ({mark})")
        print(f"  libraries  {', '.join(sorted(build['libraries']))}")
        print(f"  server     {build['server'] or 'not built'}")
        if not build["patched"]:
            print(f"\n  {build.get('note', '')}")

    targets = payload.get("targets") or []
    print()
    if targets:
        for target in targets:
            print(f"  {target['application']}  {target['runtime_dir']}")
            print(f"    replaceable: {', '.join(target['replaceable']) or 'none'}")
    else:
        print("  no LM Studio runtime directory detected")
        print("  (set LMSTUDIO_HOME if it is somewhere unusual)")

    installs = payload.get("installs") or []
    if installs:
        print("\n  installed by this tool:")
        for entry in installs:
            print(f"    {entry['target']}  ({len(entry.get('written', []))} file(s), "
                  f"backup {entry['backup']})")
        print("\n  undo with: hnx runtime restore --yes")
    return 0


def _cmd_path(args) -> int:
    try:
        build = bridge.find_build(args.build)
    except bridge.BridgeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    if args.as_json:
        _print_json(build.to_dict())
        return 0
    # Bare, so it can be used in a shell substitution:
    #   export LD_LIBRARY_PATH="$(hnx runtime path)"
    print(build.bin_dir)
    return 0


def _cmd_serve(args) -> int:
    try:
        build = bridge.find_build(args.build)
        argv = bridge.serve_argv(
            build, args.model, host=args.host, port=args.port,
            gpu_layers=args.gpu_layers, context=args.context, alias=args.alias,
        )
    except bridge.BridgeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    base_url = f"http://{args.host}:{args.port}/v1"
    if args.as_json:
        _print_json({
            "argv": argv,
            "base_url": base_url,
            "patched": build.patched,
            "clients": [{"application": a, "where": w}
                        for a, w in bridge.OPENAI_CLIENTS],
        })
    else:
        if not build.patched:
            print(f"  warning: {build.note}\n", file=sys.stderr)
        print(f"  base URL   {base_url}", file=sys.stderr)
        print("  api key    any non-empty string (it is not checked)",
              file=sys.stderr)
        print("\n  point one of these at it:", file=sys.stderr)
        for application, where in bridge.OPENAI_CLIENTS:
            print(f"    {application:<20} {where}", file=sys.stderr)
        print(file=sys.stderr)

    if args.print_only:
        print(" ".join(argv))
        return 0

    # Exec rather than spawn-and-wait: the server becomes this process,
    # so Ctrl-C reaches it directly and there is no wrapper left holding
    # a pipe. On Windows there is no exec worth the name, so spawn.
    if os.name == "nt":
        return subprocess.run(argv, check=False).returncode
    os.execv(argv[0], argv)
    return 0                                       # pragma: no cover


def _cmd_install(args) -> int:
    try:
        build = bridge.find_build(args.build)
    except bridge.BridgeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    targets = [args.target] if args.target else [
        t["runtime_dir"] for t in bridge.detect_targets()
    ]
    if not targets:
        print("  no LM Studio runtime directory detected. Pass --target to "
              "name one, or set LMSTUDIO_HOME.", file=sys.stderr)
        return 1

    results = []
    unapplied = False
    for target in targets:
        try:
            result = bridge.install(build, target, confirmed=args.yes,
                                    dry_run=args.dry_run)
        except bridge.BridgeError as exc:
            print(f"  {target}: {exc}", file=sys.stderr)
            return 1
        results.append(result)
        if not result.get("applied"):
            unapplied = True

    if args.as_json:
        _print_json(results)
    else:
        for result in results:
            if result.get("applied"):
                print(f"  installed into {result['target']}")
                for written in result["written"]:
                    print(f"    {written}")
                print(f"  backup      {result['backup']}")
                print(f"  undo with   {result['restore_with']}")
            else:
                print(f"  {result['target']}: {result['reason']}")
                for entry in result["planned"]:
                    verb = "replace" if entry["replaces_existing"] else "add"
                    print(f"    would {verb} {entry['to']}")
        if unapplied:
            print("\n  This replaces libraries inside LM Studio. It is backed "
                  "up and reversible, and LM Studio does not support it.\n"
                  "  Add --yes to go ahead.", file=sys.stderr)
    return 2 if unapplied else 0


def _cmd_restore(args) -> int:
    result = bridge.restore(confirmed=args.yes, which=args.target)
    if args.as_json:
        _print_json(result)
        return 0
    if result.get("restored"):
        print(f"  restored {result['restored']} file(s)")
        if result.get("removed"):
            print(f"  removed  {result['removed']} that this tool had added")
    elif "would_restore" in result:
        for entry in result["would_restore"]:
            print(f"  would restore {entry['files']} file(s) in {entry['target']}")
        print("\n  Add --yes to go ahead.", file=sys.stderr)
        return 2
    else:
        print(f"  {result.get('note', 'nothing to do')}")
    for problem in result.get("problems") or []:
        print(f"  problem: {problem}", file=sys.stderr)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hnx runtime",
        description="Use the HyperNix llama.cpp from LM Studio and anything "
                    "else that speaks the OpenAI API.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Declared twice on purpose: once here, so `hnx runtime --json`
    # works with no subcommand, and once on every subcommand, so
    # `hnx runtime status --json` works too. People type both, and an
    # option that is only valid on one side of the subcommand is a
    # papercut with no upside.
    #
    # SUPPRESS on the per-subcommand copies is what makes that safe: an
    # unspecified child option would otherwise write its default over
    # whatever the top level parsed.
    def add_common(target, suppress: bool) -> None:
        blank = argparse.SUPPRESS if suppress else ""
        off = argparse.SUPPRESS if suppress else False
        target.add_argument("--build", default=blank, metavar="DIR",
                            help="A built llama.cpp (default: the one "
                                 "build.sh makes, or $HNX_LLAMA_BUILD).")
        target.add_argument("--json", dest="as_json", action="store_true",
                            default=off,
                            help="Machine-readable output on stdout.")

    add_common(parser, suppress=False)
    subparsers = parser.add_subparsers(dest="command")

    status = subparsers.add_parser("status", help="What is built and detected.")
    add_common(status, suppress=True)
    status.set_defaults(handler=_cmd_status)

    path = subparsers.add_parser("path", help="Print the build's bin directory.")
    add_common(path, suppress=True)
    path.set_defaults(handler=_cmd_path)

    serve = subparsers.add_parser(
        "serve", help="Start the patched server. Changes nothing.")
    serve.add_argument("model", help="A .gguf file.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("-ngl", "--gpu-layers", type=int, default=0,
                       help="Layers to put on the GPU (0: CPU only).")
    serve.add_argument("-c", "--context", type=int, default=0,
                       help="Context length (0: the model's own).")
    serve.add_argument("--alias", default="",
                       help="The name the model answers to over the API.")
    serve.add_argument("--print-only", action="store_true",
                       help="Print the command instead of running it.")
    add_common(serve, suppress=True)
    serve.set_defaults(handler=_cmd_serve)

    install = subparsers.add_parser(
        "install", help="Put the libraries into LM Studio. Needs --yes.")
    install.add_argument("--target", default="",
                         help="A runtime directory (default: what is detected).")
    install.add_argument("--yes", action="store_true",
                         help="Confirm. Without it, this only reports.")
    install.add_argument("--dry-run", action="store_true",
                         help="Report and change nothing, even with --yes.")
    add_common(install, suppress=True)
    install.set_defaults(handler=_cmd_install)

    restore = subparsers.add_parser(
        "restore", help="Undo an install from its backup.")
    restore.add_argument("--target", default="",
                         help="Only this directory (default: all of them).")
    restore.add_argument("--yes", action="store_true")
    add_common(restore, suppress=True)
    restore.set_defaults(handler=_cmd_restore)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    # A bare `hnx runtime` is the question people mean by it, and so is
    # `hnx runtime --json`. Decided by looking for a subcommand anywhere
    # in the arguments rather than by checking whether the first one
    # starts with a dash: `--build DIR install --yes` does start with a
    # dash and is not a bare invocation, and prepending `status` to it
    # made argparse reject the whole line.
    if not any(token in _SUBCOMMANDS for token in argv):
        if not any(token in ("-h", "--help") for token in argv):
            argv = [*argv, "status"] if argv else ["status"]
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        args = parser.parse_args(["status"])
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        return 0


def cli_main(argv: list[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
