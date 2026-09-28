"""``hypernix-sync`` (also ``t1-sync`` and ``hypernix-t1 sync``).

Mirrors every model in ``~/.hypernix/models`` into
``~/.hypernix/t1api/models`` as symlinks — see :mod:`.modelsync`.
Reads the same settings the server does, from the environment and then
the server's ``.env``, so by hand and at startup they agree about both
folders.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .modelsync import default_source, default_target, sync

__all__ = ["main", "cli_main"]

_EPILOG = """\
Folders are created and every file is linked, one symlink per file, so a
Hugging Face repo or a HyperNix checkpoint looks the same in both places.
A link whose file has gone is removed. A real file or a link of your own
in the T1 folder is never replaced; it is listed as left alone.

With T1_MODEL_SYNC=1 in the server's .env, the server does this itself at
startup, after each download and before listing models, and serves from
the T1 folder.

Exit status: 0 synced, 1 could not sync, 2 synced but something was left
alone or failed (or, with --index, a model could not be read).
"""


def _setting(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    from .localserver import _env_file_values

    return _env_file_values().get(name, "")


def _index(target: Path) -> int:
    from .modelindex_cli import main as index_main

    return index_main(["--dir", str(target)])


def _print(result, *, as_json: bool, verbose: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return
    print(f"{result.source}  ->  {result.target}")
    for label, items in (("+", result.linked), ("~", result.relinked), ("-", result.removed)):
        if verbose or len(items) <= 20:
            for item in items:
                print(f"  {label} {item}")
        else:
            print(f"  {label} {len(items)} files")
    for item in result.conflicts:
        print(f"  ! {item['path']}: {item['reason']}")
    for item in result.errors:
        print(f"  x {item['path']}: {item['error']}", file=sys.stderr)
    print(result.summary() + (" (dry run: nothing changed)" if result.dry_run else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hypernix-sync",
        description="Mirror ~/.hypernix/models into ~/.hypernix/t1api/models by symlink.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", default="",
                        help="the shared models folder (default: T1_HF_DOWNLOAD_DIR, "
                             "else ~/.hypernix/models)")
    parser.add_argument("--target", default="",
                        help="the T1 server's folder (default: T1_MODELS_DIR, "
                             "else <T1_CONFIG_DIR>/models)")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="say what would change, change nothing")
    parser.add_argument("--no-prune", action="store_true",
                        help="add and relink, but keep links whose file has gone")
    parser.add_argument("--index", action="store_true",
                        help="then write the model registry from the T1 folder "
                             "(hypernix-t1 index --dir <target>)")
    parser.add_argument("--watch", type=float, default=0.0, metavar="SECONDS",
                        help="keep syncing every SECONDS until interrupted")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="list every file, however many")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    source = Path(args.source or _setting("T1_HF_DOWNLOAD_DIR") or default_source()).expanduser()
    configured_dir = _setting("T1_CONFIG_DIR")
    target = Path(
        args.target or _setting("T1_MODELS_DIR") or default_target(configured_dir or None)
    ).expanduser()

    indexed = 0
    while True:
        try:
            result = sync(source, target, prune=not args.no_prune, dry_run=args.dry_run)
        except (OSError, ValueError) as exc:
            print(f"hypernix-sync: {exc}", file=sys.stderr)
            return 1
        if not args.watch or result.changed or result.conflicts or result.errors:
            _print(result, as_json=args.as_json, verbose=args.verbose)
        if args.index and not args.dry_run and (result.changed or not args.watch):
            indexed = _index(target)
        if not args.watch:
            break
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            break
    # The indexer's own status too: 2 when a file could not be read, which
    # is a model missing from the registry somebody meant to be there.
    return 2 if (result.conflicts or result.errors) else indexed


def cli_main() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli_main()
