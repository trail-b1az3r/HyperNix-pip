"""``hypernix-t1 config web`` — the web search settings (T1 v1.1.26.10.1.0).

    hypernix-t1 config web                              # show them
    hypernix-t1 config web --set store_enabled=true     # keep searches
    hypernix-t1 config web --set store_format=parquet --set max_items=30
    hypernix-t1 config web --reset

Read by the server on every request, so a change applies at once, with
no restart. See :mod:`hypernix.t1api.websearch_v2` for what each does.
"""
from __future__ import annotations

import argparse
import json
import sys

from .websearch_v2 import WebAdminSettings, load_settings, save_settings, settings_path

__all__ = ["main"]

HELP = {
    "store_enabled": "keep searches asked to be kept (kpd?=true): AI search storage",
    "store_format": "jsonl or parquet (parquet needs pyarrow)",
    "max_items": "the most results mx?= may ask for (1-100)",
    "max_read_sites": "the most top-ranked sites sq?= may read and summarise (1-20)",
    "ai_config": "who may let a model change settings via /web/v1/ai/config: off, partial, admin",
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "web":
        args = args[1:]
    parser = argparse.ArgumentParser(prog="hypernix-t1 config web", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="change a setting; repeatable")
    parser.add_argument("--reset", action="store_true", help="back to the defaults")
    parser.add_argument("--json", action="store_true", help="print the settings as JSON")
    parsed = parser.parse_args(args)

    current = WebAdminSettings() if parsed.reset else load_settings()
    for pair in parsed.set:
        name, sep, value = pair.partition("=")
        if not sep:
            print(f"hypernix-t1 config web: --set takes NAME=VALUE, not {pair!r}", file=sys.stderr)
            return 2
        try:
            current.set(name.strip(), value.strip())
        except ValueError as exc:
            print(f"hypernix-t1 config web: {exc}", file=sys.stderr)
            return 2
    if parsed.set or parsed.reset:
        path = save_settings(current)
        print(f"Saved to {path}. The server reads it on every request; no restart needed.",
              file=sys.stderr)
    if parsed.json:
        print(json.dumps(current.to_dict(), indent=2))
        return 0
    print(f"Web search settings ({settings_path()}):")
    for name, value in current.to_dict().items():
        print(f"  {name:15} {str(value).lower() if isinstance(value, bool) else value!s:10} {HELP[name]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
