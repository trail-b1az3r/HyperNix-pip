"""``hypernix errors`` — look a code up, or list a domain's.

    hypernix errors explain R3-00020.a3
    hypernix errors list R
    hypernix errors list --min-severity 4

For the person holding a code out of a log line and nothing else.
"""
from __future__ import annotations

import argparse
import json

from . import errorcatalogue  # noqa: F401 - registers every code
from .errorcodes import CODES, Domain, Severity, codes_for, explain

__all__ = ["main", "cli_main"]


def _list(args) -> int:
    domains = [Domain(args.domain.upper())] if args.domain else list(Domain)
    rows = [
        code for domain in domains for code in codes_for(domain)
        if code.severity >= args.min_severity
    ]
    if args.json:
        print(json.dumps([c.to_dict() for c in rows], indent=2))
        return 0
    for code in rows:
        print(f"{code.code}  {code.severity.name.lower():8}  {code.explanation}")
    print(f"\n{len(rows)} of {len(CODES)} code(s)")
    return 0


def _explain(args) -> int:
    print(explain(args.code))
    return 0 if args.code.strip() in CODES else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hypernix errors",
        description="HyperNix error codes: L#-NNNNN.kS (domain, tier, "
                    "number, kind a-f, severity 1-5).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("explain", help="what a code means and what to do")
    ex.add_argument("code")
    ex.set_defaults(func=_explain)

    ls = sub.add_parser("list", help="every code, or one domain's")
    ls.add_argument("domain", nargs="?", default="",
                    help="M T D S I Q E R X")
    ls.add_argument("--min-severity", type=int, default=1,
                    choices=[int(s) for s in Severity])
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=_list)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def cli_main(argv: list[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
