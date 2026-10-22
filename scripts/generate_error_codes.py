#!/usr/bin/env python3
"""Write wiki/Error-Codes.md and docs/public/v1/error-codes.json from the
registry, so the page and the website never disagree with the code.

    python scripts/generate_error_codes.py          # write both
    python scripts/generate_error_codes.py --check  # exit 1 if either is stale

tests/system/test_error_codes.py runs --check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypernix.system import errorcatalogue  # noqa: E402,F401 - registers every code
from hypernix.system.errorcodes import ALIASES, CODES, Domain, Kind, Severity, codes_for  # noqa: E402

PAGE = ROOT / "wiki" / "Error-Codes.md"
DATA = ROOT / "docs" / "public" / "v1" / "error-codes.json"

DOMAINS = {
    "M": "Models", "T": "The T1 API", "D": "Data", "S": "System", "I": "Interfaces (hyped-pro, the TUIs)",
    "Q": "Quantisation", "E": "Elements", "R": "The runtime and runner", "X": "Training",
}
KINDS = {
    "a": "the caller's — what was asked cannot be done", "b": "configuration or environment",
    "c": "not enough of something", "d": "something outside HyperNix",
    "e": "data that is present and wrong", "f": "HyperNix's own fault — a bug",
}


def _legacy() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for old, new in sorted(ALIASES.items()):
        out.setdefault(new, []).append(old)
    return out


def page() -> str:
    legacy = _legacy()
    lines = [
        "# Error codes",
        "",
        "Every error HyperNix raises carries a code of the shape `L#-NNNNN.kS` — for",
        "example `I2-02250.a3`. Search for the code; it lands here. Generated from",
        "`hypernix.system.errorcatalogue` by `scripts/generate_error_codes.py`; do not",
        "edit by hand.",
        "",
        "```bash",
        "hypernix errors explain I2-02250.a3   # what it means and what to do",
        "hypernix errors explain HPC-CFG-001   # an old name works too",
        "hypernix errors list I                # one domain",
        "hypernix errors aliases               # every old name and its code",
        "```",
        "",
        "## Reading a code",
        "",
        "| Part | Means |",
        "|---|---|",
        "| `L` | the domain: " + ", ".join(f"`{k}` {v.split(' (')[0].lower()}" for k, v in DOMAINS.items()) + " |",
        "| `#` | the tier, 0–9: `1` is a surface people use, `9` a kernel or an internal fault |",
        "| `NNNNN` | five digits, unique in the domain; the thousands say what kind of trouble (below) |",
        "| `k` | whose problem it is: " + "; ".join(f"`{k}` {v}" for k, v in KINDS.items()) + " |",
        "| `S` | severity, 1–5: " + ", ".join(f"`{int(s)}` {s.name.lower()}" for s in Severity) + " |",
        "",
        "The thousands: `0xxxx` loading and finding, `1xxxx` memory, disk, cores and quota,",
        "`2xxxx` configuration and arguments, `3xxxx` data and formats, `4xxxx` the network",
        "and other processes, `5xxxx` permission and identity, `9xxxx` internal faults.",
        "",
        f"There are **{len(CODES)}** codes and **{len(ALIASES)}** older names that still resolve.",
        "",
        "## Contents",
        "",
    ]
    for letter, title in DOMAINS.items():
        anchor = f"{letter.lower()}--{title.lower()}"
        anchor = "".join(c for c in anchor if c.isalnum() or c in "- ").replace(" ", "-")
        lines.append(f"- [{letter} — {title}](#{anchor})")
    lines.append("- [Older names](#older-names)")
    for letter, title in DOMAINS.items():
        lines += ["", f"## {letter} — {title}", "", "| Code | Severity | Meaning | What to do |", "|---|---|---|---|"]
        for code in codes_for(Domain(letter)):
            was = legacy.get(code.code)
            meaning = code.explanation + (f" (was {', '.join(f'`{w}`' for w in was)})" if was else "")
            remedy = code.remedy.replace("|", "\\|")
            lines.append(f"| `{code.code}` | {code.severity.name.lower()} | {meaning.replace('|', chr(92) + '|')} | {remedy} |")
    lines += ["", "## Older names", "",
              "hyped-pro had its own names until 0.72.8a4. Each is an alias of a registered code",
              "now: `explain` answers for both, and hyped-pro's bridge sends the old name beside",
              "the new one as `legacy_code`.", "", "| Old name | Code |", "|---|---|"]
    for old, new in sorted(ALIASES.items()):
        lines.append(f"| `{old}` | `{new}` |")
    return "\n".join(lines) + "\n"


def data() -> str:
    legacy = _legacy()
    rows = []
    for letter in DOMAINS:
        for code in codes_for(Domain(letter)):
            rows.append({**code.to_dict(), "kind_meaning": KINDS[code.kind.value],
                         "legacy": legacy.get(code.code, [])})
    return json.dumps({"schema": 1, "domains": DOMAINS, "kinds": KINDS,
                       "severities": {int(s): s.name.lower() for s in Severity},
                       "count": len(rows), "codes": rows,
                       "aliases": dict(sorted(ALIASES.items()))}, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if a file is out of date")
    args = parser.parse_args(argv)
    assert set(Kind) and set(Domain)
    stale = []
    for path, text in ((PAGE, page()), (DATA, data())):
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path.relative_to(ROOT))
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        print("out of date (run scripts/generate_error_codes.py): " + ", ".join(map(str, stale)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
