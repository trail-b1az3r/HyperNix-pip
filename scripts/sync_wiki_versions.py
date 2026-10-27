#!/usr/bin/env python3
"""Keep the wiki's "current version" lines on the current versions.

The wiki says which HyperNix and which T1 API are current in a few
places, and nothing updated them: the release bump wrote six other
files and left these naming the previous release. Each line is matched
by a pattern here and rewritten from ``src/hypernix/__init__.py`` and
``src/hypernix/t1api/version.py``. The release's version bump runs
this; ``tests/repo/test_wiki_versions.py`` fails when they drift.

    python scripts/sync_wiki_versions.py            # rewrite in place
    python scripts/sync_wiki_versions.py --check    # exit 1 if any would change
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAGES = ("wiki/Home.md", "wiki/T1-API.md")

#: (page, pattern, replacement); replacements are formatted with
#: ``version``, ``t1`` (short form) and ``t1_long``.
RULES: tuple[tuple[str, str, str], ...] = (
    ("wiki/Home.md",
     r"Current: \*\*T1 v[\d.]+\*\* in [0-9][\w.+-]*?(?=\.? *\|)",
     "Current: **T1 v{t1}** in {version}"),
    ("wiki/T1-API.md",
     r"\*\*Status: T1 v[\d.]+\*\* \(long form `[\d.]+`\) in\n`[^`]+`",
     "**Status: T1 v{t1}** (long form `{t1_long}`) in\n`{version}`"),
    ("wiki/T1-API.md",
     r'"t1_version": "[\d.]+", "version": "[^"]+"',
     '"t1_version": "{t1}", "version": "{version}"'),
)


def package_version(root: Path = ROOT) -> str:
    text = (root / "src" / "hypernix" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__ = "([^"]+)"', text, re.M).group(1)


def t1_versions(root: Path = ROOT) -> tuple[str, str]:
    """``(short, long)``, read from the file rather than imported, so this
    runs with nothing installed."""
    text = (root / "src" / "hypernix" / "t1api" / "version.py").read_text(encoding="utf-8")
    found = re.search(r"^T1_VERSION = T1Version\(([^)]*)\)", text, re.M)
    if not found:
        raise SystemExit("T1_VERSION not found in src/hypernix/t1api/version.py")
    fields = dict(re.findall(r"(\w+)\s*=\s*(\d+)", found.group(1)))
    parts = [fields[k] for k in ("api", "major", "year", "month", "feature", "fix")]
    year = int(parts[2])
    short = ".".join([parts[0], parts[1], str(year % 100), *parts[3:]])
    long = ".".join([parts[0], parts[1], str(year), *parts[3:]])
    return short, long


def sync(pages: list[Path] | None = None, version: str = "", *, root: Path = ROOT,
         check: bool = False) -> list[str]:
    """Rewrite *pages* (default: :data:`PAGES`). The changed (or, with
    *check*, out-of-date) pages, as paths relative to *root*.

    A rule whose pattern no longer matches is an error rather than a
    silent skip: a reworded line would otherwise stop being kept.
    """
    version = version or package_version(root)
    t1, t1_long = t1_versions(root)
    wanted = {str(Path(p)) for p in (pages or PAGES)}
    changed: list[str] = []
    for page in sorted(wanted):
        path = root / page
        text = path.read_text(encoding="utf-8")
        new = text
        for rule_page, pattern, replacement in RULES:
            if str(Path(rule_page)) != page:
                continue
            value = replacement.format(version=version, t1=t1, t1_long=t1_long)
            new, count = re.subn(pattern, lambda _m, value=value: value, new)
            if not count:
                raise SystemExit(f"{page}: no line matches {pattern!r}; update "
                                 f"scripts/sync_wiki_versions.py with the new wording")
        if new != text:
            changed.append(page)
            if not check:
                path.write_text(new, encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="change nothing; exit 1 if out of date")
    parser.add_argument("--version", default="", help="the HyperNix version (default: __init__.py's)")
    args = parser.parse_args(argv)
    changed = sync(version=args.version, check=args.check)
    for page in changed:
        print(f"{'out of date' if args.check else 'updated'}: {page}")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
