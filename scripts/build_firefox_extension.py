#!/usr/bin/env python3
"""Pack browser/firefox into an .xpi Firefox can install.

    python scripts/build_firefox_extension.py [-o dist/]

The tests and the README stay out of the package. The result is unsigned:
load it from about:debugging (This Firefox -> Load Temporary Add-on), or
in Firefox Developer Edition / Nightly with xpinstall.signatures.required
set to false; for everyday Firefox, sign it on addons.mozilla.org.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "browser" / "firefox"
LEAVE_OUT = {"test", "README.md", "node_modules", ".DS_Store"}


def files(source: Path = EXT) -> list[Path]:
    return sorted(p for p in source.rglob("*")
                  if p.is_file() and not (set(p.relative_to(source).parts) & LEAVE_OUT))


def build(out_dir: Path, source: Path = EXT) -> Path:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"hypernix-browser-agent-{manifest['version']}.xpi"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as xpi:
        for path in files(source):
            xpi.write(path, path.relative_to(source).as_posix())
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "dist")
    args = parser.parse_args(argv)
    print(build(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
