#!/usr/bin/env python3
"""Strife's benchmark. The same program as ``hnx strife benchmark``.

    python benchmarks/strife_bench.py model.gguf -p 512 -n 128 -r 3 --out-dir results/

See :mod:`hypernix.strife.bench` for what it measures and writes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hypernix.strife.bench import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
