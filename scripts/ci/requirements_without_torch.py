#!/usr/bin/env python3
"""Print hypernix's requirements, minus torch, one per line.

For CI on a Python that PyTorch has no wheel for yet (CPython 3.15 as of
torch 2.14.1): install these, then `pip install --no-deps -e .`, and the
suite runs everything that does not need torch. The tests that do need
it skip with a reason that says so.

    python scripts/ci/requirements_without_torch.py dev security t1api > reqs.txt
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def requirements(extras: list[str]) -> list[str]:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    found = list(project["dependencies"])
    for extra in extras:
        found += project["optional-dependencies"][extra]
    return [r for r in found if not r.replace(" ", "").lower().startswith("torch")]


if __name__ == "__main__":
    print("\n".join(requirements(sys.argv[1:])))
