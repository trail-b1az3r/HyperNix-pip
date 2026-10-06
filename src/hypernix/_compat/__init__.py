"""Where Python-version-specific code lives, behind one import.

Some 3.15 features are new *syntax* -- PEP 798's ``[*it for it in its]``
-- and 3.12-3.14 cannot even parse a file that uses them, so it can
never sit in shared source. Each such feature has two modules here with
the same API: ``*_py315.py`` in the new syntax, imported only on 3.15+,
and ``*_legacy.py`` for 3.12-3.14. Callers import from this package and
get the right one.

Tools that parse every file in the tree (ruff, the archmap and autoscan
walkers, the docs generator) skip ``*_py315.py`` on an older
interpreter; :func:`is_py315_only` is how they tell.
"""
from __future__ import annotations

import sys
from pathlib import PurePath

__all__ = ["PY315", "PY315_SUFFIX", "flatten", "is_py315_only", "merge", "union"]

#: True on 3.15 and newer.
PY315: bool = sys.version_info >= (3, 15)

#: The file-name ending of a module written in 3.15-only syntax.
PY315_SUFFIX = "_py315.py"


def is_py315_only(path: str | PurePath) -> bool:
    """Whether *path* is a module this interpreter cannot parse because it
    is written in 3.15-only syntax. Always False on 3.15+."""
    return not PY315 and PurePath(path).name.endswith(PY315_SUFFIX)


if PY315:
    from ._iter_py315 import flatten, merge, union
else:
    from ._iter_legacy import flatten, merge, union
