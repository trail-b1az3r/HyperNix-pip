"""PEP 798 -- unpacking in comprehensions -- behind hypernix._compat.

``[*it for it in its]`` is 3.15 syntax; 3.12-3.14 cannot parse a file
that contains it. So it lives only in ``*_py315.py`` modules, and these
tests check both halves of that boundary: on 3.15 the new syntax is what
runs and gives the same answers as the old code; before 3.15 it is never
imported, and every tool that parses the whole tree is told to skip it.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from hypernix import _compat
from hypernix._compat import _iter_legacy as legacy

SRC = Path(__file__).resolve().parents[2] / "src" / "hypernix"
PY315_FILES = sorted(SRC.rglob("*_py315.py"))

needs_315 = pytest.mark.skipif(sys.version_info < (3, 15), reason="PEP 798 is Python 3.15 syntax")

#: Factories, not values: one case holds iterators, which a first use empties.
CASES = [
    lambda: [],
    lambda: [[]],
    lambda: [[1, 2], (3,), [], range(4, 6)],
    lambda: ["ab", "", "c"],
    lambda: [iter([1]), iter([2, 3])],
]


def test_there_is_3_15_syntax_and_it_is_isolated():
    assert PY315_FILES, "no *_py315.py module: nothing exercises PEP 798"
    for path in PY315_FILES:
        assert path.parent.name == "_compat", f"{path} is outside the compatibility boundary"


@pytest.mark.parametrize("path", PY315_FILES, ids=lambda p: p.name)
def test_older_pythons_are_told_to_skip_it(path):
    assert _compat.is_py315_only(path) is (sys.version_info < (3, 15))


@pytest.mark.skipif(sys.version_info >= (3, 15), reason="the boundary only matters below 3.15")
@pytest.mark.parametrize("path", PY315_FILES, ids=lambda p: p.name)
def test_it_really_cannot_be_parsed_before_3_15(path):
    """If this ever passes parsing on 3.12, the module no longer needs to
    be separate -- and the boundary is costing something for nothing."""
    with pytest.raises(SyntaxError):
        ast.parse(path.read_text(encoding="utf-8"))


def test_the_legacy_implementation_is_what_runs_before_3_15():
    expected = "hypernix._compat._iter_py315" if sys.version_info >= (3, 15) \
        else "hypernix._compat._iter_legacy"
    assert _compat.flatten.__module__ == expected
    assert _compat.union.__module__ == expected
    assert _compat.merge.__module__ == expected


@needs_315
@pytest.mark.parametrize("path", PY315_FILES, ids=lambda p: p.name)
def test_it_compiles_on_3_15(path):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


@needs_315
def test_the_syntax_itself():
    namespace: dict = {}
    exec("lists = [*xs for xs in [[1, 2], [3]]]\n"
         "sets = {*xs for xs in [[1, 2], [2, 3]]}\n"
         "dicts = {**d for d in [{'a': 1}, {'a': 2, 'b': 3}]}\n"
         "gen = list(*xs for xs in [[1], [2, 3]])\n", namespace)
    assert namespace["lists"] == [1, 2, 3]
    assert namespace["sets"] == {1, 2, 3}
    assert namespace["dicts"] == {"a": 2, "b": 3}
    assert namespace["gen"] == [1, 2, 3]


@needs_315
@pytest.mark.parametrize("case", range(len(CASES)))
def test_3_15_and_legacy_agree(case):
    from hypernix._compat import _iter_py315 as new

    fresh = CASES[case]

    assert new.flatten(fresh()) == legacy.flatten(fresh())
    assert new.union(fresh()) == legacy.union(fresh())
    maps = [{str(i): v} for i, v in enumerate(legacy.flatten(fresh()))] + [{"0": "last"}]
    assert new.merge(maps) == legacy.merge(maps)


def test_the_call_sites_get_the_same_answers_either_way():
    """What hypernix itself flattens with it."""
    pytest.importorskip("torch", reason="the optimizer module needs torch")
    from hypernix.optimizers.pressure_cooker import _flatten_params

    a, b, c = object(), object(), object()
    assert _flatten_params([{"params": [a, b]}, c, {"params": []}]) == [a, b, c]


def test_the_lmstudio_scan_merges_every_root(tmp_path, monkeypatch):
    from hypernix.quant import hyprslug_headers as hh

    monkeypatch.setattr(hh, "scan", lambda root: [{"root": str(root), "n": i} for i in range(2)])
    rows = hh.flatten(hh.scan(root) for root in (tmp_path / "a", tmp_path / "b"))
    assert [r["root"][-1] for r in rows] == ["a", "a", "b", "b"]


def test_merge_keeps_the_last_value_and_first_position():
    assert list(_compat.merge([{"a": 1, "b": 2}, {"a": 3}]).items()) == [("a", 3), ("b", 2)]
