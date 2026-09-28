"""The category layout and its back-compat guarantee.

Modules live in category subpackages (``hypernix/timing/timer.py``) and are
listed in :data:`hypernix.MODULE_CATEGORIES`. Three things have to stay true
for that to be a safe place to keep them, and each one is a test below:

1. The table matches the filesystem — no module is missing from it, no entry
   points at a file that isn't there.
2. Every module is still importable under its historical flat name, and both
   spellings give back *the same module object* (so ``isinstance`` checks and
   monkeypatching work through either one).
3. ``python -m hypernix.<flat>`` still works, because the flat name resolves
   through an alias loader rather than a real file.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import hypernix

PKG_ROOT = Path(hypernix.__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

ALL_MODULES = sorted(hypernix.CATEGORY_OF)


class TestLayoutTable:
    def test_every_category_is_a_package(self) -> None:
        for category in hypernix.MODULE_CATEGORIES:
            assert (PKG_ROOT / category / "__init__.py").is_file(), category

    def test_every_listed_module_exists_on_disk(self) -> None:
        missing = [
            f"{cat}/{mod}.py"
            for cat, mods in hypernix.MODULE_CATEGORIES.items()
            for mod in mods
            if not (PKG_ROOT / cat / f"{mod}.py").is_file()
        ]
        assert missing == []

    def test_no_stray_uncategorised_module(self) -> None:
        # __init__ and __main__ are the package's own plumbing; everything
        # else at the top level should have been given a category.
        stray = sorted(
            p.stem
            for p in PKG_ROOT.glob("*.py")
            if p.stem not in {"__init__", "__main__"}
        )
        assert stray == []

    def test_no_module_is_listed_twice(self) -> None:
        listed = [m for mods in hypernix.MODULE_CATEGORIES.values() for m in mods]
        assert len(listed) == len(set(listed))

    def test_category_of_is_the_reverse_mapping(self) -> None:
        expected = {
            mod: cat
            for cat, mods in hypernix.MODULE_CATEGORIES.items()
            for mod in mods
        }
        assert hypernix.CATEGORY_OF == expected


class TestFlatNameCompat:
    @pytest.mark.parametrize("flat", ALL_MODULES)
    def test_flat_name_resolves_to_the_categorised_module(self, flat: str) -> None:
        category = hypernix.CATEGORY_OF[flat]
        try:
            real = importlib.import_module(f"hypernix.{category}.{flat}")
        except Exception as exc:  # optional dependency missing in this env
            pytest.skip(f"hypernix.{category}.{flat} not importable here: {exc!r}")
        alias = importlib.import_module(f"hypernix.{flat}")
        assert alias is real
        assert alias.__name__ == f"hypernix.{category}.{flat}"

    def test_reload_after_a_flat_import_keeps_the_real_name(self) -> None:
        """Importing by the flat name used to leave the alias's spec on the
        real module, so a later ``importlib.reload`` re-ran it as
        ``hypernix.<flat>`` and renamed it for everyone. Whether a test
        run tripped on that depended on which file happened to go first:
        hence a fresh interpreter."""
        code = (
            "import importlib\n"
            "import hypernix.timer as alias\n"
            "import hypernix.timing.timer as real\n"
            "assert alias is real\n"
            "assert real.__spec__.name == 'hypernix.timing.timer', real.__spec__.name\n"
            "importlib.reload(real)\n"
            "assert real.__name__ == 'hypernix.timing.timer', real.__name__\n"
            "import hypernix.timer as again\n"
            "assert again is real\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": str(SRC)},
        )
        assert result.returncode == 0, result.stderr

    def test_attribute_access_matches_import(self) -> None:
        assert hypernix.timer is importlib.import_module("hypernix.timing.timer")
        assert hypernix.timing.timer is hypernix.timer

    def test_from_import_still_works(self) -> None:
        from hypernix.timer import KitchenTimer
        from hypernix.timing.timer import KitchenTimer as Categorised

        assert KitchenTimer is Categorised

    def test_unknown_module_still_raises(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("hypernix.not_a_real_module")

    def test_alias_loader_exposes_source(self) -> None:
        # inspect.getsource / linecache / runpy all need this.
        import inspect

        alias = importlib.import_module("hypernix.timer")
        assert "class KitchenTimer" in inspect.getsource(alias)


def test_runpy_through_a_flat_alias() -> None:
    """``python -m hypernix.cli`` goes through the alias loader's get_code."""
    env = {
        "PYTHONPATH": str(SRC),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HYPERNIX_AUTO_INSTALL": "0",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "hypernix.cli", "--help"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr
