"""Deprecated modules say so, on import, where it can be seen.

Four modules had said so since 0.71.5a2, with a `rich.Console().print`
at the top of the file. That got the hard part right — it was visible —
and four things wrong:

* it wrote to **stdout**, so the notice landed in whatever the caller
  was capturing; `hnx ... > out.json` got English in its JSON
* there was no `DeprecationWarning`, so `-W error` did not fail on it,
  `pytest.warns` could not assert it, and nothing could find callers
* it could not be turned off
* it imported `rich` to print eleven words

A fifth module, `monitoring.tvtop`, described itself as existing "solely
for backwards-compatibility" and then said nothing at all.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"

# module -> the successor it should name.
DEPRECATED = {
    "hypernix.models.old_oven": "hypernix.models.neo_oven",
    "hypernix.system.old_fridge": "hypernix.models.neo_oven",
    "hypernix.data.mediocre_fridge": "hypernix.models.neo_oven",
    "hypernix.evaluation.new_fridge": "hypernix.models.neo_oven",
    "hypernix.monitoring.tvtop": "hypernix.monitoring.tv",
}


def run_import(module: str, *, env_extra: dict[str, str] | None = None,
               args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Import *module* in a fresh interpreter.

    A subprocess, because `sys.modules` caches the import and the whole
    point is what happens the first time.
    """
    env = {**os.environ, "PYTHONPATH": str(SRC), "NO_COLOR": "1"}
    env.pop("HYPERNIX_DEPRECATION_WARNINGS", None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, *(args or []), "-c",
         f"import {module}; print('IMPORTED', flush=True)"],
        capture_output=True, text=True, encoding="utf-8", timeout=300, env=env,
    )


@pytest.mark.parametrize("module", sorted(DEPRECATED))
class TestEachDeprecatedModuleAnnouncesItself:
    def test_it_says_so_on_import(self, module):
        done = run_import(module)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "is deprecated" in done.stderr, (
            f"{module} imported silently\nstdout: {done.stdout!r}\n"
            f"stderr: {done.stderr!r}"
        )

    def test_it_names_the_successor(self, module):
        done = run_import(module)
        assert DEPRECATED[module] in done.stderr, done.stderr

    def test_nothing_lands_on_stdout(self, module):
        """The bug. `rich.Console()` defaults to stdout."""
        done = run_import(module)
        assert done.stdout.strip() == "IMPORTED", (
            f"{module} wrote to stdout, which is the caller's data: "
            f"{done.stdout!r}"
        )

    def test_it_is_a_real_deprecation_warning(self, module):
        """Visible to -W, to pytest.warns, and to anything auditing."""
        done = run_import(module, args=["-W", "error::DeprecationWarning"])
        assert done.returncode != 0, (
            f"{module} did not raise under -W error::DeprecationWarning"
        )
        assert "DeprecationWarning" in done.stderr

    def test_the_env_var_silences_hypernix_own_line(self, module):
        """Only the fallback. `warnings` still governs the warning.

        Both of these runs are `python -c`, which is `__main__`, where
        Python's default filters *do* show a DeprecationWarning — so the
        fallback never fires and the variable has nothing to silence.
        The hidden case, which is where it matters, is covered by
        `test_the_two_together_are_silent` below.
        """
        loud = run_import(module, args=["-W", "ignore::DeprecationWarning"])
        assert "is deprecated" in loud.stderr, (
            "with the warning ignored, the fallback should be all that is left"
        )
        quiet = run_import(
            module,
            args=["-W", "ignore::DeprecationWarning"],
            env_extra={"HYPERNIX_DEPRECATION_WARNINGS": "0"},
        )
        assert quiet.returncode == 0, quiet.stdout + quiet.stderr
        assert "is deprecated" not in quiet.stderr, quiet.stderr

    def test_the_two_together_are_silent(self, module):
        """The documented way to hear nothing at all.

        `PYTHONWARNINGS` is an environment variable too, so an operator
        who wants silence has an env-only route and does not need to
        change how the program is launched.
        """
        done = run_import(
            module,
            env_extra={
                "PYTHONWARNINGS": "ignore::DeprecationWarning",
                "HYPERNIX_DEPRECATION_WARNINGS": "0",
            },
        )
        assert done.returncode == 0, done.stdout + done.stderr
        assert "is deprecated" not in done.stderr, done.stderr
        assert done.stdout.strip() == "IMPORTED"

    def test_silencing_the_notice_does_not_silence_the_warning(self, module):
        """The env var is for noise. `-W` is for semantics.

        If the variable also suppressed the DeprecationWarning, setting
        it would quietly disarm `-W error` for the whole process.
        """
        done = run_import(
            module,
            env_extra={"HYPERNIX_DEPRECATION_WARNINGS": "0"},
            args=["-W", "error::DeprecationWarning"],
        )
        assert done.returncode != 0, done.stdout + done.stderr

    def test_it_announces_once_not_twice(self, module):
        """Shown or suppressed, exactly one message either way.

        The stderr fallback exists because DeprecationWarning is hidden
        by default; if it fired *as well as* a displayed warning, every
        developer running -W would see the notice twice.
        """
        done = run_import(module, args=["-W", "default::DeprecationWarning"])
        assert done.stderr.count("is deprecated") == 1, done.stderr


class TestTheHiddenCase:
    """DeprecationWarning shows nothing at all outside `__main__`.

    This is the case the whole stderr fallback exists for, and the one
    that would make "prints immediately on import" a lie if it were left
    to `warnings` alone.
    """

    def test_an_import_from_a_library_still_prints(self, tmp_path):
        (tmp_path / "importer.py").write_text(
            "import hypernix.monitoring.tvtop  # noqa: F401\n", encoding="utf-8"
        )
        (tmp_path / "script.py").write_text(
            "import importer  # noqa: F401\nprint('DONE')\n", encoding="utf-8"
        )
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(SRC), str(tmp_path)]),
            "NO_COLOR": "1",
        }
        env.pop("HYPERNIX_DEPRECATION_WARNINGS", None)
        done = subprocess.run(
            [sys.executable, str(tmp_path / "script.py")],
            capture_output=True, text=True, encoding="utf-8", timeout=300, env=env,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        assert done.stdout.strip() == "DONE"
        assert "is deprecated" in done.stderr, (
            "the warning was hidden and nothing took its place — which is "
            "what plain warnings.warn does outside __main__"
        )


class TestNoModuleIsMissed:
    """The guarantee: a module that says it is deprecated must announce it.

    Read from the source rather than by importing, because importing
    every module in the package pulls in torch, Qt and a FastAPI app.
    """

    @staticmethod
    def _modules() -> list[Path]:
        return [
            p for p in (SRC / "hypernix").rglob("*.py")
            if "__pycache__" not in p.parts
        ]

    @staticmethod
    def _dotted(path: Path) -> str:
        rel = path.relative_to(SRC).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    @staticmethod
    def _announces(path: Path) -> bool:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:                      # module level only
            call = node.value if isinstance(node, ast.Expr) else None
            if isinstance(call, ast.Call):
                func = call.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name == "deprecated_module":
                    return True
        return False

    def test_the_known_set_is_what_the_tree_says(self):
        announcing = {
            self._dotted(p) for p in self._modules() if self._announces(p)
        }
        assert announcing == set(DEPRECATED), (
            f"only in the tree: {sorted(announcing - set(DEPRECATED))}\n"
            f"only in this test: {sorted(set(DEPRECATED) - announcing)}"
        )

    def test_a_module_calling_itself_deprecated_announces_it(self):
        """The check that makes 'all of them' mean something.

        A docstring saying "deprecated" or "compatibility shim" and no
        `deprecated_module` call is exactly how `monitoring.tvtop` sat
        for several releases telling nobody.
        """
        claims = re.compile(
            r"\bis deprecated\b|\bcompatibility shim\b|"
            r"\bsolely for backwards[- ]compat|\bdeprecated module\b",
            re.I,
        )
        # The helper explains deprecation at length, which is not the
        # same as being deprecated. Excluded by path rather than by
        # weakening the pattern, so the pattern stays blunt.
        mechanism = SRC / "hypernix" / "system" / "deprecation.py"
        missing = []
        for path in self._modules():
            if path == mechanism:
                continue
            doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
            if claims.search(doc) and not self._announces(path):
                missing.append(self._dotted(path))
        assert not missing, (
            f"these describe themselves as deprecated but announce nothing: {missing}"
        )

    def test_rich_is_no_longer_pulled_in_to_print_a_notice(self):
        for module in DEPRECATED:
            path = SRC / Path(*module.split(".")).with_suffix(".py")
            source = path.read_text(encoding="utf-8")
            assert "rich.console" not in source, f"{module} still imports rich for this"

    def test_the_announcement_comes_before_the_heavy_imports(self):
        """So it fires even when the module below it fails to load.

        `old_oven` imports torch; a notice after that waits several
        seconds, and never appears at all if the import raises. ruff.toml
        carries an E402 exemption per module for exactly this ordering,
        so the exemption list has to keep up with the module list.
        """
        ruff = (REPO_ROOT / "ruff.toml").read_text(encoding="utf-8")
        for module in DEPRECATED:
            rel = "src/" + "/".join(module.split(".")) + ".py"
            assert f'"{rel}"' in ruff, f"{rel} is missing its E402 exemption"


class TestTheHelperItself:
    def test_it_imports_nothing_heavy(self):
        """A deprecation notice must not cost a dependency.

        The old one imported rich. This one is reached from the top of
        four modules that have not run their own imports yet.
        """
        source = (SRC / "hypernix" / "system" / "deprecation.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= {"os", "sys", "warnings", "__future__"}, imported

    def test_the_message_reads_as_a_sentence(self):
        from hypernix.system.deprecation import notice_for
        assert notice_for("a.b", instead="c.d") == "a.b is deprecated — use c.d instead."
        assert notice_for("a.b", instead="c.d", since="0.1") == (
            "a.b is deprecated since 0.1 — use c.d instead."
        )
        assert "removal in 0.9" in notice_for("a.b", instead="c.d", removed_in="0.9")

    def test_calling_it_twice_announces_once(self, capsys):
        from hypernix.system import deprecation
        deprecation.reset_for_testing()
        try:
            deprecation.deprecated_module("fake.mod", instead="other.mod")
            deprecation.deprecated_module("fake.mod", instead="other.mod")
            assert capsys.readouterr().err.count("is deprecated") == 1
        finally:
            deprecation.reset_for_testing()

    def test_it_never_writes_to_stdout(self, capsys):
        from hypernix.system import deprecation
        deprecation.reset_for_testing()
        try:
            deprecation.deprecated_module("fake.other", instead="x.y")
            captured = capsys.readouterr()
            assert captured.out == ""
            assert "is deprecated" in captured.err
        finally:
            deprecation.reset_for_testing()
