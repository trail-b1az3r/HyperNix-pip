"""The rails in conftest.py, asserted rather than assumed.

A guard nobody checks is a guard that quietly stops working. These are
cheap and they are the difference between "the suite is isolated" as a
belief and as a fact.
"""
from __future__ import annotations

import os
from pathlib import Path

import conftest
from conftest import REAL_HOME


class TestTheDatabaseIsRedirected:
    def test_the_environment_points_somewhere_else(self):
        configured = os.environ.get("T1_DB_PATH", "")
        assert configured
        assert str(REAL_HOME) not in configured

    def test_a_backend_built_with_no_path_does_not_get_the_real_one(self):
        """The case the environment variable cannot cover: a config
        constructed while T1_* was cleared carries db_path=None, and the
        backend's own default is the real file."""
        from hypernix.t1api.db import SQLiteBackend

        assert str(REAL_HOME) not in str(SQLiteBackend().db_path)

    def test_a_backend_asked_for_the_real_one_is_redirected(self):
        from hypernix.t1api.db import SQLiteBackend

        asked = REAL_HOME / "t1api" / "t1api.sqlite3"
        assert str(REAL_HOME) not in str(SQLiteBackend(str(asked)).db_path)

    def test_an_explicit_temporary_path_is_left_alone(self, tmp_path):
        """The guard must not hijack a test that isolated itself
        properly — those are the majority and they are doing it right."""
        from hypernix.t1api.db import SQLiteBackend

        wanted = tmp_path / "mine.sqlite3"
        assert SQLiteBackend(str(wanted)).db_path == wanted


class TestNothingReachedTheRealHome:
    def test_the_real_database_file_was_not_created(self):
        """Runs like any other test, so a full run asserts it and a
        single-file run asserts it too.

        Deliberately checks the *file*, not the directory: ~/.hypernix
        legitimately exists on a developer's machine and holds their
        models. What must not appear is the database.
        """
        leaked = REAL_HOME / "t1api" / "t1api.sqlite3"
        if not leaked.exists():
            return
        # It exists. That is only a failure if this run made it — an
        # existing install has one, and deleting somebody's database to
        # make a test pass would be the worst possible fix.
        import conftest

        started = Path(conftest.__file__).stat().st_mtime
        assert leaked.stat().st_mtime < started, (
            f"{leaked} was written during this test run. Something built a "
            "SQLiteBackend that escaped the redirect in tests/conftest.py."
        )


class TestTheRedirectsPointAtVariablesSomethingReads:
    """A redirect nothing honours is worse than none: it reads as
    coverage.

    ``T1_HYPERLINK_DIR`` was set here for a long time and appears
    nowhere in the package. What actually decides where HyperLink's
    server identity goes is ``T1_CONFIG_DIR``, which was not set — so
    every test that built the app wrote a seed file, which is key
    material, into the real ``~/.hypernix/t1api/hyperlink``.
    """

    def test_every_redirected_variable_is_read_by_the_package(self):
        import subprocess

        source = Path(__file__).resolve().parent.parent / "src"
        for name in conftest.STORAGE_KEYS:
            found = subprocess.run(
                ["grep", "-rl", name, str(source)],
                capture_output=True, text=True,
            )
            assert found.stdout.strip(), (
                f"tests/conftest.py redirects {name}, which nothing under "
                f"src/ reads. Either the package stopped using it or the "
                f"name was never right — and meanwhile whatever it was "
                f"meant to redirect is going to the real home."
            )

    def test_the_hyperlink_identity_lands_in_the_sandbox(self):
        """The one the missing variable was letting through."""
        from hypernix.hyperlink.identity import seed_path

        assert REAL_HOME not in seed_path().parents


class TestNoSuiteClearsTheStorageRedirects:
    """The mistake is easy, invisible, and was made again this release.

    Several suites clear every ``T1_*`` variable to get a server with no
    configuration, and they are right to want that. But the storage
    redirects live in the same namespace, so the obvious loop::

        for name in [k for k in os.environ if k.startswith("T1_")]:
            monkeypatch.delenv(name, raising=False)

    also removes ``T1_BACKUP_DIR`` and ``T1_MODULE_STORAGE_DIR``, and
    the app then creates ``~/.hypernix/t1api/backups`` and
    ``.../modules`` in the person's real home. It never fails a test,
    and it only shows up if somebody happens to look at their home
    directory afterwards — which is exactly how long it survived.

    `conftest.clear_t1_config` does the same thing and keeps the four
    paths. This test is what makes people use it.
    """

    #: Files allowed to write the loop by hand.
    #:
    #: `conftest.py` is where the correct version lives. The others
    #: match the string for an unrelated reason — a T1 key prefix — and
    #: are listed rather than excluded by a cleverer pattern, because a
    #: cleverer pattern is one somebody's new file accidentally matches.
    ALLOWED = {
        "conftest.py",
        "test_conftest_rails.py",           # this file quotes the bad loop
        "test_t1_accounts.py",              # asserts a key starts with "T1_"
        "test_v0710_gatekeeper_keymaster.py",  # same
    }

    def test_no_test_file_deletes_every_t1_variable(self):
        import re

        offenders = []
        here = Path(__file__).resolve().parent
        for path in sorted(here.glob("test_*.py")):
            if path.name in self.ALLOWED:
                continue
            text = path.read_text(encoding="utf-8")
            # The loop, not the string: `startswith("T1_")` next to a
            # `delenv` is the shape that does the damage.
            for match in re.finditer(r'startswith\("T1_"\)', text):
                window = text[match.start():match.start() + 200]
                if "STORAGE_KEYS" in window:
                    # Excluding them by name is the other correct
                    # answer, and the one a module-level helper with no
                    # monkeypatch fixture has to use.
                    continue
                if "delenv" in window or "environ.pop" in window:
                    offenders.append(f"{path.name}:{text[:match.start()].count(chr(10)) + 1}")
        assert not offenders, (
            "these clear the storage redirects along with the config, so the "
            "app writes to the real ~/.hypernix: "
            + ", ".join(offenders)
            + ". Use conftest.clear_t1_config(monkeypatch) instead."
        )

    def test_clear_t1_config_keeps_the_paths(self, monkeypatch):
        import os

        conftest.clear_t1_config(monkeypatch)
        for name in conftest.STORAGE_KEYS:
            assert os.environ.get(name), f"{name} was cleared"

    def test_clear_t1_config_clears_everything_else(self, monkeypatch):
        import os

        monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
        conftest.clear_t1_config(monkeypatch)
        assert "T1_TRUSTED_NETWORK" not in os.environ

    def test_the_paths_it_keeps_are_not_the_real_home(self, monkeypatch):
        import os

        conftest.clear_t1_config(monkeypatch)
        for name in conftest.STORAGE_KEYS:
            assert str(REAL_HOME) not in os.environ.get(name, "")
