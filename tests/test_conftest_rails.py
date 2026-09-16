"""The rails in conftest.py, asserted rather than assumed.

A guard nobody checks is a guard that quietly stops working. These are
cheap and they are the difference between "the suite is isolated" as a
belief and as a fact.
"""
from __future__ import annotations

import os
from pathlib import Path

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
