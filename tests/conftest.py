"""Test-wide safety rails.

One job: **no test writes to the real ``~/.hypernix``.**

The T1 stores default to ``~/.hypernix/t1api/t1api.sqlite3`` when nothing
says otherwise, and ``create_app()`` with no configuration is the natural
way to build a TestClient. So a suite written the obvious way reads and
writes whatever the person running it actually has — their sessions,
their memories, their keys — and, because rows persist between tests,
produces counts that are right when a file runs alone and wrong in a full
run. Both are confusing in a way that costs an afternoon.

Two layers, because the environment variable alone is not enough:

1. The paths are redirected for the whole session, before anything is
   imported.
2. ``SQLiteBackend`` itself refuses the real path. That is the one that
   matters: ``T1APIConfig`` reads ``T1_DB_PATH`` *at construction*, so a
   config built by a test that had just cleared every ``T1_*`` variable
   carries ``db_path=None`` and falls back to the real file, past every
   environment-level guard. Several suites clear the environment like
   that on purpose and are right to.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

#: Where the stores land unless told otherwise.
REAL_HOME = Path.home() / ".hypernix"

_SESSION_ROOT = Path(tempfile.mkdtemp(prefix="hypernix-tests-"))


#: The variables that decide *where things are written*, as opposed to
#: how the server behaves.
#:
#: The distinction matters because several suites clear every ``T1_*``
#: variable to get a known configuration, and they are right to — but
#: "no trusted network, no partial admin" and "use the real
#: ``~/.hypernix``" are two different requests, and clearing the lot
#: makes the second one by accident. Use :func:`clear_t1_config` to ask
#: for the first without the second.
STORAGE_KEYS = (
    "T1_DB_PATH",
    "T1_BACKUP_DIR",
    "T1_MODULE_STORAGE_DIR",
    "T1_CONFIG_DIR",
)


def _redirect_environment() -> None:
    os.environ.setdefault("T1_DB_PATH", str(_SESSION_ROOT / "t1api.sqlite3"))
    os.environ.setdefault("T1_BACKUP_DIR", str(_SESSION_ROOT / "backups"))
    os.environ.setdefault("T1_MODULE_STORAGE_DIR", str(_SESSION_ROOT / "modules"))
    # T1_CONFIG_DIR, not T1_HYPERLINK_DIR: the latter was the name this
    # file invented and nothing in the package reads it, so the redirect
    # it was making had no effect at all. What actually decides where
    # HyperLink's server identity is written is
    # `hyperlink.identity._config_root`, which reads T1_CONFIG_DIR and
    # otherwise falls back to ~/.hypernix/t1api -- so every test that
    # built the app wrote a seed file, which is *key material*, into the
    # real home. The same variable covers the registry, the launcher and
    # the training monitor.
    os.environ.setdefault("T1_CONFIG_DIR", str(_SESSION_ROOT / "config"))


def clear_t1_config(monkeypatch) -> None:
    """Drop every ``T1_*`` variable except the storage redirects.

    What a suite means by "clear the environment" is a server with no
    configuration, not a server writing to the person's real home. The
    two came apart in a full run only: the storage keys are set by this
    file at session start, so a suite that cleared them got the real
    ``~/.hypernix/t1api`` and created it — the modules directory, the
    hyperlink directory, and, whenever it got past the
    ``SQLiteBackend`` guard, the database.
    """
    for name in [k for k in os.environ if k.startswith("T1_")]:
        if name not in STORAGE_KEYS:
            monkeypatch.delenv(name, raising=False)
    _redirect_environment()


_redirect_environment()


@pytest.fixture(scope="session", autouse=True)
def _no_real_database() -> None:
    """Make ``SQLiteBackend`` incapable of opening the real database.

    Patched at class level for the whole session rather than per test,
    because the leak happens inside ``create_app()`` — which a test may
    call at any point, including from a module-level helper that no
    function-scoped fixture is wrapping.

    This is a redirect rather than a refusal on purpose: raising would
    turn "this test did not isolate itself" into a failure in an
    unrelated place, where redirecting makes it simply work on a
    throwaway file. The assertion that nothing escaped lives in
    ``test_conftest_rails.py``.
    """
    from hypernix.t1api.db import SQLiteBackend

    original = SQLiteBackend.__init__

    def guarded(self, db_path=None):  # noqa: ANN001 - mirrors the real signature
        if db_path is None or str(REAL_HOME) in str(db_path):
            db_path = os.environ.get("T1_DB_PATH") or str(
                _SESSION_ROOT / "t1api.sqlite3"
            )
        return original(self, db_path)

    SQLiteBackend.__init__ = guarded
    try:
        yield
    finally:
        SQLiteBackend.__init__ = original


@pytest.fixture(autouse=True)
def _isolate_the_environment() -> None:
    """Every ``T1_*`` variable is restored after each test.

    Two problems, one fix.

    **Configuration leaks between files.** A helper that does
    ``os.environ["T1_TRUSTED_NETWORK"] = "1"`` — rather than going through
    monkeypatch — leaves it set for everything that runs afterwards, so a
    later file's ``test_it_needs_a_key`` gets a 200 from a server that is
    now in trusted-network mode. That failure shows up in a full run and
    not when the file is run alone, which is the most expensive kind to
    diagnose. Restoring here makes it impossible, rather than asking
    every helper in every file to remember.

    **The storage redirect gets cleared.** Several suites drop every
    ``T1_*`` variable on purpose to get a known configuration, which also
    removes the paths set above. Re-applied on the way in, so "clear the
    config" cannot quietly come to mean "and use the real database".
    """
    saved = {k: v for k, v in os.environ.items() if k.startswith("T1_")}
    _redirect_environment()
    try:
        yield
    finally:
        for name in [k for k in os.environ if k.startswith("T1_")]:
            del os.environ[name]
        os.environ.update(saved)
        _redirect_environment()
