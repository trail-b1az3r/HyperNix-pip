"""``GET /version`` — what is running here, and what is installed.

The report was "it's incorrect: the server is actually running
0.72.5.post5" while HyperLink showed 0.72.5.dev3.

Both were true. ``/status`` reports ``hypernix.__version__``, which is
bound when the module is imported, so a server upgraded with pip and
never restarted keeps reporting the version it started with. The app was
faithfully showing a stale process and had no way to say so, because one
number cannot distinguish "wrong" from "out of date".

So: a second number, read from the distribution metadata on disk rather
than the loaded module, and a flag for when they disagree.
"""
from __future__ import annotations

import pytest
from conftest import clear_t1_config

fastapi = pytest.importorskip("fastapi", reason="needs the [t1api] extra (fastapi)")

from fastapi.testclient import TestClient  # noqa: E402

from hypernix.t1api.app import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Every other t1api client fixture does this and mine did not, so
    # `create_app()` went to the real ~/.hypernix and the conftest rail
    # caught it. Not every T1_* variable: the storage redirects in
    # conftest.py have to stay, or "a server with no configuration"
    # quietly becomes "a server writing to the real home directory".
    clear_t1_config(monkeypatch)
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
    monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    return TestClient(create_app())


class TestItReportsBothVersions:
    def test_it_answers_without_a_credential(self, client):
        """Like /health. It carries no configuration and no secrets, and
        /status already exposes the same version to anyone who can reach
        it, so requiring a token here would only make the endpoint
        useless in the case it exists for: a pairing that has gone
        stale because the server was upgraded."""
        assert client.get("/version").status_code == 200

    def test_it_reports_what_is_running(self, client):
        import hypernix

        body = client.get("/version").json()
        assert body["hypernix"] == hypernix.__version__

    def test_it_names_the_interpreter_and_the_module(self, client):
        """"I upgraded and nothing changed" is usually two environments
        on one machine. These two fields say which one is serving."""
        import sys

        body = client.get("/version").json()
        assert body["executable"] == sys.executable
        assert body["module_path"].endswith("__init__.py")

    def test_uptime_is_present_and_not_negative(self, client):
        body = client.get("/version").json()
        assert body["uptime_seconds"] >= 0


class TestStaleness:
    """The whole point: the two disagreeing is a *positive* signal."""

    def test_stale_when_the_installed_version_differs(self, client, monkeypatch):
        from hypernix.t1api.routers import health

        monkeypatch.setattr(health, "installed_version", lambda: "9.9.9")
        body = client.get("/version").json()
        assert body["hypernix_installed"] == "9.9.9"
        assert body["stale"] is True

    def test_not_stale_when_they_agree(self, client, monkeypatch):
        import hypernix
        from hypernix.t1api.routers import health

        monkeypatch.setattr(
            health, "installed_version", lambda: hypernix.__version__
        )
        body = client.get("/version").json()
        assert body["stale"] is False

    def test_a_source_checkout_is_not_stale(self, client, monkeypatch):
        """No distribution metadata is not the same as an old one.

        Running from a checkout on sys.path has no dist-info to read, and
        calling that "stale" would put a restart-the-server banner in
        front of every developer.
        """
        from hypernix.t1api.routers import health

        monkeypatch.setattr(health, "installed_version", lambda: "")
        body = client.get("/version").json()
        assert body["hypernix_installed"] == ""
        assert body["stale"] is False

    def test_unreadable_metadata_is_not_an_error(self, monkeypatch):
        """importlib raising must not take the endpoint down with it."""
        import importlib.metadata

        from hypernix.t1api.routers.health import installed_version

        def boom(_name):
            raise importlib.metadata.PackageNotFoundError("hypernix")

        monkeypatch.setattr(importlib.metadata, "version", boom)
        assert installed_version() == ""
