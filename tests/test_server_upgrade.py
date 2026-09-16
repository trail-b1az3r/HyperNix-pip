"""Telling somebody how to update the machine they cannot reach.

"The T1 installed thinks it is running an older version." Part of that
was `install-t1.sh` printing a hand-maintained constant that had gone
stale — fixed where it lives, in `test_install_script.py`. The other
part is real: a server falls behind, and until now finding out meant
walking over to it.

What makes this worth an endpoint rather than a documentation page is
one field: `sys.executable`.

    pip install --upgrade hypernix

on a machine with a system Python, a pyenv, and the venv the service
actually runs under upgrades whichever comes first on `PATH`, prints a
cheerful success, and leaves the server running exactly the version it
was. That is not a hypothetical — it is the single most common way a
"the update did not work" report happens, and it is indistinguishable
from the update genuinely failing.

The server knows which interpreter is running it. These commands say so.
"""
from __future__ import annotations

import shlex
import sys

import pytest
from conftest import clear_t1_config

from hypernix.t1api import upgrade


class TestItDescribesTheRealInstallation:
    def test_it_reports_the_running_interpreter(self):
        assert upgrade.describe().executable == sys.executable

    def test_it_reports_both_versions(self):
        import hypernix
        from hypernix.t1api.version import T1_VERSION

        found = upgrade.describe()
        assert found.package_version == hypernix.__version__
        assert found.t1_version == T1_VERSION.short

    def test_it_knows_whether_it_is_in_a_venv(self):
        found = upgrade.describe()
        assert found.in_venv == (sys.prefix != sys.base_prefix)

    def test_every_field_survives_a_round_trip(self):
        data = upgrade.describe().to_dict()
        for field in ("executable", "prefix", "in_venv", "editable",
                      "python_version", "package_version", "t1_version"):
            assert field in data


class TestTheCommandsNameTheInterpreter:
    """The whole reason this exists."""

    def test_the_upgrade_command_uses_this_python(self):
        commands = upgrade.plan().commands
        primary = next(c for c in commands if c.primary)
        assert shlex.quote(sys.executable) in primary.command

    def test_no_command_is_a_bare_pip(self):
        """`pip install ...` with no interpreter is the advice that
        upgrades a different installation and reports success."""
        for command in upgrade.plan().commands:
            assert not command.command.startswith("pip "), command.command

    def test_a_path_with_a_space_is_quoted(self, monkeypatch):
        """"/Users/some one/venv/bin/python" is a real path shape on a
        Mac, and an unquoted one produces a command that runs
        `/Users/some` and fails in a way nobody can read."""
        found = upgrade.describe()
        spaced = upgrade.Installation(
            executable="/Users/some one/venv/bin/python",
            prefix=found.prefix, in_venv=True, editable=False,
            location=found.location, python_version=found.python_version,
            package_version=found.package_version, t1_version=found.t1_version,
        )
        primary = next(c for c in upgrade.plan(spaced).commands if c.primary)
        assert "'/Users/some one/venv/bin/python'" in primary.command

    def test_exactly_one_command_is_primary(self):
        """A copy area with three equally-weighted commands is a
        decision the reader did not ask to make."""
        primaries = [c for c in upgrade.plan().commands if c.primary]
        assert len(primaries) == 1

    def test_every_command_has_a_label_and_a_reason(self):
        for command in upgrade.plan().commands:
            assert command.label
            assert command.command
            assert command.note


class TestItWarnsAboutWhatThePipCommandCannotDo:
    def test_it_says_the_server_is_not_restarted(self):
        """The most confusing possible outcome: the upgrade works, and
        the server keeps serving the old code because nothing restarted
        it. Someone then reports that the update did nothing."""
        warnings = " ".join(upgrade.plan().warnings).lower()
        assert "restart" in warnings

    def test_a_system_python_is_flagged(self, monkeypatch):
        found = upgrade.describe()
        system = upgrade.Installation(
            executable=found.executable, prefix=found.prefix,
            in_venv=False, editable=False, location=found.location,
            python_version=found.python_version,
            package_version=found.package_version, t1_version=found.t1_version,
        )
        warnings = " ".join(upgrade.plan(system).warnings)
        assert "virtual environment" in warnings

    def test_a_venv_is_not_flagged_for_it(self):
        found = upgrade.describe()
        in_venv = upgrade.Installation(
            executable=found.executable, prefix=found.prefix,
            in_venv=True, editable=False, location=found.location,
            python_version=found.python_version,
            package_version=found.package_version, t1_version=found.t1_version,
        )
        warnings = " ".join(upgrade.plan(in_venv).warnings)
        assert "virtual environment" not in warnings


class TestADevelopmentInstall:
    """pip will not replace an editable install, and a command that
    silently no-ops is worse than no command."""

    @staticmethod
    def _editable() -> upgrade.Installation:
        found = upgrade.describe()
        return upgrade.Installation(
            executable=found.executable, prefix=found.prefix, in_venv=True,
            editable=True, location="/home/someone/HyperNix-pip/src/hypernix",
            python_version=found.python_version,
            package_version=found.package_version, t1_version=found.t1_version,
        )

    def test_it_is_told_to_use_git_not_pip(self):
        primary = next(c for c in upgrade.plan(self._editable()).commands if c.primary)
        assert "git pull" in primary.command

    def test_it_says_why(self):
        warnings = " ".join(upgrade.plan(self._editable()).warnings)
        assert "development install" in warnings.lower()
        assert "pip will not replace it" in warnings

    def test_the_checkout_path_is_quoted(self):
        primary = next(c for c in upgrade.plan(self._editable()).commands if c.primary)
        assert "/home/someone/HyperNix-pip/src/hypernix" in primary.command


class TestTheEndpoint:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")

        from fastapi.testclient import TestClient

        # Not every T1_* variable — the storage redirects in
        # conftest.py stay, or "a server with no configuration" quietly
        # becomes "a server writing to the real ~/.hypernix".
        clear_t1_config(monkeypatch)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")

        from hypernix.t1api.app import create_app

        return TestClient(create_app(), client=("192.168.1.9", 5000))

    def test_an_ordinary_caller_can_read_it(self, client):
        """Deliberately not admin-only. "Which version is this and how
        do I move it" is the question behind most of the confusing
        behaviour people report, and making it a secret keeps the answer
        from the person who needs it."""
        assert client.get("/hyperlink/upgrade").status_code == 200

    def test_it_reports_what_is_installed(self, client):
        import hypernix

        body = client.get("/hyperlink/upgrade").json()
        assert body["installation"]["package_version"] == hypernix.__version__

    def test_it_hands_back_commands(self, client):
        body = client.get("/hyperlink/upgrade").json()
        assert body["commands"]
        assert any(command["primary"] for command in body["commands"])

    def test_the_commands_name_this_interpreter(self, client):
        body = client.get("/hyperlink/upgrade").json()
        primary = next(c for c in body["commands"] if c["primary"])
        assert shlex.quote(sys.executable) in primary["command"]

    def test_it_runs_nothing(self, client):
        """A phone button that upgraded a running server would be a
        phone button that takes a machine down in the middle of somebody
        else's conversation. The endpoint is a GET for that reason."""
        assert client.post("/hyperlink/upgrade").status_code in (404, 405)
