"""Noodle over the T1 API — mostly its boundary.

Noodle writes files, edits them, runs commands and packs archives. Said
plainly that is arbitrary code execution as a service, so almost all of
this file is about what stops it being that, and only a little is about
the feature working.

Four things hold it in, and each has tests here: it is off by default,
every owner gets their own workspace and cannot leave it, running what
was written is a *second* switch, and an ordinary read token is not
enough.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pytest

from hypernix.interfaces.noodle.tools import TOOLS, ToolContext, run_tool


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def workspace(tmp_path) -> ToolContext:
    return ToolContext(root=tmp_path / "work", allow_execute=True)


class TestTheNewToolsExist:
    @pytest.mark.parametrize("name", ["create_file", "edit_file", "run_fish", "zip", "unzip"])
    def test_it_is_registered(self, name):
        assert name in TOOLS

    def test_the_shell_and_archive_tools_are_marked_mutating(self):
        """The swarm reports mutating tools differently from reads, and a
        command that is reported as a read is a change nobody saw."""
        for name in ("run_fish", "zip", "unzip"):
            assert TOOLS[name].mutating


class TestZipping:
    def test_it_packs_a_file(self, workspace):
        run_tool(workspace, "create_file", {"path": "a.txt", "content": "hello"})
        result = run_tool(workspace, "zip", {"paths": "a.txt", "output": "out.zip"})
        assert result.ok
        assert (workspace.root / "out.zip").exists()

    def test_it_packs_a_directory(self, workspace):
        run_tool(workspace, "create_file", {"path": "d/a.txt", "content": "a"})
        run_tool(workspace, "create_file", {"path": "d/b.txt", "content": "b"})
        result = run_tool(workspace, "zip", {"paths": "d", "output": "out.zip"})
        assert result.data["members"] == 2

    def test_the_archive_reads_back(self, workspace):
        run_tool(workspace, "create_file", {"path": "a.txt", "content": "contents"})
        run_tool(workspace, "zip", {"paths": "a.txt", "output": "out.zip"})
        with zipfile.ZipFile(workspace.root / "out.zip") as archive:
            assert archive.read("a.txt") == b"contents"

    def test_the_extension_is_added(self, workspace):
        run_tool(workspace, "create_file", {"path": "a.txt", "content": "x"})
        run_tool(workspace, "zip", {"paths": "a.txt", "output": "bundle"})
        assert (workspace.root / "bundle.zip").exists()

    def test_it_will_not_pack_itself(self, workspace):
        """An archive containing itself is either an error or an infinite
        one, depending on how the writer is implemented."""
        run_tool(workspace, "create_file", {"path": "out.zip", "content": "x"})
        assert not run_tool(
            workspace, "zip", {"paths": "out.zip", "output": "out.zip"}
        ).ok

    def test_nothing_to_zip_is_refused(self, workspace):
        assert not run_tool(workspace, "zip", {"paths": []}).ok

    def test_a_missing_path_is_refused(self, workspace):
        assert not run_tool(workspace, "zip", {"paths": "nope.txt"}).ok


class TestZippingCannotReachOut:
    """The one way a zip tool turns into an exfiltration primitive."""

    def test_an_absolute_path_is_refused(self, workspace):
        assert not run_tool(workspace, "zip", {"paths": "/etc/passwd"}).ok

    def test_a_traversal_is_refused(self, workspace):
        assert not run_tool(workspace, "zip", {"paths": "../../etc/passwd"}).ok

    def test_a_symlink_out_is_refused(self, workspace, tmp_path):
        """Planted by an earlier tool call, which is why containment is
        checked after resolving rather than before."""
        secret = tmp_path / "secret.txt"
        secret.write_text("password")
        (workspace.root / "link.txt").symlink_to(secret)
        assert not run_tool(workspace, "zip", {"paths": "link.txt"}).ok


class TestUnzipping:
    def test_it_round_trips(self, workspace):
        run_tool(workspace, "create_file", {"path": "a.txt", "content": "hello"})
        run_tool(workspace, "zip", {"paths": "a.txt", "output": "out.zip"})
        (workspace.root / "a.txt").unlink()
        result = run_tool(workspace, "unzip", {"path": "out.zip"})
        assert result.ok
        assert (workspace.root / "a.txt").read_text() == "hello"

    def test_a_zip_slip_is_refused(self, workspace, tmp_path):
        """An archive entry named `../../etc/cron.d/x` extracts exactly
        where it says unless something checks, and ZipFile.extractall
        sanitises less than people assume."""
        malicious = workspace.root / "evil.zip"
        malicious.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(malicious, "w") as archive:
            archive.writestr("../../escaped.txt", "pwned")

        assert not run_tool(workspace, "unzip", {"path": "evil.zip"}).ok
        assert not (tmp_path.parent / "escaped.txt").exists()

    def test_a_file_that_is_not_a_zip_is_refused(self, workspace):
        run_tool(workspace, "create_file", {"path": "a.txt", "content": "not a zip"})
        assert not run_tool(workspace, "unzip", {"path": "a.txt"}).ok


class TestFish:
    def test_it_is_gated_on_execution(self, tmp_path):
        """`fish -c` with model-written text runs whatever the model
        wrote. Same capability as execute_file, different clothes."""
        locked = ToolContext(root=tmp_path / "w", allow_execute=False)
        refused = run_tool(locked, "run_fish", {"command": "echo hi"})
        assert not refused.ok
        assert refused.code == "execute_disabled"

    def test_an_empty_command_is_refused(self, workspace):
        assert not run_tool(workspace, "run_fish", {"command": "   "}).ok

    def test_it_runs_or_says_fish_is_missing(self, workspace):
        """Fish is not installed everywhere, and "fish is not installed"
        is a better answer than a traceback or a silent bash fallback —
        a bash one-liner run as fish usually fails loudly, which is the
        point of asking for fish."""
        result = run_tool(workspace, "run_fish", {"command": "echo hello"})
        if result.code == "fish_missing":
            pytest.skip("fish is not installed on this machine")
        assert result.ok
        assert "hello" in result.output


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


def client(tmp_path: Path, monkeypatch, **env) -> TestClient:
    from hypernix.t1api.app import create_app

    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    monkeypatch.setenv("T1_TRUSTED_NETWORK_PARTIAL_ADMIN", "1")
    monkeypatch.setenv("T1_NOODLE_WORKSPACE_DIR", str(tmp_path / "noodle"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return TestClient(create_app(), client=("192.168.1.50", 5432))


class TestItIsOffByDefault:
    def test_every_route_404s(self, tmp_path, monkeypatch):
        """404 and not 403: a server that has not enabled this has no
        reason to tell a stranger that it could."""
        app = client(tmp_path, monkeypatch)
        assert app.get("/noodle/tools").status_code == 404
        assert app.get("/noodle/workspace").status_code == 404
        assert app.post(
            "/noodle/run", json={"tool": "read_file", "arguments": {"path": "x"}}
        ).status_code == 404

    def test_the_refusal_says_how_to_turn_it_on(self, tmp_path, monkeypatch):
        body = client(tmp_path, monkeypatch).get("/noodle/tools").json()
        assert "T1_NOODLE_ENABLED" in body["error"]["message"]


class TestWhenItIsOn:
    def test_the_tool_list_is_readable(self, tmp_path, monkeypatch):
        body = client(tmp_path, monkeypatch, T1_NOODLE_ENABLED="1").get(
            "/noodle/tools"
        ).json()
        assert "create_file" in body["names"]
        assert "zip" in body["names"]

    def test_execution_is_a_second_switch(self, tmp_path, monkeypatch):
        """Writing a script and running it are different capabilities,
        and the default is the first without the second."""
        body = client(tmp_path, monkeypatch, T1_NOODLE_ENABLED="1").get(
            "/noodle/tools"
        ).json()
        assert body["execute_enabled"] is False

    def test_a_file_can_be_created_and_read_back(self, tmp_path, monkeypatch):
        app = client(tmp_path, monkeypatch, T1_NOODLE_ENABLED="1")
        written = app.post("/noodle/run", json={
            "tool": "create_file",
            "arguments": {"path": "notes.txt", "content": "remember this"},
        }).json()
        assert written["ok"]
        read = app.post("/noodle/run", json={
            "tool": "read_file", "arguments": {"path": "notes.txt"},
        }).json()
        assert "remember this" in read["content"]

    def test_running_a_command_is_refused_without_the_switch(self, tmp_path, monkeypatch):
        app = client(tmp_path, monkeypatch, T1_NOODLE_ENABLED="1")
        body = app.post("/noodle/run", json={
            "tool": "run_fish", "arguments": {"command": "echo hi"},
        })
        # 422 is this API's status for a validation error throughout.
        assert body.status_code == 422
        assert "disabled" in body.text

    def test_the_workspace_lists_what_was_written(self, tmp_path, monkeypatch):
        app = client(tmp_path, monkeypatch, T1_NOODLE_ENABLED="1")
        app.post("/noodle/run", json={
            "tool": "create_file", "arguments": {"path": "a.txt", "content": "x"},
        })
        body = app.get("/noodle/workspace").json()
        assert body["count"] == 1
        assert body["files"][0]["path"] == "a.txt"

    def test_an_unknown_tool_is_a_400_naming_the_real_ones(self, tmp_path, monkeypatch):
        app = client(tmp_path, monkeypatch, T1_NOODLE_ENABLED="1")
        response = app.post("/noodle/run", json={"tool": "rm_rf", "arguments": {}})
        assert response.status_code == 422
        assert "create_file" in response.text

    def test_escaping_the_workspace_is_the_callers_fault_not_a_500(
        self, tmp_path, monkeypatch
    ):
        """"That path is outside the workspace" is a thing the caller
        did. Reporting it as a server fault sends them looking in the
        wrong place."""
        app = client(tmp_path, monkeypatch, T1_NOODLE_ENABLED="1")
        response = app.post("/noodle/run", json={
            "tool": "create_file",
            "arguments": {"path": "../../escaped.txt", "content": "x"},
        })
        assert response.status_code == 422
        assert not (tmp_path.parent / "escaped.txt").exists()


class TestWorkspacesAreSeparate:
    def test_the_workspace_is_under_the_configured_root(self, tmp_path, monkeypatch):
        app = client(tmp_path, monkeypatch, T1_NOODLE_ENABLED="1")
        workspace = app.get("/noodle/workspace").json()["workspace"]
        assert str(tmp_path / "noodle") in workspace

    def test_an_owner_id_with_a_slash_does_not_become_a_path(self):
        """A device token's owner is a key id, and key ids have prefixes.
        A slash in a directory name is a directory somewhere else."""
        from hypernix.t1api.routers.noodle import _safe_slug

        assert "/" not in _safe_slug("T1_abc/../../etc")
        assert ".." not in _safe_slug("../../etc")

    def test_an_empty_owner_still_gets_a_directory(self):
        from hypernix.t1api.routers.noodle import _safe_slug

        assert _safe_slug("") == "anonymous"

    def test_two_owners_get_different_directories(self):
        from hypernix.t1api.routers.noodle import _safe_slug

        assert _safe_slug("alice") != _safe_slug("bob")


class TestItNeedsMoreThanARead:
    def test_a_public_origin_is_refused(self, tmp_path, monkeypatch):
        from hypernix.t1api.app import create_app

        monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
        monkeypatch.setenv("T1_NOODLE_ENABLED", "1")
        monkeypatch.setenv("T1_NOODLE_WORKSPACE_DIR", str(tmp_path / "noodle"))
        public = TestClient(create_app(), client=("8.8.8.8", 5432))
        assert public.get("/noodle/tools").status_code == 401

    def test_a_read_only_caller_is_refused(self, tmp_path, monkeypatch):
        from hypernix.t1api.app import create_app

        monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
        monkeypatch.setenv("T1_TRUSTED_NETWORK_PARTIAL_ADMIN", "0")
        monkeypatch.setenv("T1_NOODLE_ENABLED", "1")
        monkeypatch.setenv("T1_NOODLE_WORKSPACE_DIR", str(tmp_path / "noodle"))
        app = TestClient(create_app(), client=("192.168.1.50", 5432))
        assert app.get("/noodle/tools").status_code == 403
