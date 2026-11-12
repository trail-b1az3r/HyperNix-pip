"""A Strife the user built, registered with ``hnx strife build`` and served by T1."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from hypernix.strife import builds
from hypernix.strife.cli import cli_main
from hypernix.strife.errors import StrifeError
from hypernix.strife.fixtures import make_model

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERNIX_HOME", str(tmp_path / "hx"))
    monkeypatch.delenv("HNX_STRIFE_BUILD", raising=False)
    return tmp_path


@pytest.fixture
def mine(home):
    """This checkout, registered as if the user had built it."""
    return builds.register("mine", source=REPO, make_default=True)


def test_register_selftests_in_the_builds_own_interpreter(mine):
    assert mine.source == str(REPO / "src")
    assert mine.selftest["ok"] and mine.selftest["tokens"] > 0
    # The selftest imported the checkout's own server, not whatever is installed.
    assert mine.selftest["module"].startswith(str(REPO / "src"))
    assert [b.name for b in builds.list_builds()] == ["mine"]
    assert builds.default_build().name == "mine"
    mode = builds.registry_path().stat().st_mode & 0o777
    assert mode == 0o600


def test_launch_puts_the_source_first(mine):
    env = builds.launch_env(mine, {"PYTHONPATH": "/elsewhere"})
    assert env["PYTHONPATH"].split(os.pathsep) == [str(REPO / "src"), "/elsewhere"]
    assert builds.launch_argv(mine)[1:] == ["-m", "hypernix.strife.server"]
    assert builds.launch_argv(None)[0] == sys.executable


def test_a_failing_build_is_not_recorded(home, tmp_path):
    with pytest.raises(StrifeError) as missing:
        builds.register("nope", source=tmp_path)
    assert missing.value.code == "strife.build.no_source"
    with pytest.raises(StrifeError) as no_python:
        builds.register("nope", python=str(tmp_path / "no-such-python"))
    assert no_python.value.code == "strife.build.no_python"
    assert builds.list_builds() == []


@pytest.mark.parametrize("name", ["", "Has Space", "../up", "builtin", "x" * 60])
def test_names_are_plain(home, name):
    with pytest.raises(StrifeError):
        builds.register(name, source=REPO)


def test_unknown_names_and_env_choice(mine, monkeypatch):
    with pytest.raises(StrifeError, match="no Strife build"):
        builds.get_build("other")
    assert builds.get_build("builtin") is None
    monkeypatch.setenv("HNX_STRIFE_BUILD", "builtin")
    assert builds.default_build() is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_a_registry_others_can_write_is_refused(mine):
    builds.registry_path().chmod(0o666)
    with pytest.raises(StrifeError) as unsafe:
        builds.list_builds()
    assert unsafe.value.code == "strife.build.unsafe_registry"


def test_cli_list_and_remove(mine, capsys):
    assert cli_main(["build", "--list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["default"] == "mine" and listed["builds"][0]["selftest_ok"]
    assert cli_main(["build", "--remove", "mine"]) == 0
    assert builds.list_builds() == []


def test_the_runner_starts_the_named_build(mine, tmp_path, monkeypatch):
    from hypernix.hyperlink import managed

    model = make_model(tmp_path / "m.gguf", arch="qwen2", seed=3)
    seen = {}

    def fake_start(argv, **kwargs):
        seen["argv"], seen["env"] = argv, kwargs.get("env")
        raise OSError("not starting anything in a unit test")

    monkeypatch.setattr(managed, "_start_server", fake_start)
    runner = managed.ManagedRunner(port=1)
    with pytest.raises(managed.ManagedError, match="Could not start Strife"):
        runner.load(model["path"], backend="strife", strife_build="mine")
    assert seen["argv"][:3] == [mine.python, "-m", "hypernix.strife.server"]
    assert seen["env"]["PYTHONPATH"].split(os.pathsep)[0] == mine.source
    with pytest.raises(managed.ManagedError, match="no Strife build"):
        runner.load(model["path"], backend="strife", strife_build="other")


class TestOnT1:
    @pytest.fixture
    def t1(self, tmp_path, monkeypatch, mine):
        pytest.importorskip("fastapi")
        from conftest import clear_t1_config
        from fastapi.testclient import TestClient

        models = tmp_path / "models"
        models.mkdir()
        make_model(models / "tiny-qwen.gguf", arch="qwen2", seed=4)
        clear_t1_config(monkeypatch)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
        monkeypatch.setenv("T1_TRUSTED_NETWORK_PARTIAL_ADMIN", "1")
        monkeypatch.setenv("T1_LMSTUDIO_ENABLED", "0")
        monkeypatch.setenv("T1_HF_DOWNLOAD_DIR", str(models))
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        monkeypatch.setenv("T1_RUNNER_PORT", str(port))
        monkeypatch.setenv("HOME", str(tmp_path))
        from hypernix.t1api.app import create_app

        app = create_app()
        client = TestClient(app, client=("192.168.1.9", 5000))
        yield app, client
        app.state.t1_runner.unload()

    def test_status_lists_builds_and_load_serves_with_one(self, t1):
        _app, client = t1
        status = client.get("/runner/status").json()
        names = {b["name"]: b for b in status["strife_builds"]}
        assert names["mine"]["default"] and not names["builtin"]["default"]
        loaded = client.post("/runner/load", json={"model_id": "tiny-qwen", "backend": "strife",
                                                   "strife_build": "mine", "context_length": 256})
        assert loaded.status_code == 200, loaded.text
        assert "your build 'mine'" in loaded.json()["model"]["placement"]["reason"]

    def test_names_only_and_strife_only(self, t1):
        _app, client = t1
        bad = client.post("/runner/load", json={"model_id": "tiny-qwen", "backend": "strife",
                                                "strife_build": "/usr/bin/python"})
        assert bad.status_code in (400, 422)
        wrong = client.post("/runner/load", json={"model_id": "tiny-qwen", "backend": "cpu",
                                                  "strife_build": "mine"})
        assert wrong.status_code in (400, 422) and "strife" in wrong.text
        unknown = client.post("/runner/load", json={"model_id": "tiny-qwen", "backend": "strife",
                                                    "strife_build": "ghost"})
        assert unknown.status_code in (400, 422) and "no Strife build" in unknown.text
