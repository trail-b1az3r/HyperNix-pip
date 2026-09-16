"""``hypernix-t1 runner`` — the command line over ``/runner/*``.

The runner landed as an API and nothing on the machine could drive it,
so switching the served model from the keyboard of the machine running
it meant writing curl by hand with a key pasted in.

Two decisions this file is really about.

**It is a client, not a second runner.** Loading the model in this
process would start a *second* llama.cpp, which takes the VRAM the
server's copy is using — and the failure then lands on the one that was
working rather than the one being started. So it talks HTTP to the
server on the same machine, exactly as HyperLink does, and everybody
reads one answer to "what is loaded".

**A refusal is information.** A 403 from ``/runner/load`` names the
three ways to be allowed; a 404 lists what this server *can* load. Both
are more useful than the traceback that a `raise_for_status` would
produce, so they are printed rather than raised.
"""
from __future__ import annotations

import json

import pytest
from conftest import clear_t1_config

from hypernix.t1api import runner_cli


@pytest.fixture
def calls(monkeypatch):
    """Record requests and reply with whatever the test queued."""
    recorded: list[tuple[str, str, str, dict | None]] = []
    replies: list[tuple[int, dict]] = []

    def fake(method, url, key, payload=None):
        recorded.append((method, url, key, payload))
        return replies.pop(0) if replies else (200, {})

    monkeypatch.setattr(runner_cli, "_request", fake)
    fake.recorded = recorded  # type: ignore[attr-defined]
    fake.replies = replies  # type: ignore[attr-defined]
    return fake


class TestItTalksToTheServer:
    def test_status_is_a_get(self, calls, capsys):
        calls.replies.append((200, {"loaded": False, "backends": ["auto", "cpu"]}))
        assert runner_cli.main(["status"]) == 0
        method, url, _key, payload = calls.recorded[0]
        assert method == "GET"
        assert url.endswith("/runner/status")
        assert payload is None

    def test_no_command_means_status(self, calls):
        """The question people open this for is "what is running"."""
        calls.replies.append((200, {"loaded": False}))
        assert runner_cli.main([]) == 0
        assert calls.recorded[0][1].endswith("/runner/status")

    def test_plan_does_not_load(self, calls):
        """The whole point of a separate verb: loading evicts whatever
        people are currently talking to."""
        calls.replies.append((200, {"model_id": "m", "path": "/tmp/m.gguf"}))
        runner_cli.main(["plan", "m"])
        assert calls.recorded[0][1].endswith("/runner/plan")

    def test_load_sends_every_tuning_field(self, calls):
        calls.replies.append((200, {"loaded": True, "model": {"model_id": "m"}}))
        runner_cli.main([
            "load", "m", "--gpu-layers", "24", "--total-layers", "33",
            "--context-length", "8192", "--backend", "cuda",
        ])
        _method, url, _key, payload = calls.recorded[0]
        assert url.endswith("/runner/load")
        assert payload == {
            "model_id": "m", "gpu_layers": 24, "backend": "cuda",
            "context_length": 8192, "total_layers": 33,
        }

    def test_unspecified_tuning_is_none_not_zero(self, calls):
        """`gpu_layers: 0` means "no layers on the GPU" and `None` means
        "work it out". Sending 0 for "I did not say" would put every
        model on the CPU."""
        calls.replies.append((200, {}))
        runner_cli.main(["load", "m"])
        payload = calls.recorded[0][3]
        assert payload["gpu_layers"] is None
        assert payload["total_layers"] is None
        assert payload["context_length"] is None

    def test_unload_is_a_post(self, calls):
        calls.replies.append((200, {"was_running": True}))
        assert runner_cli.main(["unload"]) == 0
        method, url, _key, _payload = calls.recorded[0]
        assert method == "POST"
        assert url.endswith("/runner/unload")


class TestTheKey:
    def test_an_explicit_key_wins(self, calls, monkeypatch):
        monkeypatch.setenv("T1_ADMIN_KEY", "from-env")
        calls.replies.append((200, {}))
        runner_cli.main(["--key", "explicit", "status"])
        assert calls.recorded[0][2] == "explicit"

    def test_the_environment_is_next(self, calls, monkeypatch):
        monkeypatch.setenv("T1_ADMIN_KEY", "from-env")
        calls.replies.append((200, {}))
        runner_cli.main(["status"])
        assert calls.recorded[0][2] == "from-env"

    def test_the_servers_own_env_file_is_the_fallback(self, tmp_path, monkeypatch):
        """The common case: somebody at the keyboard of the machine that
        is running the thing. Making them paste a key that is already on
        disk is the friction that gets solved with a shell alias storing
        it somewhere worse."""
        monkeypatch.delenv("T1_ADMIN_KEY", raising=False)
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        (tmp_path / ".env").write_text(
            "# a comment\nT1_BIND_HOST=127.0.0.1\nT1_ADMIN_KEY=from-file\n",
            encoding="utf-8",
        )
        assert runner_cli._resolve_key("") == "from-file"

    def test_quotes_around_the_value_are_stripped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("T1_ADMIN_KEY", raising=False)
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        (tmp_path / ".env").write_text('T1_ADMIN_KEY="quoted"\n', encoding="utf-8")
        assert runner_cli._resolve_key("") == "quoted"

    def test_no_key_anywhere_is_not_an_error(self, tmp_path, monkeypatch):
        """A trusted-network server with partial admin accepts this with
        no credential at all. Refusing here would refuse a request the
        server would have allowed."""
        monkeypatch.delenv("T1_ADMIN_KEY", raising=False)
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        assert runner_cli._resolve_key("") == ""


class TestRefusalsArePrinted:
    def test_a_403_names_the_remedy(self, calls, capsys):
        calls.replies.append((403, {"error": {
            "message": "Changing the loaded model affects everybody.",
            "details": {"remedy": "Set T1_RUNNER_SWITCH_PERM."},
        }}))
        assert runner_cli.main(["load", "m"]) == 1
        printed = capsys.readouterr().err
        assert "Changing the loaded model" in printed
        assert "T1_RUNNER_SWITCH_PERM" in printed

    def test_a_404_lists_what_can_be_loaded(self, calls, capsys):
        """"No model 'qwen'" is half an answer. The other half is what
        this server does have."""
        calls.replies.append((404, {"error": {
            "message": "No model 'qwen' on this server.",
            "details": {"loadable": ["qwen3-8b", "llama-70b"]},
        }}))
        assert runner_cli.main(["load", "qwen"]) == 1
        printed = capsys.readouterr().err
        assert "qwen3-8b" in printed
        assert "llama-70b" in printed

    def test_a_refusal_exits_nonzero(self, calls):
        calls.replies.append((400, {"error": {"message": "does not fit"}}))
        assert runner_cli.main(["load", "m"]) == 1

    def test_an_unparseable_body_still_prints_something(self, calls, capsys):
        calls.replies.append((500, {"nothing": "recognisable"}))
        assert runner_cli.main(["status"]) == 1
        assert capsys.readouterr().err.strip()


class TestWhatItPrints:
    def test_nothing_loaded_says_so_and_lists_backends(self, calls, capsys):
        calls.replies.append((200, {"loaded": False, "backends": ["auto", "cuda"]}))
        runner_cli.main(["status"])
        printed = capsys.readouterr().out
        assert "Nothing loaded" in printed
        assert "cuda" in printed

    def test_a_loaded_model_reports_its_placement(self, calls, capsys):
        calls.replies.append((200, {
            "loaded": True,
            "model": {
                "model_id": "llama-70b", "base_url": "http://127.0.0.1:8081",
                "context_length": 8192,
                "placement": {
                    "gpu_layers": 41, "total_layers": 81,
                    "fully_offloaded": False, "reason": "24 GB of VRAM",
                },
            },
        }))
        runner_cli.main(["status"])
        printed = capsys.readouterr().out
        assert "llama-70b" in printed
        assert "41 of 81 on the GPU, 40 on the CPU" in printed
        assert "24 GB of VRAM" in printed

    def test_a_fully_offloaded_model_says_all_of_them(self, calls, capsys):
        calls.replies.append((200, {
            "loaded": True,
            "model": {"model_id": "m", "placement": {
                "gpu_layers": 33, "total_layers": 33, "fully_offloaded": True,
            }},
        }))
        runner_cli.main(["status"])
        assert "all 33 layers on the GPU" in capsys.readouterr().out

    def test_a_cpu_only_model_says_so(self, calls, capsys):
        calls.replies.append((200, {
            "loaded": True,
            "model": {"model_id": "m", "placement": {
                "gpu_layers": 0, "total_layers": 33, "fully_offloaded": False,
            }},
        }))
        runner_cli.main(["status"])
        assert "all 33 layers on the CPU" in capsys.readouterr().out

    def test_plan_says_nothing_changed(self, calls, capsys):
        """Somebody who has just watched a wall of placement output
        needs to know their server is untouched."""
        calls.replies.append((200, {
            "model_id": "m", "path": "/models/m.gguf",
            "placement": {"gpu_layers": 10, "total_layers": 20},
        }))
        runner_cli.main(["plan", "m"])
        assert "Nothing has changed" in capsys.readouterr().out

    def test_unloading_nothing_is_not_reported_as_a_success(self, calls, capsys):
        """It *is* a success — the exit code says so — but telling
        somebody "Unloaded." when nothing was running is a lie about
        what just happened to a shared machine."""
        calls.replies.append((200, {"was_running": False}))
        assert runner_cli.main(["unload"]) == 0
        assert "Nothing was running" in capsys.readouterr().out

    def test_json_is_available_for_scripts(self, calls, capsys):
        calls.replies.append((200, {"loaded": True, "model": {"model_id": "m"}}))
        runner_cli.main(["--json", "status"])
        assert json.loads(capsys.readouterr().out)["model"]["model_id"] == "m"


class TestAgainstARealServer:
    """The parts a mock cannot check: that the paths exist and the
    payload is the shape the endpoint validates."""

    def test_status_and_a_missing_model_round_trip(self, tmp_path, monkeypatch):
        fastapi = pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        # Not every T1_* variable — the storage redirects in
        # conftest.py stay, or "a server with no configuration" quietly
        # becomes "a server writing to the real ~/.hypernix".
        clear_t1_config(monkeypatch)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
        monkeypatch.setenv("T1_TRUSTED_NETWORK_PARTIAL_ADMIN", "1")

        from hypernix.t1api.app import create_app

        client = TestClient(create_app(), client=("192.168.1.9", 5000))
        assert fastapi  # the importorskip result, used

        status = client.get("/runner/status")
        assert status.status_code == 200
        assert status.json()["loaded"] is False

        missing = client.post("/runner/load", json={
            "model_id": "no-such-model", "gpu_layers": None,
            "backend": "auto", "context_length": None, "total_layers": None,
        })
        assert missing.status_code == 404
        assert "no-such-model" in missing.json()["error"]["message"]
