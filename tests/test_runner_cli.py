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


class TestStart:
    """``built-in-runner start`` — the verb for a machine with one model.

    ``load`` makes you name a model. That is right when there are forty
    of them and wrong on the commonest machine there is: one GGUF on
    disk and nothing serving it, where the name is not a decision the
    person has to make and asking for it is just a lookup they have to
    do first.

    So ``start`` names it for them when there is exactly one, and
    refuses to guess when there is not — because guessing wrong is not a
    typo, it is a minute of loading and the VRAM of a model nobody
    asked for.
    """

    def test_it_loads_the_only_loadable_model(self, calls):
        calls.replies.append((200, {"loaded": False}))
        calls.replies.append((200, {"models": [
            {"model_id": "qwen3-8b", "path": "/models/qwen3-8b.gguf"},
        ]}))
        calls.replies.append((200, {"loaded": True, "model": {"model_id": "qwen3-8b"}}))

        assert runner_cli.main(["start"]) == 0

        assert calls.recorded[0][1].endswith("/runner/status")
        assert calls.recorded[1][1].endswith("/hyperlink/models")
        method, url, _key, payload = calls.recorded[2]
        assert method == "POST"
        assert url.endswith("/runner/load")
        assert payload["model_id"] == "qwen3-8b"

    def test_a_model_with_no_file_is_not_a_candidate(self, calls):
        """An LM Studio entry or a registry row with no GGUF cannot be
        loaded, so it must not make the choice ambiguous either."""
        calls.replies.append((200, {"loaded": False}))
        calls.replies.append((200, {"models": [
            {"model_id": "remote-only", "path": ""},
            {"model_id": "on-disk", "path": "/models/on-disk.gguf"},
        ]}))
        calls.replies.append((200, {"loaded": True, "model": {"model_id": "on-disk"}}))

        assert runner_cli.main(["start"]) == 0
        assert calls.recorded[2][3]["model_id"] == "on-disk"

    def test_two_models_stops_and_lists_them(self, calls, capsys):
        calls.replies.append((200, {"loaded": False}))
        calls.replies.append((200, {"models": [
            {"model_id": "qwen3-8b", "path": "/a.gguf", "size_bytes": 4_900_000_000},
            {"model_id": "llama-70b", "path": "/b.gguf"},
        ]}))

        assert runner_cli.main(["start"]) == 1

        # Nothing was loaded — the refusal is the whole point.
        assert not any(u.endswith("/runner/load") for _m, u, _k, _p in calls.recorded)
        err = capsys.readouterr().err
        assert "qwen3-8b" in err and "llama-70b" in err

    def test_no_models_says_what_to_do_about_it(self, calls, capsys):
        calls.replies.append((200, {"loaded": False}))
        calls.replies.append((200, {"models": []}))

        assert runner_cli.main(["start"]) == 1
        err = capsys.readouterr().err
        assert "hypernix-t1 index" in err

    def test_no_models_names_which_source_was_silent(self, calls, capsys):
        """"No models" and "LM Studio is not running" produce the same
        empty list and want completely different things doing about
        them. That is why the catalogue reports its sources at all, and
        printing the list without them throws the answer away."""
        calls.replies.append((200, {"loaded": False}))
        calls.replies.append((200, {"models": [], "sources": [
            {"name": "registry", "available": True, "count": 0, "detail": ""},
            {"name": "lmstudio", "available": False, "count": 0,
             "detail": "Connection refused on 127.0.0.1:1234"},
        ]}))

        assert runner_cli.main(["start"]) == 1
        err = capsys.readouterr().err
        assert "lmstudio: unavailable" in err
        assert "Connection refused on 127.0.0.1:1234" in err
        assert "registry: 0" in err

    def test_naming_a_model_skips_the_catalogue(self, calls):
        calls.replies.append((200, {"loaded": False}))
        calls.replies.append((200, {"loaded": True, "model": {"model_id": "m"}}))

        assert runner_cli.main(["start", "m"]) == 0
        assert not any(
            u.endswith("/hyperlink/models") for _m, u, _k, _p in calls.recorded
        )
        assert calls.recorded[1][3]["model_id"] == "m"

    def test_starting_what_is_already_running_changes_nothing(self, calls, capsys):
        """Running it twice is a thing people do when they are not sure
        the first one took. Charging them a reload for that would evict
        the conversation the first one was serving."""
        calls.replies.append((200, {
            "loaded": True, "model": {"model_id": "qwen3-8b", "base_url": "http://x"},
        }))

        assert runner_cli.main(["start"]) == 0
        assert len(calls.recorded) == 1
        assert "Already running" in capsys.readouterr().out

    def test_restart_reloads_it_anyway(self, calls):
        calls.replies.append((200, {"loaded": True, "model": {"model_id": "qwen3-8b"}}))
        calls.replies.append((200, {"loaded": True, "model": {"model_id": "qwen3-8b"}}))

        assert runner_cli.main(["start", "--restart"]) == 0
        assert calls.recorded[1][1].endswith("/runner/load")
        assert calls.recorded[1][3]["model_id"] == "qwen3-8b"

    def test_naming_a_different_model_switches(self, calls, capsys):
        calls.replies.append((200, {"loaded": True, "model": {"model_id": "old"}}))
        calls.replies.append((200, {"loaded": True, "model": {"model_id": "new"}}))

        assert runner_cli.main(["start", "new"]) == 0
        assert calls.recorded[1][3]["model_id"] == "new"
        assert "Replacing old" in capsys.readouterr().err

    def test_it_sends_the_tuning_fields(self, calls):
        calls.replies.append((200, {"loaded": False}))
        calls.replies.append((200, {"loaded": True, "model": {"model_id": "m"}}))

        runner_cli.main([
            "start", "m", "--gpu-layers", "24", "--total-layers", "33",
            "--context-length", "8192", "--backend", "cuda",
        ])
        assert calls.recorded[1][3] == {
            "model_id": "m", "gpu_layers": 24, "backend": "cuda",
            "context_length": 8192, "total_layers": 33,
        }

    def test_a_refusal_reading_the_catalogue_is_printed(self, calls, capsys):
        calls.replies.append((200, {"loaded": False}))
        calls.replies.append((403, {"error": {"message": "HyperLink is off."}}))

        assert runner_cli.main(["start"]) == 1
        assert "HyperLink is off." in capsys.readouterr().err

    def test_a_refusal_loading_is_printed(self, calls, capsys):
        calls.replies.append((200, {"loaded": False}))
        calls.replies.append((400, {"error": {
            "message": "No llama.cpp build on this machine.",
            "details": {"remedy": "Build it, or use hnx-cpu."},
        }}))

        assert runner_cli.main(["start", "m"]) == 1
        err = capsys.readouterr().err
        assert "No llama.cpp build" in err
        assert "Build it, or use hnx-cpu." in err


class TestStopIsUnload:
    """``start``'s opposite has to be typeable. `unload` is the verb the
    API uses and stays, but nobody who just typed `start` reaches for
    it, and a start with no stop is a half-finished command."""

    def test_stop_posts_unload(self, calls, capsys):
        calls.replies.append((200, {"loaded": False, "was_running": True}))
        assert runner_cli.main(["stop"]) == 0
        assert calls.recorded[0][1].endswith("/runner/unload")
        assert "Unloaded." in capsys.readouterr().out

    def test_stopping_nothing_is_a_success(self, calls, capsys):
        calls.replies.append((200, {"loaded": False, "was_running": False}))
        assert runner_cli.main(["stop"]) == 0
        assert "Nothing was running." in capsys.readouterr().out


class TestTheWrapperReachesIt:
    """`hypernix-t1 built-in-runner ...` — the spelling the request
    asked for, and the one the help has to describe."""

    @staticmethod
    def _wrapper() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parents[1] / "bin" / "hypernix-t1").read_text()

    def test_both_spellings_dispatch(self):
        text = self._wrapper()
        assert "built-in-runner|builtin-runner)" in text
        assert "cmd_runner built-in-runner" in text
        assert 'runner)            cmd_runner runner "$@" ;;' in text

    def test_the_help_lists_it(self):
        assert "built-in-runner [start [MODEL]" in self._wrapper()

    def test_the_spelling_reaches_the_help_text(self):
        """Someone who typed `built-in-runner --help` must not be told
        to type `runner`."""
        assert 'HNX_RUNNER_PROG="hypernix-t1 $spelling"' in self._wrapper()

        import os

        os.environ["HNX_RUNNER_PROG"] = "hypernix-t1 built-in-runner"
        try:
            assert runner_cli.build_parser().prog == "hypernix-t1 built-in-runner"
        finally:
            del os.environ["HNX_RUNNER_PROG"]


class TestStartAgainstARealServer:
    """`start` invented a second path — it asks the catalogue which
    model to load — and the mocks above cannot tell one that exists from
    one that does not. A typo there is a 404 that reads to the person at
    the keyboard exactly like "this server has no models"."""

    def test_start_reaches_real_paths(self, tmp_path, monkeypatch, capsys):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        clear_t1_config(monkeypatch)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t.sqlite3"))
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("T1_HF_DOWNLOAD_DIR", str(tmp_path / "models"))
        monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
        monkeypatch.setenv("T1_TRUSTED_NETWORK_PARTIAL_ADMIN", "1")

        from hypernix.t1api.app import create_app

        client = TestClient(create_app(), client=("192.168.1.9", 5000))

        def through_the_app(method, url, key, payload=None):
            path = url[url.index("/", len("http://")):]
            response = (
                client.get(path) if method == "GET"
                else client.post(path, json=payload)
            )
            return response.status_code, response.json()

        monkeypatch.setattr(runner_cli, "_request", through_the_app)

        # No models on disk, so it stops — but it stops having reached
        # /runner/status and /hyperlink/models, not on a 404.
        assert runner_cli.main(["start"]) == 1
        err = capsys.readouterr().err
        assert "hypernix-t1 index" in err
        assert "Refused" not in err
