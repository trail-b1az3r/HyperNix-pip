"""Moving LM Studio's model onto the HyperNix runner (0.72.6.post1).

HyperLink's "Move to the HyperNix runner" button: unload the model LM
Studio is serving, and load the same file on this server's runner.
Admins and devices on the tailnet (or at the machine) only, because it
ejects another application's model.
"""
from __future__ import annotations

import json
import re
import subprocess
import time

import pytest
from conftest import clear_t1_config

from hypernix.hyperlink import handover
from hypernix.hyperlink.handover import HandoverError, find_lmstudio_file


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    clear_t1_config(monkeypatch)


def _gguf(root, relative, size=16):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\0" * size)
    return path


class TestFindingTheFile:
    def test_by_folder_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(handover, "_lms", lambda: None)
        want = _gguf(tmp_path, "lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf")
        _gguf(tmp_path, "lmstudio-community/gemma-3-4b-it-GGUF/mmproj-gemma-3-4b-it-F16.gguf")
        _gguf(tmp_path, "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
        found = find_lmstudio_file("google/gemma-3-4b-it", dirs=[tmp_path])
        assert found.path == want and found.found_by == "scan"

    def test_two_candidates_is_not_a_guess(self, tmp_path, monkeypatch):
        monkeypatch.setattr(handover, "_lms", lambda: None)
        _gguf(tmp_path, "a/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf")
        _gguf(tmp_path, "a/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q8_0.gguf")
        with pytest.raises(HandoverError) as exc:
            find_lmstudio_file("gemma-3-4b-it", dirs=[tmp_path])
        assert exc.value.step == "find" and "lms" in exc.value.remedy

    def test_nothing_says_where_it_looked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(handover, "_lms", lambda: None)
        with pytest.raises(HandoverError, match=re.escape(str(tmp_path))):
            find_lmstudio_file("nothing-like-it", dirs=[tmp_path])

    def test_lms_says_exactly_which_file(self, tmp_path, monkeypatch):
        want = _gguf(tmp_path, "lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
        _gguf(tmp_path, "lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf")
        monkeypatch.setattr(handover, "_lms", lambda: "/usr/bin/lms")
        listing = [{"modelKey": "qwen3-8b",
                    "path": "lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"}]
        monkeypatch.setattr(handover.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
            a[0], 0, stdout=json.dumps(listing), stderr=""))
        found = find_lmstudio_file("qwen3-8b", dirs=[tmp_path])
        assert found.path == want and found.found_by == "lms"


class Bridge:
    def __init__(self, loaded=("gemma-3-4b-it",), rest=True):
        self.loaded = list(loaded)
        self.rest = rest
        self.calls = []

    def _request(self, method, path, body=None, timeout=None):
        self.calls.append((method, path, body))
        if not self.rest:
            raise RuntimeError("404")
        if path.endswith("/unload"):
            self.loaded.remove(body["instance_id"])
        elif path.endswith("/load"):
            self.loaded.append(body["model"])
        return {}

    def loaded_models(self):
        class Model:
            def __init__(self, model_id):
                self.model_id = model_id
        return [Model(m) for m in self.loaded]


class TestUnloading:
    def test_through_the_api(self):
        bridge = Bridge()
        assert handover.unload_from_lmstudio(bridge, "gemma-3-4b-it") == "api"
        assert bridge.loaded == []

    def test_through_lms_when_the_api_will_not(self, monkeypatch):
        monkeypatch.setattr(handover, "_lms", lambda: "/usr/bin/lms")
        ran = []
        monkeypatch.setattr(handover.subprocess, "run", lambda argv, **k: (
            ran.append(argv), subprocess.CompletedProcess(argv, 0, "", ""))[1])
        assert handover.unload_from_lmstudio(Bridge(rest=False), "gemma-3-4b-it") == "lms"
        assert ran == [["/usr/bin/lms", "unload", "gemma-3-4b-it"]]

    def test_neither_is_a_refusal_with_a_remedy(self, monkeypatch):
        monkeypatch.setattr(handover, "_lms", lambda: None)
        with pytest.raises(HandoverError) as exc:
            handover.unload_from_lmstudio(Bridge(rest=False), "gemma-3-4b-it")
        assert exc.value.step == "unload" and "Eject" in exc.value.remedy


pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


class Placement:
    reason = "all 33 layers on the GPU"


class Loaded:
    def __init__(self, model_id, path):
        self.model_id, self.path = model_id, path
        self.placement = Placement()

    def to_dict(self):
        return {"model_id": self.model_id, "path": str(self.path), "port": 8781,
                "started_at": time.time(), "placement": {}}


class FakeRunner:
    base_url = "http://127.0.0.1:8781"

    def __init__(self, fail=False):
        self.fail = fail
        self.current = None
        self.loads = []

    def load(self, path, **kwargs):
        from hypernix.hyperlink.managed import ManagedError

        self.loads.append((path, kwargs))
        if self.fail:
            raise ManagedError("out of memory")
        self.current = Loaded(kwargs["model_id"], path)
        return self.current


@pytest.fixture
def server(tmp_path, monkeypatch):
    from hypernix.t1api.app import create_app
    from hypernix.t1api.routers import runner as runner_router

    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1.sqlite3"))
    bridge = Bridge()
    gguf = _gguf(tmp_path, "lms/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf")
    monkeypatch.setattr(runner_router, "_lmstudio", lambda config: bridge)
    monkeypatch.setattr(handover, "lmstudio_models_dirs", lambda: [tmp_path / "lms"])
    monkeypatch.setattr(handover, "_lms", lambda: None)

    def client(address, *, fail=False):
        app = create_app()
        app.state.t1_runner = FakeRunner(fail=fail)
        return TestClient(app, client=(address, 5000)), app.state.t1_runner

    return client, bridge, gguf


class TestWhoMay:
    def test_a_lan_device_that_is_not_admin_may_not(self, server):
        client, bridge, _ = server
        c, runner = client("192.168.1.20")
        assert c.get("/runner/adopt").json()["allowed"] is False
        response = c.post("/runner/adopt", json={})
        assert response.status_code == 403 and "tailnet" in response.text
        assert bridge.loaded == ["gemma-3-4b-it"] and runner.loads == []

    def test_the_machine_itself_may(self, server):
        client, _bridge, _ = server
        c, _runner = client("127.0.0.1")
        preview = c.get("/runner/adopt").json()
        assert preview["allowed"] and preview["available"]
        assert preview["lmstudio_loaded"] == ["gemma-3-4b-it"]


class TestTheMove:
    def test_unloaded_there_and_loaded_here(self, server):
        client, bridge, gguf = server
        c, runner = client("127.0.0.1")
        response = c.post("/runner/adopt", json={"backend": "cuda"})
        assert response.status_code == 200, response.text
        assert response.json()["model"]["model_id"] == "gemma-3-4b-it"
        assert bridge.loaded == []
        assert runner.loads[0][0] == gguf and runner.loads[0][1]["backend"] == "cuda"
        # Unloaded before the load, so the VRAM is never asked for twice.
        assert bridge.calls[0][1] == "/api/v1/models/unload"

    def test_a_failed_load_puts_it_back(self, server):
        client, bridge, _ = server
        c, _runner = client("127.0.0.1", fail=True)
        response = c.post("/runner/adopt", json={})
        assert response.status_code == 422
        details = response.json()["error"]["details"]
        assert details["step"] == "load" and details["restored_in_lmstudio"] is True
        assert bridge.loaded == ["gemma-3-4b-it"]

    def test_a_model_lm_studio_does_not_have(self, server):
        client, _bridge, _ = server
        c, _runner = client("127.0.0.1")
        assert c.post("/runner/adopt", json={"model_id": "nope"}).status_code == 404
