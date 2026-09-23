"""Beta 4: hyped-pro against a real HyperNix T1 API server.

The interesting behaviour is the division of labour — the server decides
*which* model this key may use, the client runs it and reports the tokens
back. So these tests pin: that the client never substitutes its own model
choice, that an unmappable server model_id fails loudly instead of quietly
running something else, and that a failed usage report doesn't cost the
person the reply they already waited for.
"""
from __future__ import annotations

import pytest

from hypernix.interfaces import hyped_pro_core as core


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point every config read at a scratch file, and clear the env keys."""
    import hypernix.system.config as config
    import hypernix.waiter.local_config as waiter_config

    monkeypatch.setattr(config, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "_CONFIG_FILE", tmp_path / "config.json")
    # The other two places a T1 address is recorded, so the person
    # running the suite does not have their own server found.
    monkeypatch.setattr(waiter_config, "_DEFAULT_CONFIG_PATH", tmp_path / "waiter" / "waiter.config.jsonl")
    monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path / "t1api"))
    for var in ("HNX_T1_API_KEY", "HNX_T1_API_URL"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class TestCatalog:
    def test_t1api_provider_is_registered(self):
        provider = core.PROVIDERS["t1api"]
        assert provider.kind == "t1api"
        assert provider.auth_env_var == "HNX_T1_API_KEY"

    def test_t1api_is_distinct_from_the_local_gatekeeper_vendor(self):
        """Two different things that both say "T1": the local Gatekeeper and
        a real API server. Sharing an env var would silently cross them."""
        assert core.PROVIDERS["t1"].auth_env_var != core.PROVIDERS["t1api"].auth_env_var

    def test_routed_model_names_no_weights(self):
        """`t1-routed` has no fixed repo on purpose — claiming one would be
        a lie about where the reply comes from."""
        model = core.get_model("t1-routed")
        assert model.vendor == "t1api"
        assert model.repo == ""

    def test_qwen38_27b_is_in_the_catalog(self):
        model = core.get_model("qwen3.8-27b")
        assert model.format == "gguf"
        assert model.gguf_filename

    def test_qwen38_27b_is_in_the_download_registry(self):
        from hypernix.models.download import KNOWN_MODELS

        assert "qwen3.8-27b" in KNOWN_MODELS

    def test_catalog_json_exposes_the_configured_server(self):
        assert core.catalog_json()["t1_api_url"] == core.t1_api_url()


# ---------------------------------------------------------------------------
# URL / key resolution
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_default_url_when_nothing_is_configured(self):
        assert core.t1_api_url() == core.T1_API_DEFAULT_URL

    def test_env_var_beats_the_persisted_setting(self, monkeypatch):
        core.set_t1_api_url("http://stored.example:9000")
        monkeypatch.setenv("HNX_T1_API_URL", "http://env.example:1234")
        assert core.t1_api_url() == "http://env.example:1234"

    def test_trailing_slash_is_normalized(self):
        core.set_t1_api_url("http://server.example:8000/")
        assert core.t1_api_url() == "http://server.example:8000"

    def _waiter(self, tmp_path, **fields):
        import json

        path = tmp_path / "waiter" / "waiter.config.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fields), encoding="utf-8")
        return path

    def _t1_env(self, tmp_path, text):
        (tmp_path / "t1api").mkdir(exist_ok=True)
        (tmp_path / "t1api" / ".env").write_text(text, encoding="utf-8")

    def test_the_local_server_is_found_where_hypernix_t1_put_it(self, isolated_config):
        """The bug: a server on any other port was 'connection refused'."""
        self._t1_env(isolated_config, "# comment\nT1_HOST=0.0.0.0\nT1_PORT='8123'\n")
        url, source = core.t1_api_url_source()
        assert url == "http://127.0.0.1:8123"
        assert "hypernix-t1" in source

    def test_waiters_server_is_found(self, isolated_config):
        self._waiter(isolated_config, server="100.64.0.7", port=8000, local_only=True)
        url, source = core.t1_api_url_source()
        assert url == "http://100.64.0.7:8000"
        assert "waiter" in source

    def test_the_order_is_env_setting_waiter_local_default(self, isolated_config, monkeypatch):
        self._t1_env(isolated_config, "T1_HOST=127.0.0.1\nT1_PORT=8200\n")
        assert core.t1_api_url() == "http://127.0.0.1:8200"
        self._waiter(isolated_config, server="http://waiter.example:9000")
        assert core.t1_api_url() == "http://waiter.example:9000"
        core.set_t1_api_url("http://mine.example:1")
        assert core.t1_api_url() == "http://mine.example:1"
        monkeypatch.setenv("HNX_T1_API_URL", "http://env.example:2")
        assert core.t1_api_url() == "http://env.example:2"

    def test_an_encrypted_waiter_config_is_skipped_untouched(self, isolated_config):
        """Decrypting it would create waiter's master key as a side effect."""
        path = self._waiter(isolated_config)
        path.write_text("gAAAAABnotjson", encoding="utf-8")
        assert core.t1_api_url() == core.T1_API_DEFAULT_URL
        assert sorted(p.name for p in path.parent.iterdir()) == [path.name]

    def test_client_without_a_key_names_the_fix(self):
        with pytest.raises(core.HypedProError) as excinfo:
            core.t1_api_client()
        assert excinfo.value.code == "HPC-T1API-001"
        assert "/key t1api" in excinfo.value.message

    def test_client_uses_the_configured_key_and_url(self, monkeypatch):
        monkeypatch.setenv("HNX_T1_API_KEY", "hnx-test-key")
        core.set_t1_api_url("http://server.example:8080")
        client = core.t1_api_client()
        assert client.base_url == "http://server.example:8080"


# ---------------------------------------------------------------------------
# Server model_id -> local model
# ---------------------------------------------------------------------------


class TestModelResolution:
    def test_exact_short_name_match(self):
        assert core._resolve_local_model("qwen3-4b-gguf") is core.get_model("qwen3-4b-gguf")

    def test_configured_mapping_wins_over_a_name_match(self):
        """An operator's explicit mapping is a decision, not a hint."""
        from hypernix.system.config import set_config_value

        set_config_value(core.T1_API_MODEL_MAP_CONFIG_KEY,
                         {"qwen3-4b-gguf": "gemma-4-27b"})
        assert core._resolve_local_model("qwen3-4b-gguf") is core.get_model("gemma-4-27b")

    def test_unknown_model_resolves_to_nothing(self):
        assert core._resolve_local_model("nanonix-mini-lite") is None

    def test_no_fuzzy_matching(self):
        """Running a *similar* model to the one the server authorized would
        be worse than refusing."""
        assert core._resolve_local_model("qwen3-4b") is None
        assert core._resolve_local_model("QWEN3-4B-GGUF") is None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class _FakeDecision:
    def __init__(self, model_id):
        self.model_id = model_id


class _FakeClient:
    """Stands in for T1Client. Records what it was asked and told."""

    def __init__(self, *, model_id="qwen3-4b-gguf", route_error=None, report_error=None):
        self.base_url = "http://fake.example"
        self._model_id = model_id
        self._route_error = route_error
        self._report_error = report_error
        self.routed = []
        self.reported = []

    def route(self, *, model_id=None, input_tokens=0):
        self.routed.append({"model_id": model_id, "input_tokens": input_tokens})
        if self._route_error:
            raise self._route_error
        return _FakeDecision(self._model_id)

    def report_usage(self, **kwargs):
        if self._report_error:
            raise self._report_error
        self.reported.append(kwargs)
        return {"recorded": True}


@pytest.fixture
def fake_client(monkeypatch):
    holder = {}

    def install(**kwargs):
        client = _FakeClient(**kwargs)
        holder["client"] = client
        monkeypatch.setattr(core, "t1_api_client", lambda **_kw: client)
        return client

    return install


@pytest.fixture
def stub_local_inference(monkeypatch):
    """Replace real inference with a recorded reply."""
    calls = []

    def fake_gguf(model, messages, **kwargs):
        calls.append((model.short, messages, kwargs))
        return "the local reply"

    monkeypatch.setattr(core, "ensure_downloaded", lambda model, quiet=True: "/tmp/fake")
    monkeypatch.setattr(core, "send_local_chat_gguf", fake_gguf)
    return calls


class TestDispatch:
    def test_runs_the_model_the_server_chose(self, fake_client, stub_local_inference):
        client = fake_client(model_id="qwen3-4b-gguf")
        reply = core.send_t1_api_chat([{"role": "user", "content": "hello"}])

        assert reply == "the local reply"
        assert stub_local_inference[0][0] == "qwen3-4b-gguf"
        assert client.routed == [{"model_id": None, "input_tokens": 1}]

    def test_a_requested_model_is_passed_as_a_request_not_a_choice(
        self, fake_client, stub_local_inference
    ):
        """The client asks; the server answers. Here it answers differently,
        and the client must run what the *server* said."""
        client = fake_client(model_id="gemma-4-27b")
        from hypernix.system.config import set_config_value

        set_config_value(core.T1_API_MODEL_MAP_CONFIG_KEY, {"gemma-4-27b": "qwen3-4b-gguf"})

        core.send_t1_api_chat([{"role": "user", "content": "hi"}], model_id="qwen3-4b-gguf")

        assert client.routed[0]["model_id"] == "qwen3-4b-gguf"
        assert stub_local_inference[0][0] == "qwen3-4b-gguf"

    def test_reports_usage_after_the_reply(self, fake_client, stub_local_inference):
        client = fake_client()
        core.send_t1_api_chat([{"role": "user", "content": "x" * 400}])

        assert len(client.reported) == 1
        report = client.reported[0]
        assert report["model_id"] == "qwen3-4b-gguf"
        assert report["input_tokens"] > 0
        assert report["output_tokens"] > 0
        assert report["endpoint"] == "/chat"

    def test_unmappable_model_refuses_and_says_how_to_fix_it(
        self, fake_client, stub_local_inference
    ):
        fake_client(model_id="nanonix-mini-lite")
        with pytest.raises(core.HypedProError) as excinfo:
            core.send_t1_api_chat([{"role": "user", "content": "hi"}])

        assert excinfo.value.code == "HPC-T1API-003"
        assert "nanonix-mini-lite" in excinfo.value.message
        assert core.T1_API_MODEL_MAP_CONFIG_KEY in excinfo.value.message
        # Nothing ran, so nothing was reported.
        assert stub_local_inference == []

    def test_mapping_back_to_the_pseudo_model_is_refused(
        self, fake_client, stub_local_inference
    ):
        """A map entry pointing at `t1-routed` would recurse forever."""
        from hypernix.system.config import set_config_value

        set_config_value(core.T1_API_MODEL_MAP_CONFIG_KEY, {"anything": "t1-routed"})
        fake_client(model_id="anything")

        with pytest.raises(core.HypedProError) as excinfo:
            core.send_t1_api_chat([{"role": "user", "content": "hi"}])
        assert excinfo.value.code == "HPC-T1API-003"

    def test_empty_model_id_from_the_server_is_refused(
        self, fake_client, stub_local_inference
    ):
        fake_client(model_id="")
        with pytest.raises(core.HypedProError) as excinfo:
            core.send_t1_api_chat([{"role": "user", "content": "hi"}])
        assert excinfo.value.code == "HPC-T1API-002"

    def test_a_failed_report_does_not_lose_the_reply(
        self, fake_client, stub_local_inference, caplog
    ):
        """The person already waited for this answer. Losing it because the
        accounting call failed would be the wrong trade."""
        fake_client(report_error=RuntimeError("connection reset"))
        reply = core.send_t1_api_chat([{"role": "user", "content": "hi"}])
        assert reply == "the local reply"
        assert "under-count" in caplog.text

    def test_routing_failure_preserves_the_servers_error_code(
        self, fake_client, stub_local_inference
    ):
        from hypernix.t1sdk.errors import T1Error

        fake_client(route_error=T1Error("all models exhausted", code="ROUTING_EXHAUSTED"))
        with pytest.raises(core.HypedProError) as excinfo:
            core.send_t1_api_chat([{"role": "user", "content": "hi"}])

        assert excinfo.value.code == "HPC-T1API-002"
        assert "ROUTING_EXHAUSTED" in excinfo.value.message
        assert stub_local_inference == []

    def test_unreachable_server_names_the_url(self, fake_client, stub_local_inference):
        fake_client(route_error=OSError("Connection refused"))
        with pytest.raises(core.HypedProError) as excinfo:
            core.send_t1_api_chat([{"role": "user", "content": "hi"}])
        assert excinfo.value.code == "HPC-T1API-002"
        assert core.t1_api_url() in excinfo.value.message
        # Refused means nothing is listening: say where the address came
        # from and what to run, not just the errno.
        assert "Nothing is listening" in excinfo.value.message
        assert "hnx-t1 start" in excinfo.value.message
        assert "the default" in excinfo.value.message

    def test_a_refusal_through_the_sdk_gets_the_advice_too(self, isolated_config, monkeypatch):
        """The real path: the SDK's transport error, not a bare OSError."""
        import socket

        from hypernix.t1sdk import T1Client

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        core.set_t1_api_url(f"http://127.0.0.1:{port}")
        monkeypatch.setenv("HNX_T1_API_KEY", "T1_whatever")
        assert T1Client  # the real client, no fakes
        with pytest.raises(core.HypedProError) as excinfo:
            core.t1_api_status()
        assert "Nothing is listening" in excinfo.value.message
        assert "/t1api setting" in excinfo.value.message

    def test_send_chat_message_routes_the_pseudo_model_here(
        self, fake_client, stub_local_inference
    ):
        fake_client()
        result = core.send_chat_message("t1-routed", [{"role": "user", "content": "hi"}])
        assert result["content"] == "the local reply"


# ---------------------------------------------------------------------------
# Release info
# ---------------------------------------------------------------------------


class TestReleaseInfo:
    def test_opt_out_is_honoured_even_with_force(self, monkeypatch):
        from hypernix.system import release

        monkeypatch.setenv("HYPERNIX_NO_VERSION_CHECK", "1")
        info = release.current_release(force=True)
        assert info.status == "unknown"
        assert "disabled" in (info.error or "")

    def test_network_failure_is_not_an_exception(self, monkeypatch):
        from hypernix.system import release

        monkeypatch.delenv("HYPERNIX_NO_VERSION_CHECK", raising=False)
        monkeypatch.delenv("HYPERNIX_OFFLINE", raising=False)
        monkeypatch.setattr(release, "_read_cache", lambda _ttl: None)
        monkeypatch.setattr(release, "_fetch",
                            lambda _t: (_ for _ in ()).throw(OSError("no route to host")))
        info = release.current_release()
        assert info.status == "unknown"
        assert info.latest is None
        assert not info.update_available

    @pytest.mark.parametrize("installed,latest,expected", [
        ("0.71.4", "0.71.5", "outdated"),
        ("0.71.5", "0.71.5", "current"),
        ("0.71.5rc2", "0.71.4.post5", "prerelease"),
        ("0.71.5rc2", "0.71.5", "outdated"),
        ("0.71.6", "0.71.5", "current"),
        ("0.71.5", None, "unknown"),
        ("not-a-version", "0.71.5", "unknown"),
    ])
    def test_compare(self, installed, latest, expected):
        from hypernix.system import release

        assert release.compare(installed, latest) == expected

    def test_summary_never_tells_a_prerelease_user_to_downgrade(self):
        from hypernix.system.release import ReleaseInfo

        info = ReleaseInfo(installed="0.71.5rc2", latest="0.71.4.post5",
                           status="prerelease")
        assert not info.update_available
        assert "available" not in info.summary()
