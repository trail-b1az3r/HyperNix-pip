"""Core tests for T1 v1.0.26.8.0.1 — no HTTP layer involved.

Everything here imports without ``fastapi``, matching the rest of the
t1api core suite: the version scheme, the LM Studio bridge's parsing and
error mapping, pairing, sessions, the attachment store, and the Hugging
Face link merger.

The HTTP surface is covered separately in
``tests/test_t1_v1_0_26_http.py``.
"""
from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest import mock

import pytest

from hypernix.bridge.lmstudio import (
    LMStudioBridge,
    LMStudioError,
    LMStudioModel,
    _normalise_url,
    default_endpoints,
)
from hypernix.hyperlink.files import AttachmentStore, sniff_content_type
from hypernix.hyperlink.hfmerge import (
    HFResolveError,
    guess_quantization,
    merge,
    parse_link,
    resolve,
    split_siblings,
)
from hypernix.hyperlink.pairing import (
    MAX_REDEEM_ATTEMPTS,
    PAIRING_ALPHABET,
    DeviceRegistry,
    generate_code,
    hash_token,
    normalise_code,
)
from hypernix.hyperlink.sessions import ChatSessionStore, estimate_tokens
from hypernix.t1api.db import SQLiteBackend
from hypernix.t1api.errors import T1APIError, T1ErrorCode
from hypernix.t1api.version import MIN_CLIENT_VERSION, T1_VERSION, T1Version


@pytest.fixture()
def backend(tmp_path: Path) -> SQLiteBackend:
    return SQLiteBackend(tmp_path / "t1.db")


# ---------------------------------------------------------------------------
# The version scheme
# ---------------------------------------------------------------------------


class TestT1Version:
    def test_this_build_is_the_documented_version(self):
        # 1.0.2026.9.2.1 takes feature 2 in September: the governed
        # inference surface (/inference/*), which puts generation behind
        # the registry, the plan cascade, the quota and the cost ledger
        # that /bridge/lmstudio/* passes straight through.
        #
        # Deliberately a literal. This is the tripwire that makes a
        # version bump a decision rather than a side effect.
        assert T1_VERSION.short == "1.1.26.9.0.0"
        assert T1_VERSION.long == "1.1.2026.9.0.0"
        assert T1_VERSION.display == "t1 v1.1.26.9.0.0"
        assert T1_VERSION.generation == "1.1"
        assert T1_VERSION.release == "2026-09"

    @pytest.mark.parametrize(
        "template",
        ["{short}", "{long}", "v{short}", "t1 v{long}", "T1 V{short}", "  {short}  "],
    )
    def test_both_spellings_and_every_prefix_parse_to_one_value(self, template):
        """About the parser, not about which version happens to ship.

        Built from the constant so a version bump does not need this test
        edited — editing it each time is how a spelling quietly stops
        being covered.
        """
        text = template.format(short=T1_VERSION.short, long=T1_VERSION.long)
        assert T1Version.parse(text) == T1_VERSION

    def test_short_and_long_spellings_compare_equal(self):
        assert T1Version.parse("1.0.26.8.0.1") == T1Version.parse("1.0.2026.8.0.1")
        assert hash(T1Version.parse("1.0.26.8.0.1")) == hash(T1Version.parse("1.0.2026.8.0.1"))

    def test_ordering_runs_most_significant_first(self):
        assert T1Version.parse("1.0.26.8.0.1") < T1Version.parse("1.0.26.8.0.2")
        assert T1Version.parse("1.0.26.8.0.9") < T1Version.parse("1.0.26.8.1.0")
        assert T1Version.parse("1.0.26.8.9.9") < T1Version.parse("1.0.26.9.0.0")
        assert T1Version.parse("1.0.26.12.0.0") < T1Version.parse("1.0.27.1.0.0")
        assert T1Version.parse("1.9.26.8.0.0") < T1Version.parse("2.0.26.8.0.0")

    def test_can_be_compared_against_a_string(self):
        assert T1_VERSION > "1.0.26.7.0.0"
        assert T1_VERSION >= "1.0.26.8.0.1"

    @pytest.mark.parametrize(
        "bad",
        [
            "1.0.26.8.0",          # five components
            "1.0.26.8.0.1.2",      # seven
            "1.0.202.8.0.1",       # three-digit year is a typo, not a year
            "1.0.26.13.0.1",       # month 13
            "1.0.26.0.0.1",        # month 0
            "1.0.-1.8.0.1",        # negative
            "1.0.x.8.0.1",         # not a number
            "",
        ],
    )
    def test_a_malformed_version_raises_rather_than_parsing_loosely(self, bad):
        with pytest.raises(ValueError):
            T1Version.parse(bad)

    def test_the_offending_text_is_in_the_error(self):
        with pytest.raises(ValueError, match="1.0.26.99.0.1"):
            T1Version.parse("1.0.26.99.0.1")

    def test_compatibility_is_by_generation(self):
        """Generation is `api.major`, and 0.72.6 pt2 moved it 1.0 -> 1.1.

        That is the meaning of the bump rather than a side effect of it:
        a 1.0 client is now out of generation and gets told to upgrade,
        instead of being left to fail later on a field that is not
        there.
        """
        assert T1_VERSION.compatible_with("1.1.26.12.4.0")
        assert not T1_VERSION.compatible_with("1.0.26.9.2.3")
        assert not T1_VERSION.compatible_with("2.0.26.8.0.0")

    def test_min_client_is_in_this_generation_and_not_newer_than_us(self):
        assert MIN_CLIENT_VERSION <= T1_VERSION
        assert MIN_CLIENT_VERSION.compatible_with(T1_VERSION)

    def test_bump_replaces_only_what_it_is_given(self):
        # Deliberately does *not* zero lower components: year and month
        # are dates, not counters.
        #
        # It does not zero `fix` on a feature bump either, which is worth
        # knowing: from 1.0.26.8.1.1, bumping the feature gives
        # 1.0.26.9.2.1 — a version claiming a fix that never happened to
        # that feature. Every caller today passes the components it wants
        # explicitly, so nothing relies on the zeroing; the assertion below
        # records the behaviour as it is rather than as it might be.
        #
        # A fixed base, not T1_VERSION: this tests bump(), and pinning it
        # to whatever is shipping means every release edits a test about
        # arithmetic.
        base = T1Version(api=1, major=0, year=2026, month=8, feature=1, fix=1)
        assert base.bump(fix=2).short == "1.0.26.8.1.2"
        assert base.bump(month=9, feature=2, fix=0).short == "1.0.26.9.2.0"
        assert base.bump(month=9, feature=2).short == "1.0.26.9.2.1"

    def test_to_dict_carries_both_spellings(self):
        data = T1_VERSION.to_dict()
        assert data["short"] == T1_VERSION.short
        assert data["long"] == T1_VERSION.long
        assert data["short"] != data["long"], "the two spellings must differ"
        # The structure, not the shipping date — a to_dict test should not
        # need editing because a month passed.
        assert data["year"] == T1_VERSION.year
        assert data["month"] == T1_VERSION.month
        assert 1 <= data["month"] <= 12

    def test_every_shipped_component_agrees(self):
        from hypernix.hyperlink import __hyperlink_version__
        from hypernix.t1api import __t1api_version__, __t1api_version_long__
        from hypernix.t1sdk import __sdk_version__
        from hypernix.waiter import __waiter_version__

        assert __t1api_version__ == T1_VERSION.short
        assert __t1api_version_long__ == T1_VERSION.long
        assert __sdk_version__ == __waiter_version__ == __hyperlink_version__ == T1_VERSION.short


# ---------------------------------------------------------------------------
# The LM Studio bridge
# ---------------------------------------------------------------------------


def _fake_urlopen(payloads: dict[str, object], *, fail: set[str] | None = None):
    """Serve canned JSON per path, and 404 anything unmapped."""
    fail = fail or set()

    def opener(request, timeout=None):  # noqa: ARG001
        path = request.full_url.split("://", 1)[1].split("/", 1)[1]
        path = "/" + path.split("?")[0]
        if path in fail:
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, BytesIO(
                json.dumps({"error": {"message": "No models loaded"}}).encode()
            ))
        if path not in payloads:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, BytesIO(b"{}"))
        body = json.dumps(payloads[path]).encode()

        class _Resp:
            headers = {}

            def read(self):
                return body

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    return opener


class TestLMStudioURLHandling:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("desktop:1234", "http://desktop:1234"),
            ("http://x:1234/", "http://x:1234"),
            ("http://x:1234/v1", "http://x:1234"),
            ("http://x:1234/v1/", "http://x:1234"),
            ("http://x:1234/api/v0", "http://x:1234"),
            ("https://x", "https://x"),
            ("", "http://localhost:1234"),
        ],
    )
    def test_the_two_mistakes_people_make_are_absorbed(self, given, expected):
        # Pasting the base URL LM Studio prints (ends in /v1) and
        # omitting the scheme are both common enough to be worth fixing
        # rather than 404-ing on.
        assert _normalise_url(given) == expected

    def test_an_explicit_env_url_short_circuits_discovery(self, monkeypatch):
        monkeypatch.setenv("HYPERNIX_LMSTUDIO_URL", "http://gpu-box:1234")
        assert default_endpoints() == ["http://gpu-box:1234"]

    def test_discovery_defaults_cover_loopback(self, monkeypatch):
        monkeypatch.delenv("HYPERNIX_LMSTUDIO_URL", raising=False)
        endpoints = default_endpoints(include_tailscale=False)
        assert "http://localhost:1234" in endpoints
        assert "http://127.0.0.1:1234" in endpoints


class TestLMStudioModels:
    def test_native_listing_carries_the_loaded_bit(self):
        model = LMStudioModel.from_native(
            {
                "id": "qwen/qwen3-8b",
                "type": "llm",
                "state": "loaded",
                "arch": "qwen3",
                "quantization": "Q4_K_M",
                "max_context_length": 32768,
                "publisher": "qwen",
            }
        )
        assert model.loaded is True
        assert model.quantization == "Q4_K_M"
        assert model.context_length == 32768

    def test_a_downloaded_but_unloaded_model_is_not_loaded(self):
        assert LMStudioModel.from_native({"id": "m", "state": "not-loaded"}).loaded is False

    def test_a_vision_model_is_marked_as_one(self):
        assert LMStudioModel.from_native({"id": "m", "type": "vlm"}).supports_vision is True

    def test_openai_listing_never_claims_a_model_is_loaded(self):
        # The thin API cannot distinguish; claiming otherwise produces a
        # sixty-second stall with no explanation.
        assert LMStudioModel.from_openai({"id": "m", "owned_by": "org"}).loaded is False

    def test_publisher_falls_back_to_the_owner_in_the_id(self):
        assert LMStudioModel.from_native({"id": "bartowski/model"}).publisher == "bartowski"
        assert LMStudioModel.from_native({"id": "flat-name"}).publisher == ""
        assert LMStudioModel.from_native({"id": "flat", "publisher": "x"}).publisher == "x"


class TestLMStudioBridge:
    def _bridge(self, payloads, *, fail=None):
        bridge = LMStudioBridge("http://lmstudio.test:1234", connect_timeout=0.1)
        # The connect pre-check opens a real socket; the point of these
        # tests is the protocol above it.
        object.__setattr__(bridge, "_check_connect", lambda: None)
        return bridge, mock.patch(
            "urllib.request.urlopen", _fake_urlopen(payloads, fail=fail)
        )

    def test_prefers_the_native_listing(self):
        bridge, patcher = self._bridge(
            {
                "/api/v0/models": {"data": [{"id": "a", "state": "loaded", "type": "llm"}]},
                "/v1/models": {"data": [{"id": "should-not-be-used"}]},
            }
        )
        with patcher:
            models = bridge.list_models()
        assert [m.model_id for m in models] == ["a"]
        assert models[0].loaded

    def test_falls_back_to_the_openai_listing(self):
        bridge, patcher = self._bridge({"/v1/models": {"data": [{"id": "b"}]}})
        with patcher:
            models = bridge.list_models()
        assert [m.model_id for m in models] == ["b"]

    def test_resolve_picks_the_loaded_model_when_none_is_named(self):
        bridge, patcher = self._bridge(
            {
                "/api/v0/models": {
                    "data": [
                        {"id": "cold", "state": "not-loaded"},
                        {"id": "hot", "state": "loaded"},
                    ]
                }
            }
        )
        with patcher:
            assert bridge.resolve_model() == "hot"

    def test_asking_for_a_downloaded_but_unloaded_model_says_so(self):
        bridge, patcher = self._bridge(
            {"/api/v0/models": {"data": [{"id": "cold", "state": "not-loaded"}]}}
        )
        with patcher, pytest.raises(LMStudioError) as excinfo:
            bridge.resolve_model("cold")
        assert excinfo.value.code == "no_model_loaded"
        assert "not loaded" in str(excinfo.value)

    def test_an_unknown_model_lists_what_is_available(self):
        bridge, patcher = self._bridge(
            {"/api/v0/models": {"data": [{"id": "real", "state": "loaded"}]}}
        )
        with patcher, pytest.raises(LMStudioError) as excinfo:
            bridge.resolve_model("imaginary")
        assert excinfo.value.code == "model_not_found"
        assert "real" in str(excinfo.value)

    def test_an_empty_server_is_a_specific_error(self):
        bridge, patcher = self._bridge({"/api/v0/models": {"data": []}})
        with patcher, pytest.raises(LMStudioError) as excinfo:
            bridge.resolve_model()
        assert excinfo.value.code == "no_model_loaded"

    def test_a_thin_api_server_still_yields_a_model(self):
        # On a non-LM-Studio OpenAI server "loaded" is unknowable, and
        # refusing would make the bridge useless against all of them.
        bridge, patcher = self._bridge({"/v1/models": {"data": [{"id": "only"}]}})
        with patcher:
            assert bridge.resolve_model() == "only"

    def test_a_non_model_list_is_rejected_with_a_useful_message(self):
        bridge, patcher = self._bridge({"/v1/models": {"nope": True}})
        with patcher, pytest.raises(LMStudioError) as excinfo:
            bridge.list_models()
        assert excinfo.value.code == "bad_response"

    def test_lm_studios_400_for_an_empty_server_maps_to_no_model_loaded(self):
        # LM Studio answers "no models loaded" with a 400 and a message,
        # not a dedicated status. Recognising it is what lets the CLI
        # print "load a model" instead of "HTTP 400".
        bridge, patcher = self._bridge(
            {"/api/v0/models": {"data": [{"id": "m", "state": "loaded"}]}},
            fail={"/v1/chat/completions"},
        )
        with patcher, pytest.raises(LMStudioError) as excinfo:
            bridge.chat([{"role": "user", "content": "hi"}])
        assert excinfo.value.code == "no_model_loaded"

    def test_a_dead_address_names_the_address_and_what_to_do(self):
        bridge = LMStudioBridge("http://127.0.0.1:1", connect_timeout=0.2)
        probe = bridge.probe(check_cors=False)
        assert probe.reachable is False
        assert probe.error_code == "unreachable"
        assert "127.0.0.1:1" in probe.error
        assert "Start Server" in probe.error

    def test_probe_never_raises(self):
        # Discovery probes many addresses and most are not LM Studio;
        # an exception per dead address would make callers unreadable.
        assert LMStudioBridge("http://127.0.0.1:2", connect_timeout=0.2).probe().usable is False


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


class TestPairing:
    def test_codes_avoid_the_characters_people_confuse(self):
        for _ in range(200):
            code = generate_code()
            assert len(code) == 6
            assert set(code) <= set(PAIRING_ALPHABET)
        assert not (set("O0I1L") & set(PAIRING_ALPHABET))

    @pytest.mark.parametrize(
        "typed", ["fzq-4m7", "FZQ 4M7", "fzq4m7", " f z q 4 m 7 ", "FZQ.4M7", "fzq_4m7"]
    )
    def test_a_code_is_recognised_however_it_was_punctuated(self, typed):
        assert normalise_code(typed) == "FZQ4M7"

    def test_a_mistyped_character_survives_normalisation(self):
        # Stripping it would shorten the code and produce "a pairing
        # code is six characters", which is an error about the wrong
        # thing. Keeping it gives "unknown pairing code".
        assert normalise_code("FZQ4M0") == "FZQ4M0"

    def test_a_mistyped_code_is_unknown_not_malformed(self, backend):
        registry = DeviceRegistry(backend)
        registry.create_code(created_by="admin")
        with pytest.raises(T1APIError) as excinfo:
            registry.redeem("FZQ4M0", device_name="phone")
        assert excinfo.value.code is T1ErrorCode.NOT_FOUND

    def test_redeeming_yields_a_token_that_authenticates(self, backend):
        registry = DeviceRegistry(backend)
        code = registry.create_code(created_by="admin-key", label="phone")
        record, token = registry.redeem(
            code.code, device_name="Test iPhone", app_version="1.0"
        )
        assert token.startswith("HLNK_")
        assert registry.authenticate(token).device_id == record.device_id

    def test_the_token_is_stored_only_as_a_hash(self, backend):
        registry = DeviceRegistry(backend)
        code = registry.create_code(created_by="admin")
        _, token = registry.redeem(code.code, device_name="phone")
        with backend.connect() as conn:
            rows = conn.execute("SELECT token_hash FROM hyperlink_devices").fetchall()
        assert rows[0]["token_hash"] == hash_token(token)
        assert token not in rows[0]["token_hash"]

    def test_a_device_record_never_exposes_its_hash(self, backend):
        registry = DeviceRegistry(backend)
        code = registry.create_code(created_by="admin")
        record, _ = registry.redeem(code.code, device_name="phone")
        assert "token_hash" not in record.to_dict()

    def test_a_code_is_single_use(self, backend):
        registry = DeviceRegistry(backend)
        code = registry.create_code(created_by="admin")
        registry.redeem(code.code, device_name="first")
        with pytest.raises(T1APIError) as excinfo:
            registry.redeem(code.code, device_name="second")
        assert excinfo.value.code is T1ErrorCode.CONFLICT

    def test_an_expired_code_says_so_rather_than_not_found(self, backend):
        # Those two send the user to two different places.
        registry = DeviceRegistry(backend)
        code = registry.create_code(created_by="admin", ttl_seconds=-1)
        with pytest.raises(T1APIError) as excinfo:
            registry.redeem(code.code, device_name="phone")
        assert excinfo.value.code is T1ErrorCode.VALIDATION_ERROR
        assert "expired" in str(excinfo.value)

    def test_a_code_is_cancelled_after_too_many_failed_attempts(self, backend):
        registry = DeviceRegistry(backend)
        code = registry.create_code(created_by="admin")
        for _ in range(MAX_REDEEM_ATTEMPTS):
            registry.note_failed_attempt(code.code)
        with pytest.raises(T1APIError, match="Too many failed attempts"):
            registry.redeem(code.code, device_name="phone")
        # And it is gone, not merely refused.
        with pytest.raises(T1APIError):
            registry.redeem(code.code, device_name="phone")

    def test_a_code_of_the_wrong_length_is_rejected_before_any_lookup(self, backend):
        registry = DeviceRegistry(backend)
        with pytest.raises(T1APIError, match="six characters|6 characters"):
            registry.redeem("ABC", device_name="phone")

    def test_revoking_a_device_invalidates_its_token_immediately(self, backend):
        registry = DeviceRegistry(backend)
        code = registry.create_code(created_by="admin")
        record, token = registry.redeem(code.code, device_name="phone")
        registry.revoke_device(record.device_id)
        with pytest.raises(T1APIError) as excinfo:
            registry.authenticate(token)
        assert excinfo.value.code is T1ErrorCode.AUTH_REVOKED_KEY

    def test_revoking_one_device_leaves_the_others_alone(self, backend):
        registry = DeviceRegistry(backend)
        tokens = []
        for name in ("phone", "ipad"):
            code = registry.create_code(created_by="admin")
            record, token = registry.redeem(code.code, device_name=name)
            tokens.append((record, token))
        registry.revoke_device(tokens[0][0].device_id)
        assert registry.authenticate(tokens[1][1]).name == "ipad"

    def test_an_unknown_token_is_refused(self, backend):
        registry = DeviceRegistry(backend)
        with pytest.raises(T1APIError) as excinfo:
            registry.authenticate("HLNK_not-a-real-token")
        assert excinfo.value.code is T1ErrorCode.AUTH_INVALID_KEY

    def test_an_empty_token_is_a_missing_credential_not_an_invalid_one(self, backend):
        with pytest.raises(T1APIError) as excinfo:
            DeviceRegistry(backend).authenticate("")
        assert excinfo.value.code is T1ErrorCode.AUTH_MISSING_CREDENTIALS

    def test_devices_inherit_the_scopes_the_code_was_minted_with(self, backend):
        registry = DeviceRegistry(backend)
        code = registry.create_code(created_by="admin", scopes=["models:read"])
        record, _ = registry.redeem(code.code, device_name="phone")
        assert record.scopes == ("models:read",)

    def test_a_devices_owner_is_the_key_that_paired_it(self, backend):
        # This is what makes one operator's phone and desktop share
        # sessions while another operator's stay invisible.
        registry = DeviceRegistry(backend)
        code = registry.create_code(created_by="key-abc")
        record, _ = registry.redeem(code.code, device_name="phone")
        assert record.paired_by == "key-abc"

    def test_listing_hides_revoked_devices_by_default(self, backend):
        registry = DeviceRegistry(backend)
        code = registry.create_code(created_by="admin")
        record, _ = registry.redeem(code.code, device_name="phone")
        registry.revoke_device(record.device_id)
        assert registry.list_devices() == []
        assert len(registry.list_devices(include_revoked=True)) == 1

    def test_renaming_requires_a_name(self, backend):
        registry = DeviceRegistry(backend)
        code = registry.create_code(created_by="admin")
        record, _ = registry.redeem(code.code, device_name="phone")
        assert registry.rename_device(record.device_id, "  Mason's phone ").name == "Mason's phone"
        with pytest.raises(T1APIError):
            registry.rename_device(record.device_id, "   ")


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestSessions:
    def test_a_system_prompt_becomes_the_first_message(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1", system_prompt="Be terse.")
        messages = store.messages(session.session_id)
        assert [m.role for m in messages] == ["system"]

    def test_sequence_numbers_are_dense_and_ordered(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1")
        for i in range(5):
            store.append(session.session_id, role="user", content=f"m{i}")
        assert [m.seq for m in store.messages(session.session_id)] == [1, 2, 3, 4, 5]

    def test_an_invalid_role_is_rejected(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1")
        with pytest.raises(T1APIError):
            store.append(session.session_id, role="wizard", content="x")

    def test_another_owners_session_is_a_404_not_a_403(self, backend):
        # Confirming an id exists is itself information.
        store = ChatSessionStore(backend)
        session = store.create(owner="k1")
        with pytest.raises(T1APIError) as excinfo:
            store.get(session.session_id, owner="k2")
        assert excinfo.value.code is T1ErrorCode.NOT_FOUND

    def test_owner_scoping_applies_to_messages_too(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1")
        store.append(session.session_id, role="user", content="secret")
        with pytest.raises(T1APIError):
            store.messages(session.session_id, owner="k2")

    def test_context_keeps_the_system_prompt_and_trims_from_the_front(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1", system_prompt="SYS")
        for _ in range(30):
            store.append(session.session_id, role="user", content="x" * 200)
        context = store.context_for(session.session_id, token_budget=300)
        assert context[0].role == "system"
        assert len(context) < 31
        # What survives is the *end* of the conversation.
        assert context[-1].seq == 31

    def test_context_keeps_at_least_one_message_even_on_a_tiny_budget(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1")
        store.append(session.session_id, role="user", content="x" * 10_000)
        assert len(store.context_for(session.session_id, token_budget=1)) == 1

    def test_autotitle_uses_the_first_user_message(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1", system_prompt="SYS")
        store.append(session.session_id, role="user", content="How does SIMD work?\nmore")
        assert store.autotitle(session.session_id).title == "How does SIMD work?"

    def test_autotitle_leaves_a_real_title_alone(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1", title="My notes")
        store.append(session.session_id, role="user", content="anything")
        assert store.autotitle(session.session_id).title == "My notes"

    def test_a_very_long_first_line_is_truncated_visibly(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1")
        store.append(session.session_id, role="user", content="y" * 200)
        assert store.autotitle(session.session_id).title.endswith("…")

    def test_incremental_sync_returns_only_newer_messages(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1")
        for i in range(4):
            store.append(session.session_id, role="user", content=str(i))
        assert [m.seq for m in store.messages(session.session_id, after_seq=2)] == [3, 4]

    def test_a_limit_returns_the_newest_messages_in_order(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1")
        for i in range(10):
            store.append(session.session_id, role="user", content=str(i))
        assert [m.seq for m in store.messages(session.session_id, limit=3)] == [8, 9, 10]

    def test_deleting_a_session_takes_its_messages_with_it(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1")
        store.append(session.session_id, role="user", content="x")
        store.delete(session.session_id, owner="k1")
        with backend.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM hyperlink_messages WHERE session_id = ?",
                (session.session_id,),
            ).fetchone()
        assert int(row["n"]) == 0

    def test_the_session_list_counts_messages_without_an_n_plus_one(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1")
        for _ in range(3):
            store.append(session.session_id, role="user", content="x")
        listed = store.list_sessions(owner="k1")
        assert listed[0].message_count == 3

    def test_archived_sessions_are_hidden_by_default(self, backend):
        store = ChatSessionStore(backend)
        session = store.create(owner="k1")
        store.update(session.session_id, owner="k1", archived=True)
        assert store.list_sessions(owner="k1") == []
        assert len(store.list_sessions(owner="k1", include_archived=True)) == 1

    def test_token_estimate_is_roughly_four_characters(self):
        assert estimate_tokens("") == 1
        assert 20 <= estimate_tokens("x" * 100) <= 30


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------


class TestAttachments:
    @pytest.fixture()
    def store(self, tmp_path, backend):
        return AttachmentStore(tmp_path / "blobs", backend=backend)

    def test_magic_bytes_beat_a_lying_filename(self, ):
        # A .png that is really a zip must not be treated as an image.
        assert sniff_content_type(b"PK\x03\x04zz", filename="photo.png") == "application/zip"

    def test_magic_bytes_beat_a_lying_declared_type(self):
        assert (
            sniff_content_type(b"\x89PNG\r\n\x1a\n", declared="text/plain") == "image/png"
        )

    def test_source_files_are_recognised_as_text(self):
        assert sniff_content_type(b"print(1)\n", filename="a.py") == "text/x-python"
        assert sniff_content_type(b"let x = 1\n", filename="a.swift") == "text/x-swift"

    def test_unnamed_printable_content_is_text(self):
        assert sniff_content_type(b"hello there") == "text/plain"

    def test_binary_with_no_magic_is_opaque(self):
        assert sniff_content_type(b"\x00\x01\x02\x03") == "application/octet-stream"

    def test_a_path_traversing_filename_is_neutralised(self, store):
        record = store.put(b"data", filename="../../etc/passwd", owner="k1")
        assert record.filename == "passwd"

    def test_identical_bytes_share_one_blob(self, store, tmp_path):
        data = b"\x89PNG\r\n\x1a\n" + b"x" * 50
        first = store.put(data, filename="a.png", owner="k1")
        second = store.put(data, filename="b.png", owner="k1")
        assert first.sha256 == second.sha256
        assert first.file_id != second.file_id
        assert store.usage_bytes(owner="k1") == len(data)

    def test_deleting_one_reference_keeps_the_blob_for_the_other(self, store):
        data = b"\x89PNG\r\n\x1a\n" + b"y" * 50
        first = store.put(data, filename="a.png", owner="k1")
        second = store.put(data, filename="b.png", owner="k1")
        store.delete(second.file_id)
        assert store.read(first.file_id) == data

    def test_deleting_the_last_reference_removes_the_blob(self, store):
        record = store.put(b"only", filename="a.txt", owner="k1")
        path = store.path_for(record.file_id)
        store.delete(record.file_id)
        assert not path.exists()

    def test_an_empty_file_is_refused(self, store):
        with pytest.raises(T1APIError):
            store.put(b"", filename="empty", owner="k1")

    def test_an_oversized_file_is_refused_with_its_size(self, tmp_path, backend):
        store = AttachmentStore(tmp_path / "b", backend=backend, max_bytes=10)
        with pytest.raises(T1APIError, match="limit"):
            store.put(b"x" * 100, filename="big.bin", owner="k1")

    def test_another_owners_file_is_a_404(self, store):
        record = store.put(b"secret", filename="a.txt", owner="k1")
        with pytest.raises(T1APIError) as excinfo:
            store.get(record.file_id, owner="k2")
        assert excinfo.value.code is T1ErrorCode.NOT_FOUND

    def test_only_images_can_be_inlined_for_a_vision_model(self, store):
        record = store.put(b"just text", filename="a.txt", owner="k1")
        with pytest.raises(T1APIError, match="not an image"):
            store.data_url(record.file_id)

    def test_an_image_inlines_as_a_data_url(self, store):
        record = store.put(b"\x89PNG\r\n\x1a\n" + b"z" * 20, filename="a.png", owner="k1")
        assert store.data_url(record.file_id).startswith("data:image/png;base64,")

    def test_text_extraction_marks_its_own_truncation(self, store):
        record = store.put(b"x" * 5000, filename="a.txt", owner="k1")
        text = store.text_of(record.file_id, max_chars=100)
        assert "truncated" in text
        assert len(text) < 500

    def test_listing_can_be_scoped_to_a_session(self, store):
        store.put(b"a", filename="a.txt", owner="k1", session_id="s1")
        store.put(b"b", filename="b.txt", owner="k1", session_id="s2")
        assert len(store.list_files(owner="k1", session_id="s1")) == 1


# ---------------------------------------------------------------------------
# Hugging Face link merging
# ---------------------------------------------------------------------------


class TestHFParsing:
    @pytest.mark.parametrize(
        ("link", "repo", "filename"),
        [
            ("https://huggingface.co/o/r", "o/r", ""),
            ("https://huggingface.co/o/r/tree/main", "o/r", ""),
            ("https://huggingface.co/o/r/blob/main/m.gguf", "o/r", "m.gguf"),
            ("https://huggingface.co/o/r/resolve/main/m.gguf?download=true", "o/r", "m.gguf"),
            ("https://hf.co/o/r", "o/r", ""),
            ("hf://o/r/m.gguf", "o/r", "m.gguf"),
            ("o/r", "o/r", ""),
        ],
    )
    def test_every_shape_people_paste_parses(self, link, repo, filename):
        ref = parse_link(link)
        assert ref.repo_id == repo
        assert ref.filename == filename

    def test_a_revision_in_the_url_is_kept(self):
        assert parse_link("https://hf.co/o/r/blob/v2/m.gguf").revision == "v2"

    def test_the_ollama_style_quant_shorthand_becomes_a_hint(self):
        assert parse_link("o/r:Q5_K_M").filename == "*Q5_K_M*"

    def test_a_non_huggingface_host_is_refused_by_name(self):
        with pytest.raises(HFResolveError) as excinfo:
            parse_link("https://example.com/o/r")
        assert excinfo.value.code == "not_a_hf_link"
        assert str(excinfo.value) == "Not a Hugging Face link (host: example.com)."

    def test_a_link_with_no_repository_says_what_one_looks_like(self):
        with pytest.raises(HFResolveError) as excinfo:
            parse_link("https://huggingface.co/justowner")
        assert excinfo.value.code == "no_repo"
        assert "owner/name" in str(excinfo.value)

    @pytest.mark.parametrize(
        ("filename", "quant"),
        [
            ("m-Q4_K_M.gguf", "Q4_K_M"),
            ("m-Q4_K.gguf", "Q4_K"),
            ("m.IQ3_XXS.gguf", "IQ3_XXS"),
            ("m-BF16.gguf", "BF16"),
            ("Q8-Research-model.gguf", ""),
            ("plain.gguf", ""),
        ],
    )
    def test_quantisation_is_matched_longest_first_and_bounded(self, filename, quant):
        assert guess_quantization(filename) == quant

    def test_split_parts_are_recognised(self):
        assert split_siblings("m-Q5_K_M-00002-of-00005.gguf") == ("m-Q5_K_M", 2, 5)
        assert split_siblings("m.gguf") == ("m.gguf", 0, 0)


class TestHFMerge:
    def test_a_page_and_a_file_from_one_repo_combine(self):
        ref = merge(
            "https://huggingface.co/o/r",
            "https://huggingface.co/o/r/resolve/main/m-Q4_K_M.gguf?download=true",
        )
        assert ref.repo_id == "o/r"
        assert ref.filename == "m-Q4_K_M.gguf"

    def test_a_repository_conflict_is_raised_rather_than_guessed(self):
        with pytest.raises(HFResolveError) as excinfo:
            merge("https://huggingface.co/a/b", "https://huggingface.co/c/d/resolve/main/m.gguf")
        assert excinfo.value.code == "repo_conflict"
        assert "a/b" in str(excinfo.value) and "c/d" in str(excinfo.value)

    def test_prefer_file_takes_the_repository_that_owns_the_bytes(self):
        ref = merge(
            "https://huggingface.co/a/b",
            "https://huggingface.co/c/d/resolve/main/m.gguf",
            prefer="file",
        )
        assert ref.repo_id == "c/d" and ref.filename == "m.gguf"

    def test_prefer_page_carries_the_filename_across(self):
        ref = merge(
            "https://huggingface.co/a/b",
            "https://huggingface.co/c/d/resolve/main/m.gguf",
            prefer="page",
        )
        assert ref.repo_id == "a/b" and ref.filename == "m.gguf"

    def test_a_page_at_main_does_not_override_a_pinned_file_revision(self):
        ref = merge("https://huggingface.co/o/r", "https://huggingface.co/o/r/blob/v3/m.gguf")
        assert ref.revision == "v3"

    def test_either_link_alone_is_enough(self):
        assert merge(page_link="o/r").repo_id == "o/r"
        assert merge(file_link="hf://o/r/m.gguf").filename == "m.gguf"

    def test_neither_link_is_an_error(self):
        with pytest.raises(HFResolveError):
            merge()


class TestHFResolve:
    """Resolution against a canned Hugging Face API — no network."""

    def _patch_api(self, siblings, **extra):
        info = {"siblings": [{"rfilename": name, "size": size} for name, size in siblings]}
        info.update(extra)
        return mock.patch("hypernix.hyperlink.hfmerge.fetch_repo_info", return_value=info)

    def test_a_single_file_repo_resolves_to_one_download(self):
        with self._patch_api([("m-Q4_K_M.gguf", 4_000_000_000)]):
            plan = resolve("https://huggingface.co/o/r")
        assert plan.file_count == 1
        assert plan.quantization == "Q4_K_M"
        assert plan.total_bytes == 4_000_000_000
        assert plan.files[0].url.endswith("m-Q4_K_M.gguf?download=true")

    def test_clicking_one_part_of_a_split_model_pulls_the_whole_set(self):
        siblings = [(f"m-0000{i}-of-00003.gguf", 1000) for i in (1, 2, 3)]
        with self._patch_api(siblings):
            plan = resolve(
                file_link="https://huggingface.co/o/r/resolve/main/m-00002-of-00003.gguf"
            )
        assert plan.is_split
        assert [f.part_index for f in plan.files] == [1, 2, 3]
        assert plan.primary.part_index == 1
        assert any("part 2 of 3" in w for w in plan.warnings)

    def test_a_missing_split_part_is_reported(self):
        with self._patch_api([("m-00001-of-00003.gguf", 1), ("m-00003-of-00003.gguf", 1)]):
            plan = resolve(file_link="https://huggingface.co/o/r/resolve/main/m-00001-of-00003.gguf")
        assert any("00002" in w for w in plan.warnings)

    def test_a_vision_projector_is_included(self):
        with self._patch_api([("m-Q4_K_M.gguf", 10), ("mmproj-F16.gguf", 5)]):
            plan = resolve("o/r")
        assert plan.has_vision
        assert [f.role for f in plan.files] == ["weights", "mmproj"]
        assert any("cannot read images" in w for w in plan.warnings)

    def test_the_projector_matching_the_quant_is_preferred(self):
        with self._patch_api(
            [("m-Q8_0.gguf", 10), ("mmproj-F16.gguf", 5), ("mmproj-Q8_0.gguf", 4)]
        ):
            plan = resolve("o/r")
        assert plan.files[1].filename == "mmproj-Q8_0.gguf"

    def test_a_projector_is_never_chosen_as_the_model(self):
        with self._patch_api([("mmproj-F16.gguf", 5), ("m-Q4_K_M.gguf", 10)]):
            plan = resolve("o/r")
        assert plan.primary.filename == "m-Q4_K_M.gguf"

    def test_the_default_quant_is_q4_k_m_when_nobody_said(self):
        with self._patch_api(
            [("m-Q2_K.gguf", 1), ("m-Q4_K_M.gguf", 2), ("m-Q8_0.gguf", 3)]
        ):
            assert resolve("o/r").quantization == "Q4_K_M"

    def test_the_quant_shorthand_selects_a_file(self):
        with self._patch_api([("m-Q4_K_M.gguf", 1), ("m-Q6_K.gguf", 2)]):
            assert resolve("o/r:Q6_K").primary.filename == "m-Q6_K.gguf"

    def test_an_unavailable_quant_lists_what_there_is(self):
        with self._patch_api([("m-Q4_K_M.gguf", 1)]):
            with pytest.raises(HFResolveError) as excinfo:
                resolve("o/r:Q2_K")
        assert excinfo.value.code == "file_not_found"
        assert "Q4_K_M" in str(excinfo.value)

    def test_a_safetensors_repo_suggests_the_gguf_route(self):
        with self._patch_api([("model.safetensors", 1)]):
            with pytest.raises(HFResolveError) as excinfo:
                resolve("o/r")
        assert excinfo.value.code == "no_gguf"
        assert "quantise" in str(excinfo.value) or "-GGUF" in str(excinfo.value)

    def test_a_file_link_the_repo_does_not_have_says_what_it_does(self):
        with self._patch_api([("real.gguf", 1)]):
            with pytest.raises(HFResolveError) as excinfo:
                resolve(file_link="https://huggingface.co/o/r/resolve/main/imaginary.gguf")
        assert excinfo.value.code == "file_not_found"
        assert "real.gguf" in str(excinfo.value)

    def test_a_gated_repo_is_flagged_with_what_to_do(self):
        with self._patch_api([("m-Q4_K_M.gguf", 1)], gated=True):
            plan = resolve("o/r")
        assert plan.gated
        assert any("licence" in w for w in plan.warnings)

    def test_lfs_sizes_are_used_when_present(self):
        info = {"siblings": [{"rfilename": "m-Q4_K_M.gguf", "lfs": {"size": 1234}}]}
        with mock.patch("hypernix.hyperlink.hfmerge.fetch_repo_info", return_value=info):
            assert resolve("o/r").total_bytes == 1234

    def test_offline_resolution_works_from_a_file_link_alone(self):
        plan = resolve(file_link="https://huggingface.co/o/r/resolve/main/m-Q4_K_M.gguf", offline=True)
        assert plan.metadata_from_api is False
        assert plan.primary.filename == "m-Q4_K_M.gguf"

    def test_offline_resolution_expands_split_parts_from_the_name(self):
        plan = resolve(
            file_link="https://huggingface.co/o/r/resolve/main/m-00001-of-00004.gguf", offline=True
        )
        assert len(plan.files) == 4

    def test_offline_resolution_needs_more_than_a_page(self):
        with pytest.raises(HFResolveError) as excinfo:
            resolve("https://huggingface.co/o/r", offline=True)
        assert excinfo.value.code == "offline"
        assert excinfo.value.hint

    def test_an_unreachable_hub_still_honours_an_exact_file_link(self):
        # A phone on a bad connection should still be able to start a
        # download it has the exact URL for.
        with mock.patch(
            "hypernix.hyperlink.hfmerge.fetch_repo_info",
            side_effect=HFResolveError("down", code="offline"),
        ):
            plan = resolve(file_link="https://huggingface.co/o/r/resolve/main/m.gguf")
        assert plan.metadata_from_api is False
        assert plan.files[0].filename == "m.gguf"

    def test_an_unreachable_hub_with_only_a_page_still_raises(self):
        with mock.patch(
            "hypernix.hyperlink.hfmerge.fetch_repo_info",
            side_effect=HFResolveError("down", code="offline"),
        ):
            with pytest.raises(HFResolveError):
                resolve("https://huggingface.co/o/r")
