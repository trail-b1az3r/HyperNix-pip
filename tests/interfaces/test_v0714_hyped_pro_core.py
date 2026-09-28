"""Tests for hypernix.hyped_pro_core: the model/provider catalog, thinking
extraction, and real dispatch (cloud HTTP protocols + the agentic
tool-calling loop) behind hyped-pro (hyped+).
"""
from __future__ import annotations

import json

import pytest

from hypernix import hyped_pro_core as core
from hypernix.hyped_pro_core import HypedProError

# ---------------------------------------------------------------------------
# Catalog integrity
# ---------------------------------------------------------------------------

def test_every_model_vendor_is_registered():
    for model in core.MODELS:
        assert model.vendor in core.PROVIDERS, f"{model.short}: unknown vendor {model.vendor!r}"


def test_every_model_short_name_is_unique():
    shorts = [m.short for m in core.MODELS]
    assert len(shorts) == len(set(shorts))


def test_gguf_models_have_a_filename():
    for model in core.MODELS:
        if model.format == "gguf":
            assert model.gguf_filename, f"{model.short}: format=gguf but no gguf_filename"


def test_get_model_unknown_raises_coded_error():
    with pytest.raises(HypedProError) as exc_info:
        core.get_model("definitely-not-a-real-model")
    assert exc_info.value.code == "HPC-CFG-001"


def test_get_model_known_short_resolves():
    m = core.get_model("claude-sonnet-5")
    assert m.vendor == "anthropic"
    assert core.PROVIDERS[m.vendor].kind == "cloud"


def test_qwen_and_kimi_k3_are_classified_cloud():
    # The whole point of the original catalog fix: these two were wrongly
    # marked "local" and must never regress.
    qwen = core.get_model("qwen3.7-plus")
    kimi = core.get_model("kimi-k3")
    assert qwen.kind == "cloud"
    assert qwen.vendor == "dashscope"
    assert kimi.kind == "cloud"
    assert kimi.vendor == "moonshot"


def test_kimi_k2_7_code_stays_local_open_weight():
    # Only K3 is cloud-only at launch -- the K2.7 open-weight sibling must
    # not get swept into "cloud" by a careless catalog edit.
    m = core.get_model("kimi-k2.7-code")
    assert m.kind == "local"


# v0.71.5rc2 added a third kind. It is listed explicitly rather than
# dropping the check, so a *fourth* one still has to be a deliberate edit.
PROVIDER_KINDS = ("local", "cloud", "t1api")


def test_provider_kinds_are_from_the_known_set():
    for info in core.PROVIDERS.values():
        assert info.kind in PROVIDER_KINDS


def test_exactly_one_provider_speaks_to_a_t1_api_server():
    """`t1` (in-process Gatekeeper) and `t1api` (HTTP to a real server) are
    different things; only the second one is this kind."""
    t1api_vendors = [v for v, info in core.PROVIDERS.items() if info.kind == "t1api"]
    assert t1api_vendors == ["t1api"]


def test_cloud_providers_have_api_base_and_auth_env_var():
    for info in core.PROVIDERS.values():
        if info.kind == "cloud":
            assert info.api_base, f"{info.vendor}: cloud provider missing api_base"
            assert info.auth_env_var, f"{info.vendor}: cloud provider missing auth_env_var"


def test_catalog_json_round_trips_through_json():
    data = core.catalog_json()
    encoded = json.dumps(data)
    decoded = json.loads(encoded)
    assert set(decoded.keys()) == {"providers", "models", "t1_api_url"}
    assert len(decoded["models"]) == len(core.MODELS)
    assert set(decoded["providers"].keys()) == set(core.PROVIDERS.keys())
    # The TUI shows this in /t1api, so it has to survive the JSON hop.
    assert isinstance(decoded["t1_api_url"], str) and decoded["t1_api_url"]


# ---------------------------------------------------------------------------
# Thinking extraction / stripping
# ---------------------------------------------------------------------------

def test_strip_thinking_removes_closed_block():
    assert core.strip_thinking("<think>reasoning</think>The answer.") == "The answer."


def test_strip_thinking_no_block_is_unchanged():
    assert core.strip_thinking("just an answer") == "just an answer"


def test_strip_thinking_handles_unclosed_block_with_note():
    result = core.strip_thinking("<think>cut off mid-thought")
    assert "raise max output tokens" in result


def test_strip_thinking_case_insensitive_and_alt_tags():
    assert core.strip_thinking("<THINKING>x</THINKING>done") == "done"
    assert core.strip_thinking("<reasoning>x</reasoning>done") == "done"


def test_extract_thinking_returns_both_parts():
    content, thinking = core.extract_thinking("<think>step by step</think>42")
    assert content == "42"
    assert thinking == "step by step"


def test_extract_thinking_no_block_returns_none_thinking():
    content, thinking = core.extract_thinking("no thinking here")
    assert content == "no thinking here"
    assert thinking is None


def test_extract_thinking_multiple_blocks_joined():
    content, thinking = core.extract_thinking("<think>a</think>mid<think>b</think>end")
    assert content == "midend"
    assert thinking == "a\n\nb"


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def test_resolve_api_key_override_wins(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert core._resolve_api_key("anthropic", "sk-override") == "sk-override"


def test_resolve_api_key_missing_raises_coded_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(core, "get_provider_key", lambda vendor: None, raising=False)
    # get_provider_key is imported lazily inside _resolve_api_key from
    # .config, so patch it there instead.
    import hypernix.config as config_mod
    monkeypatch.setattr(config_mod, "get_provider_key", lambda vendor: None)
    with pytest.raises(HypedProError) as exc_info:
        core._resolve_api_key("openai", None)
    assert exc_info.value.code == "HPC-CLOUD-001"


# ---------------------------------------------------------------------------
# Cloud dispatch -- mocked HTTP, both protocols
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_http_post_json():
    """Every test that monkeypatches core._http_post_json must not leak
    into other tests -- restore the real one afterward."""
    original = core._http_post_json
    yield
    core._http_post_json = original


def test_send_cloud_chat_openai_protocol(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured = {}

    def fake_post(url, headers, body, timeout=60):
        captured["url"] = url
        captured["body"] = body
        return {"choices": [{"message": {"role": "assistant", "content": "hi there"}}]}

    core._http_post_json = fake_post
    model = core.get_model("gpt-4o")
    reply = core.send_cloud_chat(model, [{"role": "user", "content": "hello"}], enable_tools=False)
    assert reply == "hi there"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["body"]["messages"][-1]["content"] == "hello"


def test_send_cloud_chat_anthropic_protocol(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    captured = {}

    def fake_post(url, headers, body, timeout=60):
        captured["headers"] = headers
        captured["body"] = body
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "42"}]}

    core._http_post_json = fake_post
    model = core.get_model("claude-sonnet-5")
    reply = core.send_cloud_chat(model, [{"role": "user", "content": "what is the answer"}], enable_tools=False)
    assert reply == "42"
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert "anthropic-version" in captured["headers"]


def test_send_cloud_chat_anthropic_extended_thinking_params(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    captured = {}

    def fake_post(url, headers, body, timeout=60):
        captured["body"] = body
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]}

    core._http_post_json = fake_post
    model = core.get_model("claude-sonnet-5")
    core.send_cloud_chat(
        model, [{"role": "user", "content": "hi"}],
        max_tokens=500, max_thinking_tokens=200, enable_tools=False,
    )
    body = captured["body"]
    # Real Anthropic API constraints: temperature must be 1, and max_tokens
    # must exceed budget_tokens (it's the *total* budget).
    assert body["temperature"] == 1
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 200}
    assert body["max_tokens"] > 200


def test_send_cloud_chat_http_error_raises_coded_error(monkeypatch):

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def fake_post(url, headers, body, timeout=60):
        raise HypedProError("HPC-CLOUD-002", "HTTP 500 from api.openai.com")

    core._http_post_json = fake_post
    model = core.get_model("gpt-4o")
    with pytest.raises(HypedProError) as exc_info:
        core.send_cloud_chat(model, [{"role": "user", "content": "hi"}], enable_tools=False)
    assert exc_info.value.code == "HPC-CLOUD-002"


def test_send_cloud_chat_malformed_response_raises_coded_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    core._http_post_json = lambda *a, **k: {"unexpected": "shape"}
    model = core.get_model("gpt-4o")
    with pytest.raises(HypedProError) as exc_info:
        core.send_cloud_chat(model, [{"role": "user", "content": "hi"}], enable_tools=False)
    assert exc_info.value.code == "HPC-CLOUD-003"


# ---------------------------------------------------------------------------
# Agentic tool-calling loop
# ---------------------------------------------------------------------------

def test_openai_tool_loop_executes_tool_and_returns_final_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("HYPED_PRO_WORKSPACE", str(tmp_path))
    calls = {"n": 0}

    def fake_post(url, headers, body, timeout=60):
        calls["n"] += 1
        if calls["n"] == 1:
            assert "tools" in body
            return {"choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "function": {
                    "name": "create_file",
                    "arguments": json.dumps({"path": "out.txt", "content": "hello"}),
                }}],
            }}]}
        last_msg = body["messages"][-1]
        assert last_msg["role"] == "tool"
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    core._http_post_json = fake_post
    model = core.get_model("gpt-4o")
    reply = core.send_cloud_chat(model, [{"role": "user", "content": "make a file"}])
    assert reply == "done"
    assert calls["n"] == 2
    assert (tmp_path / "out.txt").read_text() == "hello"


def test_anthropic_tool_loop_executes_tool_and_returns_final_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HYPED_PRO_WORKSPACE", str(tmp_path))
    (tmp_path / "existing.txt").write_text("needle in a haystack")
    calls = {"n": 0}

    def fake_post(url, headers, body, timeout=60):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "t1", "name": "search_files",
                             "input": {"query": "needle", "path": "."}}],
            }
        last = body["messages"][-1]
        assert last["role"] == "user"
        assert last["content"][0]["type"] == "tool_result"
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "found it"}]}

    core._http_post_json = fake_post
    model = core.get_model("claude-sonnet-5")
    reply = core.send_cloud_chat(model, [{"role": "user", "content": "find needle"}])
    assert reply == "found it"
    assert calls["n"] == 2


def test_tool_loop_respects_round_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("HYPED_PRO_WORKSPACE", str(tmp_path))

    def always_wants_tool(url, headers, body, timeout=60):
        return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "x", "function": {"name": "list_directory", "arguments": "{}"}},
        ]}}]}

    core._http_post_json = always_wants_tool
    model = core.get_model("gpt-4o")
    with pytest.raises(HypedProError) as exc_info:
        core.send_cloud_chat(model, [{"role": "user", "content": "loop"}], max_tool_rounds=2)
    assert exc_info.value.code == "HPC-CLOUD-004"


def test_tool_loop_disabled_omits_tools_from_request(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured = {}

    def fake_post(url, headers, body, timeout=60):
        captured["body"] = body
        return {"choices": [{"message": {"role": "assistant", "content": "no tools used"}}]}

    core._http_post_json = fake_post
    model = core.get_model("gpt-4o")
    core.send_cloud_chat(model, [{"role": "user", "content": "hi"}], enable_tools=False)
    assert "tools" not in captured["body"]


def test_send_chat_message_returns_content_and_thinking_dict(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    core._http_post_json = lambda *a, **k: {
        "choices": [{"message": {"role": "assistant", "content": "<think>hmm</think>Done."}}]
    }
    hidden = core.send_chat_message("gpt-4o", [{"role": "user", "content": "hi"}], hide_thinking=True)
    assert hidden == {"content": "Done.", "thinking": None}

    shown = core.send_chat_message("gpt-4o", [{"role": "user", "content": "hi"}], hide_thinking=False)
    assert shown == {"content": "Done.", "thinking": "hmm"}
