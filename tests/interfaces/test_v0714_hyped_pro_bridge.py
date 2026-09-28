"""Tests for hypernix.hyped_pro_bridge: the line-delimited JSON stdio
protocol the Node hyped-pro TUI shells out to for every real operation.
"""
from __future__ import annotations

import json

import pytest

from hypernix import hyped_pro_bridge as bridge
from hypernix import hyped_pro_core as core


@pytest.fixture(autouse=True)
def _isolated_config_file(tmp_path, monkeypatch):
    """hypernix.config's _CONFIG_DIR/_CONFIG_FILE are computed once from
    Path.home() at import time, so monkeypatching HOME after import has no
    effect -- the already-bound path objects have to be patched directly."""
    from hypernix import config
    config_dir = tmp_path / ".hypernix"
    monkeypatch.setattr(config, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "_CONFIG_FILE", config_dir / "config.json")

# ---------------------------------------------------------------------------
# dispatch() -- command routing and response shape
# ---------------------------------------------------------------------------

def test_dispatch_ping():
    resp = bridge.dispatch({"id": 1, "cmd": "ping"})
    assert resp["ok"] is True
    assert resp["data"]["pong"] is True
    assert resp["data"]["version"] == bridge.BRIDGE_VERSION


def test_dispatch_catalog_returns_real_catalog():
    resp = bridge.dispatch({"id": 2, "cmd": "catalog"})
    assert resp["ok"] is True
    assert resp["data"] == core.catalog_json()


def test_dispatch_unknown_command_returns_coded_error():
    resp = bridge.dispatch({"id": 3, "cmd": "not_a_real_command"})
    assert resp["ok"] is False
    assert resp["code"] == "HPB-PROTO-001"


def test_dispatch_missing_required_field_returns_coded_error():
    # "chat" requires "model" and "messages" -- omit both.
    resp = bridge.dispatch({"id": 4, "cmd": "chat"})
    assert resp["ok"] is False
    assert resp["code"] == "HPB-PROTO-001"


def test_dispatch_preserves_request_id():
    resp = bridge.dispatch({"id": 12345, "cmd": "ping"})
    assert resp["id"] == 12345


def test_dispatch_hyped_pro_error_propagates_code_and_message():
    resp = bridge.dispatch({"id": 5, "cmd": "is_downloaded", "model": "not-a-real-model"})
    assert resp["ok"] is False
    assert resp["code"] == "HPC-CFG-001"


def test_dispatch_unhandled_exception_is_caught_not_raised(monkeypatch):
    def boom(model_short):
        raise RuntimeError("something broke internally")

    monkeypatch.setattr(core, "get_model", boom)
    resp = bridge.dispatch({"id": 6, "cmd": "is_downloaded", "model": "whatever"})
    assert resp["ok"] is False
    assert resp["code"] == "HPB-INTERNAL-001"
    assert "RuntimeError" in resp["error"]


# ---------------------------------------------------------------------------
# key_get / key_set / key_clear
# ---------------------------------------------------------------------------

def test_key_set_then_get_round_trips(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    set_resp = bridge.dispatch({"id": 7, "cmd": "key_set", "vendor": "anthropic", "key": "sk-ant-test1234567890"})
    assert set_resp["ok"] is True
    assert set_resp["data"]["saved"] is True

    get_resp = bridge.dispatch({"id": 8, "cmd": "key_get", "vendor": "anthropic"})
    assert get_resp["ok"] is True
    assert get_resp["data"]["set"] is True
    assert get_resp["data"]["masked"] != "sk-ant-test1234567890"  # never returned in full


def test_key_get_unset_vendor_reports_not_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resp = bridge.dispatch({"id": 9, "cmd": "key_get", "vendor": "openai"})
    assert resp["ok"] is True
    assert resp["data"]["set"] is False
    assert resp["data"]["masked"] is None


def test_key_clear_removes_a_set_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    bridge.dispatch({"id": 10, "cmd": "key_set", "vendor": "openai", "key": "sk-test"})
    clear_resp = bridge.dispatch({"id": 11, "cmd": "key_clear", "vendor": "openai"})
    assert clear_resp["ok"] is True
    get_resp = bridge.dispatch({"id": 12, "cmd": "key_get", "vendor": "openai"})
    assert get_resp["data"]["set"] is False


# ---------------------------------------------------------------------------
# cli_main -- one-shot mode used for scripting/debugging
# ---------------------------------------------------------------------------

def test_cli_main_one_shot_prints_json(capsys):
    exit_code = bridge.cli_main(["ping"])
    assert exit_code == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["ok"] is True
    assert captured["data"]["pong"] is True


def test_cli_main_one_shot_failure_returns_nonzero(capsys):
    exit_code = bridge.cli_main(["not_a_real_command"])
    assert exit_code == 1


def test_cli_main_key_value_args_become_request_fields(capsys):
    exit_code = bridge.cli_main(["is_downloaded", "model=not-a-real-model"])
    assert exit_code == 1
    captured = json.loads(capsys.readouterr().out)
    assert captured["code"] == "HPC-CFG-001"


def test_cli_main_bad_arg_format_returns_usage_error(capsys):
    exit_code = bridge.cli_main(["chat", "not-a-key-value-pair"])
    assert exit_code == 2
