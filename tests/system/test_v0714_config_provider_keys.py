"""Tests for hypernix.config's provider API key storage
(get_provider_key/set_provider_key/clear_provider_key) -- used by hyped-pro's
/key command and the GUI's settings tab.
"""
from __future__ import annotations

import pytest

from hypernix import config


@pytest.fixture(autouse=True)
def _isolated_config_file(tmp_path, monkeypatch):
    """Redirect config.py's module-level _CONFIG_DIR/_CONFIG_FILE directly.

    These are computed once from Path.home() at import time, not
    re-resolved per call -- monkeypatching the HOME env var after import
    has no effect on them, so isolation has to patch the already-bound
    path objects themselves instead.
    """
    config_dir = tmp_path / ".hypernix"
    monkeypatch.setattr(config, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "_CONFIG_FILE", config_dir / "config.json")


def test_set_then_get_provider_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config.set_provider_key("anthropic", "sk-ant-abc123")
    assert config.get_provider_key("anthropic") == "sk-ant-abc123"


def test_env_var_takes_priority_over_persisted_key(monkeypatch):
    config.set_provider_key("anthropic", "sk-ant-persisted")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    assert config.get_provider_key("anthropic") == "sk-ant-from-env"


def test_get_unset_provider_key_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert config.get_provider_key("openai") is None


def test_clear_provider_key_removes_persisted_value(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config.set_provider_key("openai", "sk-test")
    config.clear_provider_key("openai")
    assert config.get_provider_key("openai") is None


def test_clear_provider_key_does_not_touch_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    config.set_provider_key("openai", "sk-persisted")
    config.clear_provider_key("openai")
    # env var still wins even after the persisted copy is cleared
    assert config.get_provider_key("openai") == "sk-from-env"


def test_provider_keys_for_different_vendors_are_independent(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config.set_provider_key("anthropic", "sk-ant-1")
    config.set_provider_key("openai", "sk-oai-1")
    assert config.get_provider_key("anthropic") == "sk-ant-1"
    assert config.get_provider_key("openai") == "sk-oai-1"


def test_provider_env_vars_cover_all_cloud_vendors():
    # Every cloud vendor hyped_pro_core knows about must have a real env
    # var mapping here, or /key's env-var hint would silently be wrong.
    from hypernix import hyped_pro_core as core
    for vendor, info in core.PROVIDERS.items():
        if info.kind == "cloud" or vendor == "t1":
            assert vendor in config.PROVIDER_ENV_VARS, f"{vendor}: missing from PROVIDER_ENV_VARS"
