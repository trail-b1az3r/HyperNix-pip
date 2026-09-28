"""Tests for hypernix.multilama: the backend registry, auto-selection
heuristic, and GitHub-release asset-picking logic behind hyped-pro's GGUF
dispatch (vanilla / ik_llama.cpp / PrismML / KoboldCpp / Nanbeige backends).
"""
from __future__ import annotations

import pytest

from hypernix import multilama as ml
from hypernix.multilama import MultiLlamaError

# ---------------------------------------------------------------------------
# Backend registry integrity
# ---------------------------------------------------------------------------

def test_all_backends_have_a_valid_kind():
    for name, backend in ml.BACKENDS.items():
        assert backend.kind in ("python-binding", "server-binary"), name


def test_vanilla_is_python_binding():
    assert ml.BACKENDS["vanilla"].kind == "python-binding"


def test_server_binary_backends_have_binary_names():
    for name, backend in ml.BACKENDS.items():
        if backend.kind == "server-binary":
            assert backend.binary_names, f"{name}: server-binary backend with no binary_names"


def test_prismml_has_no_prebuilt_releases_and_a_build_hint():
    # This is a deliberate, documented fact -- not an oversight. If this
    # ever flips to True without a real prebuilt release existing, fetch
    # would silently try (and fail) against a nonexistent asset instead of
    # giving the real build-from-source instructions.
    prismml = ml.BACKENDS["prismml"]
    assert prismml.has_prebuilt_releases is False
    assert prismml.build_hint
    assert "nanbeige42" not in prismml.build_hint  # sanity: not copy-pasted from the wrong backend


def test_nanbeige_has_no_prebuilt_releases_and_correct_branch():
    nanbeige = ml.BACKENDS["nanbeige"]
    assert nanbeige.has_prebuilt_releases is False
    assert nanbeige.branch == "nanbeige42"
    assert "nanbeige42" in nanbeige.build_hint


def test_get_backend_unknown_raises_coded_error():
    with pytest.raises(MultiLlamaError) as exc_info:
        ml.get_backend("not-a-real-backend")
    assert exc_info.value.code == "ML-CFG-001"


def test_get_backend_known_resolves():
    assert ml.get_backend("ik").name == "ik"


# ---------------------------------------------------------------------------
# auto_select_backend heuristic
# ---------------------------------------------------------------------------

def test_auto_select_bonsai_routes_to_prismml():
    assert ml.auto_select_backend("prism-ml/Bonsai-27B-gguf", "Bonsai-27B-Q1_0.gguf") == "prismml"


def test_auto_select_nanbeige_routes_to_nanbeige_backend():
    assert ml.auto_select_backend("owao/Nanbeige4.2-3B-GGUF", "Nanbeige4.2-3B-Q4_K_M.gguf") == "nanbeige"


def test_auto_select_standard_model_routes_to_vanilla():
    assert ml.auto_select_backend("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf") == "vanilla"


def test_auto_select_is_case_insensitive():
    assert ml.auto_select_backend("PRISM-ML/Bonsai-27B-gguf", "BONSAI-27B-Q1_0.GGUF") == "prismml"


# ---------------------------------------------------------------------------
# fetch_binary / ensure_binary error paths (no real network needed)
# ---------------------------------------------------------------------------

def test_fetch_binary_on_python_binding_backend_raises():
    with pytest.raises(MultiLlamaError) as exc_info:
        ml.fetch_binary(ml.get_backend("vanilla"))
    assert exc_info.value.code == "ML-CFG-001"


def test_fetch_binary_no_prebuilt_releases_raises_with_build_hint(monkeypatch):
    # Ensure locate_binary can't accidentally find a real one on this
    # machine and short-circuit the "no prebuilt releases" path.
    monkeypatch.setattr(ml, "locate_binary", lambda backend: None)
    with pytest.raises(MultiLlamaError) as exc_info:
        ml.fetch_binary(ml.get_backend("prismml"))
    assert exc_info.value.code == "ML-FETCH-001"
    assert "cmake" in exc_info.value.message


# ---------------------------------------------------------------------------
# Asset picking (GitHub release asset selection) -- pure logic, no network
# ---------------------------------------------------------------------------

def test_pick_asset_prefers_matching_os_and_arch(monkeypatch):
    monkeypatch.setattr(ml, "_detect_asset_tokens", lambda: ("ubuntu", ["x64", "x86_64", "amd64"]))
    assets = [
        {"name": "llama-server-macos-arm64.zip"},
        {"name": "llama-server-ubuntu-x64.zip"},
        {"name": "llama-server-win-x64.zip"},
    ]
    picked = ml._pick_asset(assets, ("llama-server",))
    assert picked["name"] == "llama-server-ubuntu-x64.zip"


def test_pick_asset_excludes_gpu_variants_on_generic_search(monkeypatch):
    monkeypatch.setattr(ml, "_detect_asset_tokens", lambda: ("ubuntu", ["x64", "x86_64", "amd64"]))
    assets = [
        {"name": "llama-server-ubuntu-x64-cuda.zip"},
        {"name": "llama-server-ubuntu-x64.zip"},
    ]
    picked = ml._pick_asset(assets, ("llama-server",))
    assert picked["name"] == "llama-server-ubuntu-x64.zip"


def test_pick_asset_returns_none_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(ml, "_detect_asset_tokens", lambda: ("ubuntu", ["x64", "x86_64", "amd64"]))
    assets = [{"name": "llama-server-macos-arm64.zip"}]
    assert ml._pick_asset(assets, ("llama-server",)) is None


def test_pick_asset_ignores_non_zip_assets(monkeypatch):
    monkeypatch.setattr(ml, "_detect_asset_tokens", lambda: ("ubuntu", ["x64", "x86_64", "amd64"]))
    assets = [{"name": "llama-server-ubuntu-x64.tar.gz"}, {"name": "checksums.txt"}]
    assert ml._pick_asset(assets, ("llama-server",)) is None


# ---------------------------------------------------------------------------
# MultiLlamaError basics
# ---------------------------------------------------------------------------

def test_multilama_error_has_code_and_message():
    err = MultiLlamaError("ML-TEST-001", "something went wrong")
    assert err.code == "ML-TEST-001"
    assert err.message == "something went wrong"
    assert "ML-TEST-001" in str(err)


def test_llama_backend_is_frozen_dataclass():
    import dataclasses
    backend = ml.BACKENDS["vanilla"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        backend.name = "renamed"  # type: ignore[misc]
