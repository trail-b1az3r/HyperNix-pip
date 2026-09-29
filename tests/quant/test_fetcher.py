"""Tests for hypernix.fetcher's asset picker + release walk-back."""
from __future__ import annotations

from unittest import mock

import pytest


def test_pick_asset_prefers_matching_cpu_ubuntu_zip():
    from hypernix import fetcher

    assets = [
        {"name": "llama-b9000-bin-ubuntu-x64.zip", "browser_download_url": "u1"},
        {"name": "llama-b9000-bin-ubuntu-x64-cuda.zip", "browser_download_url": "u2"},
        {"name": "llama-b9000-bin-macos-x64.zip", "browser_download_url": "u3"},
        {"name": "llama-b9000-bin-win-x64.zip", "browser_download_url": "u4"},
    ]
    with mock.patch.object(fetcher, "_detect_asset_tokens",
                           return_value=("ubuntu", ["x64", "x86_64", "amd64"])):
        chosen = fetcher._pick_asset(assets)
    assert chosen is not None
    # Must be the pure CPU ubuntu asset; the cuda one must be rejected.
    assert chosen["name"] == "llama-b9000-bin-ubuntu-x64.zip"


def test_pick_asset_rejects_when_no_match():
    from hypernix import fetcher

    assets = [
        {"name": "llama-b9000-bin-ubuntu-x64-cuda.zip", "browser_download_url": "u1"},
        {"name": "llama-b9000-bin-ubuntu-arm64.zip", "browser_download_url": "u2"},
    ]
    with mock.patch.object(fetcher, "_detect_asset_tokens",
                           return_value=("ubuntu", ["x64", "x86_64", "amd64"])):
        chosen = fetcher._pick_asset(assets)
    assert chosen is None


def test_fetch_walks_back_when_latest_has_no_match(tmp_path, monkeypatch):
    """The reproduction of the user's bug: latest release b8863 has no CPU
    asset, older release b8850 does. fetch_llama_quantize() must fall through
    to the older one instead of raising."""
    from hypernix import fetcher

    latest = {
        "tag_name": "b8863",
        "assets": [{"name": "llama-b8863-bin-ubuntu-x64-cuda.zip",
                    "browser_download_url": "no"}],
    }
    older = {
        "tag_name": "b8850",
        "assets": [
            {"name": "llama-b8850-bin-ubuntu-x64.zip",
             "browser_download_url": "http://fake/b8850.zip",
             "size": 1234},
        ],
    }
    monkeypatch.setattr(fetcher, "_recent_releases", lambda limit=10: [latest, older])
    monkeypatch.setattr(fetcher, "cache_dir", lambda: tmp_path / "bin")
    monkeypatch.setattr(fetcher, "cached_binary", lambda: None)
    monkeypatch.setattr(fetcher, "_detect_asset_tokens",
                        lambda: ("ubuntu", ["x64", "x86_64", "amd64"]))

    # Stub network + extraction. We record which URL was fetched so we can
    # confirm the resolver moved on to b8850 after b8863 had no match.
    fetched: dict[str, str] = {}

    def fake_download(url: str):
        fetched["url"] = url
        dest = tmp_path / "dl.zip"
        dest.write_bytes(b"x")
        return dest

    def fake_extract(zip_path, target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        bin_path = target_dir / "llama-quantize"
        bin_path.write_text("#!/bin/sh\n")
        bin_path.chmod(0o755)
        return bin_path

    monkeypatch.setattr(fetcher, "_download_to_temp", fake_download)
    monkeypatch.setattr(fetcher, "_extract_binary", fake_extract)

    result = fetcher.fetch_llama_quantize(quiet=True)
    assert result.name == "llama-quantize"
    assert fetched["url"] == "http://fake/b8850.zip"


def test_fetch_raises_helpful_error_when_no_release_has_match(monkeypatch):
    from hypernix import fetcher

    releases = [
        {"tag_name": "b1", "assets": [{"name": "foo-cuda.zip", "browser_download_url": "u"}]},
        {"tag_name": "b2", "assets": []},
    ]
    monkeypatch.setattr(fetcher, "_recent_releases", lambda limit=10: releases)
    monkeypatch.setattr(fetcher, "cached_binary", lambda: None)
    monkeypatch.setattr(fetcher, "_detect_asset_tokens",
                        lambda: ("ubuntu", ["x64", "x86_64", "amd64"]))

    with pytest.raises(RuntimeError, match="No CPU-only asset"):
        fetcher.fetch_llama_quantize(quiet=True)
