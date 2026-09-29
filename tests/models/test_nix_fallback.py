"""Tests for the ``nix`` short-name fallback chain added in 0.46.1."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def test_nix_short_name_resolves_to_2_7a() -> None:
    """`resolve_repo_id("nix")` now returns the latest Nix (2.7a),
    not the older ray0rf1re/Nix2.5."""
    from hypernix import resolve_repo_id

    assert resolve_repo_id("nix") == "Nix-ai/Nix-2.7a"


def test_fallback_chain_registered() -> None:
    from hypernix.download import FALLBACK_CHAINS

    chain = FALLBACK_CHAINS["nix"]
    assert chain[0] == "Nix-ai/Nix-2.7a"
    assert chain[1] == "Nix-ai/Nix2.6-mm"
    assert chain[2] == "ray0rf1re/Nix2.5"


def test_download_model_uses_first_choice_on_success(tmp_path: Path) -> None:
    """When the first repo in the chain resolves, the later entries
    are never touched."""
    import hypernix.download as dl

    target = tmp_path / "Nix-ai__Nix-2.7a"
    target.mkdir()
    (target / "config.json").write_text("{}")

    calls: list[str] = []

    def fake_snapshot(*, repo_id, **kwargs):
        calls.append(repo_id)
        return str(target)

    with (
        patch.object(dl, "snapshot_download", side_effect=fake_snapshot),
        patch.object(dl, "verify_snapshot", return_value=["config.json"]),
    ):
        out = dl.download_model(repo_id="nix", quiet=True)

    assert calls == ["Nix-ai/Nix-2.7a"]
    assert out == target


def test_download_model_falls_through_on_404(tmp_path: Path) -> None:
    """If 2.7a fails, the downloader tries 2.6-mm, then 2.5."""
    import hypernix.download as dl

    target = tmp_path / "nix_2_5"
    target.mkdir()
    (target / "config.json").write_text("{}")

    calls: list[str] = []

    def fake_snapshot(*, repo_id, **kwargs):
        calls.append(repo_id)
        if repo_id != "ray0rf1re/Nix2.5":
            raise RuntimeError(f"simulated 404 for {repo_id}")
        return str(target)

    with (
        patch.object(dl, "snapshot_download", side_effect=fake_snapshot),
        patch.object(dl, "verify_snapshot", return_value=["config.json"]),
    ):
        out = dl.download_model(repo_id="nix", quiet=True)

    assert calls == [
        "Nix-ai/Nix-2.7a",
        "Nix-ai/Nix2.6-mm",
        "ray0rf1re/Nix2.5",
    ]
    assert out == target


def test_download_model_raises_when_all_candidates_fail() -> None:
    """With every repo in the chain unreachable, a RuntimeError is
    raised summarising the failure (chained to the last original
    exception)."""
    import hypernix.download as dl

    def fake_snapshot(*, repo_id, **kwargs):
        raise RuntimeError(f"simulated outage for {repo_id}")

    with (
        patch.object(dl, "snapshot_download", side_effect=fake_snapshot),
        pytest.raises(RuntimeError, match="exhausted|fallback chain"),
    ):
        dl.download_model(repo_id="nix", quiet=True)


def test_explicit_repo_id_does_not_use_fallback_chain(tmp_path: Path) -> None:
    """Passing a full ``org/repo`` id bypasses the fallback chain
    completely — the user gets exactly what they asked for, or an
    error."""
    import hypernix.download as dl

    calls: list[str] = []

    def fake_snapshot(*, repo_id, **kwargs):
        calls.append(repo_id)
        raise RuntimeError("oops")

    with (
        patch.object(dl, "snapshot_download", side_effect=fake_snapshot),
        pytest.raises(RuntimeError),
    ):
        dl.download_model(repo_id="Nix-ai/Nix-2.7a", quiet=True)

    # Only the explicit repo is tried — no fallback to 2.6-mm or 2.5.
    assert calls == ["Nix-ai/Nix-2.7a"]
