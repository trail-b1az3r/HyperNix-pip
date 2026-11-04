"""0.73.0 pre-alpha 1: HyperLink for the Mac (paused until 0.73.2).

An Apple-silicon and an Intel .dmg, built with Mac Catalyst by
ios/scripts/make_macos_dmg.sh in .github/workflows/macos.yml and attached
by public-release.yml. There is no Mac here, so these check the wiring;
the macos workflow builds the real thing on a macOS runner.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ios" / "scripts" / "make_macos_dmg.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "macos.yml"
RELEASE = ROOT / ".github" / "workflows" / "public-release.yml"
PROJECT = ROOT / "ios" / "project.yml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_app_target_supports_mac_catalyst():
    target = _load(PROJECT)["targets"]["HyperLink"]["settings"]["base"]
    assert target["SUPPORTS_MACCATALYST"] in (True, "YES")
    # The iPad idiom, scaled: not the Mac idiom (6), which re-lays out everything.
    assert str(target["TARGETED_DEVICE_FAMILY"]) == "1,2"


def test_both_architectures_are_built():
    jobs = _load(WORKFLOW)["jobs"]
    archs = {entry["arch"] for entry in jobs["dmg"]["strategy"]["matrix"]["include"]}
    assert archs == {"arm64", "x86_64"}
    # The workflow can be called by the release, which is the point.
    assert "workflow_call" in _load(WORKFLOW)[True]  # yaml reads `on:` as True


def test_paused_until_0_73_2():
    """The Mac app is paused until 0.73.2: wired, but neither the release
    nor pull requests build it. Turning it back on means deleting the
    `false &&` and restoring the pull_request trigger -- and this test."""
    condition = _load(RELEASE)["jobs"]["hyperlink-dmg"]["if"]
    assert condition.strip().startswith("false &&")
    assert "pull_request" not in _load(WORKFLOW)[True]
    assert "workflow_dispatch" in _load(WORKFLOW)[True]
    assert "macos-apple-silicon.dmg" not in RELEASE.read_text(encoding="utf-8")


def test_the_release_is_still_wired_to_attach_them():
    release = _load(RELEASE)
    jobs = release["jobs"]
    assert jobs["hyperlink-dmg"]["uses"] == "./.github/workflows/macos.yml"
    assert "hyperlink-dmg" in jobs["github-release"]["needs"]
    text = RELEASE.read_text(encoding="utf-8")
    assert "dist/macos/*.dmg" in text and "dist/macos/SHA256SUMS.macos" in text
    assert "pattern: hyperlink-macos-*" in text


def test_a_missing_dmg_never_holds_the_release():
    condition = _load(RELEASE)["jobs"]["github-release"]["if"]
    assert "hyperlink-dmg" not in condition


def test_the_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@pytest.mark.parametrize("argv,needle", [
    (["armv7", "1.0"], "ARCH must be arm64 or x86_64"),
    (["arm64", "1.0;rm -rf ~"], "does not look like a version"),
    (["arm64", "1.0", "1a"], "BUILD must be a whole number"),
])
def test_bad_arguments_are_refused_before_anything_runs(argv, needle):
    done = subprocess.run(["bash", str(SCRIPT), *argv], capture_output=True, text=True)
    assert done.returncode == 2
    assert needle in done.stderr


def test_it_signs_ad_hoc_and_checks_the_slice():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "codesign --force --deep --sign -" in text
    assert "lipo -archs" in text
    assert "variant=Mac Catalyst" in text
