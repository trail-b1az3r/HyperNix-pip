"""The iOS app's version, and the three places it has to agree.

`ios/scripts/app_version.py` exists so the app and the server quote the
same string in a support question: it reads `T1_VERSION` and prints the
short spelling. `ios.yml` calls it and passes the result to xcodebuild as
`MARKETING_VERSION`.

And none of that reached the app. `project.yml` set

    CFBundleShortVersionString: "1.0.26"

as a *literal*. Overriding the `MARKETING_VERSION` build setting on the
command line cannot change a plist key that does not reference it, so
every build shipped reporting `1.0.26` whatever CI had computed —
including a build reporting a version that the script, the workflow and
the T1 source all agreed was something else. `CFBundleVersion` on the
very next line already used `$(CURRENT_PROJECT_VERSION)`; this one
simply never got the same treatment.

The tests here are the wiring, not the number: that the plist references
the build setting rather than restating it, and that the fallback baked
into `project.yml` still matches what the script computes. A default
nobody checks goes stale the first time the T1 version moves, which is
how it got wrong in the first place.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_YML = REPO_ROOT / "ios" / "project.yml"
APP_VERSION = REPO_ROOT / "ios" / "scripts" / "app_version.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ios.yml"


def _spec() -> str:
    return PROJECT_YML.read_text(encoding="utf-8")


def _computed_version() -> str:
    result = subprocess.run(
        [sys.executable, str(APP_VERSION)],
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=True,
    )
    return result.stdout.strip()


class TestTheScriptItself:
    def test_it_prints_the_t1_version_in_short_form(self):
        from hypernix.t1api.version import T1_VERSION_SHORT

        assert _computed_version() == T1_VERSION_SHORT

    def test_it_is_a_six_part_version(self):
        """api.major.year.month.feature.fix — the shape the server reports."""
        assert re.fullmatch(r"\d+\.\d+\.\d+\.\d+\.\d+\.\d+", _computed_version())


class TestThePlistUsesTheBuildSetting:
    def test_the_short_version_references_marketing_version(self):
        """Not a literal. A literal cannot be overridden by xcodebuild."""
        match = re.search(
            r"^\s*CFBundleShortVersionString:\s*(.+)$", _spec(), re.M
        )

        assert match is not None, "no CFBundleShortVersionString in project.yml"
        assert "$(MARKETING_VERSION)" in match.group(1), match.group(1)

    def test_the_build_number_still_references_its_setting(self):
        """The line that was already right, so a fix cannot break it."""
        match = re.search(r"^\s*CFBundleVersion:\s*(.+)$", _spec(), re.M)

        assert match is not None
        assert "$(CURRENT_PROJECT_VERSION)" in match.group(1)

    def test_no_hardcoded_version_literal_remains(self):
        """A three-part literal anywhere in the plist block is the bug."""
        block = _spec()
        block = block[block.index("properties:"):block.index("settings:")]
        literals = re.findall(
            r"^\s*CFBundle\w*Version\w*:\s*\"(\d[\d.]*)\"", block, re.M
        )

        assert not literals, f"hardcoded version literal(s): {literals}"


class TestTheFallbackDoesNotGoStale:
    def test_the_default_matches_what_the_script_computes(self):
        """CI overrides this, but a local Xcode build uses it, and a
        default that drifts is what shipped the wrong number before."""
        match = re.search(r"^\s*MARKETING_VERSION:\s*\"([^\"]+)\"", _spec(), re.M)

        assert match is not None, "no MARKETING_VERSION in project.yml"
        assert match.group(1) == _computed_version(), (
            f"project.yml says {match.group(1)}, app_version.py computes "
            f"{_computed_version()} — update project.yml"
        )


class TestTheWorkflowStillPassesIt:
    @pytest.mark.skipif(not WORKFLOW.exists(), reason="no ios workflow")
    def test_ci_computes_the_version_from_the_script(self):
        source = WORKFLOW.read_text(encoding="utf-8")

        assert "ios/scripts/app_version.py" in source

    @pytest.mark.skipif(not WORKFLOW.exists(), reason="no ios workflow")
    def test_ci_passes_it_to_xcodebuild(self):
        source = WORKFLOW.read_text(encoding="utf-8")

        assert "MARKETING_VERSION=" in source
