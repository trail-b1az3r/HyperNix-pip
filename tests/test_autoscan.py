"""hypernix.system.autoscan — the twice-weekly sweep and its release gate.

The gate is the part that matters. This code can publish to PyPI under
the maintainer's name, so every clause that says "no" is tested for
saying no, and the two that decide *whether it is even allowed to ask*
— a pre-release upstream, and an unresolved security finding — are
tested hardest.

The scanner's own history is in here too. Three bugs, all found by
running it against this repository rather than against fixtures:

1. `silent-except` reported 132 hits because it flagged every
   `except: pass` while its message asked for a comment it never
   looked for.
2. The security scan reported three TLS findings and all three were
   prose — two were its own pattern definitions matching themselves.
3. The fix for (2) skipped any line containing a string, which hid two
   real `torch.load(..., weights_only=False)` calls behind their own
   `map_location="cpu"` argument.

Each has a test named after it, because each was invisible until the
scanner was pointed at real code.
"""
from __future__ import annotations

import textwrap

import pytest

from hypernix.system.autoscan import (
    Finding,
    ReleaseDecision,
    ScanReport,
    _parse_verdict,
    advise,
    is_stable_version,
    next_post_version,
    scan_python,
    scan_security,
    should_release,
)


def write(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


class TestStableVersions:
    @pytest.mark.parametrize(
        "version", ["1.2.3", "0.72.5", "1.0", "0.72.5.post13", "v1.2.3"]
    )
    def test_a_plain_release_is_stable(self, version):
        assert is_stable_version(version)

    @pytest.mark.parametrize(
        "version",
        ["1.0b1", "1.0a2", "1.0rc1", "0.72.5.dev3", "1.0.0-beta", "1.0+local"],
    )
    def test_a_pre_release_is_not(self, version):
        assert not is_stable_version(version)

    def test_post_increments(self):
        assert next_post_version("0.72.5") == "0.72.5.post1"
        assert next_post_version("0.72.5.post13") == "0.72.5.post14"

    def test_a_post_of_a_pre_release_is_refused(self):
        """`1.0b1.post1` sorts *below* `1.0`, so it would publish into a
        lineage nobody installs — a release that reaches no one."""
        with pytest.raises(ValueError, match="sort below"):
            next_post_version("1.0b1")


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def report_with(*, fixed: int = 5, security: int = 0) -> ScanReport:
    report = ScanReport()
    report.fixed = [f"file{i}.py: X{i}" for i in range(fixed)]
    for index in range(security):
        report.findings.append(
            Finding("security", "eval-exec", f"s{index}.py", 1, "eval on a value")
        )
    return report


class TestTheReleaseGate:
    def test_it_releases_when_everything_holds(self):
        decision = should_release(report_with(fixed=5), "0.72.5")
        assert decision.release
        assert decision.version == "0.72.5.post1"

    def test_a_security_finding_blocks_it(self):
        """Shipping faster is not the correct response to finding a
        vulnerability."""
        decision = should_release(report_with(fixed=50, security=1), "0.72.5")
        assert not decision.release
        assert "security" in decision.reason

    def test_security_outranks_every_other_reason(self):
        """The reader needs to know there is a vulnerability, not that
        the bot was below its fix threshold."""
        decision = should_release(report_with(fixed=0, security=1), "1.0b1")
        assert "security" in decision.reason

    @pytest.mark.parametrize("latest", ["1.0b1", "0.72.5.dev3", "1.0rc2", "1.0a1"])
    def test_a_pre_release_upstream_blocks_it(self, latest):
        decision = should_release(report_with(fixed=50), latest)
        assert not decision.release
        assert "pre-release" in decision.reason

    def test_too_few_fixes_blocks_it(self):
        """One import reorder is not a release, and a bot that releases
        for one is a bot people turn off."""
        decision = should_release(report_with(fixed=1), "0.72.5")
        assert not decision.release
        assert "threshold" in decision.reason

    def test_the_refusal_always_carries_a_reason(self):
        for latest, fixed, security in (
            ("1.0b1", 9, 0), ("1.0", 0, 0), ("1.0", 9, 1),
        ):
            decision = should_release(report_with(fixed=fixed, security=security), latest)
            assert not decision.release
            assert len(decision.reason) > 20, decision

    def test_a_decision_prints_readably(self):
        assert "no release" in str(ReleaseDecision(False, "because"))
        assert "1.0.post1" in str(ReleaseDecision(True, "ok", "1.0.post1"))


# ---------------------------------------------------------------------------
# The bug scanner
# ---------------------------------------------------------------------------


class TestBugScanning:
    def test_it_finds_a_bare_except(self, tmp_path):
        write(tmp_path, "a.py", """
            def f():
                try:
                    g()
                except:
                    h()
        """)
        rules = {f.rule for f in scan_python(tmp_path)}
        assert "bare-except" in rules

    def test_it_finds_a_mutable_default(self, tmp_path):
        write(tmp_path, "a.py", """
            def f(items=[]):
                return items
        """)
        assert any(f.rule == "mutable-default" for f in scan_python(tmp_path))

    def test_it_finds_is_against_a_literal(self, tmp_path):
        write(tmp_path, "a.py", """
            def f(x):
                return x is "hello"
        """)
        assert any(f.rule == "is-literal" for f in scan_python(tmp_path))

    def test_is_none_and_is_true_are_fine(self, tmp_path):
        """`is None` is the correct idiom; flagging it would be the
        fastest possible way to get this scanner ignored."""
        write(tmp_path, "a.py", """
            def f(x):
                return x is None or x is True
        """)
        assert not any(f.rule == "is-literal" for f in scan_python(tmp_path))

    def test_an_explained_swallow_is_not_reported(self, tmp_path):
        """Bug 1: this reported 132 hits on the real tree because it
        flagged every `except: pass` while its message asked for a
        comment it never looked for. A swallow with a reason is a
        decision; without one it is a guess."""
        write(tmp_path, "a.py", """
            def f():
                try:
                    g()
                except OSError:
                    # Best effort: the cache is optional.
                    pass
        """)
        assert not any(f.rule == "silent-except" for f in scan_python(tmp_path))

    def test_an_unexplained_swallow_is_reported(self, tmp_path):
        write(tmp_path, "a.py", """
            def f():
                try:
                    g()
                except OSError:
                    pass
        """)
        assert any(f.rule == "silent-except" for f in scan_python(tmp_path))

    def test_an_unparseable_file_is_a_finding_not_a_crash(self, tmp_path):
        write(tmp_path, "broken.py", "def f(:\n")
        assert any(f.rule == "unparseable" for f in scan_python(tmp_path))


# ---------------------------------------------------------------------------
# The security scanner
# ---------------------------------------------------------------------------


class TestSecurityScanning:
    def test_it_finds_shell_true(self, tmp_path):
        write(tmp_path, "a.py", """
            import subprocess
            def f(name):
                subprocess.run("ls " + name, shell=True)
        """)
        assert any(f.rule == "shell-injection" for f in scan_security(tmp_path))

    def test_it_finds_disabled_tls(self, tmp_path):
        write(tmp_path, "a.py", """
            import requests
            def f():
                return requests.get("https://x", verify=False)
        """)
        assert any(f.rule == "tls-disabled" for f in scan_security(tmp_path))

    def test_it_finds_an_unsafe_torch_load(self, tmp_path):
        write(tmp_path, "a.py", """
            import torch
            def f(p):
                return torch.load(p, weights_only=False)
        """)
        assert any(f.rule == "unsafe-pickle" for f in scan_security(tmp_path))

    def test_a_safe_torch_load_is_left_alone(self, tmp_path):
        write(tmp_path, "a.py", """
            import torch
            def f(p):
                return torch.load(p, map_location="cpu", weights_only=True)
        """)
        assert not any(f.rule == "unsafe-pickle" for f in scan_security(tmp_path))

    def test_prose_about_a_pattern_is_not_a_finding(self, tmp_path):
        """Bug 2: three TLS findings on the real tree, all prose — two
        were the scanner's own pattern definitions matching themselves,
        one was a docstring advising *against* `verify=False`."""
        write(tmp_path, "a.py", '''
            """Point ca_certs at your CA rather than reaching for verify=False."""
            PATTERN = r"verify\\s*=\\s*False|CERT_NONE"

            def f():
                return 1
        ''')
        assert not scan_security(tmp_path)

    def test_a_string_argument_does_not_hide_the_call(self, tmp_path):
        """Bug 3, and the worse of the two. The fix for bug 2 skipped
        any line containing a string, so
        `torch.load(p, map_location="cpu", weights_only=False)` was
        hidden behind its own `"cpu"`. Two real findings disappeared.
        Hiding a true finding to suppress a false one is the worse
        trade."""
        write(tmp_path, "a.py", """
            import torch
            def f(p):
                return torch.load(p, map_location="cpu", weights_only=False)
        """)
        found = scan_security(tmp_path)
        assert any(f.rule == "unsafe-pickle" for f in found), found

    def test_a_comment_is_not_a_finding(self, tmp_path):
        write(tmp_path, "a.py", """
            def f():
                # never use shell=True here
                return 1
        """)
        assert not scan_security(tmp_path)


# ---------------------------------------------------------------------------
# The advisor
# ---------------------------------------------------------------------------


class TestTheAdvisor:
    def test_no_api_key_means_do_not_publish(self, monkeypatch):
        """A scanner that treats "nothing reviewed it" as approval is a
        scanner that publishes on an outage."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        verdict = advise(ScanReport(), "", ReleaseDecision(True, "ok", "1.0.post1"))
        assert verdict["publish"] is False
        assert "unreviewed" in verdict["concerns"]

    def test_unparseable_advice_means_do_not_publish(self):
        assert _parse_verdict("I think it's fine, honestly")["publish"] is False

    def test_broken_json_means_do_not_publish(self):
        assert _parse_verdict('{"publish": tru')["publish"] is False

    def test_a_clear_yes_is_read(self):
        verdict = _parse_verdict(
            'Here you go: {"publish": true, "confidence": "high", '
            '"reasoning": "All formatting.", "concerns": []}'
        )
        assert verdict["publish"] is True
        assert verdict["confidence"] == "high"

    def test_a_clear_no_is_read(self):
        assert _parse_verdict('{"publish": false}')["publish"] is False

    def test_the_executor_and_advisor_are_a_valid_pair(self):
        """The advisor must be at least as capable as the executor or
        the API rejects the pairing with a 400."""
        from hypernix.system.autoscan import ADVISOR_MODEL, EXECUTOR_MODEL

        assert EXECUTOR_MODEL == "claude-sonnet-5"
        assert ADVISOR_MODEL == "claude-opus-5"


class TestAgainstThisRepository:
    """The scanner has to survive the tree it ships in."""

    def test_it_runs_over_the_real_source_without_crashing(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "src"
        bugs = scan_python(root)
        security = scan_security(root)
        assert isinstance(bugs, list) and isinstance(security, list)

    def test_it_does_not_report_its_own_patterns(self):
        """The module defines every pattern it looks for, as strings.
        If it reports itself, it reports itself forever."""
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "src"
        offenders = [
            f for f in scan_security(root) if f.path.endswith("autoscan.py")
        ]
        assert not offenders, offenders
