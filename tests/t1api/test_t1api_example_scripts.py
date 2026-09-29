"""``examples/t1api/*.sh`` — the deployment scripts people actually run.

These are the first thing a new operator executes, and shell is otherwise
uncovered by this suite. The checks here are about the failures that are
silent or misleading rather than loud.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from shell_support import (
    BASH,
    NO_BASH_REASON,
    NO_PYTHON3_REASON,
    python3_path_entry,
    shell_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples" / "t1api"
LOCAL = EXAMPLES / "run_local.sh"
TAILSCALE = EXAMPLES / "run_tailscale.sh"

pytestmark = pytest.mark.skipif(BASH is None, reason=NO_BASH_REASON)


@pytest.fixture
def fake_tailscale(tmp_path) -> Path:
    """A `tailscale` that reports an address, so the script gets past it."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "tailscale"
    stub.write_text("#!/bin/sh\necho 100.64.1.23\n", encoding="utf-8")
    stub.chmod(0o755)
    return bindir


@pytest.fixture(scope="module")
def no_fastapi(tmp_path_factory) -> Path:
    """A PYTHONPATH entry that makes ``import fastapi`` fail.

    Both scripts end in ``exec uvicorn`` and would then serve forever, so
    a test that runs one to completion hangs. Both also preflight the
    [t1api] extra *after* resolving the token secret and *before* exec, so
    failing that import stops them at exactly the right point — and
    exercises the preflight rather than working around it.
    """
    stub = tmp_path_factory.mktemp("stubs")
    # It also records the environment it was imported with. That is the
    # environment the server would have been started in, which is the
    # thing worth asserting on — a trace line only says what the script
    # typed, not what it ended up exporting.
    #
    # Written to a file rather than stderr: the preflight runs the import
    # with `2>/dev/null`, so anything on stderr is discarded.
    (stub / "fastapi.py").write_text(
        "import os\n"
        'target = os.environ.get("T1_TEST_REPORT")\n'
        "if target:\n"
        '    with open(target, "w") as handle:\n'
        '        handle.write(os.environ.get("T1_TOKEN_SECRET", ""))\n'
        'raise ImportError("stubbed out so the example script stops before exec")\n',
        encoding="utf-8",
    )
    return stub


def run_script(
    script: Path,
    home: Path,
    bindir: Path | None = None,
    *,
    stubs: Path | None = None,
    **env,
):
    entries = ["/usr/bin", "/bin"]
    python3_entry, _style = python3_path_entry()
    if python3_entry:
        # The scripts call python3 -- to generate a secret, and for the
        # [t1api] preflight -- and the PATH here is deliberately minimal
        # so only the stubs below are visible. On Windows there is no
        # python3 on any PATH, so a shim stands in for it.
        entries.insert(0, python3_entry)
    if bindir:
        entries.insert(0, shell_path(bindir))
    # bash splits PATH on ":" even on Windows, where it is running under
    # MSYS rather than cmd.
    environ = {"PATH": ":".join(entries), "HOME": shell_path(home), **env}
    report = home / "resolved-secret"
    if stubs is not None:
        # PYTHONPATH and the report path are read by the interpreter, not
        # by the shell, so those stay in the platform's own spelling.
        environ["PYTHONPATH"] = str(stubs)
        environ["T1_TEST_REPORT"] = str(report)
        if report.exists():
            report.unlink()
    result = subprocess.run(
        [BASH, "-x", str(script)],
        capture_output=True,
        text=True, encoding="utf-8",
        timeout=90,
        env=environ,
    )
    result.resolved_secret = (  # type: ignore[attr-defined]
        report.read_text(encoding="utf-8") if report.exists() else None
    )
    return result


def resolved_secret(result: subprocess.CompletedProcess) -> str | None:
    """The secret the script actually exported, as the server would see it.

    Recorded by the stub in :func:`no_fastapi` from ``os.environ`` at the
    preflight, so this reflects the exported environment rather than what
    the source happens to say.
    """
    return getattr(result, "resolved_secret", None)


class TestShape:
    @pytest.mark.parametrize("script", [LOCAL, TAILSCALE])
    def test_parses_and_is_executable(self, script):
        assert script.is_file()
        assert os.access(script, os.X_OK), f"{script.name} is not executable"
        subprocess.run([BASH, "-n", str(script)], check=True)

    @pytest.mark.parametrize("script", [LOCAL, TAILSCALE])
    def test_shellcheck_is_clean_if_available(self, script):
        if shutil.which("shellcheck") is None:
            pytest.skip("shellcheck not installed")
        result = subprocess.run(
            ["shellcheck", "--severity=warning", str(script)],
            capture_output=True,
            text=True, encoding="utf-8",
        )
        assert result.returncode == 0, result.stdout + result.stderr


class TestTheErrorMessageIsPasteable:
    """``${VAR:?msg}`` strips the quotes from its own message.

    The tailnet script used to fail with:

        T1_TOKEN_SECRET: set T1_TOKEN_SECRET (python3 -c import secrets;print(...))

    and that command is a syntax error in both the shell and Python,
    because the single quotes around the -c argument were removed on the
    way out. An error whose remedy does not run is worse than no remedy:
    it costs the reader a debugging session to discover the instructions
    were wrong, not them.
    """

    def test_no_quoted_command_inside_a_colon_question_expansion(self):
        for script in (LOCAL, TAILSCALE):
            text = script.read_text(encoding="utf-8")
            for match in re.finditer(r"\$\{[A-Za-z_][A-Za-z0-9_]*:\?([^}]*)\}", text):
                message = match.group(1)
                assert "'" not in message and '"' not in message, (
                    f"{script.name}: quotes in a ${{VAR:?...}} message are stripped "
                    f"before printing, so this arrives unpasteable: {message}"
                )

    def test_the_suggested_commands_actually_run(self, tmp_path, fake_tailscale):
        """Extract every indented command from the failure text and run it.

        This is the test that would have caught the original bug: the
        message looked right in the source and was broken by the time it
        reached the terminal.
        """
        home = tmp_path / "home"
        home.mkdir()
        result = run_script(TAILSCALE, home, fake_tailscale)
        assert result.returncode != 0, "expected a failure with no secret set"

        # The remedy block, as the reader sees it.
        commands = [
            line.strip()
            for line in result.stderr.splitlines()
            if line.startswith("    ") and line.strip()
        ]
        assert commands, "the failure names no command to run"

        # Reassemble continuations, then run the whole block in one shell.
        block = "\n".join(commands)
        assert "python3 -c 'import secrets" in block or 'python3 -c "import secrets' in block, (
            f"the generate command lost its quoting:\n{block}"
        )
        check = subprocess.run(
            [BASH, "-n", "-c", block], capture_output=True, text=True, encoding="utf-8"
        )
        assert check.returncode == 0, (
            f"the suggested commands do not parse:\n{block}\n{check.stderr}"
        )

    def test_the_message_says_how_to_set_it_not_just_that_it_is_unset(
        self, tmp_path, fake_tailscale
    ):
        home = tmp_path / "home"
        home.mkdir()
        result = run_script(TAILSCALE, home, fake_tailscale)
        assert "T1_TOKEN_SECRET is not set" in result.stderr
        assert "export T1_TOKEN_SECRET" in result.stderr
        assert "install-t1.sh" in result.stderr, (
            "the installer writes this file; the message should say so"
        )


@pytest.mark.skipif(python3_path_entry()[0] is None, reason=NO_PYTHON3_REASON)
class TestSecretResolution:
    """Environment, then the installer's .env, then generate-or-fail.

    Every test here reads what the script exported at the [t1api]
    preflight, which means the script has to get as far as running
    python3. Skipped rather than failed where it cannot: "the secret was
    None" is a true statement about a run that never happened, and it
    reads as a bug in the secret resolution it is not about.
    """

    @staticmethod
    def _write_env(home: Path, value: str) -> None:
        cfg = home / ".hypernix" / "t1api"
        cfg.mkdir(parents=True, exist_ok=True)
        (cfg / ".env").write_text(f"T1_TOKEN_SECRET={value}\n", encoding="utf-8")

    def test_tailscale_uses_the_installers_env(self, tmp_path, fake_tailscale, no_fastapi):
        """The bug behind the bug.

        install-t1.sh writes a stable secret into ~/.hypernix/t1api/.env
        and this script never looked at it, so an operator who had just
        run the installer was still told to go and make one.
        """
        home = tmp_path / "home"
        home.mkdir()
        self._write_env(home, "secret_from_dot_env")
        result = run_script(TAILSCALE, home, fake_tailscale, stubs=no_fastapi)
        assert resolved_secret(result) == "secret_from_dot_env"

    def test_the_environment_wins_over_the_file(self, tmp_path, fake_tailscale, no_fastapi):
        home = tmp_path / "home"
        home.mkdir()
        self._write_env(home, "secret_from_dot_env")
        result = run_script(
            TAILSCALE,
            home,
            fake_tailscale,
            stubs=no_fastapi,
            T1_TOKEN_SECRET="secret_from_env",
        )
        assert resolved_secret(result) == "secret_from_env"

    @pytest.mark.parametrize(
        "stored,expected",
        [
            ("plainvalue", "plainvalue"),
            ("'singlequoted'", "singlequoted"),
            ('"doublequoted"', "doublequoted"),
        ],
    )
    def test_a_quoted_value_is_unwrapped(
        self, tmp_path, fake_tailscale, no_fastapi, stored, expected
    ):
        """install-t1.sh quotes some values; a quoted secret must still work.

        A secret carrying literal quote characters would not match the one
        the server signs with, and every token would fail to verify with
        no indication why.
        """
        home = tmp_path / "home"
        home.mkdir()
        self._write_env(home, stored)
        result = run_script(TAILSCALE, home, fake_tailscale, stubs=no_fastapi)
        assert resolved_secret(result) == expected

    def test_local_reuses_a_stable_secret_across_runs(self, tmp_path, no_fastapi):
        """Otherwise every restart silently invalidates every scoped token."""
        home = tmp_path / "home"
        home.mkdir()
        self._write_env(home, "stable_local_secret")
        first = resolved_secret(run_script(LOCAL, home, stubs=no_fastapi))
        second = resolved_secret(run_script(LOCAL, home, stubs=no_fastapi))
        assert first == second == "stable_local_secret"

    def test_local_still_generates_when_there_is_no_file(self, tmp_path, no_fastapi):
        """The documented zero-setup path must keep working."""
        home = tmp_path / "home"
        home.mkdir()
        first = resolved_secret(run_script(LOCAL, home, stubs=no_fastapi))
        second = resolved_secret(run_script(LOCAL, home, stubs=no_fastapi))
        assert first and second and first != second
        assert len(first) == 64, "expected 32 bytes of hex"

    def test_the_file_is_read_not_sourced(self):
        """Sourcing runs whatever is in the file and imports every setting.

        This script is asking for one value, and .env is a file an
        operator edits by hand.
        """
        for script in (LOCAL, TAILSCALE):
            text = script.read_text(encoding="utf-8")
            assert not re.search(r"^\s*(\.|source)\s+.*\.env", text, re.M), (
                f"{script.name} sources .env instead of reading the one value"
            )


class TestPreflight:
    def test_both_scripts_check_for_the_http_extra(self):
        """The tailnet script used to reach uvicorn before finding out.

        run_local.sh checked and run_tailscale.sh did not, so the same
        missing dependency produced a clear message on one path and a
        ModuleNotFoundError on the other.
        """
        for script in (LOCAL, TAILSCALE):
            text = script.read_text(encoding="utf-8")
            assert "import fastapi" in text, f"{script.name} has no [t1api] preflight"
            assert "hypernix[t1api]" in text
