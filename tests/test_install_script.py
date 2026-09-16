"""``install-t1.sh`` — the interactive T1 API installer.

Shell is not covered by the rest of the suite, and this script is the
first thing a new operator runs: a syntax error or a bash-4-ism in it is
a first impression that cannot be walked back. So the checks here are
deliberately about the properties that break silently.

Nothing here installs anything. The dry run is the whole point: it must
be provably inert.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from shell_support import BASH, NO_BASH_REASON, shell_path


def _names(line: str, path) -> bool:
    """Does *line* name *path*, in whichever spelling the shell used?

    The installer prints the interpreter it chose, as the shell resolved
    it. On Windows that is Git Bash's mount form, ``/c/Users/...``, while
    ``str(Path)`` here is ``C:\\Users\\...`` — so a test comparing the two
    asserts that two spellings of one path are the same string, and fails
    on a run where the script did exactly the right thing:

        assert 'C:\\Users\\...\\bin\\python3' in
               '  ✓ python: /c/Users/.../bin/python3 (3.11.9)'

    Which spelling appears is the shell's business. Which interpreter was
    chosen is the test's, so all the plausible spellings are accepted.
    """
    text = Path(path).as_posix()
    spellings = {str(Path(path)), text, shell_path(path)}
    if len(text) > 2 and text[1] == ":":
        spellings.add(f"/{text[0].lower()}{text[2:]}")
    return any(spelling in line for spelling in spellings)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "install-t1.sh"

pytestmark = pytest.mark.skipif(BASH is None, reason=NO_BASH_REASON)


@pytest.fixture(scope="module")
def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def code(source: str) -> str:
    """``source`` with whole-line comments dropped.

    The script documents the hazards it avoids — an unbounded read from
    ``/dev/urandom``, ``${x,,}`` — and a scanner that reads comments finds
    those descriptions and calls them violations. Only whole-line comments
    are removed: a trailing ``#`` is ambiguous in shell (``#`` is a literal
    inside a string, and this script has several) and guessing there would
    trade one false positive for another.
    """
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


class TestShape:
    def test_the_script_exists_and_is_executable(self):
        assert SCRIPT.is_file()
        # Shipped without the bit set, `curl | bash` still works but
        # `./install-t1.sh` does not, and the second is what the README says.
        assert os.access(SCRIPT, os.X_OK), "install-t1.sh is not executable"

    def test_it_parses(self):
        subprocess.run([BASH, "-n", str(SCRIPT)], check=True)

    def test_it_fails_on_error_and_on_unset(self, source):
        assert "set -euo pipefail" in source

    def test_shellcheck_is_clean_if_available(self):
        if shutil.which("shellcheck") is None:
            pytest.skip("shellcheck not installed")
        result = subprocess.run(
            ["shellcheck", "--severity=warning", str(SCRIPT)],
            capture_output=True,
            text=True, encoding="utf-8",
        )
        assert result.returncode == 0, result.stdout + result.stderr


class TestBash32Portability:
    """macOS ships bash 3.2 and will not ship a newer one.

    An operator on a Mac has to be able to run this without installing a
    shell first, so the bash 4 conveniences are off the table. These are
    the ones that are easy to reach for by accident.
    """

    def test_no_associative_arrays(self, code):
        assert "declare -A" not in code
        assert "local -A" not in code

    def test_no_case_conversion_expansion(self, code):
        assert not re.search(r"\$\{[A-Za-z_][A-Za-z0-9_]*(,,|\^\^)", code)

    def test_no_mapfile_or_readarray(self, code):
        assert not re.search(r"\b(mapfile|readarray)\b", code)

    def test_no_negative_string_index(self, code):
        assert "${x:(-" not in code


class TestNoHeredocRunsCommands:
    """A backtick inside an unquoted heredoc is command substitution.

    The ``.env`` heredoc has to be unquoted -- it expands ``$BIND_HOST``,
    ``$DEFAULT_PLAN`` and a dozen others -- so anything backticked in its
    body *runs*. A comment written as::

        # `hypernix-t1 start` reads them from here

    executes ``hypernix-t1 start`` while the config file is being
    written, and splices its output into the comment. That shipped, and
    Codacy's shellcheck is what caught it.

    This is not a style rule. Command substitution in a heredoc that
    writes a *config file* runs a server as a side effect of writing
    down where the server listens.
    """

    def _heredoc_bodies(self, source: str):
        """Every unquoted-heredoc body in the script, with its delimiter.

        ``<<'EOF'`` and ``<<"EOF"`` are literal and safe; bare ``<<EOF``
        expands. Only the second kind is interesting.
        """
        import re

        bodies = []
        lines = source.splitlines()
        opener = re.compile(r"<<-?\s*(?P<quote>['\"]?)(?P<tag>[A-Za-z_][A-Za-z0-9_]*)\1")
        index = 0
        while index < len(lines):
            found = opener.search(lines[index])
            if found:
                tag = found.group("tag")
                quoted = bool(found.group("quote"))
                start = index + 1
                end = start
                while end < len(lines) and lines[end].strip() != tag:
                    end += 1
                if not quoted:
                    bodies.append((tag, start, lines[start:end]))
                index = end
            index += 1
        return bodies

    def test_the_parser_finds_the_env_heredoc(self, source):
        """Otherwise the test below passes by looking at nothing."""
        tags = {tag for tag, _start, _body in self._heredoc_bodies(source)}
        assert "ENVEOF" in tags, tags

    def test_no_unquoted_heredoc_contains_a_backtick(self, source):
        offenders = []
        for tag, start, body in self._heredoc_bodies(source):
            for offset, line in enumerate(body):
                if "`" in line:
                    offenders.append(f"  {tag} line {start + offset + 1}: {line.strip()}")
        assert not offenders, (
            "backticks inside an unquoted heredoc are command substitution "
            "and will run while the file is written:\n" + "\n".join(offenders)
        )

    def test_the_generated_env_has_no_command_substitution(self, tmp_path):
        """End to end: write a real config and check nothing ran.

        The static check above is the one that scales; this is the one
        that proves the static check is testing the right thing.
        """
        import os
        import subprocess

        marker = tmp_path / "should-not-exist"
        env = {
            **os.environ,
            "HOME": str(tmp_path / "home"),
            "T1_CONFIG_DIR": str(tmp_path / "cfg"),
            "NO_COLOR": "1",
            # If any heredoc runs a command, this is on PATH to catch it.
            "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        }
        (tmp_path / "bin").mkdir(parents=True)
        for name in ("hypernix-t1", "hypernix"):
            shim = tmp_path / "bin" / name
            shim.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 0\n', encoding="utf-8")
            shim.chmod(0o755)

        subprocess.run(
            [BASH, str(SCRIPT), "--non-interactive", "--install", "skip",
             "--dry-run"],
            capture_output=True, text=True, encoding="utf-8", timeout=300, env=env, check=False,
        )
        assert not marker.exists(), (
            "the installer ran a command that only appears inside a heredoc"
        )


class TestNothingHereTrustsTheMachinesLocale:
    """These tests read UTF-8 shell scripts. They must say so.

    ``install-t1.sh`` carries 476 non-ASCII bytes — em dashes, tick
    marks — and prints them. ``Path.read_text()`` and
    ``subprocess.run(text=True)`` both decode with
    ``locale.getpreferredencoding()``, which is UTF-8 on Linux and macOS
    and **cp1252** on Windows. So every one of these calls was correct on
    two thirds of the CI matrix and raised on the third:

        UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f in
        position 2607: character maps to <undefined>

    Same shape as the device-placement bug in ``hnxrun``: an implicit
    default that happens to be right on the machine the tests were
    written on. A locale is not observable from inside a passing test, so
    this is checked at the source level instead — the only place the
    property lives.

    Scoped to the files that drive the shell scripts. Elsewhere a bare
    ``read_text`` may be reading something this repo wrote as ASCII, and
    a blanket rule would be noise rather than a guard.
    """

    #: Test modules that read or run the UTF-8 shell scripts.
    SHELL_TEST_FILES = (
        "test_install_script.py",
        "test_hypernix_t1_service.py",
        "test_t1api_example_scripts.py",
        "test_autofix_scripts.py",
    )

    def _calls(self, module: str):
        import ast

        tree = ast.parse((Path(__file__).parent / module).read_text(
            encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else None
            if name in ("read_text", "write_text"):
                yield node, name, {kw.arg for kw in node.keywords}
            elif name == "run" and any(
                kw.arg == "text" for kw in node.keywords
            ):
                yield node, "run(text=True)", {kw.arg for kw in node.keywords}

    @pytest.mark.parametrize("module", SHELL_TEST_FILES)
    def test_the_audit_finds_something_to_audit(self, module):
        assert len(list(self._calls(module))) >= 2

    @pytest.mark.parametrize("module", SHELL_TEST_FILES)
    def test_every_read_and_capture_names_its_encoding(self, module):
        offenders = [
            f"{module}:{node.lineno} {kind}"
            for node, kind, keywords in self._calls(module)
            if "encoding" not in keywords
        ]
        assert not offenders, (
            "these decode with the machine's locale, which is UTF-8 here "
            "and cp1252 on Windows: " + "; ".join(offenders)
        )


class TestSecretHandling:
    def test_secrets_are_read_with_echo_off(self, source):
        assert "read -rs" in source

    def test_generated_files_are_chmodded_before_they_are_written(self, source):
        # The order matters: a 0644-then-chmod .env is world-readable for
        # as long as it takes to write it, which is long enough.
        write_file = _function_body(source, "write_file")
        # Anchored on the real write. The dry-run branch's `cat > /dev/null`
        # comes first in the function and is not the one that matters.
        chmod_at = write_file.index('chmod "$mode"')
        content_at = write_file.index('cat > "$path"')
        assert chmod_at < content_at, "write_file chmods after writing content"

    def test_env_file_is_0600(self, source):
        body = _function_body(source, "write_env")
        assert "0600" in body

    def test_start_script_is_0755(self, source):
        body = _function_body(source, "write_start_script")
        assert "0755" in body

    def test_no_secret_is_passed_on_a_command_line(self, source):
        # Anything on argv is visible in `ps` to every user on the box.
        # Secrets travel in the environment or on stdin instead.
        for bad in ("--password $ADMIN_PASSWORD", "--password \"$ADMIN_PASSWORD\""):
            assert bad not in source


class TestRandomness:
    def test_urandom_is_bounded_before_it_is_filtered(self, code):
        """``tr < /dev/urandom | head`` dies of SIGPIPE under pipefail.

        ``head`` closes the pipe while ``tr`` is still writing, ``tr``
        takes SIGPIPE, and ``set -o pipefail`` turns that into a failed
        pipeline that kills the installer partway through. Reading a
        bounded chunk first and trimming with ``cut`` has no such race.
        """
        for line in code.splitlines():
            if "/dev/urandom" not in line:
                continue
            assert "head -c" in line, f"unbounded read from urandom: {line.strip()}"
            assert "| head" not in line.split("/dev/urandom", 1)[1], (
                f"head downstream of urandom filter (SIGPIPE): {line.strip()}"
            )


class TestDryRun:
    def test_dry_run_writes_nothing_and_succeeds(self, tmp_path):
        target = tmp_path / "config"
        result = subprocess.run(
            [
                BASH,
                str(SCRIPT),
                "--dry-run",
                "--non-interactive",
                "--config-dir",
                str(target),
                "--install",
                "skip",
            ],
            capture_output=True,
            text=True, encoding="utf-8",
            timeout=180,
            env={**os.environ, "NO_COLOR": "1", "HOME": str(tmp_path / "home")},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert not target.exists(), "dry run created the config directory"

    def test_help_exits_zero(self):
        result = subprocess.run(
            [BASH, str(SCRIPT), "--help"], capture_output=True, text=True, encoding="utf-8", timeout=60
        )
        assert result.returncode == 0
        assert "--non-interactive" in result.stdout

    def test_an_unknown_flag_is_refused(self):
        result = subprocess.run(
            [BASH, str(SCRIPT), "--not-a-flag"],
            capture_output=True,
            text=True, encoding="utf-8",
            timeout=60,
        )
        assert result.returncode != 0


class TestCidrValidation:
    """The allowlist prompt validates before the seeding does.

    A typo'd CIDR aborts seeding partway through, which leaves the
    whitelist *on* and half-populated — the operator is locked out of
    their own server. Catching it at the prompt is the difference.
    """

    @staticmethod
    def _validate(value: str) -> str:
        harness = (
            'set -euo pipefail\nPYTHON="$(command -v python3)"\n'
            + _function_source(SCRIPT.read_text(encoding="utf-8"), "validate_cidrs")
            + f'\nvalidate_cidrs "{value}"\n'
        )
        return subprocess.run(
            [BASH, "-c", harness], capture_output=True, text=True, encoding="utf-8", timeout=60
        ).stdout.strip()

    @pytest.mark.parametrize(
        "value",
        ["127.0.0.1/32", "100.64.0.0/10", "192.168.1.5", "fd00::/8",
         "127.0.0.1/32, 10.0.0.0/8 "],
    )
    def test_accepts_what_the_server_accepts(self, value):
        assert self._validate(value) == "OK"

    @pytest.mark.parametrize("value", ["not-a-cidr", "10.0.0.0/33", "999.1.1.1"])
    def test_rejects_what_the_server_rejects(self, value):
        assert self._validate(value) != "OK"

    def test_it_agrees_with_the_shipped_parser(self):
        """Two validators that disagree are worse than one.

        The prompt uses the standard library because it runs before the
        package is installed; the server uses ``parse_cidr``. They must
        accept the same set or the installer writes a config the server
        then refuses.
        """
        from hypernix.t1api.errors import T1APIError
        from hypernix.t1api.netpolicy import parse_cidr

        for value in ["127.0.0.1/32", "192.168.1.1/24", "fd00::/8", "10.0.0.5"]:
            parse_cidr(value)  # does not raise
            assert self._validate(value) == "OK"

        for value in ["not-a-cidr", "10.0.0.0/33"]:
            with pytest.raises(T1APIError):
                parse_cidr(value)
            assert self._validate(value) != "OK"


class TestQuestionsAsked:
    """Every subject the installer was asked to cover has a question."""

    @pytest.mark.parametrize(
        "func",
        [
            "q_identity",
            "q_network",
            "q_environment",
            "q_keys",
            "q_whitelist",
            "q_requests",
            "q_cost",
            "q_models",
            "q_features",
        ],
    )
    def test_question_exists(self, source, func):
        assert f"{func}()" in source
        assert re.search(rf"^\s+{func}\s*$", source, re.M), f"{func} is never called"

    def test_the_waiter_tui_is_offered(self, source):
        body = _function_body(source, "q_features")
        assert "TUI" in body or "tui" in body

    def test_key_families_are_a_choice(self, source):
        body = _function_body(source, "q_keys")
        for token in ("T1", "T2"):
            assert token in body



class TestKeyPolicyIsEnforceable:
    """The installer offers three key policies; the server must have three.

    "T2 only" was offered as a choice and written into the config, but the
    server had no switch for it — ``accept_t2_keys`` alone cannot refuse
    the T1 spelling — so the option silently behaved as "both". An
    installer question that does not change the deployment is worse than
    no question: the operator believes a migration is enforced.
    """

    def test_both_switches_exist(self):
        from hypernix.t1api.config import T1APIConfig

        config = T1APIConfig()
        assert hasattr(config, "accept_t1_keys")
        assert hasattr(config, "accept_t2_keys")

    def test_defaults_accept_everything(self):
        from hypernix.t1api.config import T1APIConfig

        config = T1APIConfig()
        assert config.accept_t1_keys is True
        assert config.accept_t2_keys is True

    def test_the_installer_writes_both(self, source):
        body = _function_body(source, "write_env")
        assert "T1_ACCEPT_T1_KEYS=" in body
        assert "T1_ACCEPT_T2_KEYS=" in body

    def test_t2_only_turns_off_the_t1_spelling(self, source):
        body = _function_body(source, "write_env")
        assert "t2) accept_t1=0" in body.replace("  ", " ")

    def test_refusing_both_families_is_refused(self):
        """A server nothing can authenticate to should not start.

        Left to run, every request fails with a message about the key
        rather than about the configuration, which is the hardest kind
        of misconfiguration to diagnose.
        """
        from hypernix.security.gatekeeper import Gatekeeper
        from hypernix.security.keymaster import Keymaster
        from hypernix.t1api.auth import T1AuthService

        km = Keymaster(auto_rotate=False)
        with pytest.raises(ValueError, match="at least one"):
            T1AuthService(
                km,
                Gatekeeper(keymaster=km),
                token_secret="x" * 32,
                accept_t1_keys=False,
                accept_t2_keys=False,
            )


class TestTheMintedKeyReachesTheServer:
    """The installer's admin key must live in the store the server reads.

    ``Keymaster()`` defaults to ``~/.hypernix/keymaster``, but the
    installer mints into ``$CONFIG_DIR/keymaster``. Without a way to point
    the server at that store, every install printed an admin key the
    server had never heard of — "shown once, copy this now" for a
    credential that could not authenticate.
    """

    def test_the_store_is_configurable(self):
        from hypernix.t1api.config import T1APIConfig

        assert hasattr(T1APIConfig(), "keymaster_dir")

    def test_unset_keeps_the_long_standing_default(self, monkeypatch):
        from hypernix.t1api.config import T1APIConfig

        monkeypatch.delenv("T1_KEYMASTER_DIR", raising=False)
        assert T1APIConfig().keymaster_dir is None

    def test_the_env_var_is_honoured(self, monkeypatch, tmp_path):
        from hypernix.t1api.config import T1APIConfig

        monkeypatch.setenv("T1_KEYMASTER_DIR", str(tmp_path / "ks"))
        assert T1APIConfig().keymaster_dir == str(tmp_path / "ks")

    def test_the_installer_writes_it(self, source):
        body = _function_body(source, "write_env")
        assert "T1_KEYMASTER_DIR=" in body

    def test_it_matches_where_the_key_is_minted(self, source):
        """Both halves must name the same directory.

        This is the actual bug: two places each internally consistent and
        pointing at different directories.
        """
        mint = _function_body(source, "mint_admin_key")
        env = _function_body(source, "write_env")
        assert 'T1_KEYMASTER_DIR="$CONFIG_DIR/keymaster"' in mint
        assert "T1_KEYMASTER_DIR=$CONFIG_DIR/keymaster" in env

    def test_a_key_minted_into_a_store_is_visible_to_a_server_reading_it(self, tmp_path):
        """End to end, without the shell: mint here, authenticate there."""
        from hypernix.security.gatekeeper import Gatekeeper
        from hypernix.security.keymaster import Keymaster, KeyScope, KeyType
        from hypernix.t1api.auth import T1AuthService

        store = tmp_path / "keymaster"
        minted = Keymaster(store_dir=store, auto_rotate=False).create(
            key_type=KeyType.ADMIN,
            scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE},
            prefix="installer",
        )

        # A separate Keymaster over the same directory, as the server does.
        server_km = Keymaster(store_dir=store, auto_rotate=False)
        auth = T1AuthService(
            server_km, Gatekeeper(keymaster=server_km), token_secret="x" * 32
        )
        assert auth.validate_key(minted.key).is_admin

    def test_a_t2_only_server_still_admits_the_wrapped_admin_key(self, tmp_path):
        """The T2-only lockout, in miniature.

        Under T2-only the installer hands over the T2 form of the minted
        key. Admin authority comes from the key store, not from the T2
        password component, so the wrapped key must still be an admin —
        otherwise choosing T2-only leaves the operator with no admin
        credential at all and no way to change the setting back.
        """
        from hypernix.security.gatekeeper import Gatekeeper
        from hypernix.security.keymaster import Keymaster, KeyScope, KeyType
        from hypernix.security.t2keys import T2KeyGenerator
        from hypernix.t1api.auth import T1AuthService

        store = tmp_path / "keymaster"
        minted = Keymaster(store_dir=store, auto_rotate=False).create(
            key_type=KeyType.ADMIN,
            scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE},
            prefix="installer",
        )
        wrapped = T2KeyGenerator.from_t1(minted.key, access_level=9).raw

        server_km = Keymaster(store_dir=store, auto_rotate=False)
        auth = T1AuthService(
            server_km,
            Gatekeeper(keymaster=server_km),
            token_secret="x" * 32,
            accept_t1_keys=False,
        )
        assert auth.validate_key(wrapped).is_admin

        # And the spelling it was told to refuse is refused, by message
        # as well as by status: the holder needs to know what to present.
        from hypernix.t1api.errors import T1APIError

        with pytest.raises(T1APIError) as excinfo:
            auth.validate_key(minted.key)
        assert "T2" in excinfo.value.message
        assert excinfo.value.details.get("accepted") == ["T2", "T2S"]


class TestAllowlistSeeding:
    def test_success_is_verified_by_a_read_back(self, source):
        body = _function_body(source, "seed_allowlist")
        assert "list_entries()" in body, "seeding trusts the write instead of reading back"

    def test_the_reported_list_is_tagged_not_scraped(self, source):
        """Only the deliberate line is echoed as the verified list.

        stderr is folded into the capture so a traceback survives to the
        failure message; that also picks up interpreter noise, which
        once got printed to the operator as the list of allowed CIDRs.
        """
        body = _function_body(source, "seed_allowlist")
        assert "T1SEEDOK:" in body
        assert "${seed_output##*T1SEEDOK:}" in body

    def test_failure_does_not_report_success(self, source):
        body = _function_body(source, "seed_allowlist")
        success_at = body.index("Allowlist seeded and verified")
        guard_at = body.index('if [ "$seed_status" -ne 0 ]')
        assert guard_at < success_at, "success is announced before the status is checked"


class TestTheYesFlag:
    """``--yes`` was parsed, documented, and never read.

    shellcheck's SC2034 ("appears unused") caught it, which is the whole
    reason the shellcheck check is in this file: a flag that sets a
    variable nothing looks at is invisible to every other kind of test —
    the script runs, exits 0, and quietly ignores what you asked for.
    """

    def test_it_is_read_and_not_only_assigned(self, code):
        assignments = len(re.findall(r"\bASSUME_YES=", code))
        reads = len(re.findall(r'"\$ASSUME_YES"', code))
        assert reads > 0, "--yes sets a variable nothing reads"
        assert assignments >= 1

    def test_confirmations_are_answered(self, source):
        body = _function_body(source, "ask_yes_no")
        assert '"$ASSUME_YES" = "1"' in body

    def test_open_questions_are_still_asked(self, source):
        """The difference between --yes and --non-interactive.

        If ``ask`` honoured it too, --yes would silently become
        --non-interactive and the port, allowlist and prices would be
        defaulted without the operator seeing them.
        """
        for func in ("ask", "ask_choice", "ask_secret"):
            assert "ASSUME_YES" not in _function_body(source, func), (
                f"{func} honours --yes; open questions would stop being asked"
            )

    def test_it_does_not_start_a_server(self, source):
        """Both launch branches end in exec.

        Treating "start the server now?" as a confirmation would hand a
        foreground process to someone who asked not to be interrupted.
        Setting up and starting are separate decisions.
        """
        body = _code_only(_function_body(source, "offer_launch"))
        assert '"$ASSUME_YES" = "1"' in body, "offer_launch does not check --yes"
        # `exec "` — the call, not the word, which also appears in the
        # comment right above the guard explaining why it is there.
        guard_at = body.index('"$ASSUME_YES" = "1"')
        assert guard_at < body.index('exec "'), "--yes reaches an exec"

    def test_the_help_text_describes_what_it_does(self):
        result = subprocess.run(
            [BASH, str(SCRIPT), "--help"], capture_output=True, text=True, encoding="utf-8", timeout=60
        )
        assert "--yes" in result.stdout
        # Collapse the wrapping first: the help text is hard-wrapped, so
        # the phrase spans a line break in the file.
        flowed = " ".join(result.stdout.lower().split())
        # It promises not to start the server; that promise is tested above.
        assert "does not start the server" in flowed

    def test_a_yes_run_answers_and_finishes(self, tmp_path):
        """End to end: confirmations answered, open questions asked, exit 0."""
        answers = "\n".join(
            ["4", "yes-flag-srv", "1", "8951", "", "1", "3",
             "127.0.0.1/32,10.0.0.0/8", "2", "1", "free", "3"]
        ) + "\n"
        target = tmp_path / "cfg"
        result = subprocess.run(
            [BASH, str(SCRIPT), "--yes", "--install", "skip",
             "--config-dir", str(target)],
            input=answers,
            capture_output=True,
            text=True, encoding="utf-8",
            timeout=300,
            env={**os.environ, "NO_COLOR": "1", "HOME": str(tmp_path / "home")},
        )
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
        # Confirmations answered without a prompt...
        assert "(--yes)" in result.stdout
        # ...open questions still put to the operator...
        assert "Port" in result.stdout
        # ...and nothing was launched.
        assert "Not starting the server" in result.stdout
        assert not (target / "server.pid").exists()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _function_source(source: str, name: str) -> str:
    """Return ``name() { ... }`` verbatim, brace-matched.

    Naive ``sed`` extraction to the first ``^}`` is wrong for any function
    containing a heredoc that itself ends a line with ``}``, and several
    of these do.
    """
    start = source.index(f"{name}() {{")
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def _function_body(source: str, name: str) -> str:
    return _function_source(source, name)


def _code_only(text: str) -> str:
    """Drop whole-line comments, preserving offsets well enough to order by.

    Comments are blanked rather than removed so an index into the result
    still points at roughly the right place in the original.
    """
    return "\n".join(
        "" if line.lstrip().startswith("#") else line for line in text.splitlines()
    )


class TestWhichInterpreter:
    """`--install skip` has to find the one that *has* hypernix.

    The search used to be ordered `python3.12 python3.13 python3.11
    python3 python` — newest-ish first, with the operator's own `python3`
    fourth. On any machine with several interpreters that picks one the
    operator never chose, and under `--install skip`, whose whole premise
    is that the package is already installed somewhere, it then verifies
    an installation that was never meant to be in that interpreter and
    fails while everything is in fact fine. That is exactly what happened
    in CI: the job installed into setup-python's 3.11 and the script went
    looking in the system 3.12.
    """

    def _fake_python(self, directory: Path, name: str, *, has_hypernix: bool) -> Path:
        """An interpreter that works but may not have hypernix.

        Delegates everything to the real interpreter except `import
        hypernix`, so the version and pip preflight checks pass and only
        the question under test differs.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        reject = "" if has_hypernix else '''
for arg in "$@"; do
  case "$arg" in
    *"import hypernix"*) exit 1 ;;
  esac
done
'''
        # shell_path, not str(): sys.executable on Windows is full of
        # backslashes, and a shell reads every one of them as an escape,
        # so the exec line names a path that does not exist.
        path.write_text(
            "#!/bin/sh\n" + reject + f'exec "{shell_path(sys.executable)}" "$@"\n',
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def _preflight_choice(self, tmp_path, path_entries: list[str], **env) -> str:
        """Run just as far as the interpreter report, and return that line."""
        result = subprocess.run(
            [BASH, str(SCRIPT), "--dry-run", "--non-interactive",
             "--config-dir", str(tmp_path / "cfg"), "--install", "skip"],
            capture_output=True, text=True, encoding="utf-8", timeout=300,
            env={
                **os.environ,
                "NO_COLOR": "1",
                "HOME": str(tmp_path / "home"),
                "PATH": os.pathsep.join([*path_entries, os.environ["PATH"]]),
                **env,
            },
        )
        line = next(
            (ln for ln in result.stdout.splitlines() if "python:" in ln), ""
        )
        assert line, result.stdout[-2000:] + result.stderr[-1000:]
        return line

    def test_it_prefers_the_interpreter_that_has_hypernix(self, tmp_path):
        """A newer one without the package must not win."""
        bin_dir = tmp_path / "bin"
        self._fake_python(bin_dir, "python3.14", has_hypernix=False)
        chosen = self._fake_python(bin_dir, "python3", has_hypernix=True)

        line = self._preflight_choice(
            tmp_path, [str(bin_dir)],
            PYTHONPATH=str(REPO_ROOT / "src"),
        )

        assert _names(line, chosen), line

    def test_python_override_wins_over_the_search(self, tmp_path):
        bin_dir = tmp_path / "bin"
        self._fake_python(bin_dir, "python3", has_hypernix=True)
        override = self._fake_python(bin_dir, "python3.11", has_hypernix=True)

        line = self._preflight_choice(
            tmp_path, [str(bin_dir)],
            HYPERNIX_PYTHON=str(override),
            PYTHONPATH=str(REPO_ROOT / "src"),
        )

        assert _names(line, override), line

    def test_an_unusable_override_is_refused_by_name(self, tmp_path):
        result = subprocess.run(
            [BASH, str(SCRIPT), "--dry-run", "--non-interactive",
             "--python", "/definitely/not/a/python",
             "--config-dir", str(tmp_path / "cfg")],
            capture_output=True, text=True, encoding="utf-8", timeout=180,
            env={**os.environ, "NO_COLOR": "1", "HOME": str(tmp_path / "home")},
        )
        assert result.returncode != 0
        assert "/definitely/not/a/python" in result.stderr

    def _searched_names(self, source: str) -> list[str]:
        """Every interpreter name the script will try, read from the script.

        Hard-coding the list here would rot the moment someone adds
        python3.15 to the search: the test would keep passing while no
        longer covering the name that was added.
        """
        match = re.search(r'local search="([^"]+)"', source)
        assert match, "install-t1.sh no longer declares its search list"
        return match.group(1).split()

    def test_skip_with_no_installation_says_what_to_do(self, tmp_path, source):
        """Not "install failed" — nothing was installed, and that is the point.

        Every name in the search list has to be shadowed, not just
        `python3`. Shadowing one only worked here, where the other names
        do not resolve; on a CI runner `python3.14` is setup-python's
        interpreter with hypernix installed, so the search found it, the
        import succeeded, and the script correctly did not fail.
        """
        bin_dir = tmp_path / "bin"
        for name in self._searched_names(source):
            self._fake_python(bin_dir, name, has_hypernix=False)

        result = subprocess.run(
            [BASH, str(SCRIPT), "--non-interactive", "--install", "skip",
             "--config-dir", str(tmp_path / "cfg")],
            capture_output=True, text=True, encoding="utf-8", timeout=300,
            env={
                **os.environ,
                "NO_COLOR": "1",
                "HOME": str(tmp_path / "home"),
                "PATH": os.pathsep.join([str(bin_dir), os.environ["PATH"]]),
                "PYTHONPATH": "",
            },
        )
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "--python" in combined, combined[-2000:]
        assert "after installing" not in combined, "wrong diagnosis"

    def test_the_help_documents_the_override(self, source: str):
        assert "--python PATH" in source
        assert "HYPERNIX_PYTHON" in source


class TestIndexingModelsAtInstall:
    """`hypernix-t1 index` existed and nothing ran it.

    So a person with eleven GGUFs in ~/.hypernix/models finished the
    installer with a registry containing one placeholder entry marked
    "edit before serving traffic", and a HyperLink model picker with
    nothing in it. Every model on the machine was invisible to the server
    that was just set up to serve them.
    """

    def test_both_flags_are_accepted(self, source):
        assert "--index-models)" in source
        assert "--estimate-prices)" in source

    def test_estimating_implies_indexing(self, code):
        """Pricing models the installer did not index would price
        nothing."""
        line = next(
            line for line in code.splitlines() if "--estimate-prices)" in line
        )
        assert "INDEX_MODELS=1" in line

    def test_both_are_off_by_default(self, code):
        assert "INDEX_MODELS=0" in code
        assert "ESTIMATE_PRICES=0" in code

    def test_they_are_documented(self):
        result = subprocess.run(
            [BASH, str(SCRIPT), "--help"],
            capture_output=True, text=True, encoding="utf-8", check=True,
        )
        assert "--index-models" in result.stdout
        assert "--estimate-prices" in result.stdout

    def test_it_calls_the_shipped_indexer(self, code):
        """Not a second GGUF parser in shell. The context limit the
        indexer reads is the number the server then enforces, and a
        second implementation is a second place for it to be wrong."""
        assert "hypernix.t1api.modelindex_cli" in code

    def test_it_runs_after_the_package_is_installed(self, code):
        """It invokes the installed module, so it cannot run before
        install_package."""
        body = code[code.index("main() {"):]
        assert body.index("install_package") < body.index("index_models")

    def test_a_missing_models_directory_is_not_a_failure(self, code):
        """A fresh machine has no ~/.hypernix/models, and that is the
        normal case rather than a broken install."""
        block = code[code.index("index_models() {"):]
        block = block[:block.index("\nsummary() {")]
        assert "return 0" in block
        assert "nothing to index" in SCRIPT.read_text(encoding="utf-8")

    def test_a_failed_index_does_not_abort_the_install(self, code):
        """Everything else has already been set up by that point.
        Leaving a working server uninstalled over a registry write is the
        wrong trade."""
        block = code[code.index("index_models() {"):]
        block = block[:block.index("\nsummary() {")]
        assert "warn" in block
        assert "die" not in block

    def test_a_dry_run_indexes_nothing(self, code):
        block = code[code.index("index_models() {"):]
        block = block[:block.index("\nsummary() {")]
        assert 'DRY_RUN" = "1"' in block

    def test_the_dry_run_still_says_nothing_was_written(self):
        """The whole contract of --dry-run. A step added later that
        wrote anyway would be the one thing this flag exists to
        prevent."""
        subprocess.run(
            [BASH, str(SCRIPT), "--dry-run", "--non-interactive",
             "--estimate-prices", "--config-dir", "/tmp/hnx-dry-run-check"],
            capture_output=True, text=True, encoding="utf-8", check=False,
        )
        assert not Path("/tmp/hnx-dry-run-check").exists()
