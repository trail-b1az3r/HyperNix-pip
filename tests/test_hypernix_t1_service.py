"""``bin/hypernix-t1`` — running a T1 API as a thing you manage.

The gap it fills: starting the server by hand is a uvicorn incantation
with six environment variables, and every one has to match what ``gkey``
and ``waiter`` think. Getting one wrong does not fail loudly — it
produces a server that runs and rejects your keys, which is the most
expensive way for this to go wrong.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import pytest
from shell_support import BASH, NO_BASH_REASON

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "bin" / "hypernix-t1"

pytestmark = pytest.mark.skipif(BASH is None, reason=NO_BASH_REASON)


def _function_body(source: str, name: str) -> str:
    """The body of a shell function, from its `name()` to the closing brace.

    A fixed character window would instead measure how much comment the
    function carries, and go red when someone explains themselves.
    """
    start = source.index(f"{name}()")
    end = source.index("\n}", start)
    return source[start:end]


def run(*argv: str, home: Path, config: Path, timeout: int = 60):
    """Run the script. `.output` is stdout+stderr.

    Warnings go to stderr — correct for a status line, and easy to miss
    when asserting.
    """
    result = subprocess.run(
        [BASH, str(SCRIPT), *argv],
        capture_output=True,
        text=True, encoding="utf-8",
        timeout=timeout,
        env={
            **os.environ,
            "HOME": str(home),
            "T1_CONFIG_DIR": str(config),
            "NO_COLOR": "1",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        },
    )
    result.output = result.stdout + result.stderr  # type: ignore[attr-defined]
    return result


@pytest.fixture
def configured(tmp_path):
    home = tmp_path / "home"
    config = tmp_path / "cfg"
    home.mkdir()
    config.mkdir()
    (config / ".env").write_text(
        "T1_HOST=127.0.0.1\n"
        "T1_PORT=8123\n"
        f"T1_KEYMASTER_DIR={config}/keymaster\n"
        f"T1_DB_PATH={config}/t1.sqlite3\n"
        "T1_TOKEN_SECRET=" + "d" * 64 + "\n",
        encoding="utf-8",
    )
    return home, config


class TestShape:
    def test_it_exists_and_is_executable(self):
        assert SCRIPT.is_file()
        assert os.access(SCRIPT, os.X_OK)

    def test_it_parses(self):
        subprocess.run([BASH, "-n", str(SCRIPT)], check=True)

    def test_shellcheck_is_clean_if_available(self):
        if shutil.which("shellcheck") is None:
            pytest.skip("shellcheck not installed")
        # warning, not style, and matching the other two shell tests. The
        # info and style tiers are advisory and their contents move between
        # shellcheck releases: 0.8.0 raises SC2009 on `ps -p "$pid" | grep`,
        # 0.9.0 through 0.11.0 do not, because -p already scopes it to one
        # process. Gating CI on those tiers makes the build depend on which
        # shellcheck the runner happens to ship, which is how a commit that
        # touched nothing turns red.
        result = subprocess.run(
            ["shellcheck", "--severity=warning", str(SCRIPT)],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_help_lists_every_command(self, configured):
        home, config = configured
        text = run("help", home=home, config=config).output
        for command in ("start", "stop", "kill", "restart", "status", "logs",
                        "create", "configure", "test", "key", "autostart", "remove"):
            assert command in text, command

    def test_an_unknown_command_fails(self, configured):
        home, config = configured
        assert run("nonsense", home=home, config=config).returncode == 2


class TestProcessIdentity:
    """A PID file outlives the process it named, and PIDs get reused.

    Trusting one is how a manager reports a dead server as healthy — or,
    much worse, kills whatever now holds that number.
    """

    def test_a_stale_pid_is_not_reported_as_running(self, configured):
        home, config = configured
        (config / "server.pid").write_text("999999\n", encoding="utf-8")
        assert "not running" in run("status", home=home, config=config).output

    def test_someone_elses_process_is_not_claimed(self, configured):
        """The dangerous case: the PID exists, and is not ours."""
        home, config = configured
        victim = subprocess.Popen(["sleep", "120"])
        try:
            (config / "server.pid").write_text(f"{victim.pid}\n", encoding="utf-8")
            assert "not running" in run("status", home=home, config=config).output
            # And stop must not kill it.
            run("stop", home=home, config=config)
            assert victim.poll() is None, "it killed an unrelated process"
        finally:
            victim.kill()
            victim.wait(timeout=10)

    def test_it_checks_the_command_line_not_just_the_pid(self):
        """Both checks, and read from the function, not from a byte window.

        This used to slice the first 600 characters after `server_pid()`,
        which made it a test of how much comment the function carries: a
        four-line note pushed the `ps` line past the window and the test
        failed while the code was correct.
        """
        body = _function_body(SCRIPT.read_text(encoding="utf-8"), "server_pid")
        assert "kill -0" in body, "does not check the PID is alive"
        assert "hypernix.t1api" in body, "does not check the command line"


class TestConfig:
    def test_the_env_file_is_read_not_sourced(self):
        """.env is a file people edit by hand; sourcing runs whatever
        ends up in it."""
        source = SCRIPT.read_text(encoding="utf-8")
        body = source.split("load_env()")[1].split("\n}")[0]
        # The syntax, not the word — the function's own comment explains
        # why sourcing is avoided, and matching that proves nothing.
        code = "\n".join(
            row for row in body.splitlines() if not row.strip().startswith("#")
        )
        assert '. "$ENV_FILE"' not in code
        assert "source " not in code

    def test_only_known_prefixes_are_exported(self):
        """A stray line in .env should not set PATH or LD_PRELOAD."""
        assert 'T1_*|HYPERNIX_*)' in SCRIPT.read_text(encoding="utf-8")


class TestTheSystemdUnit:
    def test_exec_start_is_absolute(self, configured):
        """systemd will not resolve a relative path, and the unit it
        wrote silently refused to start."""
        home, config = configured
        # --write-only installs the unit without touching the user bus.
        # Without it this test skipped wherever `systemctl --user` cannot
        # reach a session -- containers, plain ssh, WSL -- which is
        # everywhere CI runs, so the assertion below never ran.
        run("autostart", "on", "--write-only", home=home, config=config)
        unit = home / ".config" / "systemd" / "user" / "hypernix-t1.service"
        if not unit.exists():
            pytest.skip("systemd not available to write a unit here")
        line = next(
            row for row in unit.read_text(encoding="utf-8").splitlines() if row.startswith("ExecStart=")
        )
        path = line.split("=", 1)[1].split()[0]
        assert path.startswith("/"), line

    def test_the_unit_sets_an_explicit_path(self):
        """A user unit gets a minimal PATH, and a tailscale in
        /usr/local/bin then becomes invisible — which presents as
        "Tailscale is broken" when only PATH is."""
        source = SCRIPT.read_text(encoding="utf-8")
        assert "Environment=PATH=" in source
        assert "/usr/local/bin" in source

    def test_a_machine_with_no_user_bus_says_what_to_do(self, configured):
        """systemctl being on PATH is not the same as there being a
        session to talk to. Containers, plain ssh and WSL all fail here,
        and the bare systemd error -- "Failed to connect to bus: No
        medium found" -- says nothing about what to do next.

        The guard probes for the bus directly rather than inferring it
        from the exit code, which is what the first version of this test
        did and why it failed on every GitHub runner. A runner *has* a
        user bus, so the no-bus branch is never reached there -- but the
        run still exits non-zero, because HOME is redirected to a tmp
        directory and systemd cannot see the unit written into it
        ("Unit file hypernix-t1.service does not exist"). Reading a
        non-zero exit as "took the branch I meant" turned a skip into a
        failure on Linux, macOS and Windows at once.
        """
        import shutil
        import subprocess

        home, config = configured
        if shutil.which("systemctl") is None:
            pytest.skip("no systemctl to probe")
        probe = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            capture_output=True, timeout=30, check=False,
        )
        if probe.returncode == 0:
            pytest.skip(
                "this machine has a working user bus, so the no-bus branch "
                "cannot be exercised here"
            )

        result = run("autostart", "on", home=home, config=config)
        assert result.returncode != 0
        message = result.stdout + result.stderr
        assert "enable-linger" in message or "session startup" in message
        assert "--write-only" in message

    def test_an_unknown_autostart_argument_is_refused(self, configured):
        """It used to take ``${1:-on}`` and ignore everything else, so a
        typo silently enabled autostart instead of reporting itself."""
        home, config = configured
        result = run("autostart", "onn", home=home, config=config)
        assert result.returncode != 0

    def test_it_runs_the_foreground_form_under_systemd(self):
        """No PID file and no backgrounding: systemd supervises."""
        source = SCRIPT.read_text(encoding="utf-8")
        assert "start-foreground" in source
        foreground = _function_body(source, "cmd_start_foreground")
        assert "exec " in foreground
        assert "PID_FILE" not in foreground


class TestRemoveKeepsTheKeys:
    def test_the_key_store_is_never_deleted(self):
        """Keys are not recoverable and may still be in use elsewhere.

        Everything else in the config directory can be rebuilt.
        """
        body = _function_body(SCRIPT.read_text(encoding="utf-8"), "cmd_remove")
        assert "! -name keymaster" in body
        assert "Keeping the key store" in body

    def test_it_requires_typing_the_word(self):
        body = _function_body(SCRIPT.read_text(encoding="utf-8"), "cmd_remove")
        assert "'remove'" in body or '"remove"' in body


def _can_serve() -> bool:
    """Can this machine actually run `hypernix-t1 start`?

    Asked of a subprocess rather than with an import here, because the
    script runs whatever `python3` is on PATH and that is not
    necessarily the interpreter running pytest.

    Both modules, not just fastapi. `start` execs `python -m uvicorn`,
    so uvicorn is what decides whether a server comes up -- and a check
    for fastapi alone passes on a machine that has fastapi and no
    uvicorn, which is precisely the machine CI runs on. Three tests
    went red there for exactly that reason.
    """
    return subprocess.run(
        ["python3", "-c", "import fastapi, uvicorn"], capture_output=True
    ).returncode == 0


NEEDS_A_SERVER = pytest.mark.skipif(
    not _can_serve(), reason="needs the [t1api] extra (fastapi + uvicorn)"
)


@NEEDS_A_SERVER
class TestAgainstARealServer:
    def test_start_status_and_stop(self, configured):
        home, config = configured
        try:
            started = run("start", home=home, config=config, timeout=120)
            assert started.returncode == 0, started.stdout + started.stderr
            assert "Running" in started.stdout

            status = run("status", home=home, config=config)
            assert "running" in status.stdout
            assert "t1 v" in status.stdout, "status did not reach /status"
        finally:
            run("kill", home=home, config=config)

    def test_start_is_idempotent(self, configured):
        home, config = configured
        try:
            run("start", home=home, config=config, timeout=120)
            again = run("start", home=home, config=config, timeout=60)
            assert "Already running" in again.stdout
        finally:
            run("kill", home=home, config=config)

    def test_key_uses_the_servers_own_store(self, configured):
        """gkey and the server must agree about where the keys live.

        They did not until T1_KEYMASTER_DIR was honoured by both: keys
        the operator minted were invisible to their own server.
        """
        home, config = configured
        result = run("key", "create", "-v", "v2", home=home, config=config, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert list((config / "keymaster").glob("*.json")), "minted somewhere else"


class TestCreateWithoutACheckout:
    """`pip install hypernix` gives you the manager and no installer.

    `create` hands off to install-t1.sh, which is a checkout file. From a
    wheel there is no checkout, and a dead end there would mean the
    manager cannot create the thing it manages.
    """

    def _isolated(self, tmp_path):
        """Run the script from a copy with no install-t1.sh anywhere near it.

        That is what a wheel install looks like: the program sits in a bin
        directory beside other console scripts, and the repo is not there.
        """
        binned = tmp_path / "prefix" / "bin"
        binned.mkdir(parents=True)
        copied = binned / "hypernix-t1"
        copied.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
        copied.chmod(0o755)
        return copied

    def _run(self, copied: Path, *argv: str, home: Path, config: Path):
        return subprocess.run(
            [BASH, str(copied), *argv],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
            env={
                **os.environ,
                "HOME": str(home),
                "T1_CONFIG_DIR": str(config),
                "NO_COLOR": "1",
                "PYTHONPATH": str(REPO_ROOT / "src"),
            },
            cwd=str(tempfile.gettempdir()),
        )

    def test_it_writes_a_usable_config(self, tmp_path):
        home, config = tmp_path / "home", tmp_path / "cfg"
        home.mkdir()
        copied = self._isolated(tmp_path)

        result = self._run(copied, "create", "--port", "8971",
                           home=home, config=config)

        assert result.returncode == 0, result.stdout + result.stderr
        env = (config / ".env").read_text(encoding="utf-8")
        assert "T1_PORT=8971" in env
        assert "T1_HOST=127.0.0.1" in env
        assert f"T1_KEYMASTER_DIR={config}/keymaster" in env

    def test_the_secret_is_real_and_the_file_is_not_readable(self, tmp_path):
        home, config = tmp_path / "home", tmp_path / "cfg"
        home.mkdir()
        copied = self._isolated(tmp_path)

        self._run(copied, "create", home=home, config=config)

        env_file = config / ".env"
        if os.name != "nt":
            # Windows has no owner/group/other permission bits to set;
            # chmod there moves the read-only flag and nothing else, so
            # st_mode reads 666 however the file was created. Asserting
            # 600 on Windows tests the platform, not this script.
            assert oct(env_file.stat().st_mode)[-3:] == "600"
        secret = next(
            line.split("=", 1)[1]
            for line in env_file.read_text(encoding="utf-8").splitlines()
            if line.startswith("T1_TOKEN_SECRET=")
        )
        assert len(secret) == 64
        assert int(secret, 16)              # hex, and not a placeholder
        assert len(set(secret)) > 4

    def test_two_creates_produce_different_secrets(self, tmp_path):
        """A fixed secret would validate every other install's tokens."""
        secrets = []
        for i in range(2):
            home, config = tmp_path / f"home{i}", tmp_path / f"cfg{i}"
            home.mkdir()
            copied = self._isolated(tmp_path / f"run{i}")
            self._run(copied, "create", home=home, config=config)
            secrets.append((config / ".env").read_text(encoding="utf-8"))
        assert secrets[0] != secrets[1]

    def test_it_refuses_to_overwrite_without_force(self, tmp_path):
        home, config = tmp_path / "home", tmp_path / "cfg"
        home.mkdir()
        copied = self._isolated(tmp_path)

        self._run(copied, "create", home=home, config=config)
        first = (config / ".env").read_text(encoding="utf-8")
        again = self._run(copied, "create", home=home, config=config)

        assert again.returncode != 0
        assert "already exists" in again.stdout + again.stderr
        assert (config / ".env").read_text(encoding="utf-8") == first

        forced = self._run(copied, "create", "--force", home=home, config=config)
        assert forced.returncode == 0
        assert (config / ".env").read_text(encoding="utf-8") != first

    def test_it_says_what_the_minimal_config_does_not_cover(self, tmp_path):
        """Silence here would read as "configured", which it is not."""
        home, config = tmp_path / "home", tmp_path / "cfg"
        home.mkdir()
        copied = self._isolated(tmp_path)

        result = self._run(copied, "create", home=home, config=config)
        combined = result.stdout + result.stderr

        assert "install-t1.sh" in combined
        for missing in ("allowlist", "rate limits", "pricing"):
            assert missing in combined.lower(), missing

    def test_a_bad_port_is_refused(self, tmp_path):
        home, config = tmp_path / "home", tmp_path / "cfg"
        home.mkdir()
        copied = self._isolated(tmp_path)

        result = self._run(copied, "create", "--port", "eight",
                           home=home, config=config)

        assert result.returncode != 0
        assert not (config / ".env").exists()

    def test_the_checkout_still_hands_off_to_the_installer(self, tmp_path):
        """The guided setup stays the default wherever it is available."""
        home, config = tmp_path / "home", tmp_path / "cfg"
        home.mkdir()
        result = subprocess.run(
            [BASH, str(SCRIPT), "create", "--help"],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
            env={**os.environ, "HOME": str(home),
                 "T1_CONFIG_DIR": str(config), "NO_COLOR": "1"},
        )
        # install-t1.sh --help, not the minimal path's "Unknown option".
        assert "install-t1.sh" in result.stdout
        assert "--non-interactive" in result.stdout


class TestTheTwoCreatePathsAgree:
    """``hypernix-t1 create`` runs one of two programs.

    From a checkout it execs ``install-t1.sh``; from a wheel there is no
    installer and it writes a minimal config itself. Both are documented
    by the same ``--help``, so both have to accept the same flags and
    write the same keys — and neither of those was true.
    """

    def test_the_installer_accepts_the_flags_the_manager_documents(self):
        """``create --host H --port N --force`` is in ``hypernix-t1
        --help``. From a checkout it reached install-t1.sh, which had
        never heard of any of them and died with "Unknown option:
        --port" — so the documented interface failed on exactly the
        machine a developer is sitting at."""
        installer = REPO_ROOT / "install-t1.sh"
        if not installer.is_file():
            pytest.skip("no checkout")
        source = installer.read_text(encoding="utf-8")
        for flag in ("--host)", "--port)", "--force)"):
            assert flag in source, flag

    def test_both_paths_write_the_key_the_manager_reads(self, tmp_path):
        """The consequential half. install-t1.sh put the bind address
        only into start-t1.sh, so ``hypernix-t1 start`` found no T1_HOST
        or T1_PORT, fell back to its own 127.0.0.1:8000 default, and
        started the server somewhere else — after which status, logs,
        key and test all pointed at an address nothing was listening on.
        """
        installer = REPO_ROOT / "install-t1.sh"
        if not installer.is_file():
            pytest.skip("no checkout")
        source = installer.read_text(encoding="utf-8")
        assert "T1_HOST=$BIND_HOST" in source
        assert "T1_PORT=$BIND_PORT" in source

        manager = SCRIPT.read_text(encoding="utf-8")
        assert 'setting T1_HOST' in manager
        assert 'setting T1_PORT' in manager

    def test_the_installer_refuses_to_clobber_an_env(self, tmp_path):
        """Regenerating the token secret invalidates every key already
        minted against it — a failure that surfaces later as "the server
        rejects my keys" rather than here as "the file was replaced".
        ``create_minimal`` already refused; the installer did not."""
        installer = REPO_ROOT / "install-t1.sh"
        if not installer.is_file():
            pytest.skip("no checkout")
        source = installer.read_text(encoding="utf-8")
        body = _function_body(source, "write_env")
        assert "FORCE_OVERWRITE" in body
        assert "already exists" in body


class TestItIsActuallyInstalled:
    def test_pyproject_ships_it_on_path(self):
        """[project.scripts] cannot carry a shell program; script-files can.

        Without this the docs promise a `hypernix-t1` that `pip install
        hypernix` does not provide.
        """
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'script-files = ["bin/hypernix-t1"]' in text

    def test_the_sdist_carries_it_and_the_installer(self):
        """tests/ ships in the sdist and runs both of these files."""
        manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        assert "include install-t1.sh" in manifest
        assert "recursive-include bin *" in manifest


class TestTheInstallAdviceNamesTheInterpreter:
    """"It is installed already" — and it was, for a different python.

    ``python_bin`` prefers the private venv install-t1.sh creates. The
    old message said only::

        hypernix[t1api] is not installed for <venv>/bin/python.
        Run: pip install 'hypernix[t1api]'

    A bare ``pip`` in the operator's shell installs into whatever their
    shell resolves, which is not that venv. So the instruction can be
    followed correctly, report success, and leave the check failing --
    any number of times. That is a loop with nothing on screen to
    explain it.
    """

    def _stub(self, tmp_path, *, has_hypernix: bool, has_extra: bool):
        """A fake interpreter with a chosen import outcome."""
        config = tmp_path / "t1api"
        (config / "venv" / "bin").mkdir(parents=True)
        python = config / "venv" / "bin" / "python"
        python.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            f"  *\"import hypernix.t1api\"*) exit {0 if has_extra else 1} ;;\n"
            f"  *\"import hypernix\"*) exit {0 if has_hypernix else 1} ;;\n"
            "esac\nexit 0\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
        return config, python

    def _start(self, config, **env):
        import os

        environment = {**os.environ, "T1_CONFIG_DIR": str(config), "NO_COLOR": "1", **env}
        environment.pop("PYTHONPATH", None)
        return subprocess.run(
            [BASH, str(SCRIPT), "start"],
            capture_output=True, text=True, encoding="utf-8",
            timeout=60, env=environment,
        )

    def test_the_pip_command_names_the_interpreter_it_checked(self, tmp_path):
        config, python = self._stub(tmp_path, has_hypernix=False, has_extra=False)

        result = self._start(config)
        output = result.stdout + result.stderr

        assert f"{python} -m pip install" in output

    def test_a_bare_pip_is_no_longer_suggested(self, tmp_path):
        """The whole defect: `pip install` with no interpreter on it."""
        config, _python = self._stub(tmp_path, has_hypernix=False, has_extra=False)

        result = self._start(config)
        output = result.stdout + result.stderr

        assert "Run: pip install" not in output

    def test_a_missing_extra_is_not_reported_as_a_missing_package(self, tmp_path):
        """Different problems, different fixes. Installing the wrong one
        of the two fixes nothing."""
        config, _python = self._stub(tmp_path, has_hypernix=True, has_extra=False)

        result = self._start(config)
        output = result.stdout + result.stderr

        assert "the [t1api] extra is not" in output
        assert "hypernix is not installed" not in output

    def test_a_missing_package_says_so(self, tmp_path):
        config, _python = self._stub(tmp_path, has_hypernix=False, has_extra=False)

        result = self._start(config)
        output = result.stdout + result.stderr

        assert "hypernix is not installed" in output

    def test_it_points_at_the_other_interpreter_when_there_is_one(self, tmp_path):
        """The line that explains "but it IS installed"."""
        import os

        config, _python = self._stub(tmp_path, has_hypernix=False, has_extra=False)
        elsewhere = tmp_path / "bin"
        elsewhere.mkdir()
        (elsewhere / "python3").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        (elsewhere / "python3").chmod(0o755)

        result = self._start(
            config, PATH=os.pathsep.join([str(elsewhere), os.environ["PATH"]])
        )
        output = result.stdout + result.stderr

        assert "It *is* installed for" in output
        assert str(elsewhere / "python3") in output

    def test_it_offers_deleting_the_venv_as_the_other_way_out(self, tmp_path):
        import os

        config, _python = self._stub(tmp_path, has_hypernix=False, has_extra=False)
        elsewhere = tmp_path / "bin"
        elsewhere.mkdir()
        (elsewhere / "python3").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        (elsewhere / "python3").chmod(0o755)

        result = self._start(
            config, PATH=os.pathsep.join([str(elsewhere), os.environ["PATH"]])
        )

        assert "venv" in (result.stdout + result.stderr)

    def test_it_still_fails(self, tmp_path):
        """A better message is not a reason to start a server that cannot run."""
        config, _python = self._stub(tmp_path, has_hypernix=False, has_extra=False)

        assert self._start(config).returncode != 0


class TestStartOutlivesTheShell:
    """`start` has to leave a server running, on every supported machine.

    Two things went wrong at once and both wore the same symptom -- "it
    says it started and then there is no server."

    `setsid ... & echo $!` needs a `setsid` binary, and macOS does not
    have one. The script advertises bash 3.2, which is the bash macOS
    ships, so macOS is a supported platform; there the background job
    died on "setsid: command not found", `echo $!` still succeeded so the
    `|| die` guard never fired, and the only evidence was a raw shell
    error tailed out of the log 45 seconds later.

    And `$!` is not the server's pid in general: setsid(1) forks when it
    is already a process-group leader and its parent then exits, so the
    recorded pid can name a process that has already gone. `launch-script`
    hit exactly that and reported "unknown" for healthy jobs -- see
    `_launch_setsid` in hypernix/system/launcher.py, which stopped using
    the binary for this reason. `start` had not been given the same fix.
    """

    def test_the_setsid_binary_is_not_invoked(self):
        """Read argv positions, not the prose.

        The function explains at length *why* the binary is wrong, so a
        plain `"setsid" in source` search matches the explanation and
        passes whatever the code does. Comments go first, then the match
        has to be a command word.
        """
        source = SCRIPT.read_text(encoding="utf-8")
        code = "\n".join(
            line.split(" #")[0] if not line.lstrip().startswith("#") else ""
            for line in source.splitlines()
        )
        offenders = re.findall(r"(?:^|[;&|(]|\bthen\b|\bdo\b)\s*setsid\b", code, re.M)
        assert not offenders, (
            "the setsid binary is invoked again -- it does not exist on macOS, "
            "and $! is not the server's pid when it forks"
        )

    @NEEDS_A_SERVER
    def test_it_starts_when_there_is_no_setsid_on_path(self, configured, tmp_path):
        """The macOS shape: everything present except setsid.

        Before the fix this printed "The server exited during startup"
        and left nothing running.
        """
        home, config = configured
        needed = [
            "bash", "sh", "python3", "python", "curl", "ps", "grep", "sed",
            "awk", "cat", "tail", "head", "find", "id", "sleep", "kill",
            "env", "dirname", "basename", "tr", "date", "mkdir", "rm",
            "cut", "sort", "chmod", "stat", "ln", "touch", "ls", "uname",
        ]
        fake_bin = tmp_path / "no-setsid-bin"
        fake_bin.mkdir()
        for tool in needed:
            found = shutil.which(tool)
            if found:
                (fake_bin / tool).symlink_to(found)
        assert shutil.which("python3", path=str(fake_bin)), "no python on the fake PATH"
        assert not shutil.which("setsid", path=str(fake_bin)), "setsid leaked in"

        env = {
            **os.environ,
            "PATH": str(fake_bin),
            "HOME": str(home),
            "T1_CONFIG_DIR": str(config),
            "NO_COLOR": "1",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        }
        try:
            started = subprocess.run(
                [BASH, str(SCRIPT), "start"],
                capture_output=True, text=True, encoding="utf-8",
                timeout=180, env=env,
            )
            assert started.returncode == 0, started.stdout + started.stderr
            assert "Running" in started.stdout, started.stdout + started.stderr
            log = (config / "server.log").read_text(encoding="utf-8", errors="replace")
            assert "setsid" not in log, log[-2000:]
        finally:
            run("kill", home=home, config=config)

    @NEEDS_A_SERVER
    def test_the_recorded_pid_is_the_server_itself(self, configured):
        """Not a launcher that has already exited.

        Everything downstream reads this number: `status` reports from
        it, `stop` signals it, and `wait_healthy` decides the server died
        when it goes away. A pid file that names the wrong process makes
        a running server invisible to its own manager.
        """
        home, config = configured
        try:
            started = run("start", home=home, config=config, timeout=180)
            assert started.returncode == 0, started.stdout + started.stderr
            pid = int((config / "server.pid").read_text(encoding="utf-8").strip())
            os.kill(pid, 0)  # raises if it is gone
            argv = subprocess.run(
                ["ps", "-p", str(pid), "-o", "args="],
                capture_output=True, text=True, encoding="utf-8",
            ).stdout
            assert "hypernix.t1api" in argv, f"pid {pid} is not the server: {argv!r}"
        finally:
            run("kill", home=home, config=config)

    @NEEDS_A_SERVER
    def test_the_server_gets_a_session_of_its_own(self, configured):
        """Which is the whole point of the detach.

        A process still in the launching shell's session is sent the
        SIGHUP that follows the terminal closing or the ssh connection
        dropping. Session leadership is the observable fact -- setsid(2)
        makes the child its own session leader, so its session id equals
        its pid -- and it holds however the detach is spelled.
        """
        home, config = configured
        try:
            started = run("start", home=home, config=config, timeout=180)
            assert started.returncode == 0, started.stdout + started.stderr
            pid = int((config / "server.pid").read_text(encoding="utf-8").strip())
            assert os.getsid(pid) == pid, (
                f"server pid {pid} is in session {os.getsid(pid)}, not its own -- "
                "a SIGHUP to the launching session would take it down"
            )
            assert os.getsid(pid) != os.getsid(os.getpid()), "same session as the test"
        finally:
            run("kill", home=home, config=config)

    def test_a_launcher_that_prints_no_pid_is_a_failure(self, configured, tmp_path):
        """The old shape could not detect this at all.

        `( cmd & echo $! > pid )` ends in `echo`, which succeeds whether
        or not the command behind it ever ran, so `|| die` was dead code.
        The pid now has to arrive and has to be a number.
        """
        home, config = configured
        stub = tmp_path / "python3"
        stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)
        source = SCRIPT.read_text(encoding="utf-8")
        body = _function_body(source, "cmd_start")
        assert "no pid from the launcher" in body, (
            "cmd_start does not check that the launcher reported a pid"
        )
        assert "*[!0-9]*" in body, "the reported pid is not checked for being a number"


class TestLogoutDoesNotTakeTheServerWithIt:
    """systemd-logind's KillUserProcesses=yes is the other cause.

    When it is on, logging out kills everything the user owns -- a
    process in its own session included. setsid(2) does not exempt
    anything from it; lingering does. Nothing is written to the log when
    it happens, so from the operator's side the server simply is not
    there any more, and `start` looked like it had not worked.
    """

    def test_the_check_exists_and_treats_lingering_as_the_exemption(self):
        body = _function_body(SCRIPT.read_text(encoding="utf-8"), "logind_kills_user_processes")
        assert "Linger" in body, "does not check whether lingering is on"
        assert "KillUserProcesses" in body

    def test_it_asks_the_running_configuration_first(self):
        """Drop-ins under logind.conf.d override the main file.

        Reading only /etc/systemd/logind.conf answers a question nobody
        asked on a machine whose distribution ships a drop-in.
        """
        body = _function_body(SCRIPT.read_text(encoding="utf-8"), "logind_kills_user_processes")
        assert "busctl" in body, "does not ask logind itself"
        assert "logind.conf.d" in body, "the file fallback ignores drop-ins"

    def test_the_warning_names_both_ways_out(self):
        body = _function_body(SCRIPT.read_text(encoding="utf-8"), "cmd_start")
        assert "logind_kills_user_processes" in body, "start never runs the check"
        assert "autostart on" in body
        assert "enable-linger" in body

    def test_a_machine_without_loginctl_says_nothing(self, tmp_path):
        """No systemd, no warning. The check must not guess."""
        source = SCRIPT.read_text(encoding="utf-8")
        harness = tmp_path / "harness.sh"
        harness.write_text(
            source.split("cmd_start() {")[0]
            + '\nif logind_kills_user_processes; then echo WARNED; else echo QUIET; fi\n',
            encoding="utf-8",
        )
        empty = tmp_path / "bin"
        empty.mkdir()
        for tool in ("sed", "tail", "id", "cat", "sh"):
            found = shutil.which(tool)
            if found:
                (empty / tool).symlink_to(found)
        result = subprocess.run(
            [BASH, str(harness)],
            capture_output=True, text=True, encoding="utf-8", timeout=60,
            env={**os.environ, "PATH": str(empty), "HOME": str(tmp_path), "NO_COLOR": "1"},
        )
        assert result.stdout.strip().endswith("QUIET"), result.stdout + result.stderr


class TestStartDoesNotReportSomeoneElsesServer:
    """`start` said Running with a pid; `status` a second later said not
    running. From a screenshot, and reproduced exactly.

    The sequence was `autostart` (which installs a **systemd user
    service**) and then `start`. The service holds the port; `start` sees
    no live pid file of its own, so it spawns a second uvicorn. That
    uvicorn logs "Application startup complete", *then* tries to bind,
    gets ``[Errno 98] address already in use`` and exits.

    Meanwhile ``wait_healthy`` was asking "does anything answer /health
    on this port?" -- and something does: the first server. So it
    returned success, and the pid printed alongside it belonged to a
    process already on its way out. Every part of that is a true
    statement about a different process.
    """

    @pytest.fixture
    def isolated(self, configured):
        """`configured` pins port 8123, which every server test in this
        file shares. These tests deliberately orphan a server from its
        pid file, so they need a port of their own and a cleanup that
        does not go through the script -- otherwise a leaked server sits
        on 8123 and the *next* test fails on the port guard this change
        added, which is a confusing way to discover you wrote a leaky
        test.
        """
        import socket

        home, config = configured
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = (config / ".env").read_text(encoding="utf-8")
        (config / ".env").write_text(
            env.replace("T1_PORT=8123", f"T1_PORT={port}"), encoding="utf-8"
        )
        started: list[int] = []
        yield home, config, started
        for pid in started:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def _pid_file(self, config: Path) -> Path:
        return config / "server.pid"

    def _remember(self, config: Path, started: list) -> int:
        """Record the running pid so the fixture can clean up after a
        test has taken the pid file away."""
        pid = int(self._pid_file(config).read_text(encoding="utf-8").strip())
        started.append(pid)
        return pid

    @NEEDS_A_SERVER
    def test_it_refuses_instead_of_claiming_success(self, isolated):
        """The reproduction: a healthy server on the port, no pid file of
        ours -- which is exactly what a systemd-managed instance looks
        like to this script."""
        home, config, started = isolated
        first = run("start", home=home, config=config, timeout=120)
        assert first.returncode == 0, first.output
        self._remember(config, started)
        self._pid_file(config).unlink()              # systemd owns it now

        second = run("start", home=home, config=config, timeout=120)
        assert second.returncode != 0, (
            "start reported success for someone else's server:\n" + second.output
        )
        assert "already listening" in second.output
        assert "Running (pid" not in second.output

    @NEEDS_A_SERVER
    def test_the_refusal_names_the_usual_cause(self, isolated):
        """`autostart on` is what put the other server there, and it is
        this script that installed it -- so it can say so rather than
        leaving a port number and a shrug."""
        home, config, started = isolated
        run("start", home=home, config=config, timeout=120)
        self._remember(config, started)
        self._pid_file(config).unlink()
        result = run("start", home=home, config=config, timeout=120)
        assert "systemctl --user status hypernix-t1" in result.output
        assert "autostart off" in result.output
        assert "T1_PORT" in result.output

    @NEEDS_A_SERVER
    def test_status_says_why_it_thinks_nothing_is_running(self, isolated):
        """"not running" on its own is what sent this bug unexplained.
        With a foreign listener on the port, status has to mention it."""
        home, config, started = isolated
        run("start", home=home, config=config, timeout=120)
        self._remember(config, started)
        self._pid_file(config).unlink()
        status = run("status", home=home, config=config)
        assert "not running" in status.output
        assert "listening on" in status.output
        assert "systemctl --user status hypernix-t1" in status.output

    @NEEDS_A_SERVER
    def test_status_shows_the_log_when_stopped(self, isolated):
        """The answer is nearly always in the last few lines, and nothing
        was printing them."""
        home, config, _started = isolated
        run("start", home=home, config=config, timeout=120)
        run("stop", home=home, config=config, timeout=120)
        status = run("status", home=home, config=config)
        assert "not running" in status.output
        assert "server.log" in status.output

    @NEEDS_A_SERVER
    def test_restart_still_works(self, isolated):
        """The port guard must not refuse to start the server it has just
        stopped -- the failure mode a naive bind test introduces."""
        home, config, _started = isolated
        try:
            assert run("start", home=home, config=config, timeout=120).returncode == 0
            result = run("restart", home=home, config=config, timeout=180)
            assert result.returncode == 0, result.output
            assert "Running" in result.output
        finally:
            run("kill", home=home, config=config)

    @NEEDS_A_SERVER
    def test_a_normal_start_is_unaffected(self, isolated):
        home, config, _started = isolated
        try:
            result = run("start", home=home, config=config, timeout=120)
            assert result.returncode == 0, result.output
            assert "Running (pid" in result.output
            assert run("status", home=home, config=config).returncode == 0
        finally:
            run("kill", home=home, config=config)


class TestTheHealthProbeIsAboutOurProcess:
    """Static checks, so the reasoning survives an edit that looks
    harmless. Each pins a line whose removal restores the bug."""

    def test_wait_healthy_is_given_the_pid(self):
        source = SCRIPT.read_text(encoding="utf-8")
        body = _function_body(source, "wait_healthy")
        assert 'pid="${3:-}"' in body, "wait_healthy no longer takes a pid"

    def test_wait_healthy_rechecks_the_pid_after_a_good_probe(self):
        """A 200 from the port proves a server is there, not that ours
        is. Without this the caller cannot tell the two apart."""
        body = _function_body(SCRIPT.read_text(encoding="utf-8"), "wait_healthy")
        probe = body.index("/health")
        assert "kill -0" in body[probe:], (
            "wait_healthy accepts any answer on the port again"
        )

    def test_start_passes_the_pid_through(self):
        """The argument existing is no use if the call site drops it."""
        body = _function_body(SCRIPT.read_text(encoding="utf-8"), "cmd_start")
        call = body.split("wait_healthy")[1][:120]
        assert '"$pid"' in call, call

    def test_start_checks_the_port_before_spawning(self):
        body = _function_body(SCRIPT.read_text(encoding="utf-8"), "cmd_start")
        spawn = body.index("spawn_detached")
        assert "port_in_use" in body[:spawn], (
            "the port check must come before the spawn, or it is just a "
            "second opinion about a process that already failed"
        )

    def test_the_port_probe_sets_reuseaddr(self):
        """uvicorn sets it, so the probe has to ask the same question --
        otherwise a port in TIME_WAIT reads as busy and `restart` refuses
        to start what it just stopped."""
        body = _function_body(SCRIPT.read_text(encoding="utf-8"), "port_in_use")
        assert "SO_REUSEADDR" in body

    def test_the_port_probe_needs_no_extra_tools(self):
        """ss, netstat and lsof are all missing on some machines this
        runs on; a bind test is always available."""
        body = _function_body(SCRIPT.read_text(encoding="utf-8"), "port_in_use")
        for tool in ("ss ", "netstat", "lsof"):
            assert tool not in body, tool
