"""`hypernix-t1 launch-script -$ '<command>'` — a shell command as a job.

What these tests are for
------------------------
The feature is small; the ways to get it wrong are specific. The command
must reach the shell as *one* argument — re-split on spaces, its pipes,
`&&` and quoting stop meaning anything. It must get everything a script
job gets, or it is a lesser way to run things that happens to share a
name. And restarting it must restart the command: the restart path used
to treat the last element of the command as a script path, which for a
shell job is the command text.

These run real detached jobs, like the rest of the launch-script tests.
"""
from __future__ import annotations

import shutil
import sys
import time

import pytest

from hypernix.system.launcher import (
    JobStore,
    LaunchError,
    launch_shell,
    read_logs,
    refresh,
    resolve_shell,
    shell_command_of,
    stop,
)
from hypernix.t1api import launchscript_cli

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only, like launch-script")


def settle(job, store, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        refresh(job, store)
        if job.is_terminal:
            return job
        time.sleep(0.1)
    return job


class TestResolveShell:
    def test_auto_prefers_bash(self):
        """A launched job is scripting, and the syntax nearly every
        snippet uses — `export X=1 && ...`, `$(...)` — fails in fish."""
        if shutil.which("bash"):
            assert resolve_shell("auto").endswith("bash")

    def test_a_named_shell_that_is_missing_says_so(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(LaunchError) as caught:
            resolve_shell("fish")
        assert "fish is not installed" in str(caught.value)

    def test_auto_falls_back_to_what_exists(self, monkeypatch):
        monkeypatch.setattr(shutil, "which",
                            lambda name: "/usr/bin/fish" if name == "fish" else None)
        assert resolve_shell("auto") == "/usr/bin/fish"

    def test_an_unknown_shell_is_refused(self):
        with pytest.raises(LaunchError):
            resolve_shell("cmd.exe")


class TestLaunchShell:
    def test_the_command_runs_and_its_output_is_logged(self, tmp_path):
        store = JobStore(tmp_path / "jobs")
        job = launch_shell("echo first && echo second", store=store, cwd=tmp_path)
        settle(job, store)
        assert job.exit_status == 0
        log = read_logs(job)
        assert "first" in log and "second" in log

    def test_the_command_is_one_argument_not_resplit(self, tmp_path):
        """Pipes, quoting and `&&` are the shell's to interpret. Split on
        spaces and `'a b'` becomes two words and the pipe a literal."""
        store = JobStore(tmp_path / "jobs")
        job = launch_shell("printf '%s\\n' 'a b' | tr ' ' _", store=store, cwd=tmp_path)
        settle(job, store)
        assert job.command[1] == "-c"
        assert "a_b" in read_logs(job)

    def test_a_failing_command_reports_its_exit_status(self, tmp_path):
        store = JobStore(tmp_path / "jobs")
        job = launch_shell("exit 7", store=store, cwd=tmp_path)
        settle(job, store)
        assert job.exit_status == 7

    def test_it_gets_the_job_environment(self, tmp_path):
        """Everything a script job gets, or it is a lesser way to run
        things that happens to share a name."""
        store = JobStore(tmp_path / "jobs")
        job = launch_shell('echo "id=$HNX_JOB_ID"', store=store, cwd=tmp_path,
                           name="envcheck")
        settle(job, store)
        assert f"id={job.job_id}" in read_logs(job)

    def test_env_values_from_the_caller_arrive(self, tmp_path):
        store = JobStore(tmp_path / "jobs")
        job = launch_shell('echo "x=$MY_VALUE"', env={"MY_VALUE": "42"},
                           store=store, cwd=tmp_path)
        settle(job, store)
        assert "x=42" in read_logs(job)

    def test_an_empty_command_is_refused(self, tmp_path):
        with pytest.raises(LaunchError):
            launch_shell("   ", store=JobStore(tmp_path / "jobs"))

    def test_it_can_be_stopped(self, tmp_path):
        store = JobStore(tmp_path / "jobs")
        job = launch_shell("sleep 60", store=store, cwd=tmp_path)
        time.sleep(0.3)
        stop(job, store)
        settle(job, store)
        assert job.is_terminal

    def test_it_is_recognised_as_a_shell_job(self, tmp_path):
        store = JobStore(tmp_path / "jobs")
        job = launch_shell("true", store=store, cwd=tmp_path)
        shell, command = shell_command_of(job)
        assert shell in ("bash", "fish", "sh", "zsh")
        assert command == "true"


class TestTheCli:
    @pytest.fixture
    def cli(self, tmp_path, monkeypatch):
        monkeypatch.setattr(launchscript_cli, "authenticate", lambda args: "test")
        store = JobStore(tmp_path / "jobs")
        monkeypatch.setattr(launchscript_cli, "JobStore", lambda: store)
        monkeypatch.chdir(tmp_path)
        return store

    def test_dollar_flag_starts_a_job(self, cli, capsys):
        assert launchscript_cli.main(["-$", "echo via-cli", "--name", "viacli"]) == 0
        assert "started viacli" in capsys.readouterr().out
        job = cli.find("viacli")
        settle(job, cli)
        assert "via-cli" in read_logs(job)

    def test_the_long_form_for_fish_users(self, cli):
        """A bare `$` is a syntax error in fish, so `-$` cannot be typed
        there unquoted. The long form exists for them."""
        assert launchscript_cli.main(["--shell-command", "true", "--name", "long"]) == 0
        assert cli.find("long") is not None

    def test_a_script_and_a_command_together_is_refused(self, cli, tmp_path, capsys):
        script = tmp_path / "s.sh"
        script.write_text("true\n")
        assert launchscript_cli.main([str(script), "-$", "true"]) == 1
        assert "not both" in capsys.readouterr().err

    def test_restart_reruns_the_command_not_a_script(self, cli, capsys):
        """The bug in the restart path: it relaunched `command[-1]` as a
        script path, which for a shell job is the command text — "No
        such script: echo restarted"."""
        launchscript_cli.main(["-$", "echo restarted-ok", "--name", "again"])
        settle(cli.find("again"), cli)
        assert launchscript_cli.main(["--restart", "again"]) == 0
        fresh = cli.find("again")
        settle(fresh, cli)
        assert "restarted-ok" in read_logs(fresh)

    def test_the_help_warns_about_quoting(self, capsys):
        with pytest.raises(SystemExit):
            launchscript_cli.main(["--help"])
        text = capsys.readouterr().out
        assert "--shell-command" in text and "fish" in text
