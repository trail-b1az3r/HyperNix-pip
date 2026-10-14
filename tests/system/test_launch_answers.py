"""launch-script -2 … -9 (answers to a command's questions) and --logs -F.

Real detached jobs, like the rest of the launch-script tests. The
"questions" are `read` builtins in the shell command: each one reads the
next line of the job's input, which is where the answers go.
"""
from __future__ import annotations

import io
import os
import stat
import sys
import time

import pytest

from hypernix.system.launcher import JobStore, follow_logs, launch_shell, read_logs, refresh
from hypernix.t1api import launchscript_cli

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only, like launch-script")

ASKS = 'read a; read b; read c; echo "a=[$a] b=[$b] c=[$c]"'


def settle(job, store, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        refresh(job, store)
        if job.is_terminal:
            return job
        time.sleep(0.1)
    return job


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setattr(launchscript_cli, "authenticate", lambda args: "test")
    store = JobStore(tmp_path / "jobs")
    monkeypatch.setattr(launchscript_cli, "JobStore", lambda: store)
    monkeypatch.chdir(tmp_path)
    return store


def test_answers_go_to_the_questions_in_order(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = launch_shell(ASKS, shell="sh", answers=["yes", "", "3"], store=store, cwd=tmp_path)
    settle(job, store)
    assert job.status == "succeeded"
    assert "a=[yes] b=[] c=[3]" in read_logs(job)


def test_the_answers_file_is_private_and_kept(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = launch_shell("cat", shell="sh", answers=["secret"], store=store, cwd=tmp_path)
    settle(job, store)
    mode = stat.S_IMODE(os.stat(job.answers_file).st_mode)
    assert mode == 0o600
    assert store.load(job.job_id).answers_file == job.answers_file


def test_without_answers_input_is_empty_as_before(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = launch_shell("cat; echo end", shell="sh", store=store, cwd=tmp_path)
    settle(job, store)
    assert job.answers_file == ""
    assert read_logs(job).strip() == "end"


def test_an_answer_is_one_line(tmp_path):
    from hypernix.system.launcher import LaunchError

    with pytest.raises(LaunchError, match="one line"):
        launch_shell("true", answers=["a\nb"], store=JobStore(tmp_path / "jobs"), cwd=tmp_path)


class TestTheFlags:
    def test_dash_two_to_nine_with_a_skipped_number_meaning_enter(self, cli):
        assert launchscript_cli.main(["-1", ASKS, "--shell", "sh", "-2", "first", "-4", "third", "--name", "q"]) == 0
        job = settle(cli.find("q"), cli)
        assert "a=[first] b=[] c=[third]" in read_logs(job)

    def test_empty_quotes_press_enter(self, cli):
        assert launchscript_cli.main(["-1", ASKS, "--shell", "sh", "-2", "", "-3", "", "-4", "x", "--name", "e"]) == 0
        assert "a=[] b=[] c=[x]" in read_logs(settle(cli.find("e"), cli))

    def test_answers_need_a_command(self, cli, tmp_path, capsys):
        script = tmp_path / "s.py"
        script.write_text("print(1)\n")
        assert launchscript_cli.main([str(script), "-2", "y"]) == 1
        assert "-1 'command'" in capsys.readouterr().err

    def test_a_restart_answers_again(self, cli):
        assert launchscript_cli.main(["-1", ASKS, "--shell", "sh", "-2", "again", "--name", "r"]) == 0
        settle(cli.find("r"), cli)
        assert launchscript_cli.main(["--restart", "r"]) == 0
        fresh = settle(cli.find("r"), cli)
        assert "a=[again]" in read_logs(fresh)

    def test_follow_needs_logs(self, cli, capsys):
        assert launchscript_cli.main(["-F"]) == 1
        assert "--logs JOB -F" in capsys.readouterr().err


def test_follow_prints_what_is_written_and_says_when_the_job_ends(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = launch_shell("echo one; sleep 1; echo two; sleep 1; echo three", shell="sh", store=store, cwd=tmp_path)
    out = io.StringIO()
    deadline = time.time() + 20
    follow_logs(job, store=store, out=out, poll=0.1,
                until=lambda: "still following" in out.getvalue() or time.time() > deadline)
    text = out.getvalue()
    assert text.index("one") < text.index("two") < text.index("three")
    assert "succeeded" in text and "Ctrl-C" in text


def test_follow_stops_quietly_on_ctrl_c(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = launch_shell("echo hi", shell="sh", store=store, cwd=tmp_path)
    settle(job, store)
    out = io.StringIO()

    def interrupt():
        raise KeyboardInterrupt

    follow_logs(job, store=store, out=out, poll=0.01, until=interrupt)
    assert "hi" in out.getvalue()
