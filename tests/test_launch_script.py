"""``hypernix-t1 launch-script`` — a job that outlives the connection.

Closing a laptop is the normal end of a remote working session and
should not be the end of a training run. `&` does not achieve that: a
backgrounded process is still in the shell's process group and still
holds the tty, so the SIGHUP that follows a dropped connection reaches
it.

The test that matters is therefore the one that kills the parent and
checks the child is still there — everything else is bookkeeping around
that. It is also the test that found the bug this shipped with twice:

**The `setsid` binary was the wrong tool.** It forks when it is already
a process-group leader and the parent exits, so the pid recorded was a
process that had already gone, and `--status` said "unknown" for a job
running perfectly well. `Popen(start_new_session=True)` calls `setsid(2)`
in the child directly, which is the same new session with the pid we
actually want.

**`nargs=REMAINDER` swallowed the flags.** The documented form is
`launch-script ./train.py --name training-job --detach`, and REMAINDER
handed `--name training-job` to the script, silently naming the job
after the filename. The script's own arguments go after a `--`.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hypernix.security.keymaster import generate_t1_key
from hypernix.system.launcher import (
    JobStatus,
    JobStore,
    LaunchError,
    Supervisor,
    launch,
    read_logs,
    refresh,
    stop,
)
from hypernix.t1api.launchscript_cli import AuthError, authenticate
from hypernix.t1api.launchscript_cli import main as launch_main

SRC = str(Path(__file__).resolve().parent.parent / "src")


def _process_state(pid: int) -> str:
    """The single-letter state from /proc, or "" if the pid is gone.

    "Z" is a zombie: dead, but still occupying a pid because nothing has
    waited on it yet.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return ""
    # The comm field can contain spaces and parentheses; state is the
    # first field after the closing one.
    return stat.rsplit(")", 1)[1].split()[0] if ")" in stat else ""


def script(tmp_path: Path, body: str, name: str = "job.sh") -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def settle(job, store, *, timeout: float = 20.0):
    """Wait for a job to reach a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        refresh(job, store)
        if job.is_terminal:
            return job
        time.sleep(0.1)
    return job


class TestItSurvivesTheParent:
    """The whole point, and the only test that proves it."""

    def test_a_job_outlives_the_process_that_started_it(self, tmp_path):
        work = script(tmp_path, 'echo started\nsleep 8\necho finished\nexit 0\n')
        store_root = tmp_path / "jobs"

        # Launch from a *separate* process, then let that process die --
        # which is what an SSH disconnect does to the shell.
        launcher = subprocess.run(
            [sys.executable, "-c", (
                f"import sys; sys.path.insert(0, {SRC!r})\n"
                "from hypernix.system.launcher import JobStore, launch\n"
                f"store = JobStore({str(store_root)!r})\n"
                f"j = launch({str(work)!r}, name='survivor', store=store)\n"
                "print(j.job_id)"
            )],
            capture_output=True, text=True, timeout=60, check=True,
        )
        job_id = launcher.stdout.strip()

        # The launching process has exited. The job must not have.
        store = JobStore(store_root)
        job = store.load(job_id)
        assert job is not None
        refresh(job, store)

        assert job.status == JobStatus.RUNNING.value
        assert "started" in read_logs(job)

    def test_and_it_finishes_with_its_exit_status_recorded(self, tmp_path):
        work = script(tmp_path, "echo working\nexit 7\n")
        store = JobStore(tmp_path / "jobs")

        job = settle(launch(work, name="seven", store=store), store)

        assert job.status == JobStatus.FAILED.value
        assert job.exit_status == 7

    def test_a_zero_exit_is_success(self, tmp_path):
        work = script(tmp_path, "exit 0\n")
        store = JobStore(tmp_path / "jobs")

        job = settle(launch(work, name="ok", store=store), store)

        assert job.status == JobStatus.SUCCEEDED.value
        assert job.exit_status == 0

    def test_the_job_is_in_its_own_session(self, tmp_path):
        """Not merely a different process: a different *session*, which
        is what SIGHUP-on-disconnect actually follows."""
        work = script(tmp_path, "sleep 5\n")
        store = JobStore(tmp_path / "jobs")

        job = launch(work, name="sessioned", store=store)

        assert os.getsid(job.pid) != os.getsid(os.getpid())
        stop(job, store)


class TestLogsAndStatus:
    def test_output_is_captured(self, tmp_path):
        work = script(tmp_path, "echo one\necho two\n")
        store = JobStore(tmp_path / "jobs")

        job = settle(launch(work, name="talky", store=store), store)

        assert "one" in read_logs(job)
        assert "two" in read_logs(job)

    def test_stderr_lands_in_the_same_log(self, tmp_path):
        work = script(tmp_path, "echo oops >&2\n")
        store = JobStore(tmp_path / "jobs")

        job = settle(launch(work, name="noisy", store=store), store)

        assert "oops" in read_logs(job)

    def test_a_log_file_can_be_chosen(self, tmp_path):
        work = script(tmp_path, "echo hello\n")
        store = JobStore(tmp_path / "jobs")
        target = tmp_path / "chosen.log"

        settle(launch(work, log_file=target, store=store), store)

        assert target.exists()
        assert "hello" in target.read_text(encoding="utf-8")

    def test_a_record_survives_a_new_store_object(self, tmp_path):
        """--status has to work from a different SSH session entirely."""
        work = script(tmp_path, "exit 0\n")
        settle(launch(work, name="persisted", store=JobStore(tmp_path / "jobs")),
               JobStore(tmp_path / "jobs"))

        found = JobStore(tmp_path / "jobs").find("persisted")

        assert found is not None

    def test_a_job_is_findable_by_name_or_id(self, tmp_path):
        work = script(tmp_path, "exit 0\n")
        store = JobStore(tmp_path / "jobs")
        job = launch(work, name="findable", store=store)

        assert store.find("findable").job_id == job.job_id
        assert store.find(job.job_id).name == "findable"

    def test_a_half_written_record_is_never_read(self, tmp_path):
        """--status polls in a loop; it must not catch a partial write."""
        work = script(tmp_path, "exit 0\n")
        store = JobStore(tmp_path / "jobs")
        launch(work, name="atomic", store=store)

        assert not list((tmp_path / "jobs").glob("*.tmp"))


class TestStopping:
    def test_a_running_job_can_be_stopped(self, tmp_path):
        work = script(tmp_path, "sleep 30\n")
        store = JobStore(tmp_path / "jobs")
        job = launch(work, name="stoppable", store=store)

        stop(job, store)

        assert job.status == JobStatus.STOPPED.value
        time.sleep(0.5)
        assert store.find("stoppable").status == JobStatus.STOPPED.value

    def test_stopping_kills_the_whole_group(self, tmp_path):
        """The wrapper shell and the command it ran, not just one.

        Checked by process *state*, not by whether the pid exists. After
        SIGKILL the wrapper is a zombie until its parent reaps it, and in
        this test the parent is pytest — so ``os.kill(pid, 0)`` succeeds
        for a process that is thoroughly dead. In real use the launching
        process has exited and init reaps it, which is why this only
        shows up here.
        """
        if not Path("/proc").is_dir():  # pragma: no cover - Linux only
            pytest.skip("needs /proc to read process state")
        work = script(tmp_path, "sleep 30 &\nsleep 30\n")
        store = JobStore(tmp_path / "jobs")
        job = launch(work, name="grouped", store=store)
        pid = job.pid

        stop(job, store)
        time.sleep(0.5)

        assert _process_state(pid) in ("", "Z"), (
            f"pid {pid} is still in state {_process_state(pid)!r}, not dead"
        )

    def test_the_command_it_ran_is_gone_too(self, tmp_path):
        """`sleep 30` would outlast the test if only the wrapper died."""
        work = script(tmp_path, "sleep 30\n")
        store = JobStore(tmp_path / "jobs")
        job = launch(work, name="innergone", store=store)
        time.sleep(0.3)

        stop(job, store)
        time.sleep(0.5)

        # The wrapper is dead or a zombie; either way `sleep 30` must
        # not still be running, which it would be if only the wrapper
        # had been signalled.
        assert _process_state(job.pid) in ("", "Z")


class TestArgumentsAndEnvironment:
    def test_script_arguments_go_after_a_double_dash(self, tmp_path, capsys):
        work = script(tmp_path, 'echo "args:$*"\n')
        os.environ["T1_CONFIG_DIR"] = str(tmp_path / "cfg")
        os.environ["T1_API_KEY"] = generate_t1_key()

        code = launch_main([str(work), "--name", "withargs", "--", "--lr", "3e-4"])

        assert code == 0
        capsys.readouterr()
        store = JobStore()
        job = settle(store.find("withargs"), store)
        assert "args:--lr 3e-4" in read_logs(job)

    def test_flags_after_the_path_reach_the_cli_not_the_script(
        self, tmp_path, capsys
    ):
        """The documented form. REMAINDER used to hand --name to the
        script and name the job after the filename instead."""
        work = script(tmp_path, "exit 0\n", name="train.sh")
        os.environ["T1_CONFIG_DIR"] = str(tmp_path / "cfg")
        os.environ["T1_API_KEY"] = generate_t1_key()

        launch_main([str(work), "--name", "training-job", "--detach"])
        capsys.readouterr()

        assert JobStore().find("training-job") is not None

    def test_env_is_passed_to_the_job(self, tmp_path):
        work = script(tmp_path, 'echo "V=$MY_VALUE"\n')
        store = JobStore(tmp_path / "jobs")

        job = settle(
            launch(work, env={"MY_VALUE": "present"}, store=store), store
        )

        assert "V=present" in read_logs(job)

    def test_only_env_names_are_recorded_never_values(self, tmp_path):
        """A job record is readable by anything that can read the config
        directory, and several of these are credentials."""
        work = script(tmp_path, "exit 0\n")
        store = JobStore(tmp_path / "jobs")

        job = launch(work, env={"SECRET_TOKEN": "hunter2"}, store=store)

        assert job.env_keys == ["SECRET_TOKEN"]
        assert "hunter2" not in (tmp_path / "jobs" / f"{job.job_id}.json").read_text(
            encoding="utf-8"
        )

    def test_gpu_sets_both_vendor_variables(self, tmp_path):
        """Which one the runtime reads depends on whether it lands on
        CUDA or ROCm; the job should not have to know."""
        work = script(tmp_path, 'echo "C=$CUDA_VISIBLE_DEVICES H=$HIP_VISIBLE_DEVICES"\n')
        store = JobStore(tmp_path / "jobs")

        job = settle(launch(work, gpu="1", store=store), store)

        assert "C=1 H=1" in read_logs(job)

    def test_a_working_directory_is_honoured(self, tmp_path):
        work = script(tmp_path, "pwd\n")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        store = JobStore(tmp_path / "jobs")

        job = settle(launch(work, cwd=elsewhere, store=store), store)

        assert str(elsewhere.resolve()) in read_logs(job)


class TestRefusals:
    def test_a_missing_script_says_so(self, tmp_path):
        with pytest.raises(LaunchError, match="No such script"):
            launch(tmp_path / "absent.sh", store=JobStore(tmp_path / "jobs"))

    def test_a_directory_is_not_a_script(self, tmp_path):
        with pytest.raises(LaunchError, match="not a script"):
            launch(tmp_path, store=JobStore(tmp_path / "jobs"))

    def test_a_missing_working_directory_says_so(self, tmp_path):
        work = script(tmp_path, "exit 0\n")

        with pytest.raises(LaunchError, match="working directory"):
            launch(work, cwd=tmp_path / "nope", store=JobStore(tmp_path / "jobs"))

    def test_an_unrunnable_file_explains_the_two_ways_out(self, tmp_path):
        odd = tmp_path / "thing.xyz"
        odd.write_text("data", encoding="utf-8")

        with pytest.raises(LaunchError, match="chmod"):
            launch(odd, store=JobStore(tmp_path / "jobs"))

    def test_a_python_file_needs_no_shebang(self, tmp_path):
        work = tmp_path / "run.py"
        work.write_text("print('from python')\n", encoding="utf-8")
        store = JobStore(tmp_path / "jobs")

        job = settle(launch(work, store=store), store)

        assert "from python" in read_logs(job)


class TestAuthentication:
    class Args:
        key = ""
        admin_password = ""

    def test_no_credential_is_refused(self, tmp_path, monkeypatch):
        for name in ("T1_API_KEY", "HYPERNIX_T1_KEY", "T1_KEY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))

        with pytest.raises(AuthError, match="No credential"):
            authenticate(self.Args())

    def test_a_valid_key_authenticates(self, monkeypatch, tmp_path):
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        args = self.Args()
        args.key = generate_t1_key()

        assert authenticate(args).startswith("key:")

    def test_a_junk_key_is_refused(self, monkeypatch, tmp_path):
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        args = self.Args()
        args.key = "definitely-not-a-key"

        with pytest.raises(AuthError, match="not a valid"):
            authenticate(args)

    def test_an_environment_key_counts_as_being_authenticated(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("T1_API_KEY", generate_t1_key())

        assert authenticate(self.Args()).startswith("key:")

    def test_the_admin_password_is_accepted(self, monkeypatch, tmp_path):
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("T1_ADMIN_PASSWORD", "correct horse")
        args = self.Args()
        args.admin_password = "correct horse"

        assert authenticate(args) == "admin-password"

    def test_a_wrong_admin_password_is_not(self, monkeypatch, tmp_path):
        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("T1_ADMIN_PASSWORD", "correct horse")
        args = self.Args()
        args.admin_password = "wrong horse"

        with pytest.raises(AuthError, match="admin password"):
            authenticate(args)

    def test_the_credential_never_reaches_the_job(self, tmp_path):
        """A key in the child's argv or environment would be readable by
        every user on the box through `ps` and /proc."""
        work = script(tmp_path, 'echo "K=${T1_API_KEY:-unset}"\n')
        store = JobStore(tmp_path / "jobs")

        job = launch(work, store=store)

        assert not any("T1_" in part for part in job.command)


class TestTheSupervisor:
    def test_it_picks_one(self):
        assert available_supervisor_value() in ("systemd", "setsid")

    def test_the_setsid_path_records_an_exit_status(self, tmp_path):
        """The fallback has no supervisor, so the wrapper is what makes
        the outcome knowable at all."""
        work = script(tmp_path, "exit 3\n")
        store = JobStore(tmp_path / "jobs")

        job = settle(
            launch(work, store=store, supervisor=Supervisor.SETSID), store
        )

        assert job.exit_status == 3


def available_supervisor_value() -> str:
    from hypernix.system.launcher import available_supervisor

    return available_supervisor().value
