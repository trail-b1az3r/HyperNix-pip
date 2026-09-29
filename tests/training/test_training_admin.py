"""Training administration — 0.72.4 item 5.

Two halves, tested separately because they fail differently.

``hypernix.training.monitor`` is process control: pause is SIGSTOP,
stop is SIGTERM, and a run whose trainer segfaulted leaves a file that
still says "running". None of that can be checked by reading the code,
so these tests start real processes and read ``/proc`` to see what
actually happened to them.

``t1api.routers.training`` is an access decision. The spec makes it
admin-only *unless* the server is explicitly in trusted-network mode,
with the destructive controls behind a second opt-in on top of that —
so most of what is tested here is the set of callers who must be
refused, not the one who is allowed.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hypernix.training.monitor import (
    ProgressReporter,
    RunState,
    TrainingError,
    TrainingMonitor,
    TrainingRun,
)

# ===========================================================================
# The reporter and the run record
# ===========================================================================


class TestProgress:
    def test_a_run_with_no_declared_total_has_unknown_progress(self):
        """None, not 0.0.

        A progress bar showing 0% for a job two hours in states
        something false with confidence. Unknown has to stay unknown all
        the way to the caller.
        """
        run = TrainingRun(run_id="r", step=4000)

        assert run.progress is None
        assert run.percent is None
        assert run.eta_seconds is None

    def test_steps_win_over_epochs(self):
        """Both declared: steps are the finer measure, so they decide."""
        run = TrainingRun(run_id="r", step=50, total_steps=100, epoch=0, total_epochs=4)

        assert run.percent == 50.0

    def test_progress_is_clamped(self):
        """A trainer that overruns its declared schedule is not 130% done."""
        run = TrainingRun(run_id="r", step=130, total_steps=100)

        assert run.progress == 1.0

    def test_eta_comes_from_the_rate_actually_achieved(self):
        started = time.time() - 60.0
        run = TrainingRun(
            run_id="r", step=25, total_steps=100,
            started_at=started, updated_at=started + 60.0,
        )

        # A quarter done in 60s -> 180s left.
        assert run.eta_seconds == pytest.approx(180.0, abs=1.0)

    def test_no_eta_without_a_start_time(self):
        run = TrainingRun(run_id="r", step=25, total_steps=100)

        assert run.eta_seconds is None


class TestReporter:
    def test_it_writes_a_record_the_monitor_can_read(self, tmp_path):
        reporter = ProgressReporter(
            "run-1", total_epochs=2, total_steps=10, model="m", root=tmp_path
        )
        reporter.update(step=5, epoch=1, loss=0.25, lr=1e-4)

        run = TrainingMonitor(tmp_path).get("run-1")

        assert run is not None
        assert run.step == 5
        assert run.metrics["loss"] == 0.25
        assert run.metrics["lr"] == pytest.approx(1e-4)
        assert run.percent == 50.0

    def test_the_file_is_never_observed_half_written(self, tmp_path):
        """Each update is a rename over the previous file.

        The reader is a web request that can arrive at any point,
        including mid-update. Every intermediate state on disk has to be
        a complete document.
        """
        reporter = ProgressReporter("run-1", total_steps=100, root=tmp_path)
        for step in range(1, 40):
            reporter.update(step=step, loss=1.0 / step)
            json.loads(reporter.path.read_text(encoding="utf-8"))

        assert reporter.path.exists()
        assert not list(tmp_path.glob("*.tmp"))

    def test_loss_history_is_bounded(self, tmp_path):
        """Otherwise a week-long run writes a megabyte per update."""
        reporter = ProgressReporter("run-1", root=tmp_path)
        for step in range(ProgressReporter.HISTORY + 50):
            reporter.update(step=step, loss=float(step))

        assert len(reporter.run.loss_history) == ProgressReporter.HISTORY
        # The newest are the ones kept.
        assert reporter.run.loss_history[-1] == float(ProgressReporter.HISTORY + 49)

    def test_a_non_numeric_metric_is_dropped_not_fatal(self, tmp_path):
        reporter = ProgressReporter("run-1", root=tmp_path)
        reporter.update(step=1, loss=0.5, note="epoch boundary")

        assert reporter.run.metrics == {"loss": 0.5}

    def test_an_unwritable_root_does_not_kill_the_run(self, tmp_path, monkeypatch):
        """Losing the reporting is bad. Losing six hours of training to
        a failed status write would be absurd."""
        reporter = ProgressReporter("run-1", root=tmp_path)

        def boom(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(Path, "write_text", boom)

        reporter.update(step=1, loss=0.1)  # must not raise

    def test_checkpoints_accumulate(self, tmp_path):
        reporter = ProgressReporter("run-1", root=tmp_path)
        reporter.checkpoint("/models/ckpt-1")
        reporter.checkpoint("/models/ckpt-2")

        run = TrainingMonitor(tmp_path).get("run-1")

        assert run.checkpoints == ["/models/ckpt-1", "/models/ckpt-2"]

    def test_finished_records_the_terminal_state(self, tmp_path):
        reporter = ProgressReporter("run-1", root=tmp_path)
        reporter.finished(state=RunState.FAILED.value, error="CUDA out of memory")

        run = TrainingMonitor(tmp_path).get("run-1")

        assert run.state == RunState.FAILED.value
        assert run.error == "CUDA out of memory"
        assert run.finished_at is not None
        assert run.is_active is False


# ===========================================================================
# Reconciliation: what the file says vs what is actually running
# ===========================================================================


class TestReconciliation:
    def test_a_dead_trainer_is_reported_stale_not_running(self, tmp_path):
        """The case the whole reconciliation step exists for.

        A segfault, an OOM kill or a reboot leaves a status file that
        still claims to be running. Believing it means a dashboard shows
        a healthy run that has not existed since Tuesday.
        """
        reporter = ProgressReporter("run-1", total_steps=100, root=tmp_path)
        reporter.update(step=10, loss=0.5)
        # A pid nothing is behind. 2**22 is above the default pid_max.
        reporter.run.pid = 4194303
        reporter._write()

        run = TrainingMonitor(tmp_path).get("run-1")

        assert run.state == RunState.STALE.value
        assert "did not finish cleanly" in run.error

    def test_a_finished_run_is_left_alone(self, tmp_path):
        """Its process is supposed to be gone."""
        reporter = ProgressReporter("run-1", root=tmp_path)
        reporter.finished()
        reporter.run.pid = 4194303
        reporter._write()

        run = TrainingMonitor(tmp_path).get("run-1")

        assert run.state == RunState.FINISHED.value
        assert run.error == ""

    def test_runs_are_newest_first(self, tmp_path):
        for index, run_id in enumerate(("old", "mid", "new")):
            reporter = ProgressReporter(run_id, root=tmp_path)
            reporter.run.started_at = 1000.0 + index
            reporter.finished()

        assert [r.run_id for r in TrainingMonitor(tmp_path).runs()] == [
            "new", "mid", "old"
        ]

    def test_unknown_fields_in_a_file_are_ignored(self, tmp_path):
        """A record written by a newer version must not crash an older
        reader — the file is on disk across upgrades."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "run-1.json").write_text(
            json.dumps({"run_id": "run-1", "state": "finished", "from_the_future": 1}),
            encoding="utf-8",
        )

        run = TrainingMonitor(tmp_path).get("run-1")

        assert run is not None and run.run_id == "run-1"

    def test_a_corrupt_file_is_skipped_not_fatal(self, tmp_path):
        (tmp_path / "good.json").write_text(
            json.dumps({"run_id": "good", "state": "finished"}), encoding="utf-8"
        )
        (tmp_path / "bad.json").write_text("{ truncated", encoding="utf-8")

        assert [r.run_id for r in TrainingMonitor(tmp_path).runs()] == ["good"]

    def test_a_missing_root_lists_nothing(self, tmp_path):
        assert TrainingMonitor(tmp_path / "nope").runs() == []


# ===========================================================================
# The controls, against real processes
# ===========================================================================


def _proc_state(pid: int) -> str:
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return stat.rsplit(")", 1)[1].split()[0]


@pytest.fixture
def sleeper():
    """A real child in its own session, so killpg hits it and not us."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        start_new_session=True,
    )
    yield proc
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _run_for(proc, tmp_path) -> TrainingRun:
    reporter = ProgressReporter("run-1", total_steps=100, root=tmp_path)
    reporter.update(step=1, loss=1.0)
    reporter.run.pid = proc.pid
    reporter._write()
    return TrainingMonitor(tmp_path).get("run-1")


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs procfs")
class TestControls:
    def test_pause_actually_stops_the_process(self, tmp_path, sleeper):
        monitor = TrainingMonitor(tmp_path)
        run = _run_for(sleeper, tmp_path)

        monitor.pause(run)

        assert _proc_state(sleeper.pid) in ("T", "t")
        assert monitor.get("run-1").state == RunState.PAUSED.value

    def test_resume_puts_it_back(self, tmp_path, sleeper):
        monitor = TrainingMonitor(tmp_path)
        run = _run_for(sleeper, tmp_path)

        monitor.pause(run)
        monitor.resume(run)

        assert _proc_state(sleeper.pid) not in ("T", "t")
        assert monitor.get("run-1").state == RunState.RUNNING.value

    def test_stopping_a_paused_run_really_ends_it(self, tmp_path, sleeper):
        """The bug this test exists for: a SIGSTOPped process cannot
        handle SIGTERM. Without a SIGCONT first, "stop" reports success
        and leaves the run frozen forever."""
        monitor = TrainingMonitor(tmp_path)
        run = _run_for(sleeper, tmp_path)
        monitor.pause(run)

        monitor.stop(run)

        assert sleeper.wait(timeout=10) is not None
        assert monitor.get("run-1").state == RunState.STOPPED.value

    def test_stop_is_terminate_not_kill(self, tmp_path):
        """A trainer that handles SIGTERM gets to write a last
        checkpoint. The difference between "stopped at epoch 4" and
        "lost epoch 4" is the whole value of asking politely."""
        proc = subprocess.Popen(
            [
                sys.executable, "-c",
                "import signal,sys,time\n"
                "signal.signal(signal.SIGTERM, lambda *a: sys.exit(7))\n"
                "time.sleep(120)\n",
            ],
            start_new_session=True,
        )
        try:
            # Let the handler install before signalling it.
            for _ in range(100):
                if _proc_state(proc.pid) == "S":
                    break
                time.sleep(0.05)
            TrainingMonitor(tmp_path).stop(_run_for(proc, tmp_path))

            assert proc.wait(timeout=10) == 7
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_a_run_with_no_pid_refuses_clearly(self, tmp_path):
        monitor = TrainingMonitor(tmp_path)
        ProgressReporter("run-1", root=tmp_path).run  # noqa: B018
        run = monitor.get("run-1")
        run.pid = 0

        with pytest.raises(TrainingError, match="no recorded pid"):
            monitor.pause(run)

    def test_pausing_something_already_gone_says_so(self, tmp_path):
        monitor = TrainingMonitor(tmp_path)
        ProgressReporter("run-1", root=tmp_path)
        run = monitor.get("run-1")
        run.pid = 4194303

        with pytest.raises(TrainingError, match="not running"):
            monitor.pause(run)


class TestResources:
    def test_it_reports_the_machine_without_a_gpu_present(self):
        """CPU-only is the common case in CI and must not be an error."""
        payload = TrainingMonitor().resources()

        assert isinstance(payload["gpus"], list)
        assert set(payload) >= {
            "gpus", "cpu_percent", "ram_total_mb", "ram_used_mb", "ram_percent"
        }

    def test_it_never_raises_when_psutil_is_missing(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_psutil(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_psutil)
        payload = TrainingMonitor().resources()

        assert payload["cpu_percent"] is None
        assert payload["ram_total_mb"] is None


# ===========================================================================
# The API surface, and who may reach it
# ===========================================================================

# The exact spelling the lint-regression check looks for: adding a
# reason= kwarg breaks its substring match, and the guard then reads as
# absent even though it is right there.
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import (  # noqa: E402
    Keymaster,
    KeyScope,
    KeyType,
)
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _no_ambient_trust(monkeypatch):
    """The environment must not decide what these tests are testing."""
    for name in (
        "T1_TRUSTED_NETWORK",
        "T1_TRUSTED_NETWORK_LAN",
        "T1_TRUSTED_NETWORK_TAILNET",
        "T1_TRUSTED_NETWORK_PARTIAL_ADMIN",
        "T1_TRUSTED_PROXIES",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def km(tmp_path) -> Keymaster:
    return Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)


@pytest.fixture
def gk(km, tmp_path) -> Gatekeeper:
    return Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper", log_to_file=False)


@pytest.fixture
def runs_root(tmp_path) -> Path:
    root = tmp_path / "training"
    root.mkdir()
    return root


def build_client(
    km, gk, tmp_path, runs_root, *, peer: str = "127.0.0.1", **trust
) -> TestClient:
    config = T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough",
        db_path=str(tmp_path / "t1.sqlite3"),
        module_storage_dir=str(tmp_path / "modules"),
        **trust,
    )
    app = create_app(config=config, keymaster=km, gatekeeper=gk)
    app.state.t1_training_monitor = TrainingMonitor(runs_root)
    return TestClient(app, client=(peer, 51234))


@pytest.fixture
def client(km, gk, tmp_path, runs_root) -> TestClient:
    return build_client(km, gk, tmp_path, runs_root)


@pytest.fixture
def admin_key(km) -> str:
    return km.create(
        key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}
    ).key


@pytest.fixture
def user_key(km) -> str:
    return km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def a_run(runs_root) -> ProgressReporter:
    """A finished run.

    ``pid`` is cleared deliberately. ``ProgressReporter`` records
    ``os.getpid()``, which inside pytest is *pytest* — a control test
    that reached the signalling path would SIGTERM its own process
    group and take the whole suite with it.
    """
    reporter = ProgressReporter(
        "run-1", name="qwen-sft", total_epochs=2, total_steps=100,
        model="qwen3-8b", root=runs_root,
    )
    reporter.update(step=40, epoch=1, loss=0.42, lr=2e-5)
    reporter.run.pid = 0
    reporter.finished()
    return reporter


@pytest.fixture
def a_running_run(runs_root) -> ProgressReporter:
    """A run in progress with no process behind it.

    Same reason as above: the controls have to be reachable without a
    real pid, so the permission tests can assert a 200 without any test
    signalling anything.
    """
    reporter = ProgressReporter(
        "run-1", name="qwen-sft", total_epochs=2, total_steps=100,
        model="qwen3-8b", root=runs_root,
    )
    reporter.run.pid = 0
    reporter.update(step=40, epoch=1, loss=0.42, lr=2e-5)
    return reporter


READS = (
    "/training/runs",
    "/training/resources",
    "/training/runs/run-1",
    "/training/runs/run-1/logs",
    "/training/runs/run-1/checkpoints",
)
CONTROLS = (
    "/training/runs/run-1/pause",
    "/training/runs/run-1/resume",
    "/training/runs/run-1/stop",
)


class TestAdminOnlyByDefault:
    """With trusted-network mode off, this is an admin surface."""

    @pytest.mark.parametrize("path", READS)
    def test_no_credential_is_401(self, client, a_run, path):
        assert client.get(path).status_code == 401

    @pytest.mark.parametrize("path", READS)
    def test_an_ordinary_key_is_403(self, client, a_run, user_key, path):
        response = client.get(path, headers=_auth(user_key))

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "AUTH_ADMIN_REQUIRED"

    @pytest.mark.parametrize("path", CONTROLS)
    def test_an_ordinary_key_cannot_control_a_run(self, client, a_run, user_key, path):
        assert client.post(path, headers=_auth(user_key)).status_code == 403

    @pytest.mark.parametrize("path", READS)
    def test_an_admin_key_gets_through(self, client, a_run, admin_key, path):
        assert client.get(path, headers=_auth(admin_key)).status_code == 200


class TestTrustedNetworkMode:
    """The spec's exception: admin-only *unless* trusted mode is on."""

    def test_a_lan_caller_may_read_with_no_key(self, km, gk, tmp_path, runs_root, a_run):
        client = build_client(
            km, gk, tmp_path, runs_root, peer="192.168.1.40", trusted_network=True
        )

        assert client.get("/training/runs").status_code == 200

    def test_but_may_not_stop_a_run(self, km, gk, tmp_path, runs_root, a_run):
        """Killing six hours of training is not something a device that
        presented no credential gets to do because it is on the same
        wifi. That needs the second opt-in."""
        client = build_client(
            km, gk, tmp_path, runs_root, peer="192.168.1.40", trusted_network=True
        )

        response = client.post("/training/runs/run-1/stop")

        assert response.status_code == 403
        assert response.json()["error"]["details"]["reason"] == (
            "training_control_requires_admin"
        )

    def test_partial_admin_is_what_unlocks_the_controls(
        self, km, gk, tmp_path, runs_root, a_running_run
    ):
        client = build_client(
            km, gk, tmp_path, runs_root, peer="192.168.1.40",
            trusted_network=True, trusted_network_partial_admin=True,
        )

        assert client.post("/training/runs/run-1/stop").status_code == 200

    def test_a_public_origin_never_qualifies(self, km, gk, tmp_path, runs_root, a_run):
        """Item 14's line, and the one that matters most: no
        configuration turns an unauthenticated internet connection into
        training administration."""
        client = build_client(
            km, gk, tmp_path, runs_root, peer="203.0.113.9",
            trusted_network=True, trusted_network_partial_admin=True,
        )

        assert client.get("/training/runs").status_code == 401
        assert client.post("/training/runs/run-1/stop").status_code == 401

    def test_a_reverse_proxy_cannot_launder_a_public_caller(
        self, km, gk, tmp_path, runs_root, a_run
    ):
        """nginx on the same host makes every request in the world
        arrive from 127.0.0.1. An unverifiable forwarded header has to
        collapse the origin to public, or trusted mode publishes the
        training controls to the internet."""
        client = build_client(
            km, gk, tmp_path, runs_root, peer="127.0.0.1",
            trusted_network=True, trusted_network_partial_admin=True,
        )

        response = client.post(
            "/training/runs/run-1/stop", headers={"x-forwarded-for": "203.0.113.9"}
        )

        assert response.status_code == 401

    def test_presenting_a_read_key_does_not_lose_you_lan_access(
        self, km, gk, tmp_path, runs_root, a_run, user_key
    ):
        """Otherwise "authenticating made you less trusted than staying
        anonymous" would be a rule people design around by not sending
        their key."""
        client = build_client(
            km, gk, tmp_path, runs_root, peer="192.168.1.40", trusted_network=True
        )

        assert client.get(
            "/training/runs", headers=_auth(user_key)
        ).status_code == 200

    def test_an_admin_key_still_works_from_anywhere(
        self, km, gk, tmp_path, runs_root, a_running_run, admin_key
    ):
        client = build_client(
            km, gk, tmp_path, runs_root, peer="203.0.113.9"
        )

        assert client.post(
            "/training/runs/run-1/stop", headers=_auth(admin_key)
        ).status_code == 200


class TestReadingRuns:
    def test_the_list_reports_progress_and_the_active_count(
        self, client, a_run, admin_key
    ):
        body = client.get("/training/runs", headers=_auth(admin_key)).json()

        assert body["count"] == 1
        assert body["active"] == 0
        run = body["runs"][0]
        assert run["run_id"] == "run-1"
        assert run["name"] == "qwen-sft"
        assert run["percent"] == 40.0
        assert run["metrics"]["loss"] == 0.42

    def test_active_filters_without_lying_about_the_total(
        self, client, runs_root, a_run, admin_key
    ):
        live = ProgressReporter("run-2", total_steps=10, root=runs_root)
        live.update(step=1, loss=1.0)

        body = client.get(
            "/training/runs", params={"active": "true"}, headers=_auth(admin_key)
        ).json()

        assert [r["run_id"] for r in body["runs"]] == ["run-2"]
        assert body["active"] == 1

    def test_an_unknown_run_is_404(self, client, admin_key):
        response = client.get("/training/runs/nope", headers=_auth(admin_key))

        assert response.status_code == 404

    def test_a_run_with_no_log_answers_200_with_a_reason(
        self, client, a_run, admin_key
    ):
        """Not 404. "No such run" and "that run has no log" are
        different, and the alarming one must not stand in for the
        ordinary one."""
        body = client.get("/training/runs/run-1/logs", headers=_auth(admin_key)).json()

        assert body["log"] == ""
        assert "started outside the launcher" in body["detail"]

    def test_logs_are_tailed(self, client, runs_root, a_run, admin_key):
        log = runs_root / "run-1.log"
        log.write_text("\n".join(f"line {n}" for n in range(500)), encoding="utf-8")
        record = json.loads((runs_root / "run-1.json").read_text(encoding="utf-8"))
        record["log_path"] = str(log)
        (runs_root / "run-1.json").write_text(json.dumps(record), encoding="utf-8")

        body = client.get(
            "/training/runs/run-1/logs", params={"tail": 10}, headers=_auth(admin_key)
        ).json()

        assert body["lines"] == 10
        assert body["log"].splitlines()[-1] == "line 499"

    def test_checkpoints_say_whether_they_are_still_there(
        self, client, runs_root, a_run, admin_key
    ):
        """A checkpoint list is used to decide what to resume from, so
        a path that has since been deleted is the case worth knowing."""
        present = runs_root / "ckpt-1.safetensors"
        present.write_bytes(b"weights")
        record = json.loads((runs_root / "run-1.json").read_text(encoding="utf-8"))
        record["checkpoints"] = [str(present), str(runs_root / "gone.safetensors")]
        (runs_root / "run-1.json").write_text(json.dumps(record), encoding="utf-8")

        body = client.get(
            "/training/runs/run-1/checkpoints", headers=_auth(admin_key)
        ).json()

        assert [c["exists"] for c in body["checkpoints"]] == [True, False]
        assert body["checkpoints"][0]["size_bytes"] == 7

    def test_resources_is_a_literal_path_not_a_run_id(self, client, admin_key):
        """``/training/resources`` must not be matched as
        ``/training/runs/{run_id}``'s sibling by accident."""
        response = client.get("/training/resources", headers=_auth(admin_key))

        assert response.status_code == 200
        assert "gpus" in response.json()


class TestControlsOverHTTP:
    def test_pausing_says_the_card_is_not_released(
        self, client, runs_root, admin_key
    ):
        """The thing everyone assumes the opposite of. SIGSTOP freezes
        the process with its VRAM intact."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            start_new_session=True,
        )
        try:
            reporter = ProgressReporter("run-1", total_steps=10, root=runs_root)
            reporter.run.pid = proc.pid
            reporter._write()

            body = client.post(
                "/training/runs/run-1/pause", headers=_auth(admin_key)
            ).json()

            assert body["run"]["state"] == "paused"
            assert "not released while paused" in body["note"]
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            proc.wait(timeout=5)

    def test_pausing_a_finished_run_is_409_not_500(self, client, a_run, admin_key):
        response = client.post("/training/runs/run-1/pause", headers=_auth(admin_key))

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "CONFLICT"
        assert "already ended" in response.json()["error"]["message"]

    def test_stopping_a_finished_run_does_not_rewrite_how_it_ended(
        self, client, a_run, admin_key
    ):
        """Overwriting ``finished`` with ``stopped`` would leave the
        history saying an operator killed a job that in fact completed."""
        assert client.post(
            "/training/runs/run-1/stop", headers=_auth(admin_key)
        ).status_code == 409

        body = client.get("/training/runs/run-1", headers=_auth(admin_key)).json()
        assert body["run"]["state"] == "finished"

    def test_a_control_is_audited(self, client, a_running_run, admin_key):
        """Ending someone's training run is as privileged as rotating a
        key, and leaves the same trail."""
        client.post("/training/runs/run-1/stop", headers=_auth(admin_key))

        body = client.get(
            "/audit", params={"action": "training.stop"}, headers=_auth(admin_key)
        ).json()

        assert body["count"] >= 1
        assert body["events"][0]["resource_id"] == "run-1"
        assert body["events"][0]["category"] == "admin"


# ===========================================================================
# The hand-off: launcher -> trainer -> monitor
# ===========================================================================


class TestTheTrainerSideHook:
    """``train()`` publishes progress without the caller arranging it.

    Tested through ``_progress_reporter`` rather than through ``train``
    itself: standing up a real training run needs torch, a model
    snapshot and a dataset, and none of that is what this wiring is.
    """

    def test_no_id_anywhere_means_no_reporting(self, monkeypatch):
        from hypernix.training.train import _progress_reporter

        monkeypatch.delenv("HNX_RUN_ID", raising=False)

        assert _progress_reporter(None, total_steps=10, model="m") is None

    def test_the_launcher_environment_is_enough(self, monkeypatch, tmp_path):
        from hypernix.training.train import _progress_reporter

        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("HNX_RUN_ID", "job-abc123")
        monkeypatch.setenv("HNX_JOB_NAME", "qwen-sft")
        monkeypatch.setenv("HNX_LOG_PATH", str(tmp_path / "job.log"))

        reporter = _progress_reporter(None, total_steps=10, model="m")

        assert reporter is not None
        assert reporter.run.run_id == "job-abc123"
        assert reporter.run.name == "qwen-sft"
        assert reporter.run.log_path == str(tmp_path / "job.log")

    def test_an_explicit_id_beats_the_environment(self, monkeypatch, tmp_path):
        from hypernix.training.train import _progress_reporter

        monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("HNX_RUN_ID", "from-env")

        reporter = _progress_reporter("explicit", total_steps=10, model="m")

        assert reporter.run.run_id == "explicit"

    def test_reporting_that_cannot_start_does_not_stop_the_run(
        self, monkeypatch, tmp_path
    ):
        """The whole point of the try/except around it: losing the
        dashboard is annoying, losing six hours of training to it is
        not a trade anyone would take."""
        from hypernix.training.train import _progress_reporter

        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setenv("T1_CONFIG_DIR", str(blocker))
        monkeypatch.setenv("HNX_RUN_ID", "job-abc123")

        assert _progress_reporter(None, total_steps=10, model="m") is None

    def test_a_crash_is_recorded_as_failed_not_left_running(self, tmp_path):
        """``train()`` catches BaseException for this. A run killed by
        Ctrl-C or an OOM must not sit in the dashboard claiming to be
        making progress."""
        reporter = ProgressReporter("run-1", total_steps=100, root=tmp_path)
        reporter.update(step=3, loss=1.0)

        reporter.failed("RuntimeError: CUDA out of memory")

        run = TrainingMonitor(tmp_path).get("run-1")
        assert run.state == RunState.FAILED.value
        assert "CUDA out of memory" in run.error
        assert run.is_active is False


class TestTheLauncherHandOff:
    def test_it_tells_the_job_what_to_report_as(self, tmp_path):
        """``launch-script ./train.py`` has to be enough. If the run id
        had to be passed by hand, the runs people most want to watch —
        the detached ones — would be the ones that never appear."""
        from hypernix.system.launcher import JobStore, Supervisor, launch, read_logs

        work = tmp_path / "job.sh"
        work.write_text(
            "#!/bin/sh\n"
            'echo "RUN=${HNX_RUN_ID:-unset}"\n'
            'echo "NAME=${HNX_JOB_NAME:-unset}"\n'
            'echo "LOG=${HNX_LOG_PATH:-unset}"\n',
            encoding="utf-8",
        )
        work.chmod(0o755)
        store = JobStore(tmp_path / "jobs")

        job = launch(
            work, name="qwen-sft", store=store, supervisor=Supervisor.SETSID
        )
        deadline = time.time() + 20.0
        while time.time() < deadline and "LOG=" not in read_logs(job):
            time.sleep(0.1)
        output = read_logs(job)

        assert f"RUN={job.job_id}" in output
        assert "NAME=qwen-sft" in output
        assert f"LOG={job.log_path}" in output

    def test_the_run_id_is_the_job_id_not_the_name(self, tmp_path):
        """So relaunching a job of the same name supersedes nothing.
        The monitor still resolves the name, because ``get`` falls back
        to matching on it."""
        from hypernix.system.launcher import JobStore, Supervisor, launch

        work = tmp_path / "job.sh"
        work.write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
        work.chmod(0o755)
        store = JobStore(tmp_path / "jobs")

        first = launch(work, name="same", store=store, supervisor=Supervisor.SETSID)
        second = launch(work, name="same", store=store, supervisor=Supervisor.SETSID)

        assert first.job_id != second.job_id

    def test_a_run_is_reachable_by_its_job_name(self, runs_root):
        reporter = ProgressReporter(
            "job-abc123", name="qwen-sft", root=runs_root
        )
        reporter.update(step=1, loss=1.0)

        monitor = TrainingMonitor(runs_root)

        assert monitor.get("qwen-sft").run_id == "job-abc123"


# ===========================================================================
# The operator's command line
# ===========================================================================


class TestTheCommandLine:
    """``hypernix-t1 training``, which needs no server.

    That is the point of it: the moment you most want to know what a run
    is doing is usually the moment the API is the thing in trouble.
    """

    def run_cli(self, capsys, *argv, root):
        from hypernix.t1api.training_cli import main

        code = main(["--root", str(root), *argv])
        return code, capsys.readouterr()

    def test_it_lists_what_is_there(self, capsys, runs_root):
        reporter = ProgressReporter(
            "job-1", name="qwen-sft", total_steps=1000, root=runs_root
        )
        reporter.update(step=430, loss=1.234)

        code, out = self.run_cli(capsys, root=runs_root)

        assert code == 0
        assert "qwen-sft" in out.out
        assert "43.0%" in out.out

    def test_nothing_recorded_says_how_to_start_one(self, capsys, runs_root):
        code, out = self.run_cli(capsys, root=runs_root)

        assert code == 0
        assert "launch-script" in out.out

    def test_nothing_running_is_a_different_answer(self, capsys, runs_root, a_run):
        """"No runs recorded" and "nothing is running" send people
        looking for different things, and only one of them is a lost
        record."""
        code, out = self.run_cli(capsys, "--active", root=runs_root)

        assert code == 0
        assert "No training is running" in out.out
        assert "1 run(s) are recorded" in out.out

    def test_a_dead_run_advertises_no_eta(self, capsys, runs_root):
        """It keeps its last measured rate, so an ETA computed from it
        reads as "nearly finished" for a trainer that died an hour ago."""
        reporter = ProgressReporter("job-1", total_steps=1000, root=runs_root)
        reporter.update(step=430, loss=1.0)
        reporter.run.pid = 4194303
        reporter._write()

        code, out = self.run_cli(capsys, root=runs_root)

        assert "eta —" in out.out
        assert "stale" in out.out

    def test_show_marks_a_checkpoint_that_is_gone(self, capsys, runs_root):
        reporter = ProgressReporter("job-1", name="qwen-sft", root=runs_root)
        reporter.checkpoint(runs_root / "not-written.safetensors")

        code, out = self.run_cli(capsys, "--show", "qwen-sft", root=runs_root)

        assert code == 0
        assert "(gone)" in out.out

    def test_an_unknown_run_lists_the_known_ones(self, capsys, runs_root, a_run):
        code, out = self.run_cli(capsys, "--show", "typo", root=runs_root)

        assert code == 1
        assert "known runs: qwen-sft" in out.err

    def test_json_is_machine_readable(self, capsys, runs_root, a_run):
        code, out = self.run_cli(capsys, "--json", root=runs_root)

        body = json.loads(out.out)
        assert body["count"] == 1
        assert body["runs"][0]["run_id"] == "run-1"

    def test_controlling_a_finished_run_fails_loudly(self, capsys, runs_root, a_run):
        code, out = self.run_cli(capsys, "--stop", "qwen-sft", root=runs_root)

        assert code == 1
        assert "already ended" in out.err

    def test_resources_works_with_no_gpu(self, capsys, runs_root):
        code, out = self.run_cli(capsys, "--resources", root=runs_root)

        assert code == 0
        assert "CPU" in out.out
