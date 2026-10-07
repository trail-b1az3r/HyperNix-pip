"""The release gate: a live API, a real key, a fake model, an iPhone.

Nothing here runs GitHub Actions — it cannot. What it does check is that
the workflow says what it is supposed to say, and that the scripts those
jobs invoke actually work, which is the half that breaks silently.

The shape is not the obvious one, deliberately. Hosted runners cannot
reach each other over the network, so "job A hosts the server and job B
connects to it" is not expressible: there is no route between two
runners. Each job brings up its own server. The split that matters
survives — one job drives the API from outside the process, the other
drives the app against a real server, and no publish step runs until both
are green.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
RELEASE = REPO_ROOT / ".github" / "workflows" / "public-release.yml"
SCRIPTS = REPO_ROOT / "scripts" / "ci"

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestTheJobsExist:
    @pytest.mark.parametrize("path", [CI, RELEASE])
    def test_both_workflows_have_both_jobs(self, path):
        jobs = workflow(path)["jobs"]
        assert "integration-api" in jobs
        assert "integration-ios" in jobs

    def test_ci_runs_them_after_the_tests(self):
        jobs = workflow(CI)["jobs"]
        assert jobs["integration-api"]["needs"] == "test"
        assert jobs["integration-ios"]["needs"] == "test"

    def test_the_ios_job_runs_on_macos(self):
        """A simulator needs a Mac. Anywhere else the job would look
        green because xcodebuild never ran."""
        assert workflow(CI)["jobs"]["integration-ios"]["runs-on"] == "macos-latest"

    def test_each_job_mints_its_own_key(self):
        jobs = workflow(CI)["jobs"]
        for name in ("integration-api", "integration-ios"):
            steps = json.dumps(jobs[name]["steps"])
            assert "integration_probe.py" in steps, name

    def test_both_jobs_agree_with_gkey_about_the_key_store(self):
        """The server reads T1_KEYMASTER_DIR and so does gkey.

        Leaving either to infer it from $HOME is how a key gets minted
        into a store the server is not reading — which presents as a
        rejected key, several layers from the cause.
        """
        jobs = workflow(CI)["jobs"]
        for name in ("integration-api", "integration-ios"):
            assert "T1_KEYMASTER_DIR" in jobs[name]["env"], name


class TestTeardown:
    @pytest.mark.parametrize("name", ["integration-api", "integration-ios"])
    def test_teardown_runs_even_when_the_test_failed(self, name):
        """A job that leaves credentials behind on failure is worse than
        one that fails."""
        steps = workflow(CI)["jobs"][name]["steps"]
        teardown = [s for s in steps if "key" in (s.get("name") or "").lower()
                    and "delete" in (s.get("name") or "").lower()
                    or "Shut it" in (s.get("name") or "")]
        assert teardown, f"{name} has no teardown step"
        assert any(s.get("if") == "always()" for s in teardown), (
            f"{name} tears down only on success"
        )

    @pytest.mark.parametrize("name", ["integration-api", "integration-ios"])
    def test_teardown_fails_the_job_if_keys_remain(self, name):
        """Checked, not assumed. A teardown that cannot fail is a comment."""
        steps = json.dumps(workflow(CI)["jobs"][name]["steps"])
        assert "left ${remaining} keys behind" in steps, name

    def test_the_ios_job_shuts_the_simulator_down(self):
        steps = json.dumps(workflow(CI)["jobs"]["integration-ios"]["steps"])
        assert "simctl shutdown" in steps


class TestTheReleaseIsGated:
    @pytest.mark.parametrize(
        "job", ["github-release", "pypi-publish", "testpypi-publish"]
    )
    def test_nothing_publishes_until_both_pass(self, job):
        needs = workflow(RELEASE)["jobs"][job]["needs"]
        assert "integration-api" in needs, job
        assert "integration-ios" in needs, job

    def test_the_release_jobs_run_after_the_build(self):
        jobs = workflow(RELEASE)["jobs"]
        assert jobs["integration-api"]["needs"] == "cut"
        assert jobs["integration-ios"]["needs"] == "cut"


class TestTheFakeModel:
    """A stub over HTTP, not a mock patched into the bridge.

    The stub goes through the real bridge, the real routing engine and
    the real serialisation, which is where the interesting failures are.
    A mock would prove the mock works.
    """

    @pytest.fixture(scope="class")
    def server(self):
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        process = subprocess.Popen(
            [sys.executable, str(SCRIPTS / "fake_model_server.py"), "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        base = f"http://127.0.0.1:{port}"
        for _ in range(60):
            try:
                _OPENER.open(base + "/v1/models", timeout=1).read()
                break
            except OSError:
                time.sleep(0.2)
        else:
            process.kill()
            pytest.fail("the fake model never came up")
        yield base
        process.terminate()
        process.wait(timeout=10)

    def test_it_lists_a_model(self, server):
        body = json.loads(_OPENER.open(server + "/v1/models", timeout=5).read())
        assert [m["id"] for m in body["data"]] == ["hypernix-ci-echo"]

    def test_a_chat_returns_the_marker(self, server):
        """Distinctive on purpose: a test asserting on "hello" can pass
        against a server that echoed the prompt back."""
        request = urllib.request.Request(
            server + "/v1/chat/completions",
            data=json.dumps(
                {"model": "hypernix-ci-echo",
                 "messages": [{"role": "user", "content": "ping"}]}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(_OPENER.open(request, timeout=5).read())
        content = body["choices"][0]["message"]["content"]
        assert "CI-ECHO-OK" in content

    def test_it_proves_the_prompt_arrived(self, server):
        """Otherwise the check passes against a server that ignored the
        request body entirely."""
        request = urllib.request.Request(
            server + "/v1/chat/completions",
            data=json.dumps(
                {"messages": [{"role": "user", "content": "distinctive-phrase"}]}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(_OPENER.open(request, timeout=5).read())
        assert "distinctive-phrase" in body["choices"][0]["message"]["content"]

    def test_it_understands_vision_style_parts(self, server):
        """The app sends content as a list of parts, not a string."""
        request = urllib.request.Request(
            server + "/v1/chat/completions",
            data=json.dumps({
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": "look-at-this"}],
                }]
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(_OPENER.open(request, timeout=5).read())
        assert "look-at-this" in body["choices"][0]["message"]["content"]

    def test_it_reports_usage(self, server):
        """The usage meter records what a call cost; zero tokens would
        make every cost assertion downstream vacuous."""
        request = urllib.request.Request(
            server + "/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "x" * 40}]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(_OPENER.open(request, timeout=5).read())
        assert body["usage"]["total_tokens"] > 0


class TestTheProbeScript:
    def test_it_parses(self):
        subprocess.run(
            [sys.executable, "-m", "py_compile", str(SCRIPTS / "integration_probe.py")],
            check=True,
        )

    def test_it_deletes_its_keys_in_a_finally(self):
        """Teardown belongs in `finally`, not after the last assert: the
        interesting case is the run that failed."""
        source = (SCRIPTS / "integration_probe.py").read_text(encoding="utf-8")
        finally_block = source[source.index("    finally:"):]
        assert "revoke_key" in finally_block

    def test_it_bypasses_the_proxy(self):
        """CI runners set HTTP_PROXY more often than not, and a proxy in
        between answers a different question than "is this server up"."""
        source = (SCRIPTS / "integration_probe.py").read_text(encoding="utf-8")
        assert "ProxyHandler({})" in source

    def test_it_waits_rather_than_sleeping_a_fixed_time(self):
        """A fixed sleep is either too short on a loaded runner or wasted
        time on a fast one, and is the usual reason CI is flaky."""
        source = (SCRIPTS / "integration_probe.py").read_text(encoding="utf-8")
        assert "def wait_for(" in source


def _job_steps_text(path: Path, job: str) -> str:
    """Every `run:` block of a job, concatenated."""
    steps = workflow(path)["jobs"][job]["steps"]
    return "\n".join(step.get("run", "") for step in steps)


class TestEveryJobWaitsForEveryServer:
    """The failure that produced this class.

    Both integration jobs start two background servers and then talk to
    them. Whether each one was waited for used to be written out by hand
    per job, and the macOS job waited only for the API — so on a runner
    slow enough to matter, the probe reached the bridge before the fake
    model was listening and the build failed with MODEL_UNAVAILABLE,
    which reads as a broken bridge and is actually a race.
    """

    @pytest.mark.parametrize("path", [CI, RELEASE])
    @pytest.mark.parametrize("job", ["integration-api", "integration-ios"])
    def test_it_waits_for_the_fake_model(self, path, job):
        text = _job_steps_text(path, job)
        assert "fake_model_server.py" in text, "does not start the fake model"
        assert "wait_for_http.py http://127.0.0.1:1234" in text, (
            f"{path.name}:{job} starts the fake model and never waits for it"
        )

    @pytest.mark.parametrize("path", [CI, RELEASE])
    @pytest.mark.parametrize("job", ["integration-api", "integration-ios"])
    def test_it_waits_for_the_api(self, path, job):
        text = _job_steps_text(path, job)
        assert "wait_for_http.py http://127.0.0.1:8000" in text, (
            f"{path.name}:{job} starts the API and never waits for it"
        )

    @pytest.mark.parametrize("path", [CI, RELEASE])
    @pytest.mark.parametrize("job", ["integration-api", "integration-ios"])
    def test_the_wait_prints_the_log_on_failure(self, path, job):
        """A timeout with no log costs a second run to diagnose."""
        text = _job_steps_text(path, job)
        for waited in ("fake-model.log", "t1api.log"):
            assert f"--log {waited}" in text, f"{path.name}:{job} loses {waited}"

    @pytest.mark.parametrize("path", [CI, RELEASE])
    @pytest.mark.parametrize("job", ["integration-api", "integration-ios"])
    def test_no_job_still_hand_rolls_a_wait_loop(self, path, job):
        """One helper, or the two jobs drift apart again."""
        assert "seq 1 " not in _job_steps_text(path, job)

    def test_the_helper_exists_and_is_executable(self):
        helper = SCRIPTS / "wait_for_http.py"
        assert helper.exists()
        assert os.access(helper, os.X_OK)


class TestSkippingTheIntegrationGate:
    """`skip_integration` is for a runner outage, not a red test.

    The subtle half is downstream: a job that `needs` a skipped job is
    itself skipped by default, so gating the integration jobs without
    also teaching the publish jobs that "skipped" is acceptable would
    have made the flag silently cancel the release instead of the gate.
    """

    def test_the_input_exists(self):
        trigger = workflow(RELEASE)[True]["workflow_dispatch"]
        assert "skip_integration" in trigger["inputs"]
        assert trigger["inputs"]["skip_integration"]["default"] is False

    def test_it_says_what_it_is_giving_up(self):
        desc = workflow(RELEASE)[True]["workflow_dispatch"]["inputs"][
            "skip_integration"
        ]["description"].lower()
        assert "gate" in desc or "proves" in desc
        assert "never" in desc or "only" in desc

    @pytest.mark.parametrize("job", ["integration-api", "integration-ios"])
    def test_it_gates_both_jobs(self, job):
        condition = workflow(RELEASE)["jobs"][job]["if"]
        assert "skip_integration" in condition, job

    @pytest.mark.parametrize(
        "job", ["github-release", "pypi-publish", "testpypi-publish"]
    )
    def test_publishing_survives_a_skip_but_not_a_failure(self, job):
        condition = workflow(RELEASE)["jobs"][job]["if"]
        # always(), or a skipped dependency skips this job too.
        assert "always()" in condition, job
        # ...which means success is no longer implied and has to be asked for.
        assert "needs.cut.result == 'success'" in condition, job
        for dep in ("integration-api", "integration-ios"):
            assert f"needs.{dep}.result" in condition, f"{job} ignores {dep}"
            assert '"success","skipped"' in condition, job

    @pytest.mark.parametrize(
        "job", ["github-release", "pypi-publish", "testpypi-publish"]
    )
    def test_a_failed_gate_is_not_in_the_accepted_set(self, job):
        """The whole point: skipped is fine, failed is not."""
        condition = workflow(RELEASE)["jobs"][job]["if"]
        assert "failure" not in condition, (
            f"{job} accepts a failed integration job"
        )
