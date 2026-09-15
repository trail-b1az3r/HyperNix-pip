"""The server's hardware, and how long it has been up.

"Is the server busy, or is my model just slow?" cannot be answered from
a phone six hundred miles away by any other means, and "why did my
session vanish" is almost always "the API restarted four minutes ago".
The dashboards have sampled all of this for releases; these are the same
numbers with no terminal attached.

Two different gates, on purpose
-------------------------------
Uptime takes any HyperLink caller: it is not a description of anybody's
hardware, it is the answer to a question about their own session.

The hardware snapshot takes admin or partial admin, because it *is* a
description of somebody's hardware — how much memory the machine has,
what GPU is in it, how hot it is running.

Everything is optional
----------------------
Most of this file is about that. psutil may be absent, /proc is not on
macOS, no GPU vendor tool is guaranteed. A snapshot that filled those in
with zeros would be a dashboard confidently reporting an overloaded
machine as idle, so a reading that could not be taken is None and
`unavailable` says why.
"""
from __future__ import annotations

import logging

import pytest

from hypernix.system import hardware


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    logging.disable(logging.CRITICAL)
    for name in list(__import__("os").environ):
        if name.startswith("T1_"):
            monkeypatch.delenv(name, raising=False)
    yield
    logging.disable(logging.NOTSET)


class TestTheSnapshot:
    def test_it_samples_without_raising(self):
        assert hardware.snapshot().sampled_at > 0

    def test_it_names_the_machine(self):
        reading = hardware.snapshot()
        assert reading.hostname
        assert reading.platform

    def test_the_process_uptime_is_always_available(self):
        """Machine uptime can fail — /proc is not everywhere — but this
        server knows when this server started."""
        assert hardware.process_uptime_seconds() >= 0

    def test_process_uptime_is_not_machine_uptime(self):
        """The interesting one is usually the process. "The API restarted
        an hour ago" explains a dropped session in a way "the box has
        been up nine days" does not."""
        machine = hardware.uptime_seconds()
        if machine is None:
            pytest.skip("this platform does not report machine uptime")
        assert hardware.process_uptime_seconds() <= machine + 1

    def test_it_serialises(self):
        data = hardware.snapshot().to_dict()
        assert set(data) >= {
            "cpu", "memory", "swap", "disks", "gpus", "unavailable",
            "uptime_seconds", "process_uptime_seconds",
        }

    def test_memory_is_reported_in_bytes(self):
        """Bytes on the wire throughout: a client that formats one field
        in MB and another in bytes gets it wrong eventually, and the
        server is the place to be consistent."""
        memory = hardware.snapshot().memory
        if memory.total_bytes is None:
            pytest.skip("psutil is not installed")
        # A machine with less than 256 MB is not running this.
        assert memory.total_bytes > 256 * 1024 * 1024

    def test_a_disk_reading_adds_up(self):
        for disk in hardware.snapshot().disks:
            if disk.total_bytes and disk.used_bytes is not None:
                assert 0 <= disk.used_bytes <= disk.total_bytes

    def test_two_paths_on_one_filesystem_are_listed_once(self, tmp_path):
        """A container has dozens of mounts and a phone screen has room
        for two."""
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        reading = hardware.snapshot(disk_paths=[str(tmp_path), str(nested)])
        assert len(reading.disks) == 1


class TestNothingIsInvented:
    def test_missing_psutil_leaves_readings_none_and_says_so(self, monkeypatch):
        """The important property. A zero here is a dashboard reporting a
        machine at 100% as idle."""
        monkeypatch.setattr(hardware, "_psutil", lambda: None)
        reading = hardware.snapshot()
        assert reading.cpu.percent is None
        assert reading.memory.total_bytes is None
        assert any("psutil" in note for note in reading.unavailable)

    def test_a_missing_disk_is_reported_not_raised(self):
        reading = hardware.snapshot(disk_paths=["/definitely/not/here"])
        assert reading.disks == []
        assert any("/definitely/not/here" in note for note in reading.unavailable)

    def test_a_broken_gpu_query_does_not_take_the_snapshot_down(self, monkeypatch):
        """No vendor tool is guaranteed present, and a driver can hang."""
        import hypernix.system.gpus as gpus

        monkeypatch.setattr(
            gpus, "detect", lambda *a, **k: (_ for _ in ()).throw(OSError("no driver"))
        )
        reading = hardware.snapshot()
        assert reading.gpus == []
        assert any("gpu" in note for note in reading.unavailable)
        # And the rest of the sample still happened.
        assert reading.hostname

    def test_no_load_average_is_not_a_zero_load_average(self, monkeypatch):
        """Windows has no load average rather than a load average of
        zero, and those look very different on a dashboard."""
        monkeypatch.delattr("os.getloadavg", raising=False)
        assert hardware.snapshot().cpu.load_average == []


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


def client(**env) -> TestClient:
    import os

    from hypernix.t1api.app import create_app

    os.environ["T1_TRUSTED_NETWORK"] = "1"
    for key, value in env.items():
        os.environ[key] = value
    return TestClient(create_app(), client=("192.168.1.50", 5432))


class TestUptimeIsOpenToAnyCaller:
    def test_a_plain_keyless_caller_can_read_it(self):
        response = client().get("/hyperlink/uptime")
        assert response.status_code == 200

    def test_it_reports_both_clocks(self):
        body = client().get("/hyperlink/uptime").json()
        assert body["process_uptime_seconds"] >= 0
        assert "machine_uptime_seconds" in body

    def test_it_names_the_version(self):
        """So the app can tell an old server from a new one, which is
        what the command-copy area exists to act on."""
        body = client().get("/hyperlink/uptime").json()
        assert body["t1_version"]

    def test_started_at_is_consistent_with_the_uptime(self):
        import time

        body = client().get("/hyperlink/uptime").json()
        assert abs(
            (time.time() - body["started_at"]) - body["process_uptime_seconds"]
        ) < 5


class TestHardwareNeedsPartialAdminAtLeast:
    def test_a_read_only_caller_is_refused(self):
        response = client(T1_TRUSTED_NETWORK_PARTIAL_ADMIN="0").get(
            "/hyperlink/hardware"
        )
        assert response.status_code == 403

    def test_the_refusal_says_what_would_work(self):
        """A 403 that does not say how to stop being 403 is a dead end."""
        body = client(T1_TRUSTED_NETWORK_PARTIAL_ADMIN="0").get(
            "/hyperlink/hardware"
        ).json()
        assert "T1_TRUSTED_NETWORK_PARTIAL_ADMIN" in body["error"]["details"]["remedy"]

    def test_partial_admin_is_enough(self):
        """The case this exists for: the phone on the sofa, paired to
        nothing, asking what the PC is doing."""
        response = client(T1_TRUSTED_NETWORK_PARTIAL_ADMIN="1").get(
            "/hyperlink/hardware"
        )
        assert response.status_code == 200

    def test_it_carries_the_readings(self):
        body = client(T1_TRUSTED_NETWORK_PARTIAL_ADMIN="1").get(
            "/hyperlink/hardware"
        ).json()
        assert "cpu" in body and "memory" in body and "disks" in body
        assert body["hostname"]

    def test_unauthenticated_from_a_public_origin_is_still_refused(self):
        import os

        from hypernix.t1api.app import create_app

        os.environ["T1_TRUSTED_NETWORK"] = "1"
        os.environ["T1_TRUSTED_NETWORK_PARTIAL_ADMIN"] = "1"
        public = TestClient(create_app(), client=("8.8.8.8", 5432))
        assert public.get("/hyperlink/hardware").status_code == 401
