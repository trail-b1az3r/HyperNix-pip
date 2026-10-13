"""hypernix.monitoring.hostprobe: measured, not estimated.

Every source is fed from a fake -- a /proc/meminfo, a cgroup, a sysfs
hwmon tree, pactl's JSON -- so these check the arithmetic and the
parsing on any machine.
"""
from __future__ import annotations

import json
import math

import pytest

from hypernix.monitoring import hostprobe as hp

MEMINFO = """\
MemTotal:       16000000 kB
MemFree:         2000000 kB
MemAvailable:    6000000 kB
Buffers:          500000 kB
Cached:          4000000 kB
SwapCached:            0 kB
Shmem:            300000 kB
SReclaimable:     800000 kB
SwapTotal:       2000000 kB
SwapFree:        1500000 kB
"""


class TestMemory:
    def test_used_is_what_cannot_be_reclaimed_and_the_percent_agrees(self):
        m = hp._from_meminfo(hp._meminfo(MEMINFO))
        assert m["total_mib"] == 16_000_000 // 1024
        assert m["used_mib"] == 10_000_000 // 1024
        assert m["available_mib"] == 6_000_000 // 1024
        assert m["free_mib"] == 2_000_000 // 1024
        assert m["percent"] == pytest.approx(62.5)
        assert m["percent"] == pytest.approx(100 * 10_000_000 / 16_000_000)
        assert m["cached_mib"] == (500_000 + 4_000_000 + 800_000 - 300_000) // 1024
        assert m["swap_used_mib"] == 500_000 // 1024

    def test_a_container_limit_is_the_total(self, tmp_path):
        cg = tmp_path / "cg" / "box"
        cg.mkdir(parents=True)
        (cg / "memory.max").write_text("2147483648\n")
        (cg / "memory.current").write_text(str(1_500_000_000))
        (cg / "memory.stat").write_text("anon 900000000\nfile 600000000\ninactive_file 500000000\nshmem 0\n")
        m = hp._from_cgroup(16 * 1024**3, root=tmp_path / "cg", proc_cgroup="0::/box\n")
        assert m["source"] == "cgroup"
        assert m["total_mib"] == 2048
        assert m["used_mib"] == 1_000_000_000 // (1024 * 1024)
        assert m["percent"] == pytest.approx(100 * 1_000_000_000 / 2147483648, abs=0.1)

    def test_no_limit_means_the_host(self, tmp_path):
        cg = tmp_path / "cg"
        cg.mkdir()
        (cg / "memory.max").write_text("max\n")
        (cg / "memory.current").write_text("123")
        assert hp._from_cgroup(16 * 1024**3, root=cg, proc_cgroup="0::/\n") is None

    def test_macos_counts_like_activity_monitor(self):
        vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               10000.
Pages active:                            200000.
Pages wired down:                        100000.
Pages purgeable:                          20000.
File-backed pages:                       150000.
Anonymous pages:                         300000.
Pages occupied by compressor:             50000.
"""
        m = hp._from_vm_stat(vm_stat, 16 * 1024**3, "total = 2048.00M  used = 512.00M  free = 1536.00M")
        used = (300000 - 20000 + 100000 + 50000) * 16384
        assert m["used_mib"] == used // (1024 * 1024)
        assert m["percent"] == pytest.approx(100 * used / (16 * 1024**3), abs=0.1)
        assert (m["swap_used_mib"], m["swap_total_mib"]) == (512, 2048)

    def test_the_old_reader_now_agrees(self):
        from hypernix.monitoring import tv

        m = tv._read_memory_breakdown()
        if m is not None:
            assert m["percent"] == pytest.approx(100 * m["used_mib"] / m["total_mib"], abs=0.2)


def _hwmon(tmp_path, name: str, readings: dict[str, int]):
    d = tmp_path / "hwmon" / f"hwmon{len(list((tmp_path / 'hwmon').glob('*'))) if (tmp_path / 'hwmon').exists() else 0}"
    d.mkdir(parents=True)
    (d / "name").write_text(name)
    for i, (label, milli) in enumerate(readings.items(), start=1):
        (d / f"temp{i}_label").write_text(label)
        (d / f"temp{i}_input").write_text(str(milli))
    return tmp_path / "hwmon"


def _topology(tmp_path, cpus: list[tuple[int, int]]):
    base = tmp_path / "cpu"
    for i, (package, core) in enumerate(cpus):
        t = base / f"cpu{i}" / "topology"
        t.mkdir(parents=True)
        (t / "physical_package_id").write_text(str(package))
        (t / "core_id").write_text(str(core))
    return base


class TestTemperatures:
    def test_intel_cores_are_mapped_onto_logical_cpus(self, tmp_path):
        hwmon = _hwmon(tmp_path, "coretemp", {"Package id 0": 61000, "Core 0": 55000, "Core 1": 58000})
        # Hyper-threading: cpu0/cpu2 are core 0, cpu1/cpu3 are core 1.
        cpus = _topology(tmp_path, [(0, 0), (0, 1), (0, 0), (0, 1)])
        t = hp.cpu_temperatures(4, hwmon=hwmon, cpus=cpus)
        assert t["per_core"] == [55.0, 58.0, 55.0, 58.0]
        assert t["package_c"] == 61.0 and t["note"] == ""

    def test_amd_says_it_has_no_per_core_readings(self, tmp_path):
        hwmon = _hwmon(tmp_path, "k10temp", {"Tctl": 70500, "Tccd1": 64250, "Tccd2": 66000})
        t = hp.cpu_temperatures(2, hwmon=hwmon, cpus=_topology(tmp_path, [(0, 0), (0, 1)]))
        assert t["per_core"] == [None, None]
        assert t["package_c"] == 70.5 and t["dies"]["Tccd2"] == 66.0
        assert "CCD" in t["note"]

    def test_nothing_exposed(self, tmp_path):
        t = hp.cpu_temperatures(2, hwmon=tmp_path / "none", cpus=tmp_path / "none")
        assert t["per_core"] == [None, None] and t["package_c"] is None and t["note"]


PACTL = [{
    "name": "alsa_output.usb-headset", "description": "USB Headset", "state": "RUNNING",
    "mute": False, "active_port": "analog-output-headphones", "monitor_source": "alsa_output.usb-headset.monitor",
    "properties": {"api": "pipewire"},
    "volume": {"front-left": {"value": 60000, "value_percent": "92%", "db": "-2.30 dB"},
               "front-right": {"value": 60000, "value_percent": "92%", "db": "-2.30 dB"}},
}]


class TestAudio:
    @pytest.fixture(autouse=True)
    def no_cache(self, monkeypatch):
        monkeypatch.setitem(hp._AUDIO_CACHE, "value", None)
        monkeypatch.setattr(hp.platform, "system", lambda: "Linux")
        monkeypatch.delenv("TVTOP_MAX_LOUD_DB", raising=False)

    def _pactl(self, monkeypatch, sinks):
        def run(argv, timeout=1.5):
            if argv[:2] == ["pactl", "get-default-sink"]:
                return sinks[0]["name"] + "\n"
            if argv[:3] == ["pactl", "--format=json", "list"]:
                return json.dumps(sinks)
            return None
        monkeypatch.setattr(hp, "_run", run)

    def test_the_decibels_are_the_sound_servers(self, monkeypatch):
        self._pactl(monkeypatch, PACTL)
        monkeypatch.setattr(hp, "_signal_level", lambda monitor, seconds=0.25: (-3.0, -15.0))
        a = hp.audio()
        assert a["available"] and a["backend"] == "pipewire"
        assert a["volume_db"] == -2.3 and a["volume_percent"] == 92
        assert a["playing"] and a["headphones"]
        assert a["level"] == "loud" and "-2.3 dB" in a["warning"] and "headphones" in a["warning"]
        assert a["signal_peak_dbfs"] == -3.0

    def test_amplified_past_100_percent_is_unsafe(self, monkeypatch):
        sinks = json.loads(json.dumps(PACTL))
        for channel in sinks[0]["volume"].values():
            channel.update(value_percent="130%", db="6.84 dB")
        self._pactl(monkeypatch, sinks)
        monkeypatch.setattr(hp, "_signal_level", lambda monitor, seconds=0.25: (None, None))
        a = hp.audio()
        assert a["level"] == "unsafe" and "+6.8 dB" in a["warning"]

    def test_quiet_paused_or_muted_is_fine(self, monkeypatch):
        for change in ({"state": "IDLE"}, {"mute": True}):
            sinks = json.loads(json.dumps(PACTL))
            sinks[0].update(change)
            self._pactl(monkeypatch, sinks)
            hp._AUDIO_CACHE["value"] = None
            a = hp.audio()
            assert a["level"] == "ok" and a["warning"] == ""

    def test_a_quiet_volume_is_fine_while_playing(self, monkeypatch):
        sinks = json.loads(json.dumps(PACTL))
        for channel in sinks[0]["volume"].values():
            channel.update(value_percent="40%", db="-23.88 dB")
        self._pactl(monkeypatch, sinks)
        monkeypatch.setattr(hp, "_signal_level", lambda monitor, seconds=0.25: (-1.0, -9.0))
        assert hp.audio()["level"] == "ok"

    def test_the_threshold_can_be_set(self, monkeypatch):
        monkeypatch.setenv("TVTOP_MAX_LOUD_DB", "-1")
        self._pactl(monkeypatch, PACTL)
        monkeypatch.setattr(hp, "_signal_level", lambda monitor, seconds=0.25: (-3.0, -15.0))
        assert hp.audio()["level"] == "ok"

    def test_alsa_reports_db_too(self, monkeypatch):
        out = ("Simple mixer control 'Master',0\n  Front Left: Playback 60 [94%] [-3.00dB] [on]\n"
               "  Front Right: Playback 60 [94%] [-3.00dB] [on]\n")
        monkeypatch.setattr(hp, "_run", lambda argv, timeout=1.5: out if argv[0] == "amixer" else None)
        a = hp.audio(sample_signal=False)
        assert a["backend"] == "alsa" and a["volume_db"] == -3.0 and a["volume_percent"] == 94

    def test_silence_is_not_a_number(self, monkeypatch):
        sinks = json.loads(json.dumps(PACTL))
        for channel in sinks[0]["volume"].values():
            channel.update(value_percent="0%", db="-inf dB")
        self._pactl(monkeypatch, sinks)
        monkeypatch.setattr(hp, "_signal_level", lambda monitor, seconds=0.25: (-math.inf, -math.inf))
        a = hp.audio()
        assert a["volume_db"] is None and a["volume_db_silent"] is True
        json.dumps(a, allow_nan=False)

    def test_no_sound_server(self, monkeypatch):
        monkeypatch.setattr(hp, "_run", lambda argv, timeout=1.5: None)
        a = hp.audio()
        assert a["available"] is False and a["note"]


def test_the_other_probes_never_raise():
    assert isinstance(hp.disks(), list)
    assert isinstance(hp.network(), list)
    assert "hostname" in hp.system()
    assert isinstance(hp.memory_users(3), list)


def test_the_bridge_sends_them(tmp_path):
    from hypernix.monitoring.tvtop_max_bridge import Monitor

    frame = Monitor(log=str(tmp_path / "none.log"), discover=False).frame()
    for key in ("cpu_temps", "audio", "disks", "network", "system", "memory_users"):
        assert key in frame
    assert frame["memory"]["percent"] == pytest.approx(
        100 * frame["memory"]["used_mib"] / frame["memory"]["total_mib"], abs=0.2)
    json.dumps(frame, allow_nan=False, default=str)
