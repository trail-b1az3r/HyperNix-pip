"""Every GPU reader goes through one abstraction — beta 2 of item 13.

Beta 1 built ``hypernix.system.gpus``. This is the migration: the
modules that were shelling out to ``nvidia-smi`` themselves now ask it
instead, so an AMD card appears in all of them.

That is the whole point, and it is worth being concrete about what was
broken. On a Radeon box:

* ``thermometer.read_gpu_temp()`` returned ``None`` — the thermometer
  reported a CPU temperature and a blank where the GPU should be, on the
  hardware where a temperature reading matters most.
* ``tv``'s GPU panel was entirely empty, and ``tv`` is the thing people
  open *because* they want those numbers.
* ``livestream`` broadcast ``"gpus": []``, which reads as "the stream is
  broken" rather than "this tool supports one vendor".
* ``pascal.detect()`` fell through to a second, slightly different
  nvidia-smi parser that disagreed with the first about ``[N/A]``.

These tests stub ``gpus.detect()`` with an AMD card and assert the
numbers come out. No GPU is required, which is the point: the bug was
only ever visible on hardware nobody testing it had.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "hypernix"


@pytest.fixture
def amd_card():
    """One Radeon, fully populated, as rocm-smi would report it."""
    from hypernix.system.gpus import GPU, Vendor

    return GPU(
        index=0,
        vendor=Vendor.AMD,
        name="Radeon RX 7900 XTX",
        memory_total_mb=24560,
        memory_used_mb=8192,
        utilization_pct=62.0,
        temperature_c=71.0,
        power_w=210.0,
        power_limit_w=355.0,
        driver="6.7.0",
        compute="gfx1100",
    )


@pytest.fixture
def nvidia_partial():
    """An NVIDIA card that declined half the questions.

    nvidia-smi answers ``[N/A]`` for power draw and utilisation on
    several consumer cards. Those have to arrive as None, not 0 — a
    dashboard showing 0 W reads as an idle GPU rather than as a missing
    measurement.
    """
    from hypernix.system.gpus import GPU, Vendor

    return GPU(
        index=0,
        vendor=Vendor.NVIDIA,
        name="NVIDIA GeForce GTX 1080",
        memory_total_mb=8192,
        memory_used_mb=1024,
        utilization_pct=None,
        temperature_c=54.0,
        power_w=None,
        power_limit_w=None,
        driver="535.104.05",
        compute="6.1",
    )


class TestTheThermometerSeesAnyVendor:
    def test_an_amd_card_has_a_temperature(self, monkeypatch, amd_card):
        from hypernix.monitoring import thermometer
        from hypernix.system import gpus

        monkeypatch.setattr(gpus, "detect", lambda: [amd_card])

        assert thermometer.read_gpu_temp() == 71.0

    def test_it_reports_the_hottest_card_not_the_first(self, monkeypatch, amd_card):
        """This feeds a single number, and the only useful single number
        from a multi-GPU box is the one closest to throttling."""
        from dataclasses import replace

        from hypernix.monitoring import thermometer
        from hypernix.system import gpus

        cool = replace(amd_card, index=0, temperature_c=45.0)
        hot = replace(amd_card, index=1, temperature_c=88.0)
        monkeypatch.setattr(gpus, "detect", lambda: [cool, hot])

        assert thermometer.read_gpu_temp() == 88.0

    def test_a_card_with_no_reading_is_skipped_not_zero(self, monkeypatch, amd_card):
        from dataclasses import replace

        from hypernix.monitoring import thermometer
        from hypernix.system import gpus

        monkeypatch.setattr(
            gpus, "detect",
            lambda: [replace(amd_card, temperature_c=None),
                     replace(amd_card, index=1, temperature_c=60.0)],
        )

        assert thermometer.read_gpu_temp() == 60.0

    def test_no_gpu_at_all_is_none(self, monkeypatch):
        from hypernix.monitoring import thermometer
        from hypernix.system import gpus

        monkeypatch.setattr(gpus, "detect", lambda: [])

        assert thermometer.read_gpu_temp() is None

    def test_the_source_label_does_not_name_the_wrong_tool(
        self, monkeypatch, amd_card
    ):
        """It said ``nvidia-smi:gpu`` for every reading. On an AMD box
        that is a label naming a tool that was never run, which is worse
        than a vague one when somebody is working out why a number looks
        wrong."""
        from hypernix.monitoring import thermometer
        from hypernix.system import gpus

        monkeypatch.setattr(gpus, "detect", lambda: [amd_card])
        reading = thermometer.take_reading()

        assert "gpu" in reading.sources
        assert not any("nvidia" in key for key in reading.sources)


class TestLivestreamSeesAnyVendor:
    def test_an_amd_card_is_broadcast(self, monkeypatch, amd_card):
        from hypernix.monitoring import livestream
        from hypernix.system import gpus

        monkeypatch.setattr(gpus, "detect", lambda: [amd_card])
        sample = livestream.sample_hardware()

        assert len(sample["gpus"]) == 1
        card = sample["gpus"][0]
        assert card["name"] == "Radeon RX 7900 XTX"
        assert card["temperature_c"] == 71.0
        assert card["vram_used_mb"] == 8192
        assert card["vram_percent"] == pytest.approx(33.4, abs=0.2)

    def test_a_declined_measurement_is_none_not_zero(
        self, monkeypatch, nvidia_partial
    ):
        from hypernix.monitoring import livestream
        from hypernix.system import gpus

        monkeypatch.setattr(gpus, "detect", lambda: [nvidia_partial])
        card = livestream.sample_hardware()["gpus"][0]

        assert card["power_w"] is None
        assert card["utilization"] is None
        assert card["temperature_c"] == 54.0

    def test_a_sampling_failure_does_not_kill_the_stream(self, monkeypatch):
        """This runs every second inside a training process. An exception
        here would take down the run it is reporting on."""
        from hypernix.monitoring import livestream
        from hypernix.system import gpus

        def boom():
            raise RuntimeError("rocm-smi went away")

        monkeypatch.setattr(gpus, "detect", boom)
        sample = livestream.sample_hardware()

        assert sample["gpus"] == []
        assert "ram" in sample


class TestPascalDelegatesTheVendorToolPath:
    def test_torch_still_wins(self, monkeypatch):
        """torch reports the compute capability directly, which is what
        this module is for. The abstraction is the fallback, not a
        replacement."""
        source = (SRC / "system" / "pascal.py").read_text(encoding="utf-8")
        body = source.split("def detect(")[1]

        assert body.index("import torch") < body.index("from . import gpus")

    def test_it_no_longer_parses_nvidia_smi_itself(self):
        """Two parsers for one command's output is two places to
        disagree, and they did: this one read ``[N/A]`` as 0.0 VRAM."""
        source = (SRC / "system" / "pascal.py").read_text(encoding="utf-8")
        body = source.split("def detect(")[1].split("\ndef ")[0]

        assert "--query-gpu" not in body
        assert "subprocess" not in body

    def test_an_amd_card_gets_a_name_and_vram_but_no_compute(
        self, monkeypatch, amd_card
    ):
        """A gfx target is not a (major, minor) CUDA capability, and
        pretending otherwise would have the FP16 guard below reason
        about a number that means something else."""
        from hypernix.system import gpus, pascal

        monkeypatch.setattr(gpus, "detect", lambda: [amd_card])
        # Make the torch branch unavailable so the gpus path is taken.
        monkeypatch.setitem(__import__("sys").modules, "torch", None)

        info = pascal.detect()

        assert info.name == "Radeon RX 7900 XTX"
        assert info.vram_gb == pytest.approx(23.98, abs=0.05)
        assert info.compute is None
        assert info.source == "amd-smi"

    def test_an_nvidia_card_gets_its_compute_capability(
        self, monkeypatch, nvidia_partial
    ):
        from hypernix.system import gpus, pascal

        monkeypatch.setattr(gpus, "detect", lambda: [nvidia_partial])
        monkeypatch.setitem(__import__("sys").modules, "torch", None)

        info = pascal.detect()

        assert info.compute == (6, 1)
        assert info.matched is not None, "a GTX 1080 is a Pascal card"


class TestEthanolMatchesTheToolToTheCard:
    """`ethanol` writes, so it probes for the tool -- but the tool has to
    belong to a card that is actually there."""

    def test_nvidia_smi_on_an_amd_only_box_is_not_used(self, monkeypatch, amd_card):
        """A driver package can leave nvidia-smi behind on a machine
        whose card is a Radeon. Picking "nvidia" there writes clock
        offsets at a device index that is not the AMD card."""
        from hypernix.system import ethanol, gpus

        monkeypatch.setattr(gpus, "detect", lambda: [amd_card])
        monkeypatch.setattr(
            ethanol, "_has_binary",
            lambda name: name in {"nvidia-smi", "nvidia-settings", "rocm-smi"},
        )

        assert ethanol._detect_backend() == "rocm"

    def test_nvidia_is_chosen_when_the_card_is_nvidia(
        self, monkeypatch, nvidia_partial
    ):
        from hypernix.system import ethanol, gpus

        monkeypatch.setattr(gpus, "detect", lambda: [nvidia_partial])
        monkeypatch.setattr(
            ethanol, "_has_binary",
            lambda name: name in {"nvidia-smi", "nvidia-settings"},
        )

        assert ethanol._detect_backend() == "nvidia"

    def test_detection_finding_nothing_falls_back_to_tool_presence(self, monkeypatch):
        """A container with the devices passed through but no vendor tool
        visible is a real case. Refusing to overclock a card the operator
        knows is there would be worse than trusting them."""
        from hypernix.system import ethanol, gpus

        monkeypatch.setattr(gpus, "detect", lambda: [])
        monkeypatch.setattr(
            ethanol, "_has_binary",
            lambda name: name in {"nvidia-smi", "nvidia-settings"},
        )

        assert ethanol._detect_backend() == "nvidia"

    def test_no_tool_is_none_whatever_is_detected(self, monkeypatch, amd_card):
        from hypernix.system import ethanol, gpus

        monkeypatch.setattr(gpus, "detect", lambda: [amd_card])
        monkeypatch.setattr(ethanol, "_has_binary", lambda name: False)

        assert ethanol._detect_backend() == "none"


class TestNothingElseShellsOutToAVendorTool:
    """The migration, checked as a whole.

    ``gpus.py`` is the abstraction and runs the commands; every other
    module has to ask it. A new direct call would work on the author's
    machine and silently exclude half the hardware, which is exactly how
    the four bugs above got there.
    """

    #: Modules allowed to invoke a vendor tool. `gpus` is the
    #: abstraction. `ethanol` *writes* -- setting a power limit is not a
    #: read and has no place in a read-only abstraction.
    ALLOWED = {"system/gpus.py", "system/ethanol.py"}

    def test_no_module_runs_nvidia_smi_or_rocm_smi_directly(self):
        offenders = []
        for source in sorted(SRC.rglob("*.py")):
            relative = source.relative_to(SRC).as_posix()
            if relative in self.ALLOWED:
                continue
            text = source.read_text(encoding="utf-8")
            # Comments and docstrings mention these tools constantly and
            # legitimately -- naming the thing a rule forbids is what
            # documentation does. Only an argv list counts.
            code = re.sub(r"#[^\n]*", "", text)
            code = re.sub(r'"""(?:.|\n)*?"""', "", code)
            for tool in ("nvidia-smi", "rocm-smi", "amd-smi"):
                if re.search(rf'\[\s*["\']{re.escape(tool)}["\']', code) or \
                   re.search(rf'which\(\s*["\']{re.escape(tool)}["\']', code):
                    offenders.append(f"{relative}: {tool}")

        assert not offenders, (
            "these invoke a vendor tool directly instead of going through "
            "hypernix.system.gpus, which means they see one vendor's "
            "hardware:\n  " + "\n  ".join(offenders)
        )
