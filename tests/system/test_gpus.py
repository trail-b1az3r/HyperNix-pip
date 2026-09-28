"""One way to ask about a GPU, whoever made it.

Before `hypernix.system.gpus` there were 42 places that shelled out to
`nvidia-smi` and 12 that knew about `rocm-smi`, across seven modules.
That count is the problem item 13 describes: AMD support was not so much
missing as *unevenly present*, and every new panel reimplemented the
same parsing and met the same edge cases again.

There is no GPU in the machine these tests run on, and that is fine —
the part that goes wrong is never "can we call the tool", it is "what do
we do with what it said". So the parsers are driven with real vendor
output, including the shapes that have historically caused trouble:

- ``[N/A]``, ``N/A`` and ``Not Supported`` in numeric columns, which
  must become None and not 0. A dashboard that reports an unknown
  temperature as zero says the card is freezing.
- ``rocm-smi``'s VRAM in **bytes** where everything else is megabytes.
- field names that changed between ROCm releases.
- a short row from an older driver answering the same query.

And the CPU-only path, which is most machines: no GPU is an answer, not
a failure, and nothing here may raise for want of hardware.
"""
from __future__ import annotations

import json

import pytest

from hypernix.system.gpus import (
    GPU,
    Vendor,
    backend,
    describe,
    detect,
    parse_amd_smi,
    parse_nvidia,
    parse_nvidia_processes,
    parse_rocm_json,
    processes,
    select,
)

#: Real `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` output.
NVIDIA_TWO_CARDS = """\
0, NVIDIA GeForce RTX 3090, 24576, 1024, 15, 42, 120.55, 350.00, 550.90, GPU-abc123, 8.6
1, NVIDIA GeForce RTX 3090, 24576, 512, 0, 38, 98.20, 350.00, 550.90, GPU-def456, 8.6
"""

#: A consumer card under WSL: several fields unavailable.
NVIDIA_WITH_NA = """\
0, NVIDIA GeForce GTX 1080, 8192, 256, [N/A], [N/A], [N/A], [N/A], 550.90, GPU-xyz, 6.1
"""


class TestNvidiaParsing:
    def test_two_cards(self):
        cards = parse_nvidia(NVIDIA_TWO_CARDS)

        assert len(cards) == 2
        assert cards[0].name == "NVIDIA GeForce RTX 3090"
        assert cards[0].vendor is Vendor.NVIDIA
        assert cards[0].memory_total_mb == 24576
        assert cards[0].index == 0 and cards[1].index == 1

    def test_readings_are_numbers(self):
        card = parse_nvidia(NVIDIA_TWO_CARDS)[0]

        assert card.utilization_pct == 15
        assert card.temperature_c == 42
        assert card.power_w == pytest.approx(120.55)
        assert card.compute == "8.6"

    def test_unavailable_readings_are_none_not_zero(self):
        """The distinction the whole module turns on. 0C is a claim
        about the card; None is a claim about the driver."""
        card = parse_nvidia(NVIDIA_WITH_NA)[0]

        assert card.temperature_c is None
        assert card.utilization_pct is None
        assert card.power_w is None

    def test_the_readings_that_are_present_still_arrive(self):
        card = parse_nvidia(NVIDIA_WITH_NA)[0]

        assert card.memory_total_mb == 8192
        assert card.driver == "550.90"

    def test_free_memory_is_derived(self):
        card = parse_nvidia(NVIDIA_TWO_CARDS)[0]

        assert card.memory_free_mb == 24576 - 1024

    def test_free_memory_is_unknown_when_either_half_is(self):
        card = GPU(index=0, vendor=Vendor.NVIDIA, name="x", memory_total_mb=100)

        assert card.memory_free_mb is None

    def test_a_short_row_from_an_older_driver_still_parses(self):
        cards = parse_nvidia("0, Tesla K80, 11441, 0\n")

        assert len(cards) == 1
        assert cards[0].name == "Tesla K80"
        assert cards[0].compute == ""

    @pytest.mark.parametrize("text", ["", "\n\n", "garbage without commas\n"])
    def test_nothing_usable_is_no_cards_not_an_error(self, text):
        assert parse_nvidia(text) == []


class TestAMDParsing:
    #: rocm-smi --json, with VRAM in bytes as it really reports.
    ROCM = json.dumps({
        "card0": {
            "Card Series": "Radeon RX 7900 XTX",
            "Card SKU": "D70301",
            "GPU use (%)": "37",
            "Temperature (Sensor edge) (C)": "51.0",
            "Average Graphics Package Power (W)": "142.0",
            "Max Graphics Package Power (W)": "355.0",
            "VRAM Total Memory (B)": "25753026560",
            "VRAM Total Used Memory (B)": "1073741824",
            "Driver version": "6.7.0",
            "Unique ID": "0x1234",
            "GFX Version": "gfx1100",
        },
        "card1": {
            "Card Series": "Radeon RX 7900 XTX",
            "GPU use (%)": "0",
            "VRAM Total Memory (B)": "25753026560",
            "VRAM Total Used Memory (B)": "0",
        },
    })

    def test_both_cards(self):
        cards = parse_rocm_json(self.ROCM)

        assert len(cards) == 2
        assert all(c.vendor is Vendor.AMD for c in cards)
        assert cards[0].index == 0 and cards[1].index == 1

    def test_bytes_become_megabytes(self):
        """rocm-smi is the only source here reporting bytes. Mixing the
        units makes a dashboard unreadable and a limit check wrong."""
        card = parse_rocm_json(self.ROCM)[0]

        assert card.memory_total_mb == 24560
        assert card.memory_used_mb == 1024

    def test_the_other_readings(self):
        card = parse_rocm_json(self.ROCM)[0]

        assert card.utilization_pct == 37
        assert card.temperature_c == 51.0
        assert card.power_w == 142.0
        assert card.compute == "gfx1100"

    def test_a_card_missing_optional_fields_still_parses(self):
        card = parse_rocm_json(self.ROCM)[1]

        assert card.temperature_c is None
        assert card.memory_total_mb == 24560

    def test_renamed_fields_are_found_by_their_other_spelling(self):
        """The names changed across ROCm releases. A rename should cost
        that reading, not the card."""
        payload = json.dumps({"card0": {
            "Card Model": "Instinct MI210",
            "GPU Utilization (%)": "88",
            "Temperature (Sensor junction) (C)": "63",
        }})

        card = parse_rocm_json(payload)[0]

        assert card.name == "Instinct MI210"
        assert card.utilization_pct == 88
        assert card.temperature_c == 63

    @pytest.mark.parametrize("text", ["", "not json", "[]", "{}", "null"])
    def test_unparseable_output_is_no_cards(self, text):
        assert parse_rocm_json(text) == []

    def test_non_card_keys_are_ignored(self):
        """rocm-smi puts a "system" block alongside the cards."""
        payload = json.dumps({
            "system": {"Driver version": "6.7.0"},
            "card0": {"Card Series": "RX 7900"},
        })

        cards = parse_rocm_json(payload)

        assert len(cards) == 1


class TestAmdSmiParsing:
    AMD_SMI = json.dumps([{
        "gpu": 0,
        "asic": {"market_name": "Instinct MI300X", "target_graphics_version": "gfx942"},
        "mem_usage": {"total_vram": 196608, "used_vram": 2048},
        "usage": {"gfx_activity": 64},
        "temperature": {"edge": 47},
        "power": {"socket_power": 210, "power_limit": 750},
    }])

    def test_it_parses(self):
        cards = parse_amd_smi(self.AMD_SMI)

        assert len(cards) == 1
        assert cards[0].name == "Instinct MI300X"
        assert cards[0].memory_total_mb == 196608
        assert cards[0].utilization_pct == 64
        assert cards[0].compute == "gfx942"

    def test_a_gpus_wrapper_is_unwrapped(self):
        payload = json.dumps({"gpus": json.loads(self.AMD_SMI)})

        assert len(parse_amd_smi(payload)) == 1

    @pytest.mark.parametrize("text", ["", "not json", "3"])
    def test_junk_is_no_cards(self, text):
        assert parse_amd_smi(text) == []


class TestProcesses:
    def test_it_parses_compute_apps(self):
        found = parse_nvidia_processes("1234, python, 0, 512\n5678, train.py, 1, 2048\n")

        assert len(found) == 2
        assert found[0].pid == 1234
        assert found[1].memory_mb == 2048

    def test_rows_without_a_pid_are_skipped(self):
        assert parse_nvidia_processes("no pid here, x, 0\n") == []

    def test_it_returns_a_list_on_a_machine_with_no_gpu(self):
        assert isinstance(processes(), list)


class TestSelection:
    CARDS = [
        GPU(index=0, vendor=Vendor.NVIDIA, name="a"),
        GPU(index=1, vendor=Vendor.AMD, name="b"),
    ]

    @pytest.mark.parametrize("spec", ["", "auto", "all", "ALL"])
    def test_everything_by_default(self, spec):
        assert len(select(spec, self.CARDS)) == 2

    def test_one_by_index(self):
        assert [c.index for c in select("1", self.CARDS)] == [1]

    def test_several(self):
        assert [c.index for c in select("0,1", self.CARDS)] == [0, 1]

    def test_an_absent_index_is_an_error_not_an_empty_result(self):
        """Asking for GPU 3 on a two-card machine is a mistake worth
        hearing about; returning nothing would read as "no GPUs"."""
        with pytest.raises(ValueError, match="No GPU with index 3"):
            select("3", self.CARDS)

    def test_the_error_lists_what_there_is(self):
        with pytest.raises(ValueError, match="Available: 0, 1"):
            select("9", self.CARDS)

    def test_a_non_number_is_refused(self):
        with pytest.raises(ValueError, match="not a GPU index"):
            select("first", self.CARDS)


class TestTheCPUOnlyPath:
    """Most machines running this have no GPU. That is an answer."""

    def test_detect_never_raises(self):
        assert isinstance(detect(), list)

    def test_backend_is_unknown_rather_than_an_error(self, monkeypatch):
        monkeypatch.setattr("hypernix.system.gpus.detect", lambda: [])

        assert backend() is Vendor.UNKNOWN

    def test_describe_says_it_is_not_a_failure(self):
        assert "not a failure" in describe([])

    def test_a_vendor_tool_that_errors_is_no_cards(self, monkeypatch):
        monkeypatch.setattr("hypernix.system.gpus._run", lambda *a, **k: "")

        assert detect() == []

    def test_a_probe_that_raises_does_not_take_the_others_down(self, monkeypatch):
        def boom():
            raise RuntimeError("driver on fire")

        monkeypatch.setattr("hypernix.system.gpus._detect_nvidia", boom)
        monkeypatch.setattr(
            "hypernix.system.gpus._detect_amd",
            lambda: [GPU(index=0, vendor=Vendor.AMD, name="survivor")],
        )

        assert [c.name for c in detect()] == ["survivor"]


class TestTheVendorNeutralSurface:
    def test_each_vendor_names_its_framework(self):
        assert Vendor.NVIDIA.framework == "cuda"
        assert Vendor.AMD.framework == "rocm"
        assert Vendor.APPLE.framework == "mps"
        assert Vendor.UNKNOWN.framework == "cpu"

    def test_a_card_reports_its_own_framework(self):
        assert GPU(index=0, vendor=Vendor.AMD, name="x").framework == "rocm"

    def test_both_vendors_produce_the_same_shape(self):
        """The point of the layer: a caller never branches on vendor."""
        nvidia = parse_nvidia(NVIDIA_TWO_CARDS)[0].to_dict()
        amd = parse_rocm_json(TestAMDParsing.ROCM)[0].to_dict()

        assert set(nvidia) == set(amd)

    def test_describe_reads_for_a_mixed_machine(self):
        cards = parse_nvidia(NVIDIA_TWO_CARDS) + parse_rocm_json(TestAMDParsing.ROCM)

        text = describe(cards)

        assert "nvidia:0" in text
        assert "amd:0" in text
