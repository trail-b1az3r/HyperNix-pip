"""Running the sub-bit runtime somewhere other than the CPU.

Three things are checked here, and only one of them needs a GPU.

**The decoders.** :mod:`hypernix.models.hnxtorch` is a torch rewrite of
the numpy decoders in ``subbit`` and ``lowbit``, in ops that exist on
every backend torch supports. It is asserted **equal** to numpy, not
close: these are integer unpacking followed by one multiply, so any
difference is a bug in one of them rather than rounding. Running it on
the CPU device exercises exactly the code a CUDA card would run, which is
the only check of it available on a machine without one.

**The probe.** ``torch.cuda.is_available()`` returning True does not mean
the wheel has kernels for the card in the machine. A GTX 1080 is sm_61
and recent wheels build for sm_75 and up; the first kernel launch then
fails with ``no kernel image is available for execution on the device``,
which reads like a broken driver. The probe compares capability against
``get_arch_list()`` and the tests drive it with fake arch lists, because
that comparison is the entire value of the module and cannot be observed
on any single machine.

**The refusals.** ``--device cuda`` on a box with no CUDA must say why,
not fall back to the CPU and be slow for reasons nobody can see.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hypernix.models import hnxdevice, hnxtorch
from hypernix.models.hnxrun import PackedWeight
from hypernix.quant import lowbit, subbit

#: Every extension type, with the family that decodes it.
EXTENSION_TYPES = {
    200: ("subbit", "sign_scale_l"),
    201: ("subbit", "pair_code_m"),
    202: ("subbit", "quad_code_xxxl"),
    203: ("subbit", "quarter_code_uxl"),
    204: ("subbit", "int1_binary"),
    205: ("lowbit", "INT4"),
    206: ("lowbit", "FP2"),
}


def _packed(family: str, name: str, count: int, seed: int = 0) -> bytes:
    weights = (np.random.default_rng(seed).standard_normal(count) * 0.05).astype(
        np.float32
    )
    if family == "subbit":
        return subbit.quantize_tensor(weights.tolist(), name)
    return lowbit.quantize_array(weights, name)


class TestTheTorchDecodersMatchNumpyExactly:
    """Not approximately. Integer unpacking plus one multiply."""

    @pytest.mark.parametrize(("kind", "spec"), sorted(EXTENSION_TYPES.items()))
    def test_a_whole_tensor_decodes_identically(self, kind, spec):
        family, name = spec
        raw = _packed(family, name, 256 * 6)
        reference = (
            subbit.dequantize_array(raw, name)
            if family == "subbit"
            else lowbit.dequantize_array(raw, name)
        )
        got = hnxtorch.decode(hnxtorch.to_device_bytes(raw, "cpu"), kind)
        assert np.array_equal(got.numpy(), reference)

    @pytest.mark.parametrize("packing", sorted(subbit.PACKINGS))
    def test_stored_signs_match_without_expanding(self, packing):
        """The folded matmul consumes this shape directly, so it has to
        agree in layout as well as in value."""
        raw = _packed("subbit", packing, 256 * 4)
        reference = subbit.stored_signs(raw, packing)
        got = hnxtorch.stored_signs(hnxtorch.to_device_bytes(raw, "cpu"), packing)
        assert got.shape == reference.shape
        assert np.array_equal(got.numpy(), reference)

    @pytest.mark.parametrize(("kind", "spec"), sorted(EXTENSION_TYPES.items()))
    def test_an_empty_tensor_is_empty_not_an_error(self, kind, spec):
        got = hnxtorch.decode(hnxtorch.to_device_bytes(b"", "cpu"), kind)
        assert got.numel() == 0

    def test_the_bit_order_is_the_packers(self, ):
        """Little-endian within each byte, matching ``numpy.unpackbits``.
        Backwards produces a model that loads, runs, and is not the model
        in the file — so it is asserted against a known pattern rather
        than reasoned about."""
        codec = lowbit.CODECS["INT4"]
        levels = np.asarray(codec.levels, dtype=np.float32)
        pattern = np.resize(levels, 256).astype(np.float32)
        raw = lowbit.quantize_array(pattern, "INT4")
        got = hnxtorch.dequantize_lowbit(
            hnxtorch.to_device_bytes(raw, "cpu"), "INT4"
        ).numpy()
        assert np.allclose(got, pattern, rtol=1e-3, atol=1e-3)

    def test_an_unsupported_type_says_where_it_goes_instead(self):
        with pytest.raises(ValueError, match="numpy path"):
            hnxtorch.decode(hnxtorch.to_device_bytes(b"\0" * 32, "cpu"), 12)

    @pytest.mark.parametrize("kind", sorted(EXTENSION_TYPES))
    def test_supports_agrees_with_decode(self, kind):
        assert hnxtorch.supports(kind)

    @pytest.mark.parametrize("kind", [0, 1, 12, 14, 30])
    def test_it_does_not_claim_the_upstream_types(self, kind):
        assert not hnxtorch.supports(kind)


class TestThePackedWeightUsesIt:
    """``decode_on_device=True`` runs the accelerator path on the CPU.

    That is the whole testing strategy for GPU support on a machine with
    no GPU: it is the same code, reached the same way, and what it must
    produce is what the numpy path produces.
    """

    @pytest.mark.parametrize(("kind", "spec"), sorted(EXTENSION_TYPES.items()))
    def test_the_weights_are_bit_identical_either_way(self, kind, spec):
        family, name = spec
        raw = _packed(family, name, 8 * 512)
        host = PackedWeight(raw, kind, (8, 512), "cpu")
        device = PackedWeight(raw, kind, (8, 512), "cpu", decode_on_device=True)
        assert torch.equal(host.to_dense(), device.to_dense())

    @pytest.mark.parametrize(("kind", "spec"), sorted(EXTENSION_TYPES.items()))
    def test_the_matmul_agrees(self, kind, spec):
        family, name = spec
        raw = _packed(family, name, 8 * 512)
        host = PackedWeight(raw, kind, (8, 512), "cpu")
        device = PackedWeight(raw, kind, (8, 512), "cpu", decode_on_device=True)
        x = torch.randn(3, 512)
        assert torch.allclose(
            host.matmul_t(x), device.matmul_t(x), rtol=2e-5, atol=2e-5
        )

    def test_the_folded_path_is_still_the_one_taken(self):
        """The fold and the device decode are separate optimisations and
        both have to survive the other."""
        raw = _packed("subbit", "quad_code_xxxl", 8 * 512)
        device = PackedWeight(raw, 202, (8, 512), "cpu", decode_on_device=True)
        assert device._packing == "quad_code_xxxl"  # noqa: SLF001
        signs = device._folded_signs(0, 8)          # noqa: SLF001
        assert signs.shape == (8, 512 // 4 * 2)

    def test_the_packed_bytes_are_uploaded_once_not_per_chunk(self):
        """The reason this exists. Decoding on the host and pushing the
        expanded floats moves 34x the bytes, every forward pass, to save
        nothing."""
        raw = _packed("subbit", "sign_scale_l", 8 * 512)
        weight = PackedWeight(raw, 200, (8, 512), "cpu", decode_on_device=True)
        first = weight.packed_on_device()
        weight.matmul_t(torch.randn(2, 512))
        weight.matmul_t(torch.randn(2, 512))
        assert weight.packed_on_device() is first
        assert first.numel() == len(raw)

    def test_device_bytes_counts_what_is_actually_held(self):
        raw = _packed("subbit", "sign_scale_l", 8 * 512)
        packed = PackedWeight(raw, 200, (8, 512), "cpu")
        assert packed.device_bytes == 0
        device = PackedWeight(raw, 200, (8, 512), "cpu", decode_on_device=True)
        device.packed_on_device()
        assert device.device_bytes == len(raw)

    def test_an_upstream_type_still_takes_the_numpy_path(self):
        """There is no torch decoder for Q4_K, and claiming one would
        route it into a ValueError at the first matmul."""
        from hypernix.quant import llamaquants

        rng = np.random.default_rng(0)
        raw = llamaquants.quantize_array(
            rng.standard_normal(256 * 4).astype(np.float32), "Q4_K"
        )
        weight = PackedWeight(
            raw, llamaquants.FORMATS["Q4_K"].ggml_type, (4, 256), "cpu",
            decode_on_device=True,
        )
        assert not weight._decodes_on_device()   # noqa: SLF001
        assert weight.to_dense().shape == (4, 256)


class TestTheProbe:
    def test_cpu_is_always_usable(self):
        cpu = next(d for d in hnxdevice.probe() if d.kind == "cpu")
        assert cpu.usable and not cpu.reason

    def test_auto_never_fails(self):
        """It falls back to the CPU, which cannot be absent."""
        assert hnxdevice.select("auto").usable

    def test_every_backend_is_reported_even_when_unusable(self):
        """"CUDA is not in the list" and "CUDA is there but this wheel
        has no kernels for your card" are different problems with
        different fixes. A probe that only returned what works could not
        tell them apart."""
        kinds = {d.kind for d in hnxdevice.probe()}
        assert {"cpu", "cuda", "mps", "xpu", "vulkan"} <= kinds

    def test_an_unusable_device_carries_a_reason(self):
        for device in hnxdevice.probe():
            if not device.usable:
                assert device.reason, device.name

    def test_an_unknown_device_lists_the_real_ones(self):
        with pytest.raises(hnxdevice.DeviceError, match="cuda"):
            hnxdevice.select("tpu")

    def test_a_named_device_that_cannot_run_is_not_silently_downgraded(self):
        """Someone who typed --device cuda wants to know why they did not
        get it, not a slow run they cannot explain."""
        if any(d.kind == "cuda" and d.usable for d in hnxdevice.probe()):
            pytest.skip("this machine has a usable CUDA device")
        with pytest.raises(hnxdevice.DeviceError):
            hnxdevice.select("cuda")

    def test_describe_names_what_auto_would_pick(self):
        assert "--device auto would pick" in hnxdevice.describe()


class TestTheArchitectureCheck:
    """The reason this module exists, driven with fake arch lists.

    A wheel carries a fixed list of GPU architectures. ``sm_61`` — every
    GTX 10-series card — is absent from recent ones, and
    ``torch.cuda.is_available()`` still says True. The failure arrives at
    the first kernel launch, worded as though the driver were broken.
    """

    @pytest.mark.parametrize(
        ("capability", "built", "expected"),
        [
            ((6, 1), [(6, 1), (7, 0)], True),      # exact match
            ((8, 6), [(8, 0)], True),              # up within a major
            ((8, 0), [(8, 6)], False),             # never down
            ((6, 1), [(7, 5), (8, 0), (8, 6)], False),   # the 1080 case
            ((9, 0), [], False),                   # a CPU-only wheel
        ],
    )
    def test_which_wheels_can_launch_where(self, capability, built, expected):
        assert hnxdevice._runs_on(capability, built) is expected  # noqa: SLF001

    def test_the_pascal_hint_names_a_wheel_that_has_it(self):
        hint = hnxdevice.wheel_hint((6, 1))
        assert "cu118" in hint
        assert "sm_61" in hint or "Pascal" in hint

    @pytest.mark.parametrize("capability", [(5, 0), (6, 0), (6, 1)])
    def test_every_old_card_gets_an_index_url_not_a_shrug(self, capability):
        assert "index-url" in hnxdevice.wheel_hint(capability)

    def test_a_card_older_than_any_wheel_says_so(self):
        assert "no current pytorch" in hnxdevice.wheel_hint((3, 5)).lower()

    def test_the_arch_list_parses(self):
        """It is read from torch and compared numerically, so a format
        this cannot parse silently means 'nothing is supported'."""
        parsed = hnxdevice._built_capabilities()  # noqa: SLF001
        assert all(isinstance(c, tuple) and len(c) == 2 for c in parsed)


class TestHalfPrecisionIsNotAutomatic:
    """FP16 exists on Pascal and runs at 1/64 of FP32 on GP102/104. A
    rule as reasonable-looking as "half on CUDA" makes a GTX 1080
    dramatically slower while appearing to optimise it."""

    def test_cpu_is_float32(self):
        assert hnxdevice.default_dtype("cpu") is torch.float32

    @pytest.mark.parametrize("capability", [(6, 0), (6, 1)])
    def test_pascal_stays_float32(self, capability):
        device = hnxdevice.Device(
            "cuda:0", "cuda", "fake", usable=True, capability=capability
        )
        assert hnxdevice.default_dtype(device) is torch.float32

    @pytest.mark.parametrize("capability", [(7, 0), (8, 6), (9, 0)])
    def test_tensor_core_cards_get_float16(self, capability):
        device = hnxdevice.Device(
            "cuda:0", "cuda", "fake", usable=True, capability=capability
        )
        assert hnxdevice.default_dtype(device) is torch.float16


class TestVulkanIsAnswered:
    """There is no usable Vulkan backend in torch. Reporting one because
    an import succeeded would be a lie with a long debugging tail."""

    def test_it_is_never_reported_usable(self):
        vulkan = next(d for d in hnxdevice.probe() if d.kind == "vulkan")
        assert not vulkan.usable

    def test_asking_for_it_gives_the_route_that_works(self):
        with pytest.raises(hnxdevice.DeviceError) as caught:
            hnxdevice.select("vulkan")
        message = str(caught.value)
        assert "llama.cpp" in message
        assert "hyprslug-headers wrap" in message

    def test_the_runtime_passes_that_message_through(self, tmp_path):
        """A user typing --device vulkan must get the route, not a
        traceback from somewhere three layers down."""
        from hypernix.models import hnxrun

        with pytest.raises(hnxrun.HnxRunError, match="llama.cpp"):
            hnxrun.load_model(tmp_path / "absent.gguf", device="vulkan")


class TestTheCommandLine:
    def _run(self, *argv):
        import contextlib
        import io

        from hypernix.interfaces import cli

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(list(argv))
        return code, out.getvalue()

    def test_devices_lists_them(self):
        code, text = self._run("devices")
        assert code == 0
        for kind in ("cpu", "cuda", "vulkan"):
            assert kind in text

    def test_devices_json_parses(self):
        import json

        code, text = self._run("devices", "--json")
        assert code == 0
        payload = json.loads(text)
        assert payload["auto"]
        assert any(d["kind"] == "cuda" for d in payload["devices"])

    def test_it_explains_rather_than_only_refusing(self):
        """An unusable backend with no remedy is a dead end."""
        code, text = self._run("devices")
        assert code == 0
        assert "wrap" in text or "index-url" in text or "usable but CPU" in text

    @pytest.mark.parametrize("command", ["generate", "chat"])
    def test_both_commands_take_a_device(self, command):
        source = Path("src/hypernix/interfaces/cli.py").read_text()
        body = source.split(f"def _run_{command}(")[1].split("\ndef ")[0]
        assert "--hnx-device" in body
        assert "device=ns.hnx_device" in body

    def test_serve_takes_one_too(self):
        source = Path("src/hypernix/quant/hyprslug_headers_cli.py").read_text()
        assert '"--device"' in source
        assert "device=args.device" in source
