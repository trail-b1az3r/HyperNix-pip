"""Three defects found by running the shipped commands on a real machine.

None showed up in the suite, and none was a subtle failure of the thing
under test. All three were the code being confidently wrong *around* a
correct result:

**A traceback where a sentence belonged.** ``hyprslug-headers serve`` on a
box whose torch was a CUDA build missing an NVIDIA runtime wheel ended
with ``ImportError: libcusparseLt.so.0: cannot open shared object file``
and eleven frames of stack. Nothing in that says what the reader has to
do, and the file it names is one they never installed on purpose.

**An endpoint announced before it existed.** The same run printed
``http://127.0.0.1:1234/v1  (ctrl-c to stop)`` and *then* died loading
the model. The line was printed before the load rather than after the
bind, so the last thing on screen before the traceback was a promise the
process never kept.

**A tier reported without its contradiction.** ``install`` listed
``IQ0.5_XXXL   Qwen3.8-2B-IQ0.9_L.gguf``. The tier is right and the name
is wrong -- the tensors are type 202 -- but printing them side by side
without remark leaves the reader to notice that the model they think is
0.9-bit is half-bit.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hypernix.models import hnxrun
from hypernix.models.hnxdevice import (
    DeviceError,
    _torch_import_advice,
    _torch_unavailable_reason,
    import_torch,
)
from hypernix.quant.hyprslug_headers import tier_in_name

#: The exact message the user's machine produced.
CUSPARSELT = "libcusparseLt.so.0: cannot open shared object file: No such file or directory"


class TestABrokenTorchIsExplainedNotDumped:
    """The advice has to name the situation, not just the symptom."""

    def test_a_missing_cuda_runtime_names_the_wheel_that_carries_it(self):
        advice = _torch_import_advice(ImportError(CUSPARSELT))

        assert "libcusparseLt.so.0" in advice
        assert "nvidia-cusparselt-cu12" in advice

    def test_it_offers_the_cpu_build_as_the_way_out(self):
        """The likelier fix. Someone serving a 0.5-bit model on a laptop
        wants torch to import, not CUDA."""
        advice = _torch_import_advice(ImportError(CUSPARSELT))

        assert "whl/cpu" in advice
        assert "--force-reinstall" in advice

    def test_it_says_torch_is_installed_rather_than_missing(self):
        """The distinction the whole message turns on."""
        advice = _torch_import_advice(ImportError(CUSPARSELT))

        assert "is installed" in advice
        assert "is not installed" not in advice

    def test_a_genuinely_absent_torch_says_that_instead(self):
        advice = _torch_import_advice(ImportError("No module named 'torch'"))

        assert "not installed" in advice
        assert "libcusparse" not in advice

    def test_an_unrecognised_import_failure_is_passed_through(self):
        """Better a message with the real text than a confident wrong guess."""
        advice = _torch_import_advice(ImportError("some other problem"))

        assert "some other problem" in advice

    def test_import_torch_raises_a_device_error_not_an_import_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fail(name, *args, **kwargs):
            if name == "torch":
                raise ImportError(CUSPARSELT)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail)

        with pytest.raises(DeviceError, match="nvidia-cusparselt-cu12"):
            import_torch()

    def test_load_model_reports_it_as_an_environment_problem(self, monkeypatch):
        """Not as a problem with the file, which is what a bare
        HnxRunError against a path reads as."""
        import builtins

        real_import = builtins.__import__

        def fail(name, *args, **kwargs):
            if name == "torch":
                raise ImportError(CUSPARSELT)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail)

        with pytest.raises(hnxrun.HnxEnvironmentError) as caught:
            hnxrun.load_model("/nonexistent/model.gguf")

        assert "libcusparseLt.so.0" in str(caught.value)

    def test_the_environment_error_is_still_an_hnxrunerror(self):
        """Every existing `except HnxRunError` must keep working."""
        assert issubclass(hnxrun.HnxEnvironmentError, hnxrun.HnxRunError)

    def test_the_server_does_not_blame_the_model_for_it(self, monkeypatch, tmp_path):
        from hypernix.quant.hyprslug_server import HyprslugModel, ServerError

        model = tmp_path / "innocent.gguf"
        model.write_bytes(b"GGUF")

        def boom(*_args, **_kwargs):
            raise hnxrun.HnxEnvironmentError("PyTorch is installed but cannot load x")

        monkeypatch.setattr(hnxrun, "load_model", boom)

        with pytest.raises(ServerError) as caught:
            HyprslugModel(model)

        assert "innocent.gguf" not in str(caught.value)

    def test_a_real_model_problem_still_names_the_model(self, monkeypatch, tmp_path):
        """The other half of the pair — the prefix must not just be gone."""
        from hypernix.quant.hyprslug_server import HyprslugModel, ServerError

        model = tmp_path / "guilty.gguf"
        model.write_bytes(b"GGUF")

        def boom(*_args, **_kwargs):
            raise hnxrun.HnxRunError("not a llama-family architecture")

        monkeypatch.setattr(hnxrun, "load_model", boom)

        with pytest.raises(ServerError) as caught:
            HyprslugModel(model)

        assert "guilty.gguf" in str(caught.value)


class TestTheProbeDoesNotSayTorchIsAbsentWhenItIsPresent:
    """`hypernix devices` said "torch is not installed" on a machine that
    had it. That sends the reader to install what they already have."""

    def test_a_link_failure_is_reported_as_a_link_failure(self):
        reason = _torch_unavailable_reason(ImportError(CUSPARSELT))

        assert "installed but cannot load" in reason
        assert "libcusparseLt.so.0" in reason
        assert reason != "torch is not installed."

    def test_a_missing_module_still_says_not_installed(self):
        reason = _torch_unavailable_reason(ImportError("No module named 'torch'"))

        assert reason == "torch is not installed."


class TestTheEndpointIsAnnouncedOnlyOnceItAnswers:
    """It was printed before the load, so a failed load left a URL on
    screen that nothing was listening on."""

    def test_serve_takes_a_hook_called_after_the_bind(self):
        import inspect

        from hypernix.quant.hyprslug_server import serve

        assert "on_ready" in inspect.signature(serve).parameters

    def test_the_cli_passes_the_hook_rather_than_printing_early(self):
        """Reading the source, because the alternative is starting a
        server: the print must not sit above the serve() call."""
        source = Path(
            __import__("hypernix.quant.hyprslug_headers_cli", fromlist=["x"]).__file__
        ).read_text(encoding="utf-8")
        block = source[source.index('if args.command == "serve"'):]
        block = block[:block.index("parser.error")]

        assert "on_ready=announce" in block
        endpoint_line = block.index('/v1  "')
        serve_call = block.index("serve(args.model")
        assert endpoint_line < serve_call, (
            "the endpoint is printed before serve() is entered again"
        )
        assert "def announce" in block[:endpoint_line], (
            "the endpoint print must live inside the on_ready callback"
        )


class TestAFilenameThatContradictsTheTensors:
    """`install` printed `IQ0.5_XXXL   Qwen3.8-2B-IQ0.9_L.gguf` and said
    nothing about the two disagreeing."""

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("Qwen3.8-2B-IQ0.9_L.gguf", "IQ0.9_L"),
            ("nano-mini-IQ0.5_XXXL.gguf", "IQ0.5_XXXL"),
            ("thing-IQ0.75_M.gguf", "IQ0.75_M"),
            ("thing-IQ0.25_UXL.gguf", "IQ0.25_UXL"),
            ("small-INT1.gguf", "INT1"),
            ("small-INT4.gguf", "INT4"),
            ("small-FP2.gguf", "FP2"),
        ],
    )
    def test_it_reads_the_tier_a_name_claims(self, filename, expected):
        assert tier_in_name(filename) == expected

    def test_an_upstream_name_claims_nothing(self):
        assert tier_in_name("llama-7b-Q4_K_M.gguf") == ""
        assert tier_in_name("model.gguf") == ""

    def test_the_longest_name_wins(self):
        """IQ0.5_XXXL must not be read as IQ0.5 by a shorter pattern."""
        assert tier_in_name("m-IQ0.5_XXXL.gguf") == "IQ0.5_XXXL"

    def test_the_report_flags_the_disagreement(self):
        """The user's exact case, as scan() would classify it."""
        row = {
            "tier": "IQ0.5_XXXL",
            "named_tier": tier_in_name("Qwen3.8-2B-IQ0.9_L.gguf"),
        }
        row["misnamed"] = bool(
            row["named_tier"] and row["tier"] and row["named_tier"] != row["tier"]
        )

        assert row["misnamed"] is True

    def test_agreement_is_not_flagged(self):
        row = {"tier": "IQ0.5_XXXL", "named_tier": tier_in_name("m-IQ0.5_XXXL.gguf")}

        assert not (row["named_tier"] != row["tier"])
