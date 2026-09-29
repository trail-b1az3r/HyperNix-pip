"""VRAM optimizations: what they do, and what they refuse to do.

Every technique in `hypernix.system.vram` is invisible when it silently
fails — you get the same loss curve and the same OOM, with no way to tell
which one did not take effect. So most of what is worth testing here is
the refusals: that an allocator call made too late reports it, that the
fused optimizer will not run alongside gradient clipping, that a handle
really does undo what it did.

The torch-dependent half is guarded rather than skipped wholesale,
because the torch-free half is not a convenience: `configure_allocator`
has to be callable from a launcher that has not imported torch yet, or it
cannot work at all.
"""
from __future__ import annotations

import os

import pytest

from hypernix.system import vram

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    # A module-level importorskip would skip this whole file — including
    # the half that must keep working without torch, which is the half
    # most worth testing.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False

needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")


@pytest.fixture(autouse=True)
def clean_alloc_conf(monkeypatch):
    """Each test starts with no PYTORCH_CUDA_ALLOC_CONF of its own."""
    monkeypatch.delenv(vram.ALLOC_CONF_VAR, raising=False)


class TestImportsWithoutTorch:
    def test_importing_the_module_does_not_import_torch(self):
        """The point of the module, not a testing convenience.

        The CUDA allocator reads its configuration when it initializes.
        A launcher that has to import torch to reach the function that
        configures the allocator has already lost.
        """
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable, "-c",
                "import sys; from hypernix.system import vram; "
                "print('torch' in sys.modules)",
            ],
            capture_output=True, text=True, check=False,
            env={**os.environ, "PYTHONPATH": "src"},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"


class TestAllocatorConfiguration:
    def test_it_sets_expandable_segments(self):
        report = vram.configure_allocator()
        assert report.applied
        assert "expandable_segments:True" in report.value
        assert os.environ[vram.ALLOC_CONF_VAR] == report.value

    def test_it_keeps_what_the_caller_already_set(self, monkeypatch):
        """An explicit setting is a decision somebody made."""
        monkeypatch.setenv(vram.ALLOC_CONF_VAR, "expandable_segments:False")

        report = vram.configure_allocator()

        assert not report.applied
        assert "already set by the caller" in report.reason
        assert os.environ[vram.ALLOC_CONF_VAR] == "expandable_segments:False"

    def test_override_existing_replaces_it(self, monkeypatch):
        monkeypatch.setenv(vram.ALLOC_CONF_VAR, "expandable_segments:False")

        report = vram.configure_allocator(override_existing=True)

        assert report.applied
        assert "expandable_segments:True" in report.value

    def test_it_merges_with_unrelated_keys(self, monkeypatch):
        monkeypatch.setenv(vram.ALLOC_CONF_VAR, "max_split_size_mb:128")

        report = vram.configure_allocator()

        assert report.applied
        assert "max_split_size_mb:128" in report.value
        assert "expandable_segments:True" in report.value

    def test_a_partial_skip_still_applies_the_rest(self, monkeypatch):
        monkeypatch.setenv(vram.ALLOC_CONF_VAR, "expandable_segments:False")

        report = vram.configure_allocator(garbage_collection_threshold=0.8)

        assert report.applied
        assert "garbage_collection_threshold:0.8" in report.value
        assert "expandable_segments:False" in report.value
        assert "kept caller's expandable_segments" in report.reason

    def test_nothing_requested_is_reported_not_applied(self):
        report = vram.configure_allocator(expandable_segments=False)
        assert not report.applied
        assert report.reason == "nothing to set"

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
    def test_a_threshold_outside_zero_to_one_is_refused(self, bad):
        with pytest.raises(ValueError, match="fraction of capacity"):
            vram.configure_allocator(garbage_collection_threshold=bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_split_size_is_refused(self, bad):
        with pytest.raises(ValueError, match="must be positive"):
            vram.configure_allocator(max_split_size_mb=bad)

    def test_a_malformed_existing_value_does_not_crash(self, monkeypatch):
        """This parses an env var somebody may have typed by hand."""
        monkeypatch.setenv(vram.ALLOC_CONF_VAR, ",,garbage,:,x:,:y,ok:1,")

        report = vram.configure_allocator()

        assert report.applied
        assert "ok:1" in report.value
        assert "expandable_segments:True" in report.value

    def test_report_names_the_variable(self):
        applied = vram.configure_allocator()
        assert vram.ALLOC_CONF_VAR in applied.report()
        refused = vram.AllocatorReport(
            applied=False, value="", previous="", reason="because"
        )
        assert "because" in refused.report()


class TestRecommendations:
    def test_they_come_back_largest_first(self):
        recs = vram.recommend(
            parameters=7_000_000_000,
            layers=32,
            batch_size=4,
            context_length=4096,
            hidden_size=4096,
            grad_clip=False,
        )
        sizes = [r.saves_bytes for r in recs]
        assert sizes == sorted(sizes, reverse=True)

    def test_clipping_removes_the_fused_optimizer(self):
        """Not listed with a caveat — left out, because it cannot be used."""
        with_clip = vram.recommend(parameters=1_000_000, grad_clip=True)
        without = vram.recommend(parameters=1_000_000, grad_clip=False)

        names = {r.technique for r in with_clip}
        assert "optimizer-in-backward" not in names
        assert "optimizer-in-backward" in {r.technique for r in without}

    def test_accumulation_removes_it_too(self):
        recs = vram.recommend(
            parameters=1_000_000, grad_clip=False, accumulation_steps=4
        )
        assert "optimizer-in-backward" not in {r.technique for r in recs}

    def test_unknown_shape_skips_the_activation_estimate(self):
        """Zero means unknown, and an unknown is not guessed at."""
        recs = vram.recommend(parameters=1_000_000, layers=0)
        assert "activation checkpointing" not in {r.technique for r in recs}

    def test_a_known_shape_includes_it(self):
        recs = vram.recommend(
            parameters=1_000_000,
            layers=12,
            batch_size=2,
            context_length=1024,
            hidden_size=768,
        )
        assert "activation checkpointing" in {r.technique for r in recs}

    def test_negative_parameters_are_refused(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            vram.recommend(parameters=-1)

    def test_every_recommendation_reports_readably(self):
        for rec in vram.recommend(parameters=1_000_000_000, grad_clip=False):
            text = rec.report()
            assert rec.technique in text
            assert rec.call in text
            assert "iB" in text or " B" in text

    def test_bf16_halves_the_activation_estimate(self):
        wide = vram.recommend(
            parameters=1_000_000, layers=8, batch_size=1,
            context_length=512, hidden_size=512, param_bytes=4,
        )
        narrow = vram.recommend(
            parameters=1_000_000, layers=8, batch_size=1,
            context_length=512, hidden_size=512, param_bytes=2,
        )
        a = next(r for r in wide if r.technique == "activation checkpointing")
        b = next(r for r in narrow if r.technique == "activation checkpointing")
        assert b.saves_bytes == a.saves_bytes // 2


class TestCPUOnlyDegradation:
    """These are called on machines with no GPU. They must not raise."""

    def test_release_cache_returns_zero_without_cuda(self):
        assert vram.release_cache() >= 0

    def test_measure_peak_reports_unavailable_rather_than_lying(self):
        with vram.measure_peak() as peak:
            pass
        if not peak.available:
            assert "unavailable" in peak.report()
            assert peak.allocated == 0


class TestHandles:
    def test_a_checkpoint_handle_with_nothing_wrapped_reports_so(self):
        handle = vram.CheckpointHandle()
        assert not handle.active
        assert "not applied" in handle.report()
        assert handle.disable() == 0

    def test_a_fused_handle_step_and_zero_grad_are_safe_no_ops(self):
        """An existing loop keeps its shape; the calls do nothing."""
        handle = vram.FusedOptimizerHandle()
        handle.step()
        handle.zero_grad(set_to_none=True)
        assert handle.parameter_count == 0
        assert handle.remove() == 0

    def test_disable_is_idempotent(self):
        handle = vram.CheckpointHandle(wrapped=1, total_blocks=1)
        handle._undo.append((object(), None))
        first = handle.disable()
        assert handle.disable() == 0
        assert first >= 0
        assert not handle.active


# ---------------------------------------------------------------------------
# The half that needs a real torch
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class _Block(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.fc = nn.Linear(width, width)

        def forward(self, x):  # noqa: D102
            return torch.relu(self.fc(x)) + x

    class _Stack(nn.Module):
        def __init__(self, width: int = 8, depth: int = 4) -> None:
            super().__init__()
            self.layers = nn.ModuleList(_Block(width) for _ in range(depth))

        def forward(self, x):  # noqa: D102
            for layer in self.layers:
                x = layer(x)
            return x


@needs_torch
class TestCheckpointBlocks:
    def test_it_finds_and_wraps_the_layer_stack(self):
        model = _Stack(depth=4)
        handle = vram.checkpoint_blocks(model, prefer_native=False)
        assert handle.wrapped == 4
        assert handle.total_blocks == 4
        assert handle.active

    def test_every_n_wraps_a_subset(self):
        model = _Stack(depth=6)
        handle = vram.checkpoint_blocks(model, every=2, prefer_native=False)
        assert handle.wrapped == 3
        assert handle.total_blocks == 6

    def test_every_zero_is_refused(self):
        with pytest.raises(ValueError, match="every must be >= 1"):
            vram.checkpoint_blocks(_Stack(), every=0)

    def test_the_output_is_unchanged(self):
        """A memory optimization that changes the answer is a bug."""
        torch.manual_seed(0)
        model = _Stack().eval()
        x = torch.randn(2, 8)
        with torch.no_grad():
            before = model(x).clone()

        vram.checkpoint_blocks(model, prefer_native=False)
        with torch.no_grad():
            after = model(x)

        assert torch.allclose(before, after)

    def test_gradients_are_unchanged(self):
        """Including the first block, whose input does not require grad.

        This is what `use_reentrant=False` is for: the reentrant
        implementation silently produces no gradient at all when no input
        to the checkpointed region requires one, which for a transformer
        is the embedding output feeding block zero.
        """
        torch.manual_seed(0)
        plain = _Stack()
        torch.manual_seed(0)
        wrapped = _Stack()
        vram.checkpoint_blocks(wrapped, prefer_native=False)

        x = torch.randn(2, 8)
        plain(x).sum().backward()
        wrapped(x).sum().backward()

        for a, b in zip(plain.parameters(), wrapped.parameters(), strict=True):
            assert a.grad is not None
            assert b.grad is not None
            assert torch.allclose(a.grad, b.grad, atol=1e-6)

    def test_disable_restores_the_original_forward(self):
        model = _Stack(depth=3)
        originals = [block.forward for block in model.layers]

        handle = vram.checkpoint_blocks(model, prefer_native=False)
        assert [b.forward for b in model.layers] != originals

        assert handle.disable() == 3
        assert [b.forward for b in model.layers] == originals
        assert not handle.active

    def test_a_model_with_no_repeated_stack_is_reported_not_raised(self):
        handle = vram.checkpoint_blocks(nn.Linear(4, 4), prefer_native=False)
        assert handle.wrapped == 0
        assert handle.strategy == "none found"
        assert "not applied" in handle.report()

    def test_it_passes_through_under_no_grad(self):
        """Checkpointing with no backward to come would run twice for nothing."""
        model = _Stack()
        vram.checkpoint_blocks(model, prefer_native=False)
        with torch.no_grad():
            out = model(torch.randn(1, 8))
        assert out.shape == (1, 8)

    def test_native_path_is_preferred_when_the_model_offers_one(self):
        calls = []

        class WithNative(_Stack):
            def gradient_checkpointing_enable(self, **kwargs):
                calls.append(kwargs)

            def gradient_checkpointing_disable(self):
                calls.append("off")

        model = WithNative()
        handle = vram.checkpoint_blocks(model)

        assert handle.strategy == "native"
        assert calls and calls[0]
        handle.disable()
        assert calls[-1] == "off"

    def test_native_path_tolerates_an_older_signature(self):
        calls = []

        class OlderNative(_Stack):
            def gradient_checkpointing_enable(self):
                calls.append("on")

        handle = vram.checkpoint_blocks(OlderNative())
        assert handle.strategy == "native"
        assert calls == ["on"]


@needs_torch
class TestFuseOptimizerIntoBackward:
    def test_it_steps_every_parameter_during_backward(self):
        model = _Stack(depth=2)
        before = [p.detach().clone() for p in model.parameters()]

        handle = vram.fuse_optimizer_into_backward(
            model, lambda ps: torch.optim.SGD(ps, lr=0.1)
        )
        model(torch.randn(2, 8)).sum().backward()

        assert handle.parameter_count == len(before)
        # `any`, not `all`: a ReLU block can legitimately produce a zero
        # gradient for one bias on one batch, and a test that fails on
        # that is testing the initialization, not the hook.
        assert any(
            not torch.allclose(old, new)
            for old, new in zip(before, model.parameters(), strict=True)
        )

    def test_gradients_are_released_as_they_are_applied(self):
        """The whole point: no second full copy of the model."""
        model = _Stack(depth=2)
        vram.fuse_optimizer_into_backward(
            model, lambda ps: torch.optim.SGD(ps, lr=0.1)
        )

        model(torch.randn(2, 8)).sum().backward()

        assert all(p.grad is None for p in model.parameters())

    def test_grad_clip_is_refused(self):
        with pytest.raises(ValueError, match="global norm"):
            vram.fuse_optimizer_into_backward(
                _Stack(), lambda ps: torch.optim.SGD(ps, lr=0.1), grad_clip=1.0
            )

    def test_accumulation_is_refused(self):
        with pytest.raises(ValueError, match="accumulation_steps"):
            vram.fuse_optimizer_into_backward(
                _Stack(),
                lambda ps: torch.optim.SGD(ps, lr=0.1),
                accumulation_steps=4,
            )

    def test_a_grad_scaler_is_refused(self):
        with pytest.raises(ValueError, match="GradScaler"):
            vram.fuse_optimizer_into_backward(
                _Stack(),
                lambda ps: torch.optim.SGD(ps, lr=0.1),
                scaler=object(),
            )

    def test_remove_detaches_the_hooks(self):
        model = _Stack(depth=2)
        handle = vram.fuse_optimizer_into_backward(
            model, lambda ps: torch.optim.SGD(ps, lr=0.1)
        )
        assert handle.remove() == handle.parameter_count

        before = [p.detach().clone() for p in model.parameters()]
        model(torch.randn(2, 8)).sum().backward()

        for old, new in zip(before, model.parameters(), strict=True):
            assert torch.allclose(old, new)
        assert all(p.grad is not None for p in model.parameters())

    def test_a_frozen_model_warns_rather_than_pretending(self):
        model = _Stack()
        for p in model.parameters():
            p.requires_grad_(False)

        with pytest.warns(UserWarning, match="no parameters"):
            handle = vram.fuse_optimizer_into_backward(
                model, lambda ps: torch.optim.SGD(ps, lr=0.1)
            )
        assert handle.parameter_count == 0

    def test_state_dict_is_positional(self):
        model = _Stack(depth=2)
        handle = vram.fuse_optimizer_into_backward(
            model, lambda ps: torch.optim.SGD(ps, lr=0.1)
        )
        state = handle.state_dict()
        assert state["fused_optimizer_in_backward"] is True
        assert len(state["states"]) == handle.parameter_count


@needs_torch
class TestOffloadOptimizerState:
    def test_it_restores_state_on_the_way_out(self):
        model = _Stack(depth=2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.randn(2, 8)).sum().backward()
        opt.step()

        devices_before = {
            id(v): v.device
            for s in opt.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        }

        with vram.offload_optimizer_state(opt) as moved:
            assert moved >= 0

        devices_after = {
            id(v): v.device
            for s in opt.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        }
        assert set(devices_before.values()) == set(devices_after.values())

    def test_an_exception_inside_still_restores(self):
        """Otherwise the next step fails with an error naming neither."""
        model = _Stack(depth=2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.randn(2, 8)).sum().backward()
        opt.step()

        with pytest.raises(RuntimeError, match="boom"):
            with vram.offload_optimizer_state(opt):
                raise RuntimeError("boom")

        model(torch.randn(2, 8)).sum().backward()
        opt.step()  # must not raise about mismatched devices

    def test_cpu_only_moves_nothing_and_is_not_an_error(self):
        model = _Stack(depth=1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.randn(2, 8)).sum().backward()
        opt.step()

        with vram.offload_optimizer_state(opt) as moved:
            if not torch.cuda.is_available():
                assert moved == 0

    def test_an_optimizer_with_no_state_is_fine(self):
        opt = torch.optim.AdamW(_Stack(depth=1).parameters(), lr=1e-3)
        with vram.offload_optimizer_state(opt) as moved:
            assert moved == 0


class TestTrainingIntegration:
    """The flags are wired to the loop, not just to the library.

    A VRAM technique nobody can reach from `hypernix train run` is a
    technique nobody uses.
    """

    def test_train_accepts_the_flags(self):
        """Checked on the signature, so this runs without torch."""
        import ast
        import pathlib

        source = pathlib.Path("src/hypernix/training/train.py").read_text()
        tree = ast.parse(source)
        fn = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        names = {a.arg for a in fn.args.kwonlyargs}
        assert {
            "gradient_checkpointing",
            "checkpoint_every",
            "fuse_optimizer",
            "tune_allocator",
        } <= names

    def test_the_cli_exposes_them_and_passes_them_through(self):
        import pathlib

        source = pathlib.Path("src/hypernix/interfaces/cli.py").read_text()
        for flag in (
            "--gradient-checkpointing",
            "--checkpoint-every",
            "--fuse-optimizer",
            "--tune-allocator",
        ):
            assert flag in source, flag
        for kwarg in (
            "gradient_checkpointing=ns.gradient_checkpointing",
            "checkpoint_every=ns.checkpoint_every",
            "fuse_optimizer=ns.fuse_optimizer",
            "tune_allocator=ns.tune_allocator",
        ):
            assert kwarg in source, kwarg

    def test_clipping_plus_fusing_is_refused_before_anything_loads(self):
        """The refusal has to come before the model is read off disk.

        Otherwise a caller who passed both waits out a checkpoint load to
        be told the combination was never going to work.
        """
        import ast
        import pathlib

        source = pathlib.Path("src/hypernix/training/train.py").read_text()
        tree = ast.parse(source)
        fn = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        raise_line = next(
            node.lineno for node in ast.walk(fn)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and getattr(node.exc.func, "id", "") == "ValueError"
        )
        load_line = next(
            node.lineno for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "load_snapshot"
        )
        assert raise_line < load_line
