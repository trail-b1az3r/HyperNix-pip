"""Every tensor in a forward pass has to be on the same device.

This file exists because of a bug that passed 4365 local tests and every
Linux and Windows CI job, then failed twelve tests on macOS at once:

    RuntimeError: Expected all tensors to be on the same device, but
    found at least two devices, mps:0 and cpu!

``_rope`` built its inverse-frequency table with ``torch.arange(...)``
and no ``device=``. On a CPU run that is correct by accident, because the
default *is* the CPU. On a machine where ``--device auto`` finds an
accelerator -- which is every macOS runner, since MPS is present -- the
table lands on the CPU, the positions land on the GPU, and the multiply
between them ends the forward pass.

The mistake worth guarding is not the one line. It is that **a
device-placement bug is invisible on a single-device machine**, so the
usual suite cannot see it however many tests are added. Both tests here
are built to fail on a CPU-only box:

**The meta device.** ``torch.zeros(..., device="meta")`` allocates no
memory but still carries a device identity, and torch enforces that
identity in ops. A function that silently creates CPU tensors therefore
raises against meta inputs on any machine, with no GPU involved. That
turns "would break on MPS" into an ordinary assertion.

**The factory audit.** The meta test covers the function that was wrong.
The audit covers the class: every ``torch`` tensor factory in the
runtime modules must say where its result goes. It is a source-level
check because the property is a source-level one -- an omitted ``device=``
is only observable at runtime on hardware the test machine may not have.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from hypernix.models.hnxrun import _repeat_kv, _rms_norm, _rope

#: Modules that run inside a forward pass, and so must be device-clean.
RUNTIME_MODULES = ("hnxrun.py", "hnxtorch.py")

#: ``torch`` callables that allocate a new tensor from scratch. Each one
#: puts its result on the default device unless told otherwise, which is
#: what makes a missing ``device=`` a placement bug rather than a style
#: question. ``*_like`` factories are deliberately absent: they inherit
#: the device of their argument, which is the behaviour being asked for.
FACTORIES = frozenset({
    "arange", "as_tensor", "empty", "eye", "full", "linspace", "logspace",
    "ones", "rand", "randint", "randn", "randperm", "tensor", "zeros",
})

#: ``from_numpy`` and ``frombuffer`` cannot take ``device=`` at all --
#: they wrap host memory, so the result is always on the CPU. They are
#: correct only when the result is moved, so they are checked for a
#: ``.to(...)`` instead of for a keyword.
HOST_WRAPPERS = frozenset({"from_numpy", "frombuffer"})


def _source_root() -> Path:
    import hypernix.models

    return Path(hypernix.models.__file__).parent


def _torch_factory_calls(tree: ast.AST):
    """Every ``torch.<factory>(...)`` call, with the keywords it passes."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if getattr(func.value, "id", None) != "torch":
            continue
        if func.attr in FACTORIES or func.attr in HOST_WRAPPERS:
            yield node, func.attr, {kw.arg for kw in node.keywords}


def _wrapped_in_a_move(tree: ast.AST) -> set[int]:
    """Line numbers of calls that are the receiver of a ``.to(...)``."""
    moved: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "to":
            inner = func.value
            if isinstance(inner, ast.Call):
                moved.add(inner.lineno)
    return moved


class TestTheMetaDeviceCatchesItWithoutAGpu:
    """Placement bugs, made visible on a machine with one device."""

    def test_rope_keeps_its_arithmetic_where_the_input_lives(self):
        """The bug, reduced to one assertion.

        Pre-fix this raises ``Tensor on device cpu is not on the expected
        device meta!`` -- the same class of error the macOS jobs hit,
        reached without an accelerator.
        """
        x = torch.zeros(2, 3, 8, device="meta")
        positions = torch.arange(3, device="meta")

        out = _rope(x, positions, 10000.0, 8)

        assert out.device.type == "meta"
        assert tuple(out.shape) == (2, 3, 8)

    def test_rope_on_a_partial_rotation_too(self):
        """``rope_dims`` under ``head_dim`` takes the concatenating path."""
        x = torch.zeros(2, 3, 8, device="meta")
        positions = torch.arange(3, device="meta")

        out = _rope(x, positions, 10000.0, 4)

        assert out.device.type == "meta"
        assert tuple(out.shape) == (2, 3, 8)

    def test_rms_norm_stays_put(self):
        x = torch.zeros(3, 8, device="meta")
        weight = torch.zeros(8, device="meta")

        assert _rms_norm(x, weight, 1e-5).device.type == "meta"

    def test_repeat_kv_stays_put(self):
        x = torch.zeros(2, 3, 8, device="meta")

        assert _repeat_kv(x, 2).device.type == "meta"

    def test_rope_still_computes_the_right_numbers_on_the_cpu(self):
        """The fix must not have changed the arithmetic.

        A rotation of the zero vector is zero, so this uses real values
        and checks the invariant that makes it a rotation: pairwise norms
        survive it.
        """
        torch.manual_seed(0)
        x = torch.randn(2, 5, 8)
        positions = torch.arange(5)

        out = _rope(x, positions, 10000.0, 8)

        before = (x[..., 0::2] ** 2 + x[..., 1::2] ** 2).sqrt()
        after = (out[..., 0::2] ** 2 + out[..., 1::2] ** 2).sqrt()
        assert torch.allclose(before, after, atol=1e-5)


class TestEveryTensorFactorySaysWhereItGoes:
    """The class of bug, checked at the source level.

    A missing ``device=`` is only observable at runtime on hardware that
    has a second device, so a machine without one cannot test for it any
    other way.
    """

    @pytest.mark.parametrize("module", RUNTIME_MODULES)
    def test_the_audit_finds_something_to_audit(self, module):
        """Guard against the audit passing by looking at nothing."""
        tree = ast.parse((_source_root() / module).read_text())

        found = list(_torch_factory_calls(tree))

        assert len(found) >= 4, f"only found {found} in {module}"

    @pytest.mark.parametrize("module", RUNTIME_MODULES)
    def test_no_factory_relies_on_the_default_device(self, module):
        tree = ast.parse((_source_root() / module).read_text())
        moved = _wrapped_in_a_move(tree)

        naive = [
            f"{module}:{node.lineno} torch.{attr}()"
            for node, attr, keywords in _torch_factory_calls(tree)
            if attr in FACTORIES and "device" not in keywords
        ]
        unmoved = [
            f"{module}:{node.lineno} torch.{attr}() is never .to(...)"
            for node, attr, _ in _torch_factory_calls(tree)
            if attr in HOST_WRAPPERS and node.lineno not in moved
        ]

        assert not naive + unmoved, (
            "these land on the default device, which is the CPU only by "
            "accident on a CPU-only machine: " + "; ".join(naive + unmoved)
        )


class TestTheSeededDrawHappensOnTheCpu:
    """``torch.multinomial`` refuses a generator from another device.

    ``generate_tokens`` seeds a ``torch.Generator(device="cpu")``, so a
    probability vector left on an accelerator raises *Expected a 'mps:0'
    generator device but found 'cpu'* rather than sampling. Moving the
    vector rather than the generator also means a seed picks the same
    draws on every backend.

    Checked at the source level for the same reason as the audit above:
    the mismatch cannot be constructed on a single-device machine, and a
    runtime assertion that the vector is on the CPU passes whether or not
    the fix is present.
    """

    def test_the_probability_vector_is_moved_before_it_is_drawn_from(self):
        tree = ast.parse((_source_root() / "hnxrun.py").read_text())

        draws = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "multinomial"
        ]

        assert len(draws) == 1, f"expected one draw, found {len(draws)}"
        first = draws[0].args[0]
        assert (
            isinstance(first, ast.Call)
            and isinstance(first.func, ast.Attribute)
            and first.func.attr == "cpu"
        ), (
            f"hnxrun.py:{draws[0].lineno}: multinomial is given a tensor that "
            f"was never moved to the CPU, so it will refuse the CPU generator "
            f"on any accelerator"
        )

    def test_the_generator_is_a_cpu_one(self):
        """The other half of the pair, so the two cannot drift apart."""
        tree = ast.parse((_source_root() / "hnxrun.py").read_text())

        generators = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Generator"
        ]

        assert len(generators) == 1
        devices = [
            kw.value.value for kw in generators[0].keywords if kw.arg == "device"
        ]
        assert devices == ["cpu"]
