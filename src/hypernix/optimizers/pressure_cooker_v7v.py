"""hypernix.pressure_cooker_v7v — Pressure Cooker V7V: V7 with the accelerated paths.

V7V is :class:`~hypernix.optimizers.pressure_cooker_v7.PressureCookerV7`
with two optional execution optimisations layered on the same update rule.
It tries them in order and settles on the first that works::

    CUDA graph        capture one whole training step, replay it
        |             (CUDA only; you call warmup_graph() yourself)
    torch.compile     compile the per-bucket update
        |
    foreach           the fused multi-tensor path V7 always has
        |
    per-tensor        the portable fallback

Nothing here is mandatory. V7V constructs and trains on a CPU, on a GPU
without ``torch.compile``, and on a GPU where compilation fails: it warns
once, falls to the next tier, and :meth:`describe` says which tier is
running (``kernel``) and why the ones above it are not.

What it does not do is pretend. On a CPU there is no CUDA graph, so
:meth:`warmup_graph` raises instead of quietly doing nothing; and
``torch.compile`` is attempted only on CUDA, where it has something to
fuse (on a CPU it measured no gain for these memory-bound loops and adds
a long first step).
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch

from hypernix.optimizers.pressure_cooker_v7 import PressureCookerV7, _Bucket

__all__ = ["PressureCookerV7V", "V7V"]


class PressureCookerV7V(PressureCookerV7):
    """V7 with ``torch.compile`` and CUDA-graph replay where they are available.

    Args:
        params: Parameters or param-group dicts, on any device.
        compile: ``"auto"`` (compile on CUDA only), ``True`` (try wherever
            ``torch.compile`` exists) or ``False``.
        **kwargs: Everything :class:`PressureCookerV7` takes.
    """

    variant = "balanced"

    def __init__(self, params: Any, *, compile: bool | str = "auto", **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self._compiled: Callable[..., Any] | None = None
        self._graph: Any = None
        self._graph_result: Any = None
        self.fallback_reasons: list[str] = []
        on_cuda = any(p.device.type == "cuda" for g in self.param_groups for p in g["params"])
        self._on_cuda = on_cuda
        want = compile is True or (compile == "auto" and on_cuda)
        if compile == "auto" and not on_cuda:
            self.fallback_reasons.append("torch.compile: not attempted on a CPU")
        elif compile is False:
            self.fallback_reasons.append("torch.compile: turned off")
        if want:
            self._try_compile()
        if not on_cuda:
            self.fallback_reasons.append("CUDA graph: no CUDA parameters")

    def _try_compile(self) -> None:
        if not hasattr(torch, "compile"):
            self.fallback_reasons.append(f"torch.compile: not in torch {torch.__version__}")
            return
        try:
            self._compiled = torch.compile(self._step_bucket)
        except Exception as exc:  # noqa: BLE001 - depends on the local toolchain
            self.fallback_reasons.append(f"torch.compile: could not wrap the step ({exc!r})")
            self._compiled = None

    def _apply_update(self, buckets: list[_Bucket], measure: bool,
                      raw_gnorms: list[torch.Tensor] | None) -> list[torch.Tensor]:
        if self._compiled is not None:
            try:
                stats = [self._compiled(bucket, measure, raw_gnorms[i] if raw_gnorms else None)
                         for i, bucket in enumerate(buckets)]
                return [torch.stack([s[i] for s in stats]).mean() for i in range(3)]
            except Exception as exc:  # noqa: BLE001 - a compile that fails at run time
                warnings.warn(f"the compiled V7V step failed ({exc!r}); continuing on the "
                              "foreach path for the rest of this run", stacklevel=2)
                self.fallback_reasons.append(f"torch.compile: failed at run time ({exc!r})")
                self._compiled = None
        return super()._apply_update(buckets, measure, raw_gnorms)

    # -- CUDA graphs ---------------------------------------------------------

    def warmup_graph(self, step_fn: Callable[[], Any]) -> None:
        """Capture a CUDA graph of one whole training step.

        *step_fn* runs forward, backward and ``step()`` on a fixed-shape
        batch. A graph replays the kernels it recorded and nothing else, so
        every decision made in Python is frozen at capture. For V7 that
        means three things are refused here rather than silently frozen:

        * ``grad_accum_steps > 1`` and a ``grad_scaler``: whether to apply
          the update is decided per call;
        * ``skip_on_nonfinite``: whether to skip is decided per call;
        * a ``stat_interval`` above 1: whether to measure is decided per
          call.

        Construct the optimizer with ``skip_on_nonfinite=False`` and
        ``stat_interval=1`` to capture a graph.
        """
        if not self._on_cuda or not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs need CUDA parameters; on a CPU V7V runs the "
                               "foreach path (see describe()['kernel'])")
        blockers = []
        if self.grad_accum_steps > 1:
            blockers.append("grad_accum_steps > 1")
        if self.grad_scaler is not None:
            blockers.append("a grad_scaler")
        if self.skip_on_nonfinite:
            blockers.append("skip_on_nonfinite=True")
        if self.stat_interval > 1:
            blockers.append(f"stat_interval={self.stat_interval}")
        if self.diffusion:
            blockers.append("diffusion (its strength is decided from a host read)")
        if blockers:
            raise RuntimeError("a CUDA graph would freeze decisions V7 makes per step: "
                               + ", ".join(blockers) + ". Turn those off to capture a graph.")
        for _ in range(3):
            step_fn()
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._graph_result = step_fn()

    def replay_graph(self) -> Any:
        """Replay the captured step; returns what ``step_fn`` returned."""
        if self._graph is None:
            raise RuntimeError("call warmup_graph(step_fn) first")
        self._graph.replay()
        return self._graph_result

    def _kernel_name(self) -> str:
        if self._graph is not None:
            return "cuda-graph"
        if self._compiled is not None:
            return "compiled"
        return super()._kernel_name()

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["compiled"] = self._compiled is not None
        base["cuda_graph_captured"] = self._graph is not None
        base["fallback_reasons"] = list(self.fallback_reasons)
        return base


V7V = PressureCookerV7V
