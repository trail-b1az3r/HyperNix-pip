"""hypernix.optimizers.benchmark — measure the optimizers against each other.

    hnx optimizer benchmark                         # the quick set, a table
    hnx optimizer benchmark --workloads all --out-dir results/
    hnx optimizer benchmark --optimizers adamw,v6,v7,v7q --lrs 1e-3,3e-3,1e-2

Every optimizer runs the same thing: the same model from the same seed,
the same batches in the same order, the same precision, the same number of
steps. Nothing is tuned per optimizer except the learning rate, which is
swept over the same grid for all of them, because they do not share a
learning-rate scale and comparing them at one rate would only measure
which of them that rate suits.

Two kinds of measurement, kept apart:

**optimizer only**
    ``step()`` on fixed, pre-computed gradients: what the optimizer itself
    costs. Median milliseconds per step after warm-up, and the state it
    holds in bytes, counted from its tensors.

**full training step**
    forward, backward and ``step()`` together, and the share of that which
    is the optimizer. Whole-training speed is measured here; it is never
    inferred from the optimizer-only number.

**quality**, from the full runs: the final training loss, the loss on
held-out batches, the loss at a fixed step and at a fixed wall-clock time,
whether the run diverged, and how much the final loss moves across the
learning-rate grid (sensitivity).

What is not measured is said, not filled in: CPU and GPU utilisation,
kernel launches and synchronisation counts are reported as ``None``
(host reads per step are reported from each optimizer's own description
where it states one). An optimizer that cannot run here (V6V without
CUDA) is recorded as unavailable with the reason, and is not counted as a
loss for anybody.

``--ablations`` adds V7, V7S and V7M with one mechanism changed at a time,
so a difference can be put down to that mechanism. ``--band`` times
``step()`` alone at parameter counts around V7Mini's band (39M to 750M),
where a whole training run on a CPU is impractical; ``band-transformer``
(41.6M parameters) is a real training workload inside the band, run only
when named. Raw per-step times and the loss at every step are kept in the
JSON.

The numbers are this machine's. Compare runs from one machine only.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hypernix.optimizers.optimizer_framework_v2 import OPTIMIZERS, OptimizerFrameworkV2

__all__ = ["WORKLOADS", "OPT_IN_WORKLOADS", "ABLATIONS", "BAND_SIZES", "run_band",
           "DEFAULT_OPTIMIZERS", "run_benchmark", "render_report", "write_csv",
           "measured_table", "main"]

DEFAULT_OPTIMIZERS = ("adamw", "v5", "v5plus", "v5s", "v6", "v6v", "v7", "v7mini", "v7q",
                      "v7s", "v7m")
DEFAULT_LRS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)

#: One mechanism turned off (or on) at a time, so a difference can be put
#: down to that mechanism. name -> (optimizer, the options that differ).
ABLATIONS: dict[str, tuple[str, dict[str, Any]]] = {
    "v7-no-factored": ("v7", {"factored": False}),
    "v7-no-stability": ("v7", {"stability": False}),
    "v7-no-pressure": ("v7", {"adaptive_pressure": False}),
    "v7-no-diffusion": ("v7", {"diffusion": False}),
    "v7-int8": ("v7", {"state_dtype": "int8"}),
    "v7s-no-diffusion": ("v7s", {"diffusion": False}),
    "v7s-no-stability": ("v7s", {"stability": False}),
    "v7s-no-factored": ("v7s", {"factored": False}),
    "v7s-fp32": ("v7s", {"state_dtype": "fp32"}),
    "v7m-no-factored": ("v7m", {"factored": False}),
    "v7m-fp32": ("v7m", {"state_dtype": "fp32"}),
    "v7m-ema": ("v7m", {"ema_decay": 0.999}),
}


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

class _TinyLM(nn.Module):
    """A small causal transformer language model."""

    def __init__(self, vocab: int, dim: int, layers: int, heads: int, seq: int,
                 tie: bool = False) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(seq, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        if tie:
            self.head.weight = self.embed.weight
        self.register_buffer("mask", torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1),
                             persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids) + self.pos[: ids.shape[1]]
        mask: torch.Tensor = self.mask  # type: ignore[assignment]
        x = self.blocks(x, mask=mask[: ids.shape[1], : ids.shape[1]])
        return self.head(self.norm(x))


class _EmbeddingHeavy(nn.Module):
    """Most of the parameters in the embedding table: a bag-of-tokens classifier."""

    def __init__(self, vocab: int, dim: int, classes: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, classes))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.embed(ids).mean(dim=1))


class _Mixed(nn.Module):
    """Matrices, vectors, scalars-per-channel gains and a conv: a mix of shapes."""

    def __init__(self, dim: int, classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 8, 5, padding=2)
        self.gain = nn.Parameter(torch.ones(8))
        self.norm = nn.LayerNorm(dim)
        self.body = nn.Sequential(nn.Linear(8 * dim, dim), nn.GELU(), nn.Linear(dim, dim),
                                  nn.GELU())
        self.scale = nn.Parameter(torch.ones(dim))
        self.out = nn.Linear(dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x[:, None, :]) * self.gain[None, :, None]
        h = self.body(self.norm(h).flatten(1))
        return self.out(h * self.scale)


def _lm_data(vocab: int, seq: int, batch: int, batches: int, seed: int) -> list[tuple]:
    """Sequences with learnable structure: each token follows from the two before it."""
    g = torch.Generator().manual_seed(seed)
    table = torch.randint(0, vocab, (vocab, 8), generator=g)
    out = []
    for _ in range(batches):
        ids = torch.zeros(batch, seq + 1, dtype=torch.long)
        ids[:, :2] = torch.randint(0, vocab, (batch, 2), generator=g)
        for t in range(2, seq + 1):
            choice = (ids[:, t - 2] + t) % 8
            ids[:, t] = table[ids[:, t - 1], choice]
        out.append((ids[:, :-1], ids[:, 1:]))
    return out


def _regression_data(dim: int, classes: int, batch: int, batches: int, seed: int) -> list[tuple]:
    g = torch.Generator().manual_seed(seed)
    teacher = torch.randn(dim, classes, generator=g) / math.sqrt(dim)
    out = []
    for _ in range(batches):
        x = torch.randn(batch, dim, generator=g)
        out.append((x, (x @ teacher).tanh().argmax(dim=1)))
    return out


def _bag_data(vocab: int, seq: int, classes: int, batch: int, batches: int, seed: int
              ) -> list[tuple]:
    g = torch.Generator().manual_seed(seed)
    label_of = torch.randint(0, classes, (vocab,), generator=g)
    out = []
    for _ in range(batches):
        ids = torch.randint(0, vocab, (batch, seq), generator=g)
        votes = torch.zeros(batch, classes).scatter_add_(1, label_of[ids],
                                                         torch.ones(batch, seq))
        out.append((ids, votes.argmax(dim=1)))
    return out


def _lm_loss(model: nn.Module, batch: tuple) -> torch.Tensor:
    logits = model(batch[0])
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       batch[1].reshape(-1))


def _class_loss(model: nn.Module, batch: tuple) -> torch.Tensor:
    return nn.functional.cross_entropy(model(batch[0]), batch[1])


def _transformer(vocab: int, dim: int, layers: int, heads: int, seq: int, batch: int) -> dict:
    return {"build": lambda: _TinyLM(vocab, dim, layers, heads, seq),
            "data": lambda n, seed: _lm_data(vocab, seq, batch, n, seed),
            "loss": _lm_loss, "tokens_per_batch": seq * batch}


#: name -> how to build the model, its data and its loss.
WORKLOADS: dict[str, dict[str, Any]] = {
    "small-transformer": _transformer(128, 48, 2, 4, 24, 16),
    "medium-transformer": _transformer(256, 96, 3, 4, 32, 16),
    "large-transformer": _transformer(512, 192, 4, 6, 48, 16),
    "mlp": {"build": lambda: nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 256),
                                           nn.GELU(), nn.Linear(256, 16)),
            "data": lambda n, seed: _regression_data(64, 16, 64, n, seed),
            "loss": _class_loss, "tokens_per_batch": 64},
    "embedding-heavy": {"build": lambda: _EmbeddingHeavy(20_000, 64, 16),
                        "data": lambda n, seed: _bag_data(20_000, 24, 16, 32, n, seed),
                        "loss": _class_loss, "tokens_per_batch": 24 * 32},
    "mixed-1d-2d": {"build": lambda: _Mixed(64, 16),
                    "data": lambda n, seed: _regression_data(64, 16, 64, n, seed),
                    "loss": _class_loss, "tokens_per_batch": 64},
}
QUICK_WORKLOADS = ("small-transformer", "mlp", "mixed-1d-2d")
#: Named explicitly or not run: ``--workloads all`` leaves these out.
#: ``band-transformer`` has 41.6M parameters, inside V7Mini's band, and a
#: step of it takes a fraction of a second on a CPU.
OPT_IN_WORKLOADS = ("band-transformer",)
WORKLOADS["band-transformer"] = _transformer(16_000, 512, 8, 8, 64, 4)


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

def _base(name: str) -> tuple[str, dict[str, Any]]:
    """The registry key behind *name*, and the options an ablation adds."""
    if name in ABLATIONS:
        return ABLATIONS[name]
    return OptimizerFrameworkV2.spec(name).key, {}


def _label(name: str) -> str:
    key, options = _base(name)
    if not options:
        return OPTIMIZERS[key].label
    return OPTIMIZERS[key].label + " with " + ", ".join(f"{k}={v}" for k, v in options.items())


def _make(key: str, params: Any, lr: float) -> torch.optim.Optimizer:
    base, options = _base(key)
    return OptimizerFrameworkV2.create(base, params, lr=lr, **options)


def _checkpoint_cost(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """What saving and loading this optimizer's state costs."""
    import io

    buffer = io.BytesIO()
    started = time.perf_counter()
    torch.save(optimizer.state_dict(), buffer)
    saved = time.perf_counter()
    buffer.seek(0)
    optimizer.load_state_dict(torch.load(buffer, weights_only=False))
    return {"bytes": buffer.getbuffer().nbytes,
            "save_ms": round(1e3 * (saved - started), 3),
            "load_ms": round(1e3 * (time.perf_counter() - saved), 3)}


def _spread(times: list[float]) -> dict[str, Any]:
    ordered = sorted(times)
    return {"ms_per_step": round(1e3 * statistics.median(ordered), 4),
            "ms_per_step_mean": round(1e3 * statistics.fmean(ordered), 4),
            "ms_per_step_p10": round(1e3 * ordered[int(0.1 * (len(ordered) - 1))], 4),
            "ms_per_step_p90": round(1e3 * ordered[int(0.9 * (len(ordered) - 1))], 4),
            "ms_raw": [round(1e3 * t, 4) for t in times]}


def _unavailable(key: str, device: str) -> str:
    spec = OPTIMIZERS[_base(key)[0]]
    if spec.requires_cuda and not device.startswith("cuda"):
        return f"{spec.label} needs CUDA; this run is on {device}"
    return ""


def _state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return int(OptimizerFrameworkV2.diagnostics(optimizer)["state_bytes"])


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def bench_optimizer_only(key: str, workload: dict[str, Any], *, steps: int, warmup: int,
                         seed: int, device: str, lr: float) -> dict[str, Any]:
    """``step()`` alone, on gradients computed once and replayed."""
    torch.manual_seed(seed)
    model = workload["build"]().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    batches = [tuple(t.to(device) for t in b) for b in workload["data"](4, seed + 1)]
    grads = []
    for batch in batches:                      # real gradients, of the real shapes
        model.zero_grad(set_to_none=True)
        workload["loss"](model, batch).backward()
        grads.append([p.grad.detach().clone() if p.grad is not None else None for p in params])
    optimizer = _make(key, params, lr)
    times = []
    for i in range(steps + warmup):
        for p, g in zip(params, grads[i % len(grads)], strict=True):
            p.grad = None if g is None else g.clone()
        _sync(device)
        started = time.perf_counter()
        optimizer.step()
        _sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - started)
    n_params = sum(p.numel() for p in params)
    state = _state_bytes(optimizer)
    described = optimizer.describe() if hasattr(optimizer, "describe") else {}
    try:
        checkpoint: dict[str, Any] | None = _checkpoint_cost(optimizer)
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal to the timing
        checkpoint = {"error": f"{type(exc).__name__}: {exc}"}
    return {**_spread(times), "checkpoint": checkpoint,
            "state_bytes": state, "parameters": n_params,
            "state_bytes_per_parameter": round(state / max(1, n_params), 4),
            "kernel": described.get("kernel"), "state_dtype": described.get("state_dtype"),
            "host_reads_per_step": described.get("host_reads_per_step")}


def bench_training(key: str, workload: dict[str, Any], *, steps: int, seed: int, device: str,
                   lr: float, time_budget: float | None = None) -> dict[str, Any]:
    """A whole training run: forward, backward and step, every step timed."""
    torch.manual_seed(seed)
    model = workload["build"]().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # One draw, split: the held-out batches come from the same task (the same
    # teacher, the same transition table) and are never trained on.
    n_train = max(8, steps // 4)
    drawn = [tuple(t.to(device) for t in b) for b in workload["data"](n_train + 4, seed + 1)]
    train, held_out = drawn[:n_train], drawn[n_train:]
    optimizer = _make(key, params, lr)
    losses: list[float] = []
    step_seconds: list[float] = []
    opt_seconds: list[float] = []
    grad_norms: list[float] = []
    loss_at_time: float | None = None
    elapsed = 0.0
    diverged = False
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for i in range(steps):
        batch = train[i % len(train)]
        _sync(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = workload["loss"](model, batch)
        loss.backward()
        _sync(device)
        backward_done = time.perf_counter()
        # Outside the timed regions: the size of the gradient this step.
        grad_norms.append(float(torch.stack(
            [p.grad.detach().float().norm() for p in params if p.grad is not None]).norm()))
        _sync(device)
        before = time.perf_counter()
        optimizer.step()
        _sync(device)
        ended = time.perf_counter()
        value = loss.item()
        if hasattr(optimizer, "report_loss"):
            optimizer.report_loss(value)
        step_seconds.append((backward_done - started) + (ended - before))
        opt_seconds.append(ended - before)
        elapsed += step_seconds[-1]
        losses.append(value)
        if time_budget is not None and loss_at_time is None and elapsed >= time_budget:
            loss_at_time = value
        if not math.isfinite(value):
            diverged = True
            break
    model.eval()
    with torch.no_grad():
        validation = statistics.fmean(float(workload["loss"](model, b)) for b in held_out)
    tail = losses[-max(1, len(losses) // 10):]
    skip = min(len(step_seconds) - 1, max(1, len(step_seconds) // 10))   # leave warm-up out
    ms = 1e3 * statistics.median(step_seconds[skip:] or step_seconds)
    opt_ms = 1e3 * statistics.median(opt_seconds[skip:] or opt_seconds)
    fixed = min(len(losses), max(1, steps // 2)) - 1
    finite_norms = [g for g in grad_norms if math.isfinite(g)]
    return {
        "lr": lr, "steps": len(losses), "diverged": diverged or not math.isfinite(validation),
        "final_loss": round(statistics.fmean(tail), 6) if not diverged else None,
        "validation_loss": round(validation, 6) if math.isfinite(validation) else None,
        "loss_at_step": {"step": fixed + 1, "loss": round(losses[fixed], 6)},
        "loss_at_seconds": None if time_budget is None else
        {"seconds": time_budget, "loss": None if loss_at_time is None else round(loss_at_time, 6)},
        "ms_per_step": round(ms, 4), "steps_per_second": round(1e3 / ms, 3) if ms else None,
        "tokens_per_second": round(workload["tokens_per_batch"] * 1e3 / ms, 1) if ms else None,
        "optimizer_ms_per_step": round(opt_ms, 4),
        "optimizer_share": round(opt_ms / ms, 4) if ms else None,
        "ms_per_step_p10": round(1e3 * sorted(step_seconds)[int(0.1 * (len(step_seconds) - 1))],
                                 4),
        "ms_per_step_p90": round(1e3 * sorted(step_seconds)[int(0.9 * (len(step_seconds) - 1))],
                                 4),
        "losses": [round(v, 6) for v in losses],
        # Gradient stability: the typical gradient norm, and how far the
        # largest one stood above it (1.0 is perfectly even).
        "grad_norm_median": round(statistics.median(finite_norms), 6) if finite_norms else None,
        "grad_norm_peak_ratio": round(max(finite_norms) / max(statistics.median(finite_norms),
                                                              1e-30), 3)
        if finite_norms else None,
        "state_bytes": _state_bytes(optimizer),
        "peak_device_bytes": int(torch.cuda.max_memory_allocated())
        if device.startswith("cuda") else None,
    }


def _peak_rss() -> int:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        return 0


def run_benchmark(*, optimizers: tuple[str, ...] = DEFAULT_OPTIMIZERS,
                  workloads: tuple[str, ...] = QUICK_WORKLOADS,
                  lrs: tuple[float, ...] = DEFAULT_LRS, steps: int = 120,
                  optimizer_steps: int = 30, warmup: int = 5, seed: int = 0,
                  device: str = "cpu", progress: Any = None) -> dict[str, Any]:
    """Run the comparison and return the record."""
    from hypernix import __version__

    for name in workloads:
        if name not in WORKLOADS:
            raise ValueError(f"unknown workload {name!r}. Known: {', '.join(WORKLOADS)}")
    keys = tuple(name if name in ABLATIONS else OptimizerFrameworkV2.spec(name).key
                 for name in optimizers)
    record: dict[str, Any] = {
        "hypernix_version": __version__, "torch": torch.__version__,
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": {"os": f"{platform.system()} {platform.release()}",
                    "cpu": platform.processor() or platform.machine(),
                    "threads": torch.get_num_threads(), "python": platform.python_version(),
                    "device": device,
                    "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None},
        "settings": {"optimizers": list(keys), "workloads": list(workloads), "lrs": list(lrs),
                     "steps": steps, "optimizer_steps": optimizer_steps, "warmup": warmup,
                     "seed": seed, "precision": "fp32"},
        "not_measured": ["cpu_utilization", "gpu_utilization", "kernel_launches",
                         "host_device_synchronization_count"],
        "workloads": {},
    }
    for name in workloads:
        workload = WORKLOADS[name]
        torch.manual_seed(seed)
        n_params = sum(p.numel() for p in workload["build"]().parameters())
        # The wall-clock checkpoint: a third of what AdamW takes for the run.
        budget = None
        entry: dict[str, Any] = {"parameters": n_params, "optimizers": {}}
        for key in keys:
            if progress is not None:
                progress(f"{name}: {key}")
            why = _unavailable(key, device)
            if why:
                entry["optimizers"][key] = {"available": False, "reason": why}
                continue
            try:
                only = bench_optimizer_only(key, workload, steps=optimizer_steps, warmup=warmup,
                                            seed=seed, device=device, lr=lrs[len(lrs) // 2])
                runs = []
                for lr in lrs:
                    run = bench_training(key, workload, steps=steps, seed=seed, device=device,
                                         lr=lr, time_budget=budget)
                    runs.append(run)
                    if budget is None and key == keys[0]:
                        budget = round(run["ms_per_step"] * steps / 3e3, 4)
            except Exception as exc:  # noqa: BLE001 - one optimizer failing is a result
                entry["optimizers"][key] = {"available": False,
                                            "reason": f"failed: {type(exc).__name__}: {exc}"}
                continue
            good = [r for r in runs if not r["diverged"] and r["final_loss"] is not None]
            best = min(good, key=lambda r: r["final_loss"]) if good else None
            finals = [r["final_loss"] for r in good]
            entry["optimizers"][key] = {
                "available": True, "label": _label(key),
                "optimizer_only": only, "runs": runs,
                "best": best,
                "diverged_at": [r["lr"] for r in runs if r["diverged"]],
                # Its best rate is the largest or smallest tried: the real
                # best may lie outside the grid, and this row understates it.
                "best_at_grid_edge": bool(best is not None and len(lrs) > 1
                                          and best["lr"] in (min(lrs), max(lrs))),
                # How far the final loss moves across the grid: max / min.
                "lr_sensitivity": round(max(finals) / max(min(finals), 1e-12), 3)
                if len(finals) > 1 else None,
            }
        entry["wall_clock_checkpoint_seconds"] = budget
        record["workloads"][name] = entry
    record["peak_rss_bytes"] = _peak_rss()
    return record


# ---------------------------------------------------------------------------
# V7Mini's band: optimizer-only, at the sizes that matter
# ---------------------------------------------------------------------------

#: Below the band, exactly at its lower edge, just inside it, and well inside
#: it. The upper edge (749M, 750M) needs about 10 GiB and is run by naming it.
BAND_SIZES = (30_000_000, 39_000_000, 39_100_000, 150_000_000)
BAND_OPTIMIZERS = ("adamw", "v6", "v7", "v7mini", "v7q")


def _band_shapes(target: int) -> list[tuple[int, ...]]:
    """Transformer-shaped tensors that add up to exactly *target* parameters."""
    dim = 512 if target < 60_000_000 else 768 if target < 300_000_000 else 1024
    vocab = 16_000 if target < 60_000_000 else 32_000
    per_layer = 4 * dim * dim + 2 * 4 * dim * dim + 4 * dim + 4 * dim + 4 * dim
    layers = max(1, (target - vocab * dim) // per_layer)
    shapes: list[tuple[int, ...]] = [(vocab, dim)]
    for _ in range(layers):
        shapes += [(dim, dim)] * 4 + [(4 * dim, dim), (dim, 4 * dim)]
        shapes += [(dim,)] * 4 + [(4 * dim,), (dim,), (dim,)]
    left = target - sum(math.prod(shape) for shape in shapes)
    while left < 0:                              # one layer too many
        del shapes[-13:]
        left = target - sum(math.prod(shape) for shape in shapes)
    if left >= dim:
        shapes.append((left // dim, dim))
        left -= (left // dim) * dim
    if left:
        shapes.append((left,))
    return shapes


def _available_bytes() -> int | None:
    """Memory a new allocation can have: Linux's ``MemAvailable``, which
    counts reclaimable cache, or else free pages; ``None`` if unknown."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import os

        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def run_band(*, sizes: tuple[int, ...] = BAND_SIZES,
             optimizers: tuple[str, ...] = BAND_OPTIMIZERS, steps: int = 6, warmup: int = 2,
             seed: int = 0, device: str = "cpu", progress: Any = None) -> dict[str, Any]:
    """``step()`` alone on models sized around V7Mini's band.

    The tensors have a transformer's shapes and exactly the parameter count
    asked for; the gradients are random and redrawn each step (outside the
    timed region), so this measures the optimizer's cost at that size and
    nothing about training quality.
    """
    from hypernix.optimizers.pressure_cooker_v7 import MINI_MAX_PARAMS, MINI_MIN_PARAMS

    out: dict[str, Any] = {"lower": MINI_MIN_PARAMS, "upper": MINI_MAX_PARAMS, "steps": steps,
                           "warmup": warmup, "gradients": "random, redrawn each step",
                           "sizes": {}}
    for size in sizes:
        chosen = OptimizerFrameworkV2.auto_select(parameter_count=size,
                                                  parameter_count_verified=True)
        entry: dict[str, Any] = {"in_band": MINI_MIN_PARAMS < size < MINI_MAX_PARAMS,
                                 "auto_select": chosen.key, "optimizers": {}}
        for name in optimizers:
            if progress is not None:
                progress(f"band {size:,}: {name}")
            why = _unavailable(name, device)
            # Parameters and gradients, plus the state: two fp32 copies for
            # AdamW, at most one for the others here; a tenth over for slack.
            copies = 4 if _base(name)[0] == "adamw" else 3
            need = int(size * 4 * copies * 1.1)
            free = _available_bytes()
            if not why and free is not None and need > free:
                why = f"needs about {need / 2**30:.1f} GiB; {free / 2**30:.1f} GiB is free"
            if why:
                entry["optimizers"][name] = {"available": False, "reason": why}
                continue
            torch.manual_seed(seed)
            params = [torch.nn.Parameter(torch.randn(shape, device=device) * 0.02)
                      for shape in _band_shapes(size)]
            for p in params:
                p.grad = torch.empty_like(p)
            optimizer: torch.optim.Optimizer | None = None
            try:
                optimizer = _make(name, params, 1e-3)
                times = []
                for i in range(steps + warmup):
                    for p in params:
                        p.grad.normal_(0.0, 0.01)      # type: ignore[union-attr]
                    _sync(device)
                    started = time.perf_counter()
                    optimizer.step()
                    _sync(device)
                    if i >= warmup:
                        times.append(time.perf_counter() - started)
                state = _state_bytes(optimizer)
                described = optimizer.describe() if hasattr(optimizer, "describe") else {}
                entry["optimizers"][name] = {
                    "available": True, **_spread(times), "state_bytes": state,
                    "state_bytes_per_parameter": round(state / size, 4),
                    "state_dtype": described.get("state_dtype"),
                    "state_dtype_reason": described.get("state_dtype_reason")}
            except Exception as exc:  # noqa: BLE001 - one optimizer failing is a result
                entry["optimizers"][name] = {"available": False,
                                             "reason": f"failed: {type(exc).__name__}: {exc}"}
            # Free this size's tensors before the next optimizer measures
            # what memory is left: the optimizer holds the parameters and
            # its state, so it has to go too.
            optimizer = None
            del params
            gc.collect()
        entry["parameters"] = size
        out["sizes"][str(size)] = entry
    return out


def measured_table(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The record averaged over its workloads, in the shape
    :meth:`OptimizerFrameworkV2.table` and ``auto_select`` take."""
    out: dict[str, dict[str, Any]] = {}
    for key in record["settings"]["optimizers"]:
        only, full, state, loss, final = [], [], [], [], []
        for entry in record["workloads"].values():
            got = entry["optimizers"].get(key, {})
            if not got.get("available") or not got.get("best"):
                continue
            only.append(got["optimizer_only"]["ms_per_step"])
            full.append(got["best"]["ms_per_step"])
            state.append(got["optimizer_only"]["state_bytes_per_parameter"])
            loss.append(got["best"]["validation_loss"])
            final.append(got["best"]["final_loss"])
        if only:
            out[key] = {"ms_per_step": round(statistics.fmean(only), 4),
                        "full_step_ms": round(statistics.fmean(full), 4),
                        "state_bytes_per_parameter": round(statistics.fmean(state), 4),
                        "final_loss": round(statistics.fmean(final), 6),
                        "validation_loss": round(statistics.fmean(v for v in loss
                                                                  if v is not None), 6)
                        if any(v is not None for v in loss) else None,
                        "workloads": len(only), "device": record["machine"]["device"]}
    return out


def write_csv(record: dict[str, Any], path: Path) -> None:
    import csv

    columns = ("workload", "parameters", "optimizer", "available", "reason",
               "optimizer_only_ms", "state_bytes_per_parameter", "best_lr", "final_loss",
               "validation_loss", "full_step_ms", "optimizer_share", "tokens_per_second",
               "lr_sensitivity", "diverged_at")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, entry in record["workloads"].items():
            for key, got in entry["optimizers"].items():
                row: dict[str, Any] = {"workload": name, "parameters": entry["parameters"],
                                       "optimizer": key, "available": got.get("available"),
                                       "reason": got.get("reason", "")}
                if got.get("available"):
                    best = got.get("best") or {}
                    row.update({
                        "optimizer_only_ms": got["optimizer_only"]["ms_per_step"],
                        "state_bytes_per_parameter":
                            got["optimizer_only"]["state_bytes_per_parameter"],
                        "best_lr": best.get("lr"), "final_loss": best.get("final_loss"),
                        "validation_loss": best.get("validation_loss"),
                        "full_step_ms": best.get("ms_per_step"),
                        "optimizer_share": best.get("optimizer_share"),
                        "tokens_per_second": best.get("tokens_per_second"),
                        "lr_sensitivity": got.get("lr_sensitivity"),
                        "diverged_at": " ".join(str(v) for v in got.get("diverged_at", []))})
                writer.writerow(row)


def render_report(record: dict[str, Any]) -> str:
    machine, settings = record["machine"], record["settings"]
    lines = ["# Optimizer benchmark", "",
             f"- When: {record['when']}; HyperNix {record['hypernix_version']}, "
             f"torch {record['torch']}",
             f"- Machine: {machine['os']}, {machine['cpu']}, {machine['threads']} threads, "
             f"device {machine['device']}" + (f" ({machine['gpu']})" if machine["gpu"] else ""),
             f"- Each optimizer: {settings['steps']} training steps at each of "
             f"{', '.join(f'{lr:g}' for lr in settings['lrs'])}; {settings['optimizer_steps']} "
             f"optimizer-only steps after {settings['warmup']} warm-up; seed "
             f"{settings['seed']}; {settings['precision']}",
             "- Not measured: " + ", ".join(record["not_measured"]).replace("_", " "), ""]
    for name, entry in record["workloads"].items():
        lines += [f"## {name} ({entry['parameters']:,} parameters)", "",
                  "| Optimizer | Step only, ms | State B/param | Best lr | Final loss | "
                  "Held-out loss | Full step, ms | Optimizer share | lr sensitivity |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key, got in entry["optimizers"].items():
            if not got.get("available"):
                lines.append(f"| {key} | N/A: {got.get('reason', '')} | | | | | | | |")
                continue
            only, best = got["optimizer_only"], got.get("best")
            if best is None:
                lines.append(f"| {key} | {only['ms_per_step']:.3f} | "
                             f"{only['state_bytes_per_parameter']:.2f} | diverged at every "
                             "rate | | | | | |")
                continue
            lines.append(
                f"| {key} | {only['ms_per_step']:.3f} | {only['state_bytes_per_parameter']:.2f} "
                f"| {best['lr']:g} | {best['final_loss']:.4f} | {best['validation_loss']:.4f} | "
                f"{best['ms_per_step']:.3f} | {100 * best['optimizer_share']:.0f}% | "
                f"{got['lr_sensitivity'] if got['lr_sensitivity'] is not None else ''}"
                + (f" (diverged at {', '.join(f'{v:g}' for v in got['diverged_at'])})"
                   if got["diverged_at"] else "") + " |")
        lines.append("")
    if record.get("band"):
        band = record["band"]
        lines += [f"## V7Mini's band ({band['lower']:,} < n < {band['upper']:,})", "",
                  f"`step()` alone, median ms over {band['steps']} steps, on tensors with a "
                  "transformer's shapes and random gradients. This is the optimizer's cost "
                  "at that size; it says nothing about training quality.", "",
                  *_band_lines(band), ""]
    edge = sorted({key for entry in record["workloads"].values()
                   for key, got in entry["optimizers"].items() if got.get("best_at_grid_edge")})
    if edge:
        lines += ["**Best rate at the edge of the grid** on at least one workload: "
                  + ", ".join(edge) + ". Their real best may lie outside the grid, so those "
                  "rows may understate them.", ""]
    lines += ["`Step only` is the optimizer's `step()` on fixed gradients. `Full step` is "
              "forward, backward and step. `lr sensitivity` is the largest final loss over "
              "the smallest across the learning-rate grid: closer to 1 is less sensitive.",
              "", "Every figure was measured in this run, on this machine."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table(record: dict[str, Any]) -> None:
    for name, entry in record["workloads"].items():
        print(f"\n{name} ({entry['parameters']:,} parameters)")
        print(f"  {'optimizer':<17} {'step ms':>9} {'B/param':>8} {'best lr':>8} {'final':>9} "
              f"{'held-out':>9} {'full ms':>9} {'opt %':>6}")
        for key, got in entry["optimizers"].items():
            if not got.get("available"):
                print(f"  {key:<17} N/A: {got.get('reason', '')}")
                continue
            only, best = got["optimizer_only"], got.get("best")
            if best is None:
                print(f"  {key:<17} {only['ms_per_step']:>9.3f} "
                      f"{only['state_bytes_per_parameter']:>8.2f}  diverged at every rate")
                continue
            print(f"  {key:<17} {only['ms_per_step']:>9.3f} "
                  f"{only['state_bytes_per_parameter']:>8.2f} {best['lr']:>8g} "
                  f"{best['final_loss']:>9.4f} {best['validation_loss']:>9.4f} "
                  f"{best['ms_per_step']:>9.3f} {100 * best['optimizer_share']:>5.0f}%")


def _band_lines(band: dict[str, Any]) -> list[str]:
    names = list(next(iter(band["sizes"].values()))["optimizers"]) if band["sizes"] else []
    lines = ["| Parameters | In band | Auto-selects | "
             + " | ".join(f"{n}, ms" for n in names) + " |",
             "|---|---|---|" + "---|" * len(names)]
    for entry in band["sizes"].values():
        cells = []
        for name in names:
            got = entry["optimizers"][name]
            if not got.get("available"):
                cells.append("N/A")
                continue
            dtype = f" ({got['state_dtype']})" if got.get("state_dtype") not in (None, "fp32") \
                else ""
            cells.append(f"{got['ms_per_step']:.1f}{dtype}")
        lines.append(f"| {entry['parameters']:,} | {'yes' if entry['in_band'] else 'no'} | "
                     f"{entry['auto_select']} | " + " | ".join(cells) + " |")
    reasons = sorted({f"{entry['parameters']:,} {name}: {got.get('reason', '')}"
                      for entry in band["sizes"].values()
                      for name, got in entry["optimizers"].items() if not got.get("available")})
    if reasons:
        lines += ["", "N/A: " + "; ".join(reasons)]
    return lines


def _print_band(band: dict[str, Any]) -> None:
    print(f"\nstep() alone around V7Mini's band ({band['lower']:,} < n < {band['upper']:,}), "
          "random gradients")
    for line in _band_lines(band):
        print("  " + line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hnx optimizer benchmark",
        description="Compare the optimizers on the same models, data and steps. Measures "
                    "the optimizer alone and the whole training step, state memory and "
                    "training quality across a learning-rate grid.")
    parser.add_argument("--optimizers", default=",".join(DEFAULT_OPTIMIZERS),
                        help="comma-separated keys (default: %(default)s)")
    parser.add_argument("--workloads", default=",".join(QUICK_WORKLOADS),
                        help=f"comma-separated, or 'all'. Known: {', '.join(WORKLOADS)}")
    parser.add_argument("--lrs", default=",".join(f"{lr:g}" for lr in DEFAULT_LRS),
                        help="the learning-rate grid every optimizer is swept over")
    parser.add_argument("--steps", type=int, default=120, help="training steps per run")
    parser.add_argument("--optimizer-steps", type=int, default=30,
                        help="measured optimizer-only steps")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ablations", action="store_true",
                        help="also run V7, V7S and V7M with one mechanism changed at a time: "
                             + ", ".join(ABLATIONS))
    parser.add_argument("--band", nargs="?", const=",".join(str(v) for v in BAND_SIZES),
                        default=None, metavar="SIZES",
                        help="also time step() alone at these parameter counts around "
                             "V7Mini's band (default: %(const)s)")
    parser.add_argument("--json", metavar="OUT", default=None)
    parser.add_argument("--csv", metavar="OUT", default=None)
    parser.add_argument("--report", metavar="OUT", default=None)
    parser.add_argument("--out-dir", metavar="DIR", default=None,
                        help="write optimizer-bench.json, .csv and .md there")
    args = parser.parse_args(argv)
    if args.steps < 4 or args.optimizer_steps < 1:
        parser.error("--steps must be at least 4 and --optimizer-steps at least 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("optimizer benchmark: there is no CUDA device here", file=sys.stderr)
        return 2
    workloads = tuple(w for w in WORKLOADS if w not in OPT_IN_WORKLOADS) \
        if args.workloads == "all" else tuple(
        w.strip() for w in args.workloads.split(",") if w.strip())
    sizes: tuple[int, ...] = ()
    if args.band:
        try:
            sizes = tuple(int(v.replace("_", "")) for v in args.band.split(",") if v.strip())
        except ValueError:
            print("optimizer benchmark: --band takes parameter counts, e.g. "
                  "30000000,39100000", file=sys.stderr)
            return 2
    chosen = tuple(o.strip() for o in args.optimizers.split(",") if o.strip())
    if args.ablations:
        chosen += tuple(name for name in ABLATIONS if name not in chosen)
    try:
        record = run_benchmark(
            optimizers=chosen,
            workloads=workloads, lrs=tuple(float(v) for v in args.lrs.split(",")),
            steps=args.steps, optimizer_steps=args.optimizer_steps, warmup=args.warmup,
            seed=args.seed, device=args.device,
            progress=lambda text: print(text, file=sys.stderr, flush=True))
    except ValueError as exc:
        print(f"optimizer benchmark: {exc}", file=sys.stderr)
        return 2
    record["measured"] = measured_table(record)
    _print_table(record)
    if sizes:
        record["band"] = run_band(
            sizes=sizes, seed=args.seed, device=args.device,
            progress=lambda text: print(text, file=sys.stderr, flush=True))
        _print_band(record["band"])
    outputs = {"json": args.json, "csv": args.csv, "report": args.report}
    if args.out_dir:
        folder = Path(args.out_dir)
        folder.mkdir(parents=True, exist_ok=True)
        outputs = {"json": outputs["json"] or str(folder / "optimizer-bench.json"),
                   "csv": outputs["csv"] or str(folder / "optimizer-bench.csv"),
                   "report": outputs["report"] or str(folder / "optimizer-bench.md")}
    if outputs["json"]:
        Path(outputs["json"]).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if outputs["csv"]:
        write_csv(record, Path(outputs["csv"]))
    if outputs["report"]:
        Path(outputs["report"]).write_text(render_report(record), encoding="utf-8")
    for kind, where in outputs.items():
        if where:
            print(f"{kind}: {where}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
