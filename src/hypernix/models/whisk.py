"""whisk — combine multiple checkpoints into one.

A whisk blends.  In ML terms: take N saved snapshots / state dicts
and produce a single merged set of weights.  Two modes ship by
default; both work in place on plain ``dict[str, Tensor]`` so they
compose with anything.

* :func:`swa_average`       — uniform Stochastic Weight Average.
                              Mean across all N inputs.
* :func:`ema`               — Exponential Moving Average.  Later
                              entries weighted more heavily by
                              ``decay ** (N - 1 - i)``.
* :func:`geometric_mean`    — element-wise geometric mean (rare,
                              useful for log-scale tensors like
                              attention biases).

All three accept either a list of state dicts or a list of paths
(``.pt`` / ``.safetensors``).  Mismatched keys are ignored with a
warning unless ``strict=True``.

One-shot helper:

    from hypernix.whisk import whisk
    merged = whisk(["ckpt-1.pt", "ckpt-2.pt", "ckpt-3.pt"], mode="swa")

The output is a state dict you can hand to ``model.load_state_dict``
or feed to :func:`hypernix.save_snapshot` to write a full HF-style
snapshot directory.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch


def _coerce(item: Any) -> dict[str, torch.Tensor]:
    """Accept a state dict, a path to ``.pt`` / ``.safetensors``, or
    anything torch.load can read.  Returns the state dict on CPU."""
    if isinstance(item, dict):
        return item
    p = Path(item)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(p))
    return torch.load(p, map_location="cpu", weights_only=True)


def _intersect_keys(states: list[dict[str, torch.Tensor]], strict: bool) -> list[str]:
    keys = set(states[0])
    for s in states[1:]:
        if set(s) != keys:
            if strict:
                raise ValueError(
                    "state dicts have mismatched keys; use strict=False to "
                    "intersect (drops keys that aren't in every input).",
                )
            keys &= set(s)
    return [k for k in states[0] if k in keys]


# ---------------------------------------------------------------------------
# Public modes
# ---------------------------------------------------------------------------

def swa_average(
    items: Iterable[dict | str | Path],
    *,
    strict: bool = False,
) -> dict[str, torch.Tensor]:
    """Uniform mean across all inputs.  ``strict=False`` ignores keys
    not present in every input."""
    states = [_coerce(it) for it in items]
    if not states:
        raise ValueError("whisk: need at least one input")
    keys = _intersect_keys(states, strict)
    n = len(states)
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        ref = states[0][k]
        if not torch.is_floating_point(ref):
            # Integer tensors (e.g. token id buffers) are taken from
            # the first checkpoint; averaging them is meaningless.
            out[k] = ref.clone()
            continue
        acc = ref.clone().float()
        for s in states[1:]:
            acc.add_(s[k].float())
        acc.div_(n)
        out[k] = acc.to(ref.dtype)
    return out


def ema(
    items: Iterable[dict | str | Path],
    *,
    decay: float = 0.99,
    strict: bool = False,
) -> dict[str, torch.Tensor]:
    """Exponential moving average.  Earlier inputs are weighted by
    ``decay ** (N - 1 - i)``; the last input gets weight 1.

    ``decay`` in [0, 1).  decay=0 returns the last input unchanged;
    decay close to 1 approaches a uniform mean.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError("ema decay must be in [0, 1)")
    states = [_coerce(it) for it in items]
    if not states:
        raise ValueError("whisk: need at least one input")
    keys = _intersect_keys(states, strict)
    n = len(states)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    weight_sum = sum(weights)
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        ref = states[0][k]
        if not torch.is_floating_point(ref):
            out[k] = states[-1][k].clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for s, w in zip(states, weights, strict=False):
            acc.add_(s[k].float(), alpha=w)
        acc.div_(weight_sum)
        out[k] = acc.to(ref.dtype)
    return out


def geometric_mean(
    items: Iterable[dict | str | Path],
    *,
    strict: bool = False,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Element-wise geometric mean.  Negative or zero values are
    clamped to ``eps`` so the log is finite — use only on tensors
    you know are positive."""
    states = [_coerce(it) for it in items]
    if not states:
        raise ValueError("whisk: need at least one input")
    keys = _intersect_keys(states, strict)
    n = len(states)
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        ref = states[0][k]
        if not torch.is_floating_point(ref):
            out[k] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for s in states:
            t = s[k].float().clamp_min(eps)
            acc.add_(t.log())
        acc.div_(n).exp_()
        out[k] = acc.to(ref.dtype)
    return out


# ---------------------------------------------------------------------------
# One-shot helper
# ---------------------------------------------------------------------------

def whisk(
    items: Iterable[dict | str | Path],
    *,
    mode: str = "swa",
    decay: float = 0.99,
    strict: bool = False,
) -> dict[str, torch.Tensor]:
    """Pick a whisking mode by name."""
    mode = mode.lower().replace("_", "-")
    if mode in ("swa", "average", "mean"):
        return swa_average(items, strict=strict)
    if mode == "ema":
        return ema(items, decay=decay, strict=strict)
    if mode in ("geometric-mean", "geo-mean"):
        return geometric_mean(items, strict=strict)
    raise ValueError(
        f"unknown whisk mode {mode!r}; valid: 'swa', 'ema', 'geometric-mean'",
    )


def whisk_to_snapshot(
    items: Iterable[dict | str | Path],
    out_dir: Path | str,
    tokenizer_source: Path | str | None = None,
    *,
    mode: str = "swa",
    decay: float = 0.99,
    strict: bool = False,
) -> Path:
    """Whisk + write a full snapshot directory in one call.

    ``out_dir`` should be empty or a fresh path.  When
    ``tokenizer_source`` is given, the tokenizer files from that
    snapshot are copied alongside the merged weights.
    """
    from hypernix.training.train import HyperNixModel, save_snapshot

    state = whisk(items, mode=mode, decay=decay, strict=strict)

    # Try to recover the config from the first input — for paths this
    # means looking for a sibling config.json.
    cfg = _try_load_config(items)
    if cfg is None:
        warnings.warn(
            "whisk_to_snapshot: no config.json found alongside any "
            "input; you'll need to write one yourself.",
            stacklevel=2,
        )
        # Save a bare safetensors file so the caller can finish manually.
        from safetensors.torch import save_file
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_file(state, str(out / "model.safetensors"))
        return out

    model = HyperNixModel(cfg)
    model.load_state_dict(state, strict=strict)
    return save_snapshot(model, out_dir, tokenizer_source=tokenizer_source)


def _try_load_config(items):
    """Best-effort config recovery — looks for ``config.json`` next to
    each path-shaped input.  Returns the first valid config found."""
    import json

    from hypernix.training.train import HyperNixConfig

    for it in items:
        if isinstance(it, dict):
            continue
        p = Path(it)
        candidates = [p.parent / "config.json", p.with_suffix(".json")]
        for c in candidates:
            if c.exists():
                try:
                    return HyperNixConfig.from_dict(json.loads(c.read_text(encoding="utf-8")))
                except Exception:  # noqa: BLE001
                    continue
    return None


MODES: tuple[str, ...] = ("swa", "ema", "geometric-mean")
