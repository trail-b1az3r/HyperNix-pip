"""strife.sampler — the next token from the logits.

Greedy when ``temperature`` is 0. Otherwise, in llama.cpp's order:
repetition penalty, top-k, temperature, top-p, min-p, then a draw from a
:class:`numpy.random.Generator` seeded per request. The same seed, model
and prompt give the same tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import StrifeError

__all__ = ["SamplerConfig", "Sampler"]


@dataclass(frozen=True)
class SamplerConfig:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    seed: int | None = None

    def validate(self) -> SamplerConfig:
        if not 0 <= self.temperature <= 10:
            raise StrifeError("strife.config.bad_sampling", "temperature must be 0-10")
        if not 0 < self.top_p <= 1 or not 0 <= self.min_p < 1 or self.top_k < 0:
            raise StrifeError("strife.config.bad_sampling", "top_p must be in (0, 1], "
                              "min_p in [0, 1) and top_k at least 0")
        if self.repeat_penalty <= 0:
            raise StrifeError("strife.config.bad_sampling", "repeat_penalty must be positive")
        return self


class Sampler:
    def __init__(self, config: SamplerConfig) -> None:
        self.config = config.validate()
        self.rng = np.random.default_rng(config.seed)

    def __call__(self, logits: np.ndarray, history: list[int]) -> int:
        c = self.config
        logits = np.asarray(logits, dtype=np.float64).copy()
        if c.repeat_penalty != 1.0 and history:
            recent = np.unique(np.asarray(history[-c.repeat_last_n:], dtype=np.int64))
            recent = recent[(recent >= 0) & (recent < logits.size)]
            vals = logits[recent]
            logits[recent] = np.where(vals > 0, vals / c.repeat_penalty, vals * c.repeat_penalty)
        if c.temperature <= 0:
            return int(np.argmax(logits))
        candidates = np.arange(logits.size)
        if 0 < c.top_k < logits.size:
            candidates = np.argpartition(-logits, c.top_k - 1)[: c.top_k]
        vals = logits[candidates] / c.temperature
        order = np.argsort(-vals, kind="stable")
        candidates, vals = candidates[order], vals[order]
        probs = np.exp(vals - vals.max())
        probs /= probs.sum()
        if c.top_p < 1.0:
            keep = int(np.searchsorted(np.cumsum(probs), c.top_p) + 1)
            candidates, probs = candidates[:keep], probs[:keep]
        if c.min_p > 0:
            mask = probs >= c.min_p * probs[0]
            candidates, probs = candidates[mask], probs[mask]
        probs = probs / probs.sum()
        return int(candidates[self.rng.choice(len(probs), p=probs)])
