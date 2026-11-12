"""Strife — HyperNix's own GGUF inference runtime (0.73.0 alpha 1).

Strife reads a GGUF itself and runs it: no llama.cpp process, no T1 key,
no network. ::

    from hypernix.strife import Engine, SamplerConfig, inspect_model

    report = inspect_model("model.gguf")      # checks before loading anything
    with Engine("model.gguf", n_ctx=4096) as engine:
        result = engine.generate("Once upon a time", max_tokens=64,
                                 sampling=SamplerConfig(temperature=0, seed=1))
        print(result.text, result.stats["decode_tokens_per_second"])

The layers, each its own module:

=================  ===========================================================
:mod:`.inspect`    header, tensor table and architecture checks, before any
                   weight is touched
:mod:`.quants`     the tensor types with verified decoders (and only those)
:mod:`.weights`    the memory-mapped file and a RAM-budgeted decode cache
:mod:`.tokenizer`  the GGUF's own tokenizer (byte-level BPE, SentencePiece)
:mod:`.model`      the transformer forward pass and KV cache
:mod:`.sampler`    greedy / top-k / top-p / min-p / repetition penalty, seeded
:mod:`.engine`     sessions, streaming, stop strings, cancellation, timeouts
:mod:`.placement`  hardware detection and where each layer runs
=================  ===========================================================

Alpha 1 computes on the CPU. See ``wiki/Strife.md`` for the capability
matrix and what is not there yet.
"""
from __future__ import annotations

from .engine import Engine, GenerationResult
from .errors import StrifeError
from .inspect import ARCHITECTURES, ModelReport, inspect_model
from .quants import supported_types
from .sampler import SamplerConfig

__all__ = ["ARCHITECTURES", "Engine", "GenerationResult", "ModelReport", "SamplerConfig",
           "StrifeError", "inspect_model", "supported_types"]
