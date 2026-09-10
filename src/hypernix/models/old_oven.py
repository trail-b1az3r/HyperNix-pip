"""old_oven: code-generation wrapper around HyperNix.

The oven metaphor: a raw HuggingFace snapshot is dough. :func:`preheat`
downloads / loads it into a fully-ready :class:`CodeOven` you can bake
code out of — low-temperature sampling defaults, stop-sequence awareness,
and an optional fill-in-the-middle API for code infilling tasks.

Typical use (Python)::

    from hypernix import old_oven

    oven = old_oven.preheat("ray0rf1re/hyper-nix.1")
    print(oven.complete("def fibonacci(n):"))
    print(oven.fill(prefix="def add(a, b):\\n    return ",
                    suffix="\\n\\nresult = add(1, 2)"))
    oven.save_pt("./hypernix.pt")   # self-contained torch.load()-able bundle

Typical use (CLI)::

    hypernix --auto-oven --prompt "def fib(n):"
    hypernix oven --model-dir ./snapshot --prompt "def fib(n):"
    hypernix oven --model-dir ./snapshot --fill-prefix "def add(a,b):\\n    return " \\
                                         --fill-suffix "\\n\\nprint(add(1,2))"
"""
from __future__ import annotations

from hypernix.system.deprecation import deprecated_module

# Announced before the heavy imports below, so the notice reaches the
# operator immediately rather than after torch has finished loading.
deprecated_module(
    "hypernix.models.old_oven",
    instead="hypernix.models.neo_oven",
    since="0.71.5a2",
)

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hypernix.models.download import download_model, verify_snapshot
from hypernix.models.generate import _load_tokenizer, _sample_next
from hypernix.models.neo_oven import unwrap_model as _unwrap_model_fn
from hypernix.training.train import (
    HyperNixConfig,
    HyperNixModel,
    _iter_chunks,
    init_from_scratch,
    load_snapshot,
    save_snapshot,
)

# Heuristic stop strings we trim off a code completion. Users can pass an
# explicit `stop=...` to override; set `stop=()` to disable trimming.
DEFAULT_STOPS: tuple[str, ...] = ("\nclass ", "\ndef ", "\n\n\n", "</s>")

# FIM tokens used by the common "starcoder-style" convention; generate.fill()
# falls back to plain prefix-only continuation if these aren't in the vocab.
_FIM_PREFIX = "<fim_prefix>"
_FIM_SUFFIX = "<fim_suffix>"
_FIM_MIDDLE = "<fim_middle>"


# ---------------------------------------------------------------------------
# Architecture presets
# ---------------------------------------------------------------------------
# The HyperNix model class is a parametric Llama-style causal LM. By flipping
# a handful of config knobs it can also be used as the Qwen2 / Qwen2.5
# architecture: the only structural difference is that Qwen2 places a bias on
# q_proj / k_proj / v_proj (but not o_proj). Defaults also differ for
# rope_theta and the RMSNorm epsilon. These presets encode those differences
# so callers can request either architecture by name.
#
# The presets below are seeds for :func:`new_oven` (creating a fresh,
# untrained snapshot in a chosen arch). Actually *loading* any HF-family
# model — Gemma, Phi, DeepSeek, GLM, GPT-OSS, Nemotron, Llama 3+, etc. —
# goes through :func:`hypernix.train.load_snapshot` which falls back to
# ``transformers.AutoModelForCausalLM`` for any non-Llama-shaped arch.
# You don't need a preset here to consume those models; you only need one
# to hand-spec a fresh parametric model in that style.
ARCH_PRESETS: dict[str, dict[str, Any]] = {
    "hypernix": {
        "attention_bias": False,
        "model_type": "hypernix",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    # HyperNix v2 — chat-tuned, ChatML-style template baked into the
    # tokenizer, otherwise identical Llama-shape to v1.  Use this preset
    # when initialising a fresh chat model from scratch.
    "hypernix2": {
        "attention_bias": False,
        "model_type": "hypernix",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    "hyper-nix.2": {
        "attention_bias": False,
        "model_type": "hypernix",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    # ---- Llama family -----------------------------------------------------
    # Llama 2 defaults.
    "llama": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    # Llama 3 / 3.1 / 3.2 / 3.3 — same shape, rope_theta bumped to 500k.
    "llama3": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    "llama3.1": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    "llama3.2": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
    },
    "llama3.3": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    "llama4": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
    },
    # ---- Qwen family ------------------------------------------------------
    "qwen2": {
        "attention_bias": True,
        "model_type": "qwen2",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    },
    "qwen2.5": {
        "attention_bias": True,
        "model_type": "qwen2",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    },
    # Qwen3 keeps q/k/v bias off and uses an even larger rope_theta.
    # Consuming a real Qwen3 checkpoint goes through AutoModel; this seed
    # is only for creating a new Qwen3-shaped model from scratch.
    "qwen3": {
        "attention_bias": False,
        "model_type": "qwen2",  # runtime class reuses our qwen2 path
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    },
    # ---- Mistral ----------------------------------------------------------
    "mistral": {
        "attention_bias": False,
        "model_type": "mistral",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    # ---- DeepSeek-R1 (distilled Llama-shape) ------------------------------
    # The real MoE / MLA checkpoints must go through AutoModel; this is the
    # distilled-Llama seed used by the popular R1-distill variants.
    "deepseek-r1": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    "deepseek": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
}
# Backing presets for architectures that have a distinct model_type at the
# HF level but still fit the Llama-shape we expose here — provided as
# short-name aliases so `new_oven(arch="gemma2")` does something sensible.
# For actually loading a pretrained checkpoint of these families, we route
# through transformers.AutoModelForCausalLM in load_snapshot(); this just
# gives users a shortcut for hand-building a fresh-initialized one.
ARCH_PRESETS.update({
    "gemma": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    },
    "gemma2": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    },
    "gemma3": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    },
    # Gemma 4 (Apr 2026): local/global hybrid attention, per-layer embeddings on
    # the E-series, 256k vocab. The preset here just reproduces the dense-path
    # shape; real Gemma 4 checkpoints always load via AutoModelForCausalLM.
    "gemma4": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    },
    "phi3": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    "phi4": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 250000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    "glm4": {
        "attention_bias": True,
        "model_type": "qwen2",  # GLM4's attention has a qkv bias like Qwen2
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    # GLM 5 / 5.1 (zai-org): MoE with dynamic sparse attention. For seeding a
    # fresh parametric model we collapse to the dense Llama shape; real GLM-5
    # weights go through AutoModel (model_type="glm_moe_dsa").
    "glm5": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    "glm5.1": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    "nemotron": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    "gpt-oss": {
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    "gptoss": {  # alias
        "attention_bias": False,
        "model_type": "llama",
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    },
    # ---- Qwen 3.5 / 3.6 --------------------------------------------------
    # Qwen3.5 (model_type "qwen3_5"): Qwen-shape but with attention_bias=False
    # and a much larger rope_theta (10M). Interleaves linear + full attention
    # layers — real checkpoints must load via AutoModel; this preset is just
    # the dense-shape seed used by ``new_oven``.
    "qwen3.5": {
        "attention_bias": False,
        "model_type": "qwen2",
        "rope_theta": 10000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    },
    # Qwen3.6 ("qwen3_5_moe"): MoE variant with tie_word_embeddings=False.
    "qwen3.6": {
        "attention_bias": False,
        "model_type": "qwen2",
        "rope_theta": 10000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
    },
    # ---- Nix (ray0rf1re/nix collection) ----------------------------------
    # Nix models (1.0 through 2.7) are Qwen2-shape but with attention_bias
    # disabled and tied embeddings. They fit our existing qwen2 code path
    # directly; no AutoModel round-trip needed.
    "nix": {
        "attention_bias": False,
        "model_type": "qwen2",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    },
    "nix2": {  # alias for Nix 2.x
        "attention_bias": False,
        "model_type": "qwen2",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    },
})


def _dtype_from_str(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _trim_at_stop(text: str, stops: tuple[str, ...]) -> str:
    if not stops:
        return text
    earliest = len(text)
    for s in stops:
        idx = text.find(s)
        if idx >= 0:
            earliest = min(earliest, idx)
    return text[:earliest]


@dataclass
class CodeOven:
    """A loaded HyperNix-family model + tokenizer ready for generation.

    Returned by :func:`preheat`. Safe to reuse across many generate calls —
    the model is loaded once and kept resident on ``device``.

    ``model`` is typed as :class:`torch.nn.Module` so the oven can also
    host non-HyperNix architectures loaded via :func:`hypernix.train.load_snapshot`
    — currently ``NanoNanoModel`` from :mod:`hypernix.nano_nano`. Both
    classes share the same forward signature
    (``forward(input_ids, labels=None) -> {"logits": ..., "loss": ...}``)
    and expose a ``config`` attribute with ``max_position_embeddings`` and
    ``model_type``, which is all this class relies on.
    """

    model: nn.Module
    tokenizer: Any
    tokenizer_kind: str  # "hf" or "byte"
    device: torch.device
    dtype: torch.dtype
    model_dir: Path
    repo_id: str | None = None  # source HF repo, when known

    # ------------------------------------------------------------------
    # Tokenizer helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> list[int]:
        if self.tokenizer_kind == "hf":
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        return self.tokenizer.encode(text)

    def _decode(self, ids: list[int]) -> str:
        if self.tokenizer_kind == "hf":
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        return self.tokenizer.decode(ids)

    def _fim_token_id(self, marker: str) -> int | None:
        if self.tokenizer_kind != "hf":
            return None
        tid = self.tokenizer.convert_tokens_to_ids(marker)
        # HF returns `unk_token_id` (or a bare int like 0/None) on miss;
        # treat anything falling back to the unk token as "not present".
        unk = getattr(self.tokenizer, "unk_token_id", None)
        if tid is None or (unk is not None and tid == unk):
            return None
        return int(tid)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run(
        self,
        input_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_ids: tuple[int, ...] = (),
    ) -> list[int]:
        # Patch (0.51.4): defensive coercion + a clear error message
        # instead of the cryptic ``ValueError: too many dimensions
        # 'str'`` torch raises when something upstream slipped a
        # string / tensor / BatchEncoding past _format_chat.
        if isinstance(input_ids, str):
            input_ids = self._encode(input_ids)
        elif isinstance(input_ids, torch.Tensor):
            input_ids = [int(x) for x in input_ids.flatten().tolist()]
        elif not isinstance(input_ids, list):
            input_ids = list(input_ids)
        if input_ids and not isinstance(input_ids[0], int):
            raise TypeError(
                "_run expected list[int] for input_ids; got "
                f"{type(input_ids[0]).__name__} (head={input_ids[0]!r}). "
                "This usually means tokenizer.apply_chat_template returned "
                "an unexpected shape — please report the tokenizer.",
            )
        ctx = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        max_ctx = self.model.config.max_position_embeddings
        generated: list[int] = []
        for _ in range(max_new_tokens):
            if ctx.size(1) > max_ctx:
                ctx = ctx[:, -max_ctx:]
            logits = self.model(ctx)["logits"][:, -1, :].float()
            nxt = _sample_next(
                logits[0], temperature=temperature, top_k=top_k, top_p=top_p,
            )
            tok = int(nxt.item())
            generated.append(tok)
            if tok in eos_ids:
                break
            ctx = torch.cat([ctx, nxt.view(1, 1)], dim=1)
        return generated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_k: int = 40,
        top_p: float = 0.95,
        stop: tuple[str, ...] = DEFAULT_STOPS,
        seed: int | None = None,
    ) -> str:
        """Continue ``prompt`` as code.

        Defaults favour code generation: low temperature, moderate top-p,
        and a stop-sequence trimmer that cuts the output at the first
        ``\\nclass `` / ``\\ndef `` boundary so you usually get one
        function back rather than a whole file.
        """
        if seed is not None:
            torch.manual_seed(seed)

        ids = self._encode(prompt) if prompt else []
        bos = getattr(self.tokenizer, "bos_token_id", None) if self.tokenizer_kind == "hf" else None
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos, *ids]
        if not ids:
            ids = [0]

        eos: tuple[int, ...] = ()
        if self.tokenizer_kind == "hf":
            eid = getattr(self.tokenizer, "eos_token_id", None)
            if isinstance(eid, int):
                eos = (eid,)

        out_ids = self._run(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            eos_ids=eos,
        )
        new_text = self._decode(out_ids)
        return _trim_at_stop(new_text, stop)

    def fill(
        self,
        prefix: str,
        suffix: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_k: int = 40,
        top_p: float = 0.95,
        seed: int | None = None,
    ) -> str:
        """Fill-in-the-middle: generate text that fits between ``prefix``
        and ``suffix``.

        Uses the starcoder-style ``<fim_prefix> ... <fim_suffix> ... <fim_middle>``
        convention when the tokenizer has those special tokens; otherwise
        falls back to prefix-only completion (ignoring ``suffix``). Useful
        for editor-style code infilling.
        """
        if seed is not None:
            torch.manual_seed(seed)

        pre_id = self._fim_token_id(_FIM_PREFIX)
        suf_id = self._fim_token_id(_FIM_SUFFIX)
        mid_id = self._fim_token_id(_FIM_MIDDLE)

        if pre_id is not None and suf_id is not None and mid_id is not None:
            ids = [pre_id, *self._encode(prefix), suf_id, *self._encode(suffix), mid_id]
        else:
            # Graceful fallback: ignore the suffix and continue the prefix.
            # Users can detect this by comparing the tokenizer vocabulary.
            ids = self._encode(prefix) or [0]

        out_ids = self._run(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        return self._decode(out_ids)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def _format_chat(self, messages: list[dict[str, str]]) -> list[int]:
        """Render a list of ``{role, content}`` messages into token ids.

        Preference order:
        1. ``tokenizer.apply_chat_template`` when the HF tokenizer has one
           wired up (this is what ``nano-nano-v4`` and ``nano-mini`` ship
           with, so we get the canonical Llama/Qwen chat format for free).
        2. A :mod:`hypernix.cookbook` template chosen by ``self.repo_id``
           — this is how ``hyper-Nix.2`` gets a ChatML prompt even when
           the tokenizer doesn't ship a ``chat_template``.
        3. A plain ``role: content`` transcript fallback that works with
           any tokenizer including the byte fallback.

        Patch (0.51.4): ``apply_chat_template`` is allowed to return any
        of: a flat list of ints, a 1-D / 2-D ``torch.Tensor``, a
        ``BatchEncoding``, or even (broken-but-seen-in-the-wild) a plain
        rendered string.  All five shapes are coerced into a flat
        ``list[int]`` here so :meth:`_run` never sees strings.  The
        previous implementation ``return list(ids)`` happily passed
        ``list("hello") == ['h','e',…]`` straight through, which then
        hit ``ValueError: too many dimensions 'str'`` inside
        ``torch.tensor``.
        """
        if self.tokenizer_kind == "hf":
            apply = getattr(self.tokenizer, "apply_chat_template", None)
            tmpl = getattr(self.tokenizer, "chat_template", None)
            if callable(apply) and tmpl:
                try:
                    ids = apply(
                        messages, tokenize=True, add_generation_prompt=True,
                    )
                    coerced = self._coerce_token_ids(ids)
                    if coerced is not None:
                        return coerced
                except Exception:  # noqa: BLE001
                    # Fall through to the cookbook / plain path so a
                    # broken template never bricks chat().
                    pass

        # Cookbook fallback — pick a template from the repo id, fall
        # through to plain role: content if nothing matches.
        if self.repo_id:
            try:
                from hypernix.chat import cookbook as _cb
                tmpl_obj = _cb.for_model(self.repo_id, default="plain")
                rendered = tmpl_obj.apply(messages, add_generation_prompt=True)
                return self._encode(rendered)
            except Exception:  # noqa: BLE001
                pass

        parts: list[str] = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}")
        parts.append("assistant:")
        return self._encode("\n".join(parts))

    def _coerce_token_ids(self, ids: Any) -> list[int] | None:
        """Normalise whatever ``apply_chat_template`` returned into a
        flat ``list[int]``.

        Returns ``None`` (so the caller can fall through to the cookbook
        / plain path) when the input shape isn't recognised — better
        than crashing inside :meth:`_run` with ``too many dimensions
        'str'``.
        """
        # Plain string — re-encode through the tokenizer.
        if isinstance(ids, str):
            return self._encode(ids)

        # HF ``BatchEncoding`` exposes ``input_ids``; pull that out and recurse.
        input_ids_attr = getattr(ids, "input_ids", None)
        if input_ids_attr is not None and not isinstance(ids, list | tuple):
            return self._coerce_token_ids(input_ids_attr)

        # torch.Tensor (any rank).  Flatten + convert to list of ints.
        if isinstance(ids, torch.Tensor):
            if ids.numel() == 0:
                return []
            return [int(x) for x in ids.flatten().tolist()]

        # Sequence — could be list[int], list[list[int]] (batched), or
        # list[str] (the bug we patched).
        if isinstance(ids, list | tuple):
            if not ids:
                return []
            head = ids[0]
            if isinstance(head, int):
                return [int(x) for x in ids]
            if isinstance(head, list | tuple):
                # Batched — take the first batch.
                return [int(x) for x in head]
            if isinstance(head, torch.Tensor):
                return [int(x) for x in (head[0] if head.dim() else head).flatten().tolist()] \
                    if head.dim() > 0 else [int(x.item()) for x in ids]
            # list[str] or similar — not usable.
            return None

        return None

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
        seed: int | None = None,
    ) -> str:
        """Run a chat turn. ``messages`` is a list of ``{"role", "content"}``.

        Works with all three new HyperNix-family models:

        * ``Nano-nano-v4`` and ``Nano-mini-6.99-v2`` ship a HF Llama-style
          chat template, so ``apply_chat_template`` takes the fast path.
        * ``nano-nano-927-v3`` / HyperNix v1 / freshly-initialized ovens
          don't — those fall back to a simple ``role: content`` transcript.

        Defaults use ``temperature=0.7`` (chat-typical), higher than the
        code-completion default of 0.2.
        """
        if seed is not None:
            torch.manual_seed(seed)

        ids = self._format_chat(messages)
        if not ids:
            ids = [0]

        eos: tuple[int, ...] = ()
        if self.tokenizer_kind == "hf":
            eid = getattr(self.tokenizer, "eos_token_id", None)
            if isinstance(eid, int):
                eos = (eid,)

        out_ids = self._run(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            eos_ids=eos,
        )
        return self._decode(out_ids)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset_path: Path | str,
        out_dir: Path | str | None = None,
        *,
        steps: int = 1000,
        batch_size: int = 2,
        context_length: int = 512,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        log_every: int = 10,
        save_every: int = 500,
        seed: int | None = None,
        quiet: bool = False,
        optimizer_class: type | None = None,
        use_abbicus: bool = False,
        use_turbo_abbicus: bool = False,
        use_stml: bool = False,
        untrained_max_context: int = 8192,
        segment_length: int = 512,
        compute_framework: Any | None = None,
    ) -> Path:
        """Continue-pretrain this oven on a raw-text file.

        Works for both architectures (HyperNix and Qwen2/Qwen2.5) — the
        underlying model class is the same, the preset just flips
        ``attention_bias`` and a few hyperparameters. Uses the snapshot's
        HF tokenizer if one is present, otherwise falls back to the
        byte-level tokenizer so the training loop is always runnable
        (handy for smoke-testing / CI).

        Args:
            dataset_path: Path to a raw-text file.
            out_dir: Where to save the trained snapshot. Defaults to
                ``<model_dir>-trained`` next to the current model dir.
            steps, batch_size, context_length, lr, weight_decay, grad_clip:
                Standard training knobs.
            log_every, save_every: Console / checkpoint cadence.
            seed: Optional torch seed for reproducible runs.
            quiet: Suppress per-step logging.

        Returns:
            Path to the written snapshot directory (a HF-style snapshot
            that feeds straight back into ``preheat(local_dir=...)``).
        """
        import math

        dataset_path = Path(dataset_path)
        if out_dir is None:
            out = self.model_dir.parent / (self.model_dir.name + "-trained")
        else:
            out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            torch.manual_seed(seed)

        # Training mode may run in a higher-precision path than the
        # generation dtype; keep the user's dtype but flip on train().
        self.model.train()

        bos_id: int | None = None
        if self.tokenizer_kind == "hf":
            bos_id = getattr(self.tokenizer, "bos_token_id", None)

        chunks = list(
            _iter_chunks(dataset_path, self.tokenizer, context_length, bos_id=bos_id)
        )
        if not chunks:
            raise RuntimeError(
                f"dataset {dataset_path} produced no training chunks "
                f"(needs > context_length={context_length} tokens)"
            )

        optimizer_kwargs: dict[str, Any] = {
            "betas": (0.9, 0.95),
            "weight_decay": weight_decay,
        }
        if optimizer_class is not None:
            import inspect

            optimizer_params = inspect.signature(optimizer_class).parameters
            if "lr" in optimizer_params:
                optimizer_kwargs["lr"] = lr
            elif "peak_lr" in optimizer_params:
                optimizer_kwargs["peak_lr"] = lr
            if "grad_clip" in optimizer_params:
                optimizer_kwargs["grad_clip"] = grad_clip
            core = _unwrap_model_fn(self.model)
            opt = optimizer_class(core.parameters(), **optimizer_kwargs)
        else:
            core = _unwrap_model_fn(self.model)
            opt = torch.optim.AdamW(core.parameters(), lr=lr, **optimizer_kwargs)
            
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))

        abb = None
        stml_mgr = None
        if use_turbo_abbicus:
            from hypernix.training.abbicus import TurboAbbicus, TurboAbbicusConfig
            abb = TurboAbbicus(TurboAbbicusConfig(
                base_context_length=context_length,
                hard_cap=untrained_max_context,
            ))
        elif use_abbicus:
            from hypernix.training.abbicus import Abbicus, AbbicusConfig
            abb = Abbicus(AbbicusConfig(base_context_length=context_length))
        if use_stml:
            from hypernix.models.stml import STML
            stml_mgr = STML(
                trained_context=context_length,
                untrained_max_context=untrained_max_context,
                segment_length=segment_length,
                regulator=abb,
            )

        step = 0
        while step < steps:
            batch_tensors = [
                chunks[(step * batch_size + i) % len(chunks)] for i in range(batch_size)
            ]
            batch_stacked = torch.stack(batch_tensors).to(self.device)
            inputs = batch_stacked[:, :-1]
            labels = batch_stacked[:, 1:]

            # Apply context regulation (STML wraps the regulator, or use regulator alone)
            if stml_mgr is not None:
                if abb is not None:
                    abb.step(step)
                batch_dict = {"input_ids": inputs, "labels": labels}
                batch_dict = stml_mgr.regulate(batch_dict)
                inputs = batch_dict["input_ids"]
                labels = batch_dict["labels"]
            elif abb:
                abb.step(step)
                batch_dict = {"input_ids": inputs, "labels": labels}
                batch_dict = abb.regulate(batch_dict)
                inputs = batch_dict["input_ids"]
                labels = batch_dict["labels"]

            out_dict = self.model(inputs, labels=labels)
            loss = out_dict["loss"]

            # ComputeFramework backprop if provided
            if compute_framework:
                compute_framework.backward(loss)
                compute_framework.step(opt)
                opt.zero_grad(set_to_none=True)
            else:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt.step()

            sched.step()
            step += 1

            if not quiet and step % log_every == 0:
                ppl = math.exp(min(loss.item(), 20))
                ctx_len = inputs.shape[1] if hasattr(inputs, "shape") else context_length
                print(
                    f"[old_oven.train] arch={self.model.config.model_type} "
                    f"step {step}/{steps}  loss={loss.item():.4f}  ppl={ppl:.2f}  ctx={ctx_len}"
                )
            if save_every and step % save_every == 0:
                save_snapshot(self.model, out, tokenizer_source=self.model_dir)

        save_snapshot(self.model, out, tokenizer_source=self.model_dir)
        self.model.eval()
        # Point the oven at the newly-saved snapshot so subsequent
        # .save_pt() / re-preheat calls pick up the trained weights.
        self.model_dir = out
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pt(self, out_path: Path | str) -> Path:
        """Save a self-contained ``torch.load``-able bundle to ``out_path``.

        The resulting ``.pt`` contains ``config`` (dict), ``state_dict``,
        and the source ``model_dir`` for later tokenizer recovery — so
        callers can ship a single file around and reconstruct the model
        with :func:`load_pt`.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "hypernix_format_version": 1,
            "config": self.model.config.to_dict(),
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "source_model_dir": str(self.model_dir),
        }
        torch.save(bundle, out_path)
        return out_path


# ---------------------------------------------------------------------------
# Top-level functional API
# ---------------------------------------------------------------------------

def preheat(
    repo_id: str = "ray0rf1re/hyper-nix.1",
    *,
    local_dir: Path | str | None = None,
    revision: str | None = None,
    token: str | None = None,
    device: str | None = None,
    dtype: str = "float32",
    quiet: bool = False,
) -> CodeOven:
    """Download (if necessary) + load a HyperNix snapshot into a ``CodeOven``.

    If ``local_dir`` already contains a valid snapshot (``config.json`` +
    weights), no network call is made; otherwise the snapshot is fetched
    from the HF Hub. This is the one-call "get me a working PyTorch model
    right now" entry point; pairs with ``hypernix --auto-oven`` on the CLI.
    """
    # v0.61.1: surface the MAJOR undertrained warning the moment the
    # caller resolves to hyper-Nix.2, even when local_dir hits cache.
    try:
        from hypernix.system.utils import warn_hyper_nix_2
        warn_hyper_nix_2(repo_id)
    except Exception:  # noqa: BLE001
        pass
    path: Path
    if local_dir is not None:
        ld = Path(local_dir)
        try:
            if ld.exists():
                verify_snapshot(ld)
                path = ld
            else:
                raise FileNotFoundError(ld)
        except FileNotFoundError:
            path = download_model(
                repo_id=repo_id, revision=revision,
                local_dir=str(ld), token=token, quiet=quiet,
            )
    else:
        path = download_model(
            repo_id=repo_id, revision=revision, token=token, quiet=quiet,
        )

    model, _cfg = load_snapshot(path)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tdtype = _dtype_from_str(dtype)
    model.to(dev, dtype=tdtype)
    model.eval()

    tok, kind = _load_tokenizer(path)
    return CodeOven(
        model=model, tokenizer=tok, tokenizer_kind=kind,
        device=dev, dtype=tdtype, model_dir=path, repo_id=repo_id,
    )


def new_oven(
    out_dir: Path | str,
    *,
    arch: str = "hypernix",
    vocab_size: int = 32000,
    hidden_size: int = 1024,
    intermediate_size: int = 4096,
    num_hidden_layers: int = 16,
    num_attention_heads: int = 16,
    num_key_value_heads: int | None = None,
    max_position_embeddings: int = 2048,
    tokenizer_source: Path | str | None = None,
    device: str | None = None,
    dtype: str = "float32",
    seed: int | None = None,
) -> CodeOven:
    """Create a new untrained oven in HyperNix or Qwen2 (Qwen2.5) architecture.

    Writes a fresh HuggingFace-style snapshot at ``out_dir`` (so the model
    can be re-loaded from disk later), then returns a :class:`CodeOven`
    pointing at it. The returned oven is ready for ``.train(...)``.

    Args:
        out_dir: Where to write the new snapshot.
        arch: ``"hypernix"`` (default) or ``"qwen2"`` / ``"qwen2.5"``.
            The Qwen presets enable q/k/v bias, switch ``model_type`` to
            ``"qwen2"``, and use Qwen's canonical rope/eps defaults.
        vocab_size, hidden_size, ... max_position_embeddings: Standard
            causal-LM shape knobs.
        tokenizer_source: Existing snapshot to copy tokenizer files from.
            If omitted, the snapshot has no HF tokenizer and training /
            generation falls back to the byte-level tokenizer.
        device, dtype: Runtime placement for the returned oven.
        seed: Optional torch seed for reproducible initialization.

    Returns:
        A trainable :class:`CodeOven`. Call ``.train(dataset_path, ...)``
        to pretrain, then ``.complete(...)`` / ``.fill(...)`` as usual.
    """
    if arch not in ARCH_PRESETS:
        raise ValueError(
            f"unknown arch {arch!r}; choose from {sorted(ARCH_PRESETS)}"
        )
    preset = ARCH_PRESETS[arch]
    cfg = HyperNixConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        **preset,
    )
    path = init_from_scratch(
        out_dir, cfg, tokenizer_source=tokenizer_source, seed=seed,
    )

    model, _ = load_snapshot(path)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tdtype = _dtype_from_str(dtype)
    model.to(dev, dtype=tdtype)
    model.eval()

    tok, kind = _load_tokenizer(path)
    return CodeOven(
        model=model, tokenizer=tok, tokenizer_kind=kind,
        device=dev, dtype=tdtype, model_dir=path,
    )


def bake_code(source: CodeOven | Path | str, prompt: str, **kwargs: Any) -> str:
    """Convenience: accepts a preheated oven or a snapshot path/URL."""
    oven = source if isinstance(source, CodeOven) else preheat(local_dir=str(source))
    return oven.complete(prompt, **kwargs)


def fill_middle(
    source: CodeOven | Path | str, prefix: str, suffix: str, **kwargs: Any,
) -> str:
    """Convenience FIM API — accepts a preheated oven or a snapshot path."""
    oven = source if isinstance(source, CodeOven) else preheat(local_dir=str(source))
    return oven.fill(prefix, suffix, **kwargs)


def load_pt(pt_path: Path | str, *, device: str | None = None) -> CodeOven:
    """Rehydrate a CodeOven from a ``save_pt`` bundle.

    Tokenizer files are pulled from the original ``source_model_dir`` when
    still available; otherwise falls back to the byte-level tokenizer.
    """
    pt_path = Path(pt_path)
    bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
    from hypernix.training.train import HyperNixConfig

    cfg = HyperNixConfig.from_dict(bundle["config"])
    model = HyperNixModel(cfg)
    model.load_state_dict(bundle["state_dict"], strict=False)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    model.eval()

    src_dir = Path(bundle.get("source_model_dir") or "")
    tok, kind = _load_tokenizer(src_dir) if src_dir.exists() else _load_tokenizer(Path("."))
    return CodeOven(
        model=model, tokenizer=tok, tokenizer_kind=kind,
        device=dev, dtype=next(model.parameters()).dtype,
        model_dir=src_dir if src_dir.exists() else pt_path.parent,
    )
