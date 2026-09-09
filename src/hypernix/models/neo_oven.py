"""neo_oven — Unified, production-ready model management, generation, and data utilities.

NeoOven is the single successor to the old_oven / CodeOven / new_oven,
old_fridge, mediocre_fridge, and new_fridge modules. Everything those modules
did is here, done better:

* :class:`NeoOven` — load, train, generate, fill, chat, save. Supports the
  full HyperNix / Qwen / Llama / Gemma / Phi / Mistral / DeepSeek arch
  family plus any HF AutoModel checkpoint.

* :func:`preheat` / :func:`new_oven` — top-level shortcuts that now return a
  :class:`NeoOven` instead of a ``CodeOven`` (same call signature).

* Memory management: :func:`freeze`, :func:`unfreeze`, :func:`parameter_stats`,
  :func:`offload_to_cpu`, :func:`chill_cache`, :func:`unwrap_model`.
  Previously scattered in old_fridge.

* Judge dataset tools: :class:`JudgeCorpus` (replaces mediocre_fridge). Uses
  proper LLM-based or heuristic hard-negative generation, JSONL output,
  configurable prompt/response/label delimiters, and HF datasets integration.

* Training analytics: :class:`TrainingMetrics` (replaces new_fridge). Native
  callback interface for TensorBoard, W&B, and matplotlib; parses training
  log text as before but now emits structured dicts not just step/loss pairs.

Typical use::

    from hypernix import neo_oven

    oven = neo_oven.preheat("ray0rf1re/hyper-nix.1")
    print(oven.complete("def fibonacci(n):"))

    # Train
    oven.train("dataset.txt", "./out", steps=500)

    # Memory management
    neo_oven.freeze(oven.model, ["embed_tokens"])
    print(neo_oven.parameter_stats(oven.model))

    # Judge dataset
    corpus = neo_oven.JudgeCorpus.from_oven(oven, prompts=[...])
    corpus.save("judge_data.jsonl")

    # Training analytics
    metrics = neo_oven.TrainingMetrics(use_tensorboard=True)
    metrics.on_step(step=1, loss=2.3, lr=3e-4)
"""
from __future__ import annotations

import fnmatch
import gc
import json
import logging
import math
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hypernix.models.download import download_model, verify_snapshot
from hypernix.models.generate import _load_tokenizer, _sample_next
from hypernix.training.train import (
    HyperNixConfig,
    HyperNixModel,
    _iter_chunks,
    init_from_scratch,
    load_snapshot,
    save_snapshot,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (shared with old_oven for compatibility)
# ---------------------------------------------------------------------------

DEFAULT_STOPS: tuple[str, ...] = ("\nclass ", "\ndef ", "\n\n\n", "</s>")
# Fill-in-the-middle has different natural terminators than a completion:
# the code around the hole already exists, so "\ndef " is a perfectly
# ordinary thing to generate rather than a signal to stop.
FIM_STOPS: tuple[str, ...] = ("<fim_prefix>", "<fim_suffix>", "<fim_middle>",
                              "<|endoftext|>", "</s>")
_FIM_PREFIX = "<fim_prefix>"
_FIM_SUFFIX = "<fim_suffix>"
_FIM_MIDDLE = "<fim_middle>"

JUDGE_PROMPT_SEP = "<JUDGE_PROMPT>"
JUDGE_RESPONSE_SEP = "<JUDGE_RESPONSE>"
JUDGE_LABEL_SEP = "<JUDGE_LABEL>"
LABEL_GOOD = "GOOD"
LABEL_BAD = "BAD"

# Architecture presets — full set from old_oven, all in one place.
ARCH_PRESETS: dict[str, dict[str, Any]] = {
    "hypernix": {"attention_bias": False, "model_type": "hypernix", "rope_theta": 10000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "hypernix2": {"attention_bias": False, "model_type": "hypernix", "rope_theta": 10000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "hyper-nix.2": {"attention_bias": False, "model_type": "hypernix", "rope_theta": 10000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "llama": {"attention_bias": False, "model_type": "llama", "rope_theta": 10000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "llama3": {"attention_bias": False, "model_type": "llama", "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "llama3.1": {"attention_bias": False, "model_type": "llama", "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "llama3.2": {"attention_bias": False, "model_type": "llama", "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": True},
    "llama3.3": {"attention_bias": False, "model_type": "llama", "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "llama4": {"attention_bias": False, "model_type": "llama", "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": True},
    "qwen2": {"attention_bias": True, "model_type": "qwen2", "rope_theta": 1000000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True},
    "qwen2.5": {"attention_bias": True, "model_type": "qwen2", "rope_theta": 1000000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True},
    "qwen3": {"attention_bias": False, "model_type": "qwen2", "rope_theta": 1000000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True},
    "qwen3.5": {"attention_bias": False, "model_type": "qwen2", "rope_theta": 10000000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True},
    "qwen3.6": {"attention_bias": False, "model_type": "qwen2", "rope_theta": 10000000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": False},
    "mistral": {"attention_bias": False, "model_type": "mistral", "rope_theta": 1000000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "deepseek-r1": {"attention_bias": False, "model_type": "llama", "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "deepseek": {"attention_bias": False, "model_type": "llama", "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "gemma": {"attention_bias": False, "model_type": "llama", "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True},
    "gemma2": {"attention_bias": False, "model_type": "llama", "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True},
    "gemma3": {"attention_bias": False, "model_type": "llama", "rope_theta": 1000000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True},
    "gemma4": {"attention_bias": False, "model_type": "llama", "rope_theta": 1000000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True},
    "phi3": {"attention_bias": False, "model_type": "llama", "rope_theta": 10000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "phi4": {"attention_bias": False, "model_type": "llama", "rope_theta": 250000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "glm4": {"attention_bias": True, "model_type": "qwen2", "rope_theta": 10000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "glm5": {"attention_bias": False, "model_type": "llama", "rope_theta": 1000000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "glm5.1": {"attention_bias": False, "model_type": "llama", "rope_theta": 1000000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "nemotron": {"attention_bias": False, "model_type": "llama", "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "gpt-oss": {"attention_bias": False, "model_type": "llama", "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "gptoss": {"attention_bias": False, "model_type": "llama", "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False},
    "nix": {"attention_bias": False, "model_type": "qwen2", "rope_theta": 1000000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True},
    "nix2": {"attention_bias": False, "model_type": "qwen2", "rope_theta": 1000000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True},
}


# ---------------------------------------------------------------------------
# Memory / weight management (from old_fridge)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamStats:
    """Counts of parameters and their memory footprint."""
    total: int
    trainable: int
    frozen: int
    bytes: int

    @property
    def megabytes(self) -> float:
        return self.bytes / (1024 * 1024)

    def __str__(self) -> str:
        return (
            f"total={self.total:,}  trainable={self.trainable:,}  "
            f"frozen={self.frozen:,}  ({self.megabytes:.1f} MB)"
        )


def _match_any(name: str, patterns: tuple[str, ...]) -> bool:
    for p in patterns:
        if fnmatch.fnmatch(name, p) or p in name:
            return True
    return False


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Peel DDP / FSDP / DataParallel / torch.compile wrappers."""
    current = model
    seen: set[int] = set()
    while True:
        oid = id(current)
        if oid in seen:
            break
        seen.add(oid)
        for attr in ("module", "_fsdp_wrapped_module", "_orig_mod"):
            inner = getattr(current, attr, None)
            if inner is not None and inner is not current:
                current = inner
                break
        else:
            break
    return current


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the inner model, unwrapping DDP / FSDP / DataParallel."""
    return _unwrap_model(model)


def freeze(model: nn.Module, patterns: Iterable[str] = ("embed_tokens",)) -> int:
    """Set ``requires_grad=False`` on parameters matching ``patterns``.

    Returns the count of frozen parameters.
    """
    model = _unwrap_model(model)
    pats = tuple(patterns)
    n = 0
    for name, param in model.named_parameters():
        if _match_any(name, pats) and param.requires_grad:
            param.requires_grad = False
            n += param.numel()
    return n


def unfreeze(model: nn.Module, patterns: Iterable[str] = ("*",)) -> int:
    """Inverse of :func:`freeze`. Default unfreezes everything."""
    model = _unwrap_model(model)
    pats = tuple(patterns)
    n = 0
    for name, param in model.named_parameters():
        if _match_any(name, pats) and not param.requires_grad:
            param.requires_grad = True
            n += param.numel()
    return n


def parameter_stats(model: nn.Module) -> ParamStats:
    """Return :class:`ParamStats` for ``model``."""
    model = _unwrap_model(model)
    total = trainable = frozen = nbytes = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        nbytes += n * p.element_size()
        if p.requires_grad:
            trainable += n
        else:
            frozen += n
    return ParamStats(total=total, trainable=trainable, frozen=frozen, bytes=nbytes)


def offload_to_cpu(model: nn.Module, patterns: Iterable[str] = ("embed_tokens",)) -> int:
    """Move submodules matched by ``patterns`` to CPU. Returns how many moved."""
    model = _unwrap_model(model)
    pats = tuple(patterns)
    moved = 0
    for name, module in model.named_modules():
        if name and _match_any(name, pats):
            module.to("cpu")
            moved += 1
    return moved


def chill_cache() -> None:
    """Run ``gc.collect()`` and empty the CUDA allocator cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_stats() -> dict[str, float]:
    """Return current CUDA memory stats in MB. Empty dict on CPU-only systems."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


# ---------------------------------------------------------------------------
# Training analytics (from new_fridge, improved)
# ---------------------------------------------------------------------------

_LOG_LINE_RE = re.compile(r"step\s+(\d+)\s*/\s*\d+\s+loss=([-+0-9.eE]+)")
_LR_RE = re.compile(r"lr=([-+0-9.eE]+)")
_PPL_RE = re.compile(r"ppl=([-+0-9.eE]+)")


def parse_training_log(text: str) -> list[dict[str, Any]]:
    """Parse structured dicts from a hypernix training log string.

    Returns a list of ``{"step": int, "loss": float, "lr": float|None,
    "ppl": float|None}`` dicts — one per matched log line.

    Compatible with the old ``new_fridge.parse_training_log`` but richer.
    """
    results: list[dict[str, Any]] = []
    for m in _LOG_LINE_RE.finditer(text):
        entry: dict[str, Any] = {"step": int(m.group(1)), "loss": float(m.group(2))}
        lr_m = _LR_RE.search(text[max(0, m.start() - 5):m.end() + 80])
        ppl_m = _PPL_RE.search(text[max(0, m.start() - 5):m.end() + 80])
        entry["lr"] = float(lr_m.group(1)) if lr_m else None
        entry["ppl"] = float(ppl_m.group(1)) if ppl_m else None
        results.append(entry)
    return results


class TrainingMetrics:
    """Callback-based training metrics collector.

    Replaces new_fridge's static matplotlib functions with a proper callback
    interface. Optionally integrates with TensorBoard, Weights & Biases, and
    MLflow. Always buffers locally for later plotting.

    Usage::

        metrics = TrainingMetrics(use_tensorboard=True, log_dir="./runs")
        metrics.on_step(step=1, loss=2.3, lr=3e-4)
        metrics.plot_loss("loss.png")
        metrics.close()
    """

    def __init__(
        self,
        *,
        log_dir: str | Path | None = None,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        use_mlflow: bool = False,
        run_name: str = "neo_oven_run",
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir else None
        self.run_name = run_name
        self._steps: list[dict[str, Any]] = []

        self._tb_writer = None
        self._wandb_run = None
        self._mlflow_active = False

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                self._tb_writer = SummaryWriter(log_dir=str(log_dir or "runs/neo_oven"))
            except ImportError:
                logger.warning("TensorBoard not available. Install tensorboard to enable.")

        if use_wandb:
            try:
                import wandb  # type: ignore
                self._wandb_run = wandb.init(project="neo_oven", name=run_name, reinit=True)
            except ImportError:
                logger.warning("Weights & Biases not available. Install wandb to enable.")

        if use_mlflow:
            try:
                import mlflow  # type: ignore
                mlflow.start_run(run_name=run_name)
                self._mlflow_active = True
            except ImportError:
                logger.warning("MLflow not available. Install mlflow to enable.")

    def on_step(self, step: int, loss: float, **kwargs: Any) -> None:
        """Record a training step. Pass any extra metrics as kwargs (lr, ppl, etc.)."""
        entry = {"step": step, "loss": loss, **kwargs}
        self._steps.append(entry)

        if self._tb_writer:
            self._tb_writer.add_scalar("Loss/train", loss, step)
            for k, v in kwargs.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, step)

        if self._wandb_run:
            try:
                self._wandb_run.log({"loss": loss, "step": step, **kwargs})
            except Exception:
                pass

        if self._mlflow_active:
            try:
                import mlflow  # type: ignore
                mlflow.log_metrics({"loss": loss, **{k: v for k, v in kwargs.items() if isinstance(v, (int, float))}}, step=step)
            except Exception:
                pass

    def plot_loss(self, out_path: str | Path, *, title: str = "Training Loss") -> Path:
        """Save a loss curve PNG."""
        return _plot_pairs(
            [(e["step"], e["loss"]) for e in self._steps],
            out_path, title=title, ylabel="loss",
        )

    def plot_score_distribution(
        self, scores: Sequence[float], out_path: str | Path, *, title: str = "Score Distribution", bins: int = 20
    ) -> Path:
        """Histogram of judge / reward scores."""
        return _plot_histogram(scores, out_path, title=title, bins=bins)

    def plot_round_losses(
        self,
        rounds: Sequence[Sequence[tuple[int, float]]],
        out_path: str | Path,
        *,
        title: str = "Multi-Round Loss",
        round_labels: Sequence[str] | None = None,
    ) -> Path:
        """Loss curves for multiple training rounds."""
        return _plot_multi_round(rounds, out_path, title=title, round_labels=round_labels)

    def to_jsonl(self, out_path: str | Path) -> Path:
        """Write all recorded steps to a JSONL file for offline analysis."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for entry in self._steps:
                f.write(json.dumps(entry) + "\n")
        return out

    def close(self) -> None:
        """Flush and close any open writers."""
        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb_run:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
        if self._mlflow_active:
            try:
                import mlflow  # type: ignore
                mlflow.end_run()
            except Exception:
                pass


def _load_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except ModuleNotFoundError:
        from hypernix.system import deps
        deps.ensure(["matplotlib>=3.7"], reimport=["matplotlib", "matplotlib.pyplot"])
        import matplotlib.pyplot as plt  # type: ignore
        return plt


def _plot_pairs(
    pairs: Sequence[tuple[int, float]], out_path: str | Path,
    *, title: str, ylabel: str = "loss",
) -> Path:
    plt = _load_matplotlib()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    xs = [s for s, _ in pairs]
    ys = [y for _, y in pairs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_histogram(
    scores: Sequence[float], out_path: str | Path,
    *, title: str, bins: int = 20,
) -> Path:
    plt = _load_matplotlib()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(list(scores), bins=bins, edgecolor="black")
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_multi_round(
    rounds: Sequence[Sequence[tuple[int, float]]],
    out_path: str | Path,
    *, title: str,
    round_labels: Sequence[str] | None = None,
) -> Path:
    plt = _load_matplotlib()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pairs in enumerate(rounds):
        if not pairs:
            continue
        label = round_labels[i] if round_labels and i < len(round_labels) else f"round {i + 1}"
        xs = [s for s, _ in pairs]
        ys = [y for _, y in pairs]
        ax.plot(xs, ys, marker="o", markersize=3, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Judge dataset utilities (from mediocre_fridge, properly done)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeExample:
    """A (prompt, response, label) training triple for a judge / reward model."""
    prompt: str
    response: str
    label: str  # LABEL_GOOD or LABEL_BAD

    def format_text(self) -> str:
        """Serialize to the canonical plain-text delimiter format."""
        return (
            f"{JUDGE_PROMPT_SEP}{self.prompt}"
            f"{JUDGE_RESPONSE_SEP}{self.response}"
            f"{JUDGE_LABEL_SEP}{self.label}"
        )

    def to_dict(self) -> dict[str, str]:
        return {"prompt": self.prompt, "response": self.response, "label": self.label}


def _mangle_response(response: str, rng: random.Random) -> str:
    """Produce a plausibly-bad variant of a known-good response."""
    choices = ["truncate", "shuffle_words", "shuffle_chars", "wrong", "empty", "repeat", "substitute"]
    choice = rng.choice(choices)
    if choice == "truncate" and len(response) > 2:
        return response[: max(1, len(response) // 2)]
    if choice == "shuffle_words":
        words = response.split()
        if len(words) > 1:
            rng.shuffle(words)
            return " ".join(words)
    if choice == "shuffle_chars" and len(response) > 2:
        chars = list(response)
        rng.shuffle(chars)
        return "".join(chars)
    if choice == "wrong":
        return rng.choice(["I don't know.", "42", "blue", "maybe later", "N/A", "error"])
    if choice == "empty":
        return ""
    if choice == "substitute":
        # Replace digits with letters and vice versa
        result = ""
        for c in response:
            if c.isdigit():
                result += rng.choice("abcdefghij")
            elif c.isalpha():
                result += str(rng.randint(0, 9))
            else:
                result += c
        return result
    # "repeat"
    return (response + " ") * 3


class JudgeCorpus:
    """A collection of JudgeExamples for reward model / judge training.

    Replaces mediocre_fridge with a proper class-based API that supports:
    - Arbitrarily large custom prompt/response pools (not just 16 hardcoded items)
    - Multiple augmentation strategies for hard negatives
    - JSONL output (plus the legacy plain-text format)
    - HuggingFace datasets integration
    - LLM-based response collection from a NeoOven
    """

    def __init__(self, examples: list[JudgeExample] | None = None) -> None:
        self.examples: list[JudgeExample] = examples or []

    @classmethod
    def from_pairs(
        cls,
        pairs: Sequence[tuple[str, str]],
        n: int,
        *,
        seed: int | None = 0,
        good_ratio: float = 0.5,
    ) -> JudgeCorpus:
        """Build a corpus of ``n`` examples from ``(prompt, good_response)`` pairs.

        About ``good_ratio`` of examples get the reference answer (GOOD);
        the rest are mangled into BAD examples via :func:`_mangle_response`.
        """
        rng = random.Random(seed)
        examples: list[JudgeExample] = []
        for i in range(n):
            prompt, good = pairs[i % len(pairs)]
            if rng.random() < good_ratio:
                examples.append(JudgeExample(prompt, good, LABEL_GOOD))
            else:
                examples.append(JudgeExample(prompt, _mangle_response(good, rng), LABEL_BAD))
        return cls(examples)

    @classmethod
    def from_oven(
        cls,
        oven: NeoOven,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        label_rule: Callable[[str, str], str] | None = None,
    ) -> JudgeCorpus:
        """Collect real responses from a NeoOven and wrap them as JudgeExamples.

        ``label_rule`` is a ``(prompt, response) -> "GOOD" | "BAD"`` callable.
        When omitted every response is labelled GOOD — useful for building a
        teacher corpus you'll pair with mangled hard-negatives later.
        """
        examples: list[JudgeExample] = []
        for prompt in prompts:
            response = oven.complete(prompt, max_new_tokens=max_new_tokens,
                                     temperature=temperature, stop=())
            label = label_rule(prompt, response) if label_rule else LABEL_GOOD
            examples.append(JudgeExample(prompt, response, label))
        return cls(examples)

    @classmethod
    def from_hf_dataset(
        cls,
        dataset_name: str,
        prompt_col: str = "prompt",
        response_col: str = "response",
        label_col: str | None = None,
        *,
        split: str = "train",
        max_examples: int | None = None,
    ) -> JudgeCorpus:
        """Build a corpus from a HuggingFace dataset.

        If ``label_col`` is None every example is labelled GOOD.
        """
        try:
            import datasets as hf_datasets  # type: ignore
        except ImportError as e:
            raise ImportError(
                "JudgeCorpus.from_hf_dataset requires the `datasets` library. "
                "Install it with: pip install datasets"
            ) from e

        ds = hf_datasets.load_dataset(dataset_name, split=split)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))

        examples: list[JudgeExample] = []
        for row in ds:
            prompt = row[prompt_col]
            response = row[response_col]
            label = row[label_col] if label_col and label_col in row else LABEL_GOOD
            examples.append(JudgeExample(str(prompt), str(response), str(label)))
        return cls(examples)

    def save(self, out_path: str | Path, *, fmt: str = "jsonl") -> Path:
        """Write corpus to disk.

        Args:
            out_path: Destination file path.
            fmt: ``"jsonl"`` (default, one JSON object per line) or
                 ``"text"`` (legacy delimiter format).
        """
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "jsonl":
            with out.open("w", encoding="utf-8") as f:
                for ex in self.examples:
                    f.write(json.dumps(ex.to_dict()) + "\n")
        else:
            out.write_text("\n".join(e.format_text() for e in self.examples) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path, *, fmt: str = "jsonl") -> JudgeCorpus:
        """Load a corpus written by :meth:`save`."""
        path = Path(path)
        examples: list[JudgeExample] = []
        if fmt == "jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                examples.append(JudgeExample(d["prompt"], d["response"], d["label"]))
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                if JUDGE_PROMPT_SEP not in line:
                    continue
                p1 = line.split(JUDGE_RESPONSE_SEP)
                prompt = p1[0].replace(JUDGE_PROMPT_SEP, "")
                p2 = p1[1].split(JUDGE_LABEL_SEP) if len(p1) > 1 else ["", LABEL_GOOD]
                examples.append(JudgeExample(prompt, p2[0], p2[1] if len(p2) > 1 else LABEL_GOOD))
        return cls(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __repr__(self) -> str:
        good = sum(1 for e in self.examples if e.label == LABEL_GOOD)
        return f"JudgeCorpus(n={len(self.examples)}, good={good}, bad={len(self.examples)-good})"


# ---------------------------------------------------------------------------
# NeoOven — the unified model wrapper
# ---------------------------------------------------------------------------

def _dtype_from_str(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _trim_at_stop(text: str, stops: tuple[str, ...]) -> str:
    if not stops:
        return text
    earliest = len(text)
    for s in stops:
        idx = text.find(s)
        if 0 <= idx < earliest:
            earliest = idx
    return text[:earliest]


@dataclass
class NeoOven:
    """Production-ready model wrapper for generation, training, and evaluation.

    The successor to :class:`hypernix.old_oven.CodeOven`. All ``CodeOven``
    methods are preserved (``complete``, ``fill``, ``chat``, ``train``,
    ``save_pt``) plus new capabilities:

    - :meth:`stream` — token-by-token streaming generator
    - :meth:`generate_batch` — batched generation over multiple prompts
    - :meth:`freeze_backbone` / :meth:`memory_stats` — inline resource management
    - :class:`JudgeCorpus` integration via :meth:`build_judge_corpus`
    - :class:`TrainingMetrics` produced by :meth:`train` when ``metrics=True``
    """

    model: nn.Module
    tokenizer: Any
    tokenizer_kind: str  # "hf" or "byte"
    device: torch.device
    dtype: torch.dtype
    model_dir: Path
    repo_id: str | None = None

    # ------------------------------------------------------------------
    # Tokenizer helpers (identical to CodeOven for compatibility)
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
        unk = getattr(self.tokenizer, "unk_token_id", None)
        if tid is None or (unk is not None and tid == unk):
            return None
        return int(tid)

    def _coerce_token_ids(self, ids: Any) -> list[int] | None:
        """Normalise apply_chat_template output into a flat list[int]."""
        if isinstance(ids, str):
            return self._encode(ids)
        input_ids_attr = getattr(ids, "input_ids", None)
        if input_ids_attr is not None and not isinstance(ids, (list, tuple)):
            return self._coerce_token_ids(input_ids_attr)
        if isinstance(ids, torch.Tensor):
            return [] if ids.numel() == 0 else [int(x) for x in ids.flatten().tolist()]
        if isinstance(ids, (list, tuple)):
            if not ids:
                return []
            head = ids[0]
            if isinstance(head, int):
                return [int(x) for x in ids]
            if isinstance(head, (list, tuple)):
                return [int(x) for x in head]
            if isinstance(head, torch.Tensor):
                flat = head.flatten() if head.dim() > 0 else head.unsqueeze(0)
                return [int(x.item()) for x in flat]
        return None

    def _format_chat(self, messages: list[dict[str, str]]) -> list[int]:
        if self.tokenizer_kind == "hf":
            apply = getattr(self.tokenizer, "apply_chat_template", None)
            tmpl = getattr(self.tokenizer, "chat_template", None)
            if callable(apply) and tmpl:
                try:
                    ids = apply(messages, tokenize=True, add_generation_prompt=True)
                    coerced = self._coerce_token_ids(ids)
                    if coerced is not None:
                        return coerced
                except Exception:
                    pass
        if self.repo_id:
            try:
                from hypernix.chat import cookbook as _cb
                tmpl_obj = _cb.for_model(self.repo_id, default="plain")
                rendered = tmpl_obj.apply(messages, add_generation_prompt=True)
                return self._encode(rendered)
            except Exception:
                pass
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        parts.append("assistant:")
        return self._encode("\n".join(parts))

    # ------------------------------------------------------------------
    # Core sampling
    # ------------------------------------------------------------------

    def _max_context(self) -> int:
        """The model's positional limit.

        ``max_position_embeddings`` is a HyperNixConfig field and a
        near-universal HF one, but not every architecture's config carries
        it — reading it unguarded turned a loadable model into an
        AttributeError at the first generated token rather than at load
        time. 2048 is the conservative fallback.
        """
        return int(getattr(self.model.config, "max_position_embeddings", 2048) or 2048)

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
        should_stop: Callable[[], bool] | None = None,
    ) -> list[int]:
        """Sample up to *max_new_tokens*, stopping at EOS or on *should_stop*.

        *should_stop* is polled once per token. It exists so a caller that
        does not own this thread — the hyped-pro bridge, whose TUI wants
        Escape to actually stop a local generation rather than only discard
        its result — can end the loop cooperatively instead of leaving the
        GPU busy producing tokens nobody will read. Whatever was generated
        before the stop is returned; a partial answer is more useful than
        none, and the caller knows it asked.
        """
        if isinstance(input_ids, str):
            input_ids = self._encode(input_ids)
        elif isinstance(input_ids, torch.Tensor):
            input_ids = [int(x) for x in input_ids.flatten().tolist()]
        elif not isinstance(input_ids, list):
            input_ids = list(input_ids)
        if input_ids and not isinstance(input_ids[0], int):
            raise TypeError(
                f"NeoOven._run expected list[int], got {type(input_ids[0]).__name__}"
            )
        ctx = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        max_ctx = self._max_context()
        generated: list[int] = []
        for _ in range(max_new_tokens):
            if ctx.size(1) > max_ctx:
                ctx = ctx[:, -max_ctx:]
            if should_stop is not None and should_stop():
                break
            logits = self.model(ctx)["logits"][:, -1, :].float()
            nxt = _sample_next(logits[0], temperature=temperature, top_k=top_k, top_p=top_p)
            tok = int(nxt.item())
            # EOS terminates the sequence, it isn't part of it. Appending it
            # first only worked because HF decode is asked to skip special
            # tokens; a byte tokenizer, or an EOS the tokenizer doesn't
            # class as special, would have emitted it verbatim.
            if tok in eos_ids:
                break
            generated.append(tok)
            ctx = torch.cat([ctx, nxt.view(1, 1)], dim=1)
        return generated

    def _prompt_ids(self, prompt: str) -> list[int]:
        """Encode *prompt* the way every generation entry point should.

        Shared rather than duplicated because :meth:`stream` used to skip
        the BOS that :meth:`complete` prepends, which made the two sample
        different sequences from the same seed — the streamed answer and
        the non-streamed one for one prompt were simply different text.
        """
        ids = self._encode(prompt) if prompt else []
        if self.tokenizer_kind == "hf":
            bos = getattr(self.tokenizer, "bos_token_id", None)
            if isinstance(bos, int) and (not ids or ids[0] != bos):
                ids = [bos, *ids]
        return ids or [0]

    @staticmethod
    def _stable_prefix(text: str) -> str:
        """*text* minus any trailing replacement characters.

        A trailing U+FFFD means a multi-byte character is still missing
        bytes that a later token will supply — decoding it again once they
        arrive produces a *different* character in that position. Emitting
        it now would put a character in the stream that the finished text
        does not contain, so it is held back until it resolves.
        """
        return text.rstrip("\ufffd")

    @staticmethod
    def _emit_safe_length(text: str, stops: tuple[str, ...]) -> int:
        """How much of *text* is safe to emit without leaking half a stop marker.

        A stop sequence arrives one character at a time. Emitting eagerly
        means ``"\nclass "`` goes out as ``\n``, then ``c``, then ``l`` —
        each of which looks harmless on its own, and by the time the whole
        marker is recognisable its prefix is already in the stream. So any
        trailing text that *could* still turn into a stop marker is held
        back until the next token settles it either way.

        Same shape as :meth:`_stable_prefix`, one level up: that one holds
        back an incomplete character, this one holds back an incomplete
        marker.
        """
        if not stops:
            return len(text)
        held = 0
        for marker in stops:
            for n in range(min(len(marker) - 1, len(text)), 0, -1):
                if text.endswith(marker[:n]):
                    held = max(held, n)
                    break
        return len(text) - held

    def _eos_ids(self) -> tuple[int, ...]:
        """EOS token ids for this tokenizer, or ``()`` if it has none."""
        if self.tokenizer_kind != "hf":
            return ()
        eid = getattr(self.tokenizer, "eos_token_id", None)
        return (eid,) if isinstance(eid, int) else ()

    # ------------------------------------------------------------------
    # Generation API
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
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """Continue ``prompt`` as code / text."""
        if seed is not None:
            torch.manual_seed(seed)
        out_ids = self._run(self._prompt_ids(prompt), max_new_tokens=max_new_tokens,
                            temperature=temperature, top_k=top_k, top_p=top_p,
                            eos_ids=self._eos_ids(), should_stop=should_stop)
        return _trim_at_stop(self._decode(out_ids), stop)

    def fill(
        self,
        prefix: str,
        suffix: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_k: int = 40,
        top_p: float = 0.95,
        stop: tuple[str, ...] = FIM_STOPS,
        seed: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """Fill-in-the-middle generation (starcoder FIM convention).

        Stops at EOS and at the FIM markers themselves. Previously this
        passed no ``eos_ids`` at all, so a fill could never terminate early
        — every call ran the full ``max_new_tokens`` and then returned
        whatever the model had rambled into after finishing the middle.
        """
        if seed is not None:
            torch.manual_seed(seed)
        pre_id = self._fim_token_id(_FIM_PREFIX)
        suf_id = self._fim_token_id(_FIM_SUFFIX)
        mid_id = self._fim_token_id(_FIM_MIDDLE)
        eos = self._eos_ids()
        if pre_id is not None and suf_id is not None and mid_id is not None:
            ids = [pre_id, *self._encode(prefix), suf_id, *self._encode(suffix), mid_id]
            # A model that re-opens a FIM section has finished the middle.
            eos = (*eos, pre_id, suf_id, mid_id)
        else:
            ids = self._encode(prefix) or [0]
        out_ids = self._run(ids, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_k=top_k, top_p=top_p,
                            eos_ids=eos, should_stop=should_stop)
        return _trim_at_stop(self._decode(out_ids), stop)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
        seed: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """Run a chat turn given a ``[{"role": ..., "content": ...}]`` list."""
        if seed is not None:
            torch.manual_seed(seed)
        ids = self._format_chat(messages) or [0]
        out_ids = self._run(ids, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_k=top_k, top_p=top_p,
                            eos_ids=self._eos_ids(), should_stop=should_stop)
        return self._decode(out_ids)

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
        stop: tuple[str, ...] = DEFAULT_STOPS,
        seed: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ):
        """Token-by-token generator. Yields decoded text as it arrives.

        Usage::

            for chunk in oven.stream("def hello():"):
                print(chunk, end="", flush=True)

        Concatenating everything this yields gives the same string
        :meth:`complete` would return for the same prompt and seed. Getting
        that right needs two things it used to skip:

        **Incremental decoding.** A token is not a character. Decoding each
        token on its own — which is what this did — splits every multi-byte
        UTF-8 character across chunks, so a byte tokenizer streamed "café"
        as ``caf`` + ``\ufffd`` + ``\ufffd``, and a BPE tokenizer mangled
        anything non-ASCII the same way. Instead the whole sequence is
        decoded each step and only the *new* suffix is emitted; a character
        whose bytes aren't all present yet simply doesn't appear until the
        token that completes it arrives.

        **Stop sequences.** :meth:`complete` trims at *stop*, so a streamed
        turn that ignored them returned different text than the
        non-streamed one for the same input. Text past a stop marker is
        never emitted, and a chunk that would straddle one is truncated at
        it.
        """
        if seed is not None:
            torch.manual_seed(seed)
        ctx = torch.tensor([self._prompt_ids(prompt)], dtype=torch.long, device=self.device)
        max_ctx = self._max_context()
        eos = self._eos_ids()

        generated: list[int] = []
        emitted = ""

        def _advance(text: str) -> str:
            """The chunk to yield for *text*, given what's already out."""
            if not text.startswith(emitted) or len(text) <= len(emitted):
                return ""
            return text[len(emitted):]

        for _ in range(max_new_tokens):
            if should_stop is not None and should_stop():
                break
            if ctx.size(1) > max_ctx:
                ctx = ctx[:, -max_ctx:]
            logits = self.model(ctx)["logits"][:, -1, :].float()
            nxt = _sample_next(logits[0], temperature=temperature, top_k=top_k, top_p=top_p)
            tok = int(nxt.item())
            if tok in eos:
                break
            generated.append(tok)
            ctx = torch.cat([ctx, nxt.view(1, 1)], dim=1)

            text = self._decode(generated)
            trimmed = _trim_at_stop(text, stop)
            if trimmed != text:
                # A stop marker landed inside this chunk: emit whatever
                # precedes it and end the stream there.
                chunk = _advance(trimmed)
                if chunk:
                    yield chunk
                return
            stable = self._stable_prefix(text)
            chunk = _advance(stable[:self._emit_safe_length(stable, stop)])
            if chunk:
                yield chunk
                emitted += chunk

        # Flush whatever was held back — a trailing character that never
        # resolved, or the tail of a generation that hit max_new_tokens.
        # Without this, joining the stream would be short of complete().
        chunk = _advance(_trim_at_stop(self._decode(generated), stop))
        if chunk:
            yield chunk

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
        stop: tuple[str, ...] = DEFAULT_STOPS,
        seed: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[str]:
        """Generate completions for multiple prompts sequentially.

        For true batched GPU inference, the model's own ``generate`` method
        should be used; this is a convenience wrapper that calls
        :meth:`complete` for each prompt. It accepts the same *stop*, *seed*
        and *should_stop* arguments so a batch behaves like the individual
        calls it is made of — *seed* is applied once, before the first
        prompt, so the batch as a whole is reproducible.

        A *should_stop* that fires mid-batch ends it: the prompts already
        generated keep their text and the rest come back empty, rather than
        the batch ignoring the stop until it happens to finish.
        """
        if seed is not None:
            torch.manual_seed(seed)
        out: list[str] = []
        for p in prompts:
            if should_stop is not None and should_stop():
                out.extend([""] * (len(prompts) - len(out)))
                break
            out.append(self.complete(
                p, max_new_tokens=max_new_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, stop=stop, should_stop=should_stop,
            ))
        return out

    # ------------------------------------------------------------------
    # Memory management (in-object shortcuts to module-level functions)
    # ------------------------------------------------------------------

    def freeze_backbone(self, prefix: str = "embed_tokens") -> int:
        """Freeze parameters matching ``prefix``. Returns frozen count."""
        return freeze(self.model, [prefix])

    def unfreeze_backbone(self, prefix: str = "*") -> int:
        """Unfreeze parameters matching ``prefix``. Returns unfrozen count."""
        return unfreeze(self.model, [prefix])

    def memory_stats(self) -> ParamStats:
        """Return :class:`ParamStats` for the loaded model."""
        return parameter_stats(self.model)

    def vram(self) -> dict[str, float]:
        """Return CUDA VRAM usage in MB. Empty dict on CPU-only systems."""
        return vram_stats()

    # ------------------------------------------------------------------
    # Judge corpus shortcut
    # ------------------------------------------------------------------

    def build_judge_corpus(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        label_rule: Callable[[str, str], str] | None = None,
    ) -> JudgeCorpus:
        """Generate responses and return a :class:`JudgeCorpus` for reward model training."""
        return JudgeCorpus.from_oven(self, prompts, max_new_tokens=max_new_tokens,
                                      temperature=temperature, label_rule=label_rule)

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
        metrics: bool | TrainingMetrics = False,
    ) -> Path:
        """Continue-pretrain on a raw-text file.

        Args:
            dataset_path: Path to raw text file.
            out_dir: Snapshot output directory. Defaults to ``<model_dir>-trained``.
            metrics: Pass ``True`` to auto-create a :class:`TrainingMetrics`
                instance, or pass an existing one. The metrics object is
                available at ``oven.last_metrics`` after training.

        Returns:
            Path to the written snapshot directory.
        """
        dataset_path = Path(dataset_path)
        out = Path(out_dir) if out_dir else self.model_dir.parent / (self.model_dir.name + "-trained")
        out.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            torch.manual_seed(seed)

        self.model.train()
        bos_id: int | None = getattr(self.tokenizer, "bos_token_id", None) if self.tokenizer_kind == "hf" else None

        chunks = list(_iter_chunks(dataset_path, self.tokenizer, context_length, bos_id=bos_id))
        if not chunks:
            raise RuntimeError(f"dataset {dataset_path} produced no training chunks (context_length={context_length})")

        core = _unwrap_model(self.model)
        optimizer_kwargs: dict[str, Any] = {"betas": (0.9, 0.95), "weight_decay": weight_decay}
        if optimizer_class is not None:
            import inspect
            sig = inspect.signature(optimizer_class).parameters
            optimizer_kwargs["lr" if "lr" in sig else "peak_lr"] = lr
            if "grad_clip" in sig:
                optimizer_kwargs["grad_clip"] = grad_clip
            opt = optimizer_class(core.parameters(), **optimizer_kwargs)
        else:
            opt = torch.optim.AdamW(core.parameters(), lr=lr, **optimizer_kwargs)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))

        abb = None
        if use_turbo_abbicus:
            from hypernix.training.abbicus import TurboAbbicus, TurboAbbicusConfig
            abb = TurboAbbicus(TurboAbbicusConfig(base_context_length=context_length, hard_cap=untrained_max_context))
        elif use_abbicus:
            from hypernix.training.abbicus import Abbicus, AbbicusConfig
            abb = Abbicus(AbbicusConfig(base_context_length=context_length))

        stml_mgr = None
        if use_stml:
            from hypernix.models.stml import STML
            stml_mgr = STML(trained_context=context_length, untrained_max_context=untrained_max_context,
                            segment_length=segment_length, regulator=abb)

        _metrics: TrainingMetrics | None = None
        if metrics is True:
            _metrics = TrainingMetrics()
        elif isinstance(metrics, TrainingMetrics):
            _metrics = metrics

        for step in range(1, steps + 1):
            batch_tensors = [chunks[(step * batch_size + i) % len(chunks)] for i in range(batch_size)]
            batch_stacked = torch.stack(batch_tensors).to(self.device)
            inputs, labels = batch_stacked[:, :-1], batch_stacked[:, 1:]

            if stml_mgr is not None:
                if abb is not None:
                    abb.step(step)
                bd = stml_mgr.regulate({"input_ids": inputs, "labels": labels})
                inputs, labels = bd["input_ids"], bd["labels"]
            elif abb:
                abb.step(step)
                bd = abb.regulate({"input_ids": inputs, "labels": labels})
                inputs, labels = bd["input_ids"], bd["labels"]

            out_dict = self.model(inputs, labels=labels)
            loss = out_dict["loss"]

            if compute_framework:
                compute_framework.backward(loss)
                compute_framework.step(opt)
                opt.zero_grad(set_to_none=True)
            else:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt.step()

            sched.step()
            current_lr = sched.get_last_lr()[0]

            if not quiet and step % log_every == 0:
                ppl = math.exp(min(loss.item(), 20))
                ctx_len = inputs.shape[1] if hasattr(inputs, "shape") else context_length
                print(
                    f"[neo_oven.train] arch={self.model.config.model_type} "
                    f"step {step}/{steps}  loss={loss.item():.4f}  ppl={ppl:.2f}  "
                    f"lr={current_lr:.2e}  ctx={ctx_len}"
                )

            if _metrics and step % log_every == 0:
                _metrics.on_step(step=step, loss=loss.item(), lr=current_lr,
                                  ppl=math.exp(min(loss.item(), 20)))

            if save_every and step % save_every == 0:
                save_snapshot(self.model, out, tokenizer_source=self.model_dir)

        save_snapshot(self.model, out, tokenizer_source=self.model_dir)
        self.model.eval()
        self.model_dir = out
        if _metrics is not None:
            self.last_metrics = _metrics  # type: ignore[attr-defined]
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pt(self, out_path: Path | str) -> Path:
        """Save a self-contained torch.load()-able bundle."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "neo_oven_format_version": 1,
            "hypernix_format_version": 1,  # back-compat
            "config": self.model.config.to_dict(),
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "source_model_dir": str(self.model_dir),
        }
        torch.save(bundle, out_path)
        return out_path


# ---------------------------------------------------------------------------
# Top-level functional API (drop-in replacements for old_oven functions)
# ---------------------------------------------------------------------------

def preheat_brewed(
    path: Path | str,
    *,
    tokenizer_source: Path | str | None = None,
    device: str | None = None,
    dtype: str = "float32",
) -> NeoOven:
    """Load a ``hyperNix0x-v2`` model into a :class:`NeoOven`.

    Accepts either layout :mod:`hypernix.training.brewer` writes: a
    ``.pt`` checkpoint holding ``{"config", "model_state_dict"}``, or a
    save directory with a ``config.json``.

    The model comes back wrapped by
    :mod:`hypernix.models.brewer_adapter`, which is what lets every
    NeoOven method reach it: ``BrewerModel.forward`` returns a bare
    logits tensor and takes ``attn_mask`` in the position NeoOven passes
    ``labels``, so calling it unwrapped does not fail -- it trains the
    model into noise. See that module for why this is a wrapper rather
    than a few checks at the call sites.

    A brewed model has no tokenizer of its own; ``tokenizer_source``
    points at one. Without it the oven loads with the tokenizer NeoOven
    falls back to, which is fine for measuring shapes and wrong for
    generating text -- so it says so.
    """
    from . import brewer_adapter

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tdtype = _dtype_from_str(dtype)

    model, cfg = brewer_adapter.load(path)
    model.to(dev, dtype=tdtype)
    model.eval()

    source = Path(tokenizer_source) if tokenizer_source else Path(path)
    try:
        tok, kind = _load_tokenizer(source)
    except Exception:  # noqa: BLE001 - a missing tokenizer is not fatal here
        logger.warning(
            "neo_oven: no tokenizer at %s. The model will load and its shapes "
            "are usable, but generated text will be nonsense until you pass "
            "tokenizer_source= pointing at one.", source,
        )
        tok, kind = None, "none"

    return NeoOven(model=model, tokenizer=tok, tokenizer_kind=kind,
                   device=dev, dtype=tdtype, model_dir=Path(path),
                   repo_id=getattr(cfg, "name", brewer_adapter.ARCH_NAME))


def new_brewed(
    out_dir: Path | str,
    *,
    preset: str = "small",
    device: str | None = None,
    dtype: str = "float32",
    tokenizer_source: Path | str | None = None,
) -> NeoOven:
    """A fresh, untrained ``hyperNix0x-v2`` model in a :class:`NeoOven`.

    The counterpart to :func:`new_oven` for this architecture. Kept
    separate rather than folded into ``ARCH_PRESETS`` because a
    hyperNix0x-v2 model is not a :class:`HyperNixConfig` with different
    numbers -- it is a different module with its own config type, and
    listing it as a preset would make ``new_oven`` claim it can build
    something it cannot.
    """
    from ..training import brewer
    from . import brewer_adapter

    cfg = brewer_adapter.preset_config(preset)
    model = brewer_adapter.wrap(brewer.BrewerModel(cfg))

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tdtype = _dtype_from_str(dtype)
    model.to(dev, dtype=tdtype)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    torch.save(model.inner.state_dict(), out / "model.pt")

    tok, kind = (None, "none")
    if tokenizer_source is not None:
        try:
            tok, kind = _load_tokenizer(Path(tokenizer_source))
        except Exception:  # noqa: BLE001
            logger.warning("neo_oven: could not load %s", tokenizer_source)

    return NeoOven(model=model, tokenizer=tok, tokenizer_kind=kind,
                   device=dev, dtype=tdtype, model_dir=out, repo_id=cfg.name)


def preheat(
    repo_id: str = "ray0rf1re/hyper-nix.1",
    *,
    local_dir: Path | str | None = None,
    revision: str | None = None,
    token: str | None = None,
    device: str | None = None,
    dtype: str = "float32",
    quiet: bool = False,
) -> NeoOven:
    """Download (if necessary) + load a HyperNix snapshot into a :class:`NeoOven`.

    Drop-in replacement for ``old_oven.preheat``; returns a :class:`NeoOven`
    instead of a ``CodeOven`` — the API is fully compatible.
    """
    try:
        from hypernix.system.utils import warn_hyper_nix_2
        warn_hyper_nix_2(repo_id)
    except Exception:
        pass

    # A brewed model pointed at with local_dir= would otherwise fail
    # verify_snapshot and then be re-downloaded from the Hub, which is
    # both wrong and slow. Recognised by looking at the file rather than
    # its extension -- see brewer_adapter.is_brewer_checkpoint.
    if local_dir is not None:
        from . import brewer_adapter

        if brewer_adapter.is_brewer_checkpoint(local_dir):
            return preheat_brewed(local_dir, device=device, dtype=dtype)

    if local_dir is not None:
        ld = Path(local_dir)
        try:
            if ld.exists():
                verify_snapshot(ld)
                path = ld
            else:
                raise FileNotFoundError(ld)
        except FileNotFoundError:
            path = download_model(repo_id=repo_id, revision=revision, local_dir=str(ld), token=token, quiet=quiet)
    else:
        path = download_model(repo_id=repo_id, revision=revision, token=token, quiet=quiet)

    model, _cfg = load_snapshot(path)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tdtype = _dtype_from_str(dtype)
    model.to(dev, dtype=tdtype)
    model.eval()

    tok, kind = _load_tokenizer(path)
    return NeoOven(model=model, tokenizer=tok, tokenizer_kind=kind,
                   device=dev, dtype=tdtype, model_dir=path, repo_id=repo_id)


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
) -> NeoOven:
    """Create a fresh untrained :class:`NeoOven` from an architecture preset.

    Drop-in replacement for ``old_oven.new_oven``; returns a :class:`NeoOven`.
    """
    if arch not in ARCH_PRESETS:
        raise ValueError(f"Unknown arch {arch!r}; choose from {sorted(ARCH_PRESETS)}")
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
    path = init_from_scratch(out_dir, cfg, tokenizer_source=tokenizer_source, seed=seed)
    model, _ = load_snapshot(path)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tdtype = _dtype_from_str(dtype)
    model.to(dev, dtype=tdtype)
    model.eval()
    tok, kind = _load_tokenizer(path)
    return NeoOven(model=model, tokenizer=tok, tokenizer_kind=kind,
                   device=dev, dtype=tdtype, model_dir=path)


def bake_code(source: NeoOven | Path | str, prompt: str, **kwargs: Any) -> str:
    """Convenience: accepts a preheated NeoOven or a snapshot path/URL."""
    oven = source if isinstance(source, NeoOven) else preheat(local_dir=str(source))
    return oven.complete(prompt, **kwargs)


def fill_middle(source: NeoOven | Path | str, prefix: str, suffix: str, **kwargs: Any) -> str:
    """Convenience FIM API — accepts a preheated NeoOven or a snapshot path."""
    oven = source if isinstance(source, NeoOven) else preheat(local_dir=str(source))
    return oven.fill(prefix, suffix, **kwargs)


def load_pt(pt_path: Path | str, *, device: str | None = None) -> NeoOven:
    """Rehydrate a NeoOven from a ``save_pt`` bundle.

    Also works with bundles written by the old ``CodeOven.save_pt``
    (``hypernix_format_version=1``).
    """
    pt_path = Path(pt_path)
    bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = HyperNixConfig.from_dict(bundle["config"])
    model = HyperNixModel(cfg)
    model.load_state_dict(bundle["state_dict"], strict=False)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    model.eval()
    src_dir = Path(bundle.get("source_model_dir") or "")
    tok, kind = _load_tokenizer(src_dir) if src_dir.exists() else _load_tokenizer(Path("."))
    return NeoOven(
        model=model, tokenizer=tok, tokenizer_kind=kind,
        device=dev, dtype=next(model.parameters()).dtype,
        model_dir=src_dir if src_dir.exists() else pt_path.parent,
    )


# ---------------------------------------------------------------------------
# Backward-compatibility shims (so code that imported from old_oven/old_fridge
# via neo_oven still works)
# ---------------------------------------------------------------------------

# Plot shims (new_fridge compat)
def plot_loss_curve(pairs: Sequence[tuple[int, float]], out_path: Path | str, *, title: str = "Training loss") -> Path:
    """Alias for new_fridge.plot_loss_curve — kept for compatibility."""
    return _plot_pairs(pairs, out_path, title=title)


def plot_score_distribution(scores: Sequence[float], out_path: Path | str, *, title: str = "Judge score distribution", bins: int = 20) -> Path:
    """Alias for new_fridge.plot_score_distribution — kept for compatibility."""
    return _plot_histogram(scores, out_path, title=title, bins=bins)


def plot_round_losses(rounds: Sequence[Sequence[tuple[int, float]]], out_path: Path | str, *, title: str = "Multi-round training loss", round_labels: Sequence[str] | None = None) -> Path:
    """Alias for new_fridge.plot_round_losses — kept for compatibility."""
    return _plot_multi_round(rounds, out_path, title=title, round_labels=round_labels)


# Judge corpus shim (mediocre_fridge compat)
def synthesize_judge_corpus(n: int, out_path: Path | str, *, seed: int | None = 0, good_ratio: float = 0.5) -> Path:
    """Alias for the old mediocre_fridge.synthesize_judge_corpus.

    Uses the default 16-item seed from mediocre_fridge for compatibility.
    Prefer :class:`JudgeCorpus` for new code.
    """
    _DEFAULT_PAIRS: list[tuple[str, str]] = [
        ("What is 2 + 2?", "4"), ("Capital of France?", "Paris"),
        ("Name a primary color.", "Red"), ("What gas do plants absorb?", "Carbon dioxide"),
        ("Largest planet in our solar system?", "Jupiter"), ("Who wrote Hamlet?", "William Shakespeare"),
        ("Speed of light in m/s (approx)?", "299792458"), ("What is H2O commonly called?", "Water"),
        ("5 * 6 = ?", "30"), ("Smallest prime number?", "2"),
        ("Sum of angles in a triangle (deg)?", "180"), ("Boiling point of water at sea level (°C)?", "100"),
        ("Language of Guido van Rossum's first release?", "Python"), ("How many continents?", "7"),
        ("What year did WW2 end?", "1945"), ("Protein-building cell structure?", "Ribosome"),
    ]
    corpus = JudgeCorpus.from_pairs(_DEFAULT_PAIRS, n, seed=seed, good_ratio=good_ratio)
    return corpus.save(out_path, fmt="text")
