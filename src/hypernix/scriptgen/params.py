"""scriptgen.params — the parameter schema behind the GUI and the script.

One definition per parameter, used three times: to build the widget, to
validate what someone typed, and to emit the training script. Defining
them separately is how a GUI ends up offering a learning rate the
generated script does not accept.

Modelled on the nano-nano 5.1 trainer's surface — learning rates, warmup
ratios, epoch and step controls, batch and micro-batch sizing, gradient
accumulation, loss functions, optimisers — plus the HyperNix-specific
parts: the Pressure Cooker family, the 6-bit momentum modes, and the
Pascal auto-tuner.

Ranges carry reasons
--------------------
Every numeric parameter has a ``hint`` saying what the range means and
what happens outside it, because a slider from 0 to 1 with no context is
a slider people set to 0.5. :meth:`Param.validate` returns advice, not
just a verdict — a learning rate of 1e-2 is *allowed* and almost
certainly wrong, and saying so is more useful than refusing it.

0.72.8rc1 adds a seventh group (LoRA adapters), the data, device and
reporting options a runnable loop needs, and :func:`search` /
:func:`suggest`, which the GUI's filter box and the CLI's "did you mean"
share.
"""
from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "ParamKind",
    "Param",
    "ParamGroup",
    "GROUPS",
    "ALL_PARAMS",
    "get_param",
    "defaults",
    "validate_all",
    "OPTIMIZERS",
    "LOSS_FUNCTIONS",
    "SCHEDULERS",
    "PRECISIONS",
    "DEVICES",
    "REPORTERS",
    "search",
    "suggest",
]


class ParamKind(StrEnum):
    TEXT = "text"
    PATH = "path"
    INT = "int"
    FLOAT = "float"
    LOG_FLOAT = "log_float"      # a slider that moves in decades
    CHOICE = "choice"
    TOGGLE = "toggle"
    RANGE = "range"              # a dual slider: (low, high)
    MULTILINE = "multiline"


#: Optimisers the generated script can import. The Pressure Cooker
#: family is HyperNix's own; the rest are what people expect to find.
OPTIMIZERS: tuple[tuple[str, str], ...] = (
    ("PressureCookerV5", "ORCP core; quantised momentum + factored curvature, ~1.7x SGD memory"),
    ("PressureCookerV5Plus", "ORCP-Ultra; a few extra per-tensor rows, ~2.1x SGD memory"),
    ("PressureCookerV5S", "V5 slimmed for small VRAM; the same update rule, fewer buffers"),
    ("PressureCookerV6", "SSTM core with fused multi-tensor steps"),
    ("PressureCookerV6V", "V6 plus CUDA graph capture and torch.compile"),
    ("Agedcookerv5", "V5 tuned for Pascal (sm_61) — no tensor cores assumed"),
    ("ULTRAagedcookerv5", "V5Plus tuned for Pascal"),
    ("AdamW", "The baseline. ~3x SGD memory"),
    ("Adafactor", "Factored second moment; the memory-frugal classic"),
    ("Lion", "Sign-based; less memory than AdamW, wants a lower LR"),
    ("SGD", "Momentum SGD"),
)

LOSS_FUNCTIONS: tuple[tuple[str, str], ...] = (
    ("cross_entropy", "Standard next-token loss"),
    ("label_smoothed_ce", "Cross-entropy with label smoothing; steadier on small data"),
    ("focal", "Down-weights easy tokens; for heavily imbalanced targets"),
    ("z_loss_ce", "Cross-entropy plus a logit-magnitude penalty; stabilises long runs"),
    ("kl_distill", "KL against a teacher's logits — distillation"),
    ("mse", "For regression heads, not language modelling"),
)

SCHEDULERS: tuple[tuple[str, str], ...] = (
    ("cosine", "Decay to near zero on a cosine curve. The usual choice"),
    ("linear", "Straight-line decay"),
    ("constant_with_warmup", "Warm up, then hold. For short runs and debugging"),
    ("cosine_with_restarts", "Cosine, restarted periodically"),
    ("polynomial", "Polynomial decay to an end LR"),
    ("wsd", "Warmup-stable-decay: hold, then decay late. Good for resumable runs"),
)

DEVICES: tuple[tuple[str, str], ...] = (
    ("auto", "CUDA if there is a GPU, then Apple's MPS, then the CPU"),
    ("cuda", "An NVIDIA GPU"),
    ("mps", "Apple silicon"),
    ("cpu", "The CPU: slow, but always there, and right for a smoke test"),
)

REPORTERS: tuple[tuple[str, str], ...] = (
    ("jsonl", "metrics.jsonl in the output directory; nothing to install"),
    ("tensorboard", "TensorBoard event files (needs tensorboard)"),
    ("wandb", "Weights & Biases (needs wandb and a login)"),
    ("none", "Only the log lines"),
)

PRECISIONS: tuple[tuple[str, str], ...] = (
    ("auto", "Let the Pascal tuner decide (see hypernix.system.pascal)"),
    ("fp32", "Full precision. Slowest, never overflows"),
    ("fp16", "Half. Needs loss scaling; the usual source of NaN"),
    ("bf16", "Brain float. Ampere and newer only"),
    ("fp16_fp32_compute", "FP16 storage, FP32 compute — the right answer on a GTX 1080"),
)


@dataclass(frozen=True)
class Param:
    """One parameter: how to show it, how to check it, how to emit it."""

    name: str
    label: str
    kind: ParamKind
    default: Any
    hint: str = ""
    #: Numeric bounds. For RANGE these bound both ends.
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, str], ...] = ()
    #: The variable name used in the generated script. Defaults to
    #: ``name``; set when the script's spelling differs from the UI's.
    emit_as: str = ""
    #: Shown only when this other parameter is on (a toggle), or, as
    #: ``"name=a|b"``, set to one of those values. Keeps the form dense
    #: without hiding things that matter.
    depends_on: str = ""
    advanced: bool = False

    @property
    def script_name(self) -> str:
        return self.emit_as or self.name

    def shown_with(self, values: dict[str, Any]) -> bool:
        """Whether this parameter applies, given the others' *values*."""
        if not self.depends_on:
            return True
        name, _, wanted = self.depends_on.partition("=")
        gate = values.get(name)
        if wanted:
            return str(gate) in wanted.split("|")
        return bool(gate)

    def coerce(self, value: Any) -> Any:
        """Turn a widget's string into the parameter's real type."""
        if self.kind in (ParamKind.INT,):
            return int(float(value))
        if self.kind in (ParamKind.FLOAT, ParamKind.LOG_FLOAT):
            return float(value)
        if self.kind is ParamKind.TOGGLE:
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if self.kind is ParamKind.RANGE:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return (float(value[0]), float(value[1]))
            raise ValueError(f"{self.name} needs a (low, high) pair")
        return value

    def validate(self, value: Any) -> tuple[bool, str]:
        """``(ok, message)``. A non-empty message on a valid value is advice.

        The distinction matters: 1e-2 is a legal learning rate and is
        almost certainly a mistake. Refusing it would be wrong and saying
        nothing would be unhelpful, so it is accepted with a warning.
        """
        try:
            coerced = self.coerce(value)
        except (TypeError, ValueError) as exc:
            return False, f"{self.label}: {exc}"

        if self.kind is ParamKind.RANGE:
            low, high = coerced
            if low > high:
                return False, f"{self.label}: low ({low}) is above high ({high})"
            coerced = high      # bound-check the wider end

        if isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
            if math.isnan(coerced) or math.isinf(coerced):
                return False, f"{self.label}: must be a finite number"
            if self.minimum is not None and coerced < self.minimum:
                return False, f"{self.label}: must be at least {self.minimum}"
            if self.maximum is not None and coerced > self.maximum:
                return False, f"{self.label}: must be at most {self.maximum}"

        if self.kind is ParamKind.CHOICE and self.choices:
            valid = [c[0] for c in self.choices]
            if str(value) not in valid:
                return False, f"{self.label}: must be one of {', '.join(valid)}"

        return True, self._advice(coerced)

    def _advice(self, value: Any) -> str:
        """Warnings for values that are legal and probably wrong."""
        if self.name == "learning_rate" and isinstance(value, float):
            if value > 1e-3:
                return (
                    f"{value:.1e} is high for fine-tuning a language model. Above ~1e-3 the "
                    "loss usually diverges in the first few hundred steps."
                )
            if value < 1e-6:
                return f"{value:.1e} is very low; the run may not move at all."
        if self.name == "warmup_ratio" and isinstance(value, float) and value > 0.2:
            return (
                f"{value:.0%} of the run spent warming up is unusually long — it is normally "
                "under 10%."
            )
        if self.name == "gradient_accumulation" and isinstance(value, int) and value > 64:
            return (
                f"{value} accumulation steps means {value} forward/backward passes per "
                "optimiser step; check the effective batch is what you meant."
            )
        return ""


@dataclass
class ParamGroup:
    """A titled section of the form."""

    key: str
    title: str
    description: str
    params: list[Param] = field(default_factory=list)


GROUPS: tuple[ParamGroup, ...] = (
    ParamGroup(
        "model", "Model and data",
        "What is being trained, and on what.",
        [
            Param("model_id", "Base model", ParamKind.TEXT, "Compactbot/hypernix-3.1-mini",
                  hint="A Hugging Face repo id or a local folder. A HyperNix model is loaded "
                       "through hypernix, anything else through transformers."),
            Param("dataset", "Dataset", ParamKind.PATH, "",
                  hint="A Hugging Face dataset id, or a local .jsonl, .json or .txt file."),
            Param("output_dir", "Output directory", ParamKind.PATH, "./runs/hypernix",
                  hint="Checkpoints, logs and metrics land here."),
            Param("sequence_length", "Sequence length", ParamKind.INT, 512,
                  minimum=32, maximum=131072, step=32,
                  hint="Tokens per training example; capped at the model's own context. "
                       "Attention memory grows with its square: the first thing to lower "
                       "when you run out of VRAM."),
            Param("packing", "Pack examples", ParamKind.TOGGLE, True,
                  hint="Join short examples into full-length blocks, so no compute is spent "
                       "on padding."),
            Param("val_split", "Held-out fraction", ParamKind.FLOAT, 0.02,
                  minimum=0.0, maximum=0.5, step=0.01,
                  hint="Kept back for evaluation. 0 evaluates on nothing, and early "
                       "stopping has nothing to watch."),
            Param("max_samples", "Max examples", ParamKind.INT, 0,
                  minimum=0, maximum=100_000_000,
                  hint="0 uses the whole dataset. A few hundred makes a quick test run."),
            Param("dataset_split", "Split", ParamKind.TEXT, "train", advanced=True),
            Param("text_field", "Text field", ParamKind.TEXT, "text", advanced=True,
                  hint="Which column of the dataset holds the text. Chat data with a "
                       "'messages' column is flattened to text."),
            Param("trust_remote_code", "Trust remote code", ParamKind.TOGGLE, False,
                  advanced=True,
                  hint="Let transformers run code shipped in the model repo. Only for repos "
                       "you trust."),
        ],
    ),
    ParamGroup(
        "schedule", "Learning rate and schedule",
        "The parameters that decide whether the run converges at all.",
        [
            Param("learning_rate", "Learning rate", ParamKind.LOG_FLOAT, 2e-5,
                  minimum=1e-7, maximum=1e-2,
                  hint="Log scale. 1e-5 to 5e-5 is the usual band for fine-tuning; full "
                       "pre-training runs higher."),
            Param("lr_range", "LR sweep range", ParamKind.RANGE, (1e-5, 5e-5),
                  minimum=1e-7, maximum=1e-2,
                  hint="Both ends of a hyperparameter sweep. Ignored unless sweeping is on.",
                  depends_on="enable_sweep"),
            Param("min_lr_ratio", "Minimum LR ratio", ParamKind.FLOAT, 0.1,
                  minimum=0.0, maximum=1.0, step=0.01,
                  hint="The floor the scheduler decays to, as a fraction of the peak."),
            Param("lr_scheduler", "Scheduler", ParamKind.CHOICE, "cosine", choices=SCHEDULERS),
            Param("warmup_ratio", "Warmup ratio", ParamKind.FLOAT, 0.03,
                  minimum=0.0, maximum=0.5, step=0.005,
                  hint="Fraction of total steps spent ramping the LR up. A ratio rather than "
                       "a step count, so it survives a change in dataset size."),
            Param("warmup_steps", "Warmup steps (override)", ParamKind.INT, 0,
                  minimum=0, maximum=100000, advanced=True,
                  hint="Non-zero overrides the ratio. Use when resuming a run mid-schedule."),
        ],
    ),
    ParamGroup(
        "batch", "Epochs and batching",
        "How much data, in what size pieces.",
        [
            Param("epochs", "Epochs", ParamKind.FLOAT, 3.0,
                  minimum=0.01, maximum=1000.0, step=0.25,
                  hint="Fractional epochs are allowed; 0.5 is half a pass."),
            Param("max_steps", "Max steps (override)", ParamKind.INT, 0,
                  minimum=0, maximum=10_000_000,
                  hint="Non-zero stops at this many optimiser steps regardless of epochs."),
            Param("micro_batch", "Micro-batch", ParamKind.INT, 1,
                  minimum=1, maximum=512,
                  hint="Sequences per forward pass. Bounded by VRAM."),
            Param("gradient_accumulation", "Gradient accumulation", ParamKind.INT, 16,
                  minimum=1, maximum=1024,
                  hint="Effective batch = micro-batch x this. Costs time, not memory."),
            Param("eval_batch", "Eval batch", ParamKind.INT, 1,
                  minimum=1, maximum=512, advanced=True),
            Param("shuffle", "Shuffle", ParamKind.TOGGLE, True),
            Param("drop_last", "Drop last partial batch", ParamKind.TOGGLE, True, advanced=True),
        ],
    ),
    ParamGroup(
        "optim", "Optimiser and loss",
        "The update rule and what it is minimising.",
        [
            Param("optimizer", "Optimiser", ParamKind.CHOICE, "PressureCookerV5",
                  choices=OPTIMIZERS),
            Param("loss_function", "Loss", ParamKind.CHOICE, "cross_entropy",
                  choices=LOSS_FUNCTIONS),
            Param("label_smoothing", "Label smoothing", ParamKind.FLOAT, 0.0,
                  minimum=0.0, maximum=0.5, step=0.01, depends_on="loss_function=label_smoothed_ce"),
            Param("weight_decay", "Weight decay", ParamKind.LOG_FLOAT, 0.01,
                  minimum=1e-6, maximum=1.0),
            Param("max_grad_norm", "Gradient clipping", ParamKind.FLOAT, 1.0,
                  minimum=0.0, maximum=100.0, step=0.1,
                  hint="0 disables clipping. 1.0 is the near-universal default."),
            Param("beta1", "Beta 1", ParamKind.FLOAT, 0.9,
                  minimum=0.0, maximum=0.999, step=0.001, advanced=True),
            Param("beta2", "Beta 2", ParamKind.FLOAT, 0.95,
                  minimum=0.0, maximum=0.9999, step=0.0001, advanced=True),
            Param("eps", "Epsilon", ParamKind.LOG_FLOAT, 1e-8,
                  minimum=1e-12, maximum=1e-4, advanced=True),
            Param("six_bit_mode", "6-bit momentum", ParamKind.CHOICE, "off",
                  choices=(
                      ("off", "Full int8 momentum, as before"),
                      ("packed", "4 values in 3 bytes — 6.0 bits, most shift work"),
                      ("aligned", "1 value per byte — wastes 25%, fastest unpack"),
                      ("hybrid", "Packed lanes, wider per-group scale"),
                  ),
                  hint="Pressure Cooker v5/v5+/v5s/v6 only. Trades optimiser memory for "
                       "pack/unpack time."),
        ],
    ),
    ParamGroup(
        "adapters", "LoRA adapters",
        "Train small low-rank adapters instead of every weight.",
        [
            Param("use_lora", "Use LoRA", ParamKind.TOGGLE, False,
                  hint="Freeze the model and train small adapters beside it: a fraction of "
                       "the memory and the optimiser state. Works on HyperNix and "
                       "transformers models alike, with nothing extra to install."),
            Param("lora_rank", "Rank", ParamKind.INT, 16, minimum=1, maximum=512,
                  depends_on="use_lora",
                  hint="Width of each adapter. 8-32 is the usual band; higher learns more "
                       "and costs more."),
            Param("lora_alpha", "Alpha", ParamKind.INT, 32, minimum=1, maximum=1024,
                  depends_on="use_lora",
                  hint="Adapter scale is alpha / rank; twice the rank is the common choice."),
            Param("lora_dropout", "Dropout", ParamKind.FLOAT, 0.05, minimum=0.0, maximum=0.5,
                  step=0.01, depends_on="use_lora"),
            Param("lora_targets", "Target layers", ParamKind.TEXT, "q_proj,k_proj,v_proj,o_proj",
                  depends_on="use_lora",
                  hint="Comma-separated layer names. Add gate_proj,up_proj,down_proj to adapt "
                       "the feed-forward layers too."),
            Param("merge_lora", "Merge when done", ParamKind.TOGGLE, True, depends_on="use_lora",
                  hint="Fold the adapters into the weights before the final save, so the "
                       "result loads like any model."),
        ],
    ),
    ParamGroup(
        "precision", "Precision and hardware",
        "How numbers are stored and computed, and on what.",
        [
            Param("device", "Device", ParamKind.CHOICE, "auto", choices=DEVICES),
            Param("precision", "Precision", ParamKind.CHOICE, "auto", choices=PRECISIONS),
            Param("pascal_autotune", "Pascal auto-tune", ParamKind.TOGGLE, True,
                  hint="Detect an sm_61 card and set dtype, batch and attention for it. "
                       "See hypernix.system.pascal."),
            Param("loss_scale", "Initial loss scale", ParamKind.LOG_FLOAT, 65536.0,
                  minimum=1.0, maximum=2 ** 24, depends_on="precision=fp16|auto", advanced=True,
                  hint="FP16 only. Halved on overflow, doubled after a quiet period."),
            Param("gradient_checkpointing", "Gradient checkpointing", ParamKind.TOGGLE, False,
                  hint="Recompute activations instead of storing them: ~30% slower, far less "
                       "memory."),
            Param("quantization", "Weight quantisation", ParamKind.CHOICE, "none",
                  choices=(
                      ("none", "Full weights"),
                      ("NF4", "4-bit NormalFloat — QLoRA"),
                      ("INT8", "8-bit with per-channel scales"),
                      ("FP8", "8-bit float (needs sm_89+)"),
                      ("GPTQ", "One-shot second-order rounding (runs on Pascal)"),
                      ("AWQ", "Activation-aware"),
                  )),
            Param("compile_model", "torch.compile", ParamKind.TOGGLE, False, advanced=True,
                  hint="Fuses kernels on torch 2.x. No effect on Pascal's attention path."),
        ],
    ),
    ParamGroup(
        "run", "Checkpointing and logging",
        "What gets written, and how often.",
        [
            Param("save_steps", "Save every N steps", ParamKind.INT, 500,
                  minimum=0, maximum=1_000_000, hint="0 saves only at the end."),
            Param("eval_steps", "Evaluate every N steps", ParamKind.INT, 500,
                  minimum=0, maximum=1_000_000),
            Param("logging_steps", "Log every N steps", ParamKind.INT, 10,
                  minimum=1, maximum=10000),
            Param("save_total_limit", "Keep N checkpoints", ParamKind.INT, 3,
                  minimum=0, maximum=100,
                  hint="0 keeps all of them, which fills disks."),
            Param("seed", "Seed", ParamKind.INT, 42, minimum=0, maximum=2 ** 31 - 1),
            Param("resume", "Resume from checkpoint", ParamKind.TOGGLE, False),
            Param("report_to", "Metrics", ParamKind.CHOICE, "jsonl", choices=REPORTERS),
            Param("early_stopping_patience", "Early stopping patience", ParamKind.INT, 0,
                  minimum=0, maximum=1000,
                  hint="Stop after this many evaluations without a better held-out loss. "
                       "0 never stops early."),
            Param("stream_metrics", "Stream to the live view", ParamKind.TOGGLE, False,
                  hint="Publish loss, progress and hardware over hypernix's WebSocket live "
                       "view (hypernix.monitoring.livestream), on loopback."),
            Param("stream_port", "Live view port", ParamKind.INT, 8765, minimum=1024,
                  maximum=65535, depends_on="stream_metrics", advanced=True),
            Param("push_to_hub", "Upload when done", ParamKind.TOGGLE, False,
                  hint="Upload the final model to the Hugging Face Hub (needs a login)."),
            Param("hub_repo", "Hub repo", ParamKind.TEXT, "", depends_on="push_to_hub",
                  hint="owner/name to upload to; created private if it does not exist."),
            Param("enable_sweep", "Learning-rate sweep", ParamKind.TOGGLE, False,
                  hint="Try several learning rates across the sweep range with short runs, "
                       "then train fully with the best."),
            Param("sweep_trials", "Sweep trials", ParamKind.INT, 4, minimum=2, maximum=32,
                  depends_on="enable_sweep",
                  hint="Learning rates tried, spaced evenly on a log scale."),
            Param("sweep_steps", "Steps per trial", ParamKind.INT, 100, minimum=10,
                  maximum=100_000, depends_on="enable_sweep"),
            Param("training_loop", "Script body", ParamKind.CHOICE, "full",
                  choices=(
                      ("full", "A complete loop: loads the model and data, trains, evaluates, saves"),
                      ("skeleton", "The helpers only; bring your own model and data loading"),
                  )),
            Param("notes", "Notes", ParamKind.MULTILINE, "",
                  hint="Written into the script's header. What you were trying."),
        ],
    ),
)

ALL_PARAMS: dict[str, Param] = {p.name: p for group in GROUPS for p in group.params}


def get_param(name: str) -> Param:
    if name not in ALL_PARAMS:
        raise KeyError(f"Unknown parameter {name!r}")
    return ALL_PARAMS[name]


def defaults() -> dict[str, Any]:
    return {name: param.default for name, param in ALL_PARAMS.items()}


def validate_all(values: dict[str, Any]) -> tuple[list[str], list[str]]:
    """``(errors, warnings)`` across the whole form.

    Cross-parameter checks live here rather than on individual params,
    because "effective batch is 1" is a property of two fields and
    neither one is wrong on its own.
    """
    errors: list[str] = []
    warnings: list[str] = []

    for name, value in values.items():
        param = ALL_PARAMS.get(name)
        if param is None:
            continue
        ok, message = param.validate(value)
        if not ok:
            errors.append(message)
        elif message:
            warnings.append(message)

    micro = _as_int(values.get("micro_batch"), 1)
    accum = _as_int(values.get("gradient_accumulation"), 1)
    effective = micro * accum
    if effective == 1:
        warnings.append(
            "Effective batch is 1 (micro-batch 1, no accumulation). Gradient noise will be "
            "very high; raise accumulation, which costs time rather than memory."
        )
    elif effective > 4096:
        warnings.append(
            f"Effective batch is {effective}. Very large batches usually want a higher "
            "learning rate than the default."
        )

    if _as_int(values.get("max_steps"), 0) and float(values.get("epochs") or 0) > 1:
        warnings.append(
            "Both epochs and max_steps are set; max_steps wins and the run will stop early."
        )
    if values.get("six_bit_mode", "off") != "off":
        optimizer = str(values.get("optimizer", ""))
        if not optimizer.lower().startswith(("pressurecooker", "agedcooker", "ultraagedcooker")):
            errors.append(
                f"6-bit momentum is a Pressure Cooker feature; {optimizer} has no quantised "
                "momentum buffer to pack."
            )
    if str(values.get("precision")) == "bf16" and values.get("pascal_autotune"):
        warnings.append(
            "BF16 does not exist on Pascal. With auto-tune on, the tuner will override this."
        )
    if str(values.get("training_loop", "full")) == "full" and not str(values.get("dataset") or "").strip():
        warnings.append(
            "No dataset yet. The script will ask for one when it runs; name it here, or "
            "pass --dataset PATH then."
        )
    if values.get("push_to_hub") and not str(values.get("hub_repo") or "").strip():
        errors.append("Upload when done is on but there is no Hub repo to upload to.")
    if _as_int(values.get("early_stopping_patience"), 0) and not float(values.get("val_split") or 0):
        warnings.append(
            "Early stopping watches the held-out loss, and the held-out fraction is 0; "
            "it will never stop early."
        )
    if str(values.get("device")) == "cpu" and str(values.get("precision")) in ("fp16", "bf16"):
        warnings.append(
            f"{values.get('precision')} on the CPU is slower than fp32 on most CPUs; "
            "the script trains in fp32 there."
        )
    if values.get("use_lora") and not str(values.get("lora_targets") or "").strip():
        errors.append("LoRA is on but no target layers are named.")
    if values.get("use_lora") and str(values.get("training_loop")) == "skeleton":
        warnings.append("LoRA is applied by the full loop; the skeleton only carries the helper.")
    if _as_int(values.get("save_total_limit"), 0) == 0 and _as_int(values.get("save_steps"), 0):
        warnings.append(
            "Keeping every checkpoint with saves every "
            f"{_as_int(values.get('save_steps'), 0)} steps will fill the disk on a long run."
        )
    return errors, warnings


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def search(query: str) -> list[Param]:
    """Parameters whose name, label or hint mentions every word of *query*.
    The GUI's filter box and ``hnx scriptgen --list-params QUERY``."""
    words = [w for w in query.lower().split() if w]
    if not words:
        return list(ALL_PARAMS.values())
    found = []
    for group in GROUPS:
        for param in group.params:
            haystack = f"{param.name} {param.label} {param.hint} {group.title}".lower()
            if all(word in haystack for word in words):
                found.append(param)
    return found


def suggest(name: str, limit: int = 3) -> list[str]:
    """The parameter names closest to a misspelt *name*: ``lr`` finds
    learning_rate, ``batch`` finds micro_batch."""
    wanted = name.strip().lower().replace("-", "_")
    close = difflib.get_close_matches(wanted, list(ALL_PARAMS), n=limit, cutoff=0.5)
    for param in ALL_PARAMS.values():
        initials = "".join(part[0] for part in param.name.split("_"))
        if param.name not in close and (wanted == initials or wanted in param.name):
            close.append(param.name)
    return close[:limit]

