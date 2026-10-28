"""scriptgen.recipes — starting points, so nobody begins at 62 blank fields.

A recipe is a name, one line on what it is for, and the values that
differ from the defaults. Applying one starts from the defaults and lays
the recipe over them, so two recipes never mix; what is changed after
that is the user's. ``hnx scriptgen --recipe NAME`` and the GUI's
"Start from" picker read this table.

The numbers are the ordinary choices for each case, not tuned results:
a 48M-parameter model wants a higher learning rate than a 7B one, LoRA
wants a higher one again, and a GTX 1080 wants FP32 compute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .params import ALL_PARAMS, defaults

__all__ = ["Recipe", "RECIPES", "get_recipe", "apply_recipe"]


@dataclass(frozen=True)
class Recipe:
    key: str
    title: str
    description: str
    values: dict[str, Any] = field(default_factory=dict)


RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "hypernix-finetune", "Fine-tune HyperNix.3.1-mini",
        "The newest HyperNix model (48.7M, 512 tokens) on your own text. Fits any GPU, "
        "and trains on a CPU too.",
        {
            "model_id": "Compactbot/hypernix-3.1-mini", "sequence_length": 512,
            "learning_rate": 3e-4, "warmup_ratio": 0.05, "epochs": 3.0,
            "micro_batch": 8, "gradient_accumulation": 4, "optimizer": "PressureCookerV5",
            "output_dir": "./runs/hypernix-3.1-mini-ft",
        },
    ),
    Recipe(
        "quick-test", "Smoke test on the CPU",
        "Twenty steps on a few hundred examples, to see the whole script run before "
        "paying for a GPU.",
        {
            "device": "cpu", "precision": "fp32", "pascal_autotune": False,
            "sequence_length": 128, "max_samples": 400, "max_steps": 20, "epochs": 1.0,
            "micro_batch": 2, "gradient_accumulation": 1, "learning_rate": 3e-4,
            "optimizer": "AdamW", "eval_steps": 10, "save_steps": 20, "logging_steps": 2,
            "val_split": 0.1, "output_dir": "./runs/smoke-test",
        },
    ),
    Recipe(
        "lora-low-vram", "LoRA on an 8 GB card",
        "A larger transformers model with 4-bit weights, LoRA adapters and gradient "
        "checkpointing. Set the base model to the one you want.",
        {
            "use_lora": True, "lora_rank": 16, "lora_alpha": 32, "quantization": "NF4",
            "gradient_checkpointing": True, "learning_rate": 2e-4, "sequence_length": 2048,
            "micro_batch": 1, "gradient_accumulation": 16, "optimizer": "AdamW",
            "precision": "auto", "output_dir": "./runs/lora",
        },
    ),
    Recipe(
        "pascal-1080", "GTX 1080 / Pascal",
        "FP16 storage with FP32 compute, the Pascal-tuned optimiser, and the auto-tuner "
        "choosing the batch.",
        {
            "precision": "fp16_fp32_compute", "pascal_autotune": True,
            "optimizer": "Agedcookerv5", "gradient_checkpointing": True, "compile_model": False,
            "output_dir": "./runs/pascal",
        },
    ),
    Recipe(
        "continued-pretraining", "Continued pretraining",
        "More pretraining on raw text: a long warmup-stable-decay schedule, packed "
        "sequences, and frequent saves so the run can be resumed.",
        {
            "lr_scheduler": "wsd", "learning_rate": 1e-4, "warmup_ratio": 0.01, "epochs": 1.0,
            "packing": True, "save_steps": 1000, "save_total_limit": 5, "resume": True,
            "loss_function": "z_loss_ce", "output_dir": "./runs/pretrain",
        },
    ),
    Recipe(
        "lr-sweep", "Find a learning rate",
        "Four short runs across 1e-5 to 1e-3, then a full run at the best one.",
        {
            "enable_sweep": True, "lr_range": (1e-5, 1e-3), "sweep_trials": 4,
            "sweep_steps": 100, "val_split": 0.05, "output_dir": "./runs/sweep",
        },
    ),
)


def get_recipe(key: str) -> Recipe:
    wanted = key.strip().lower()
    for recipe in RECIPES:
        if recipe.key == wanted:
            return recipe
    names = ", ".join(r.key for r in RECIPES)
    raise KeyError(f"No recipe {key!r}. Recipes: {names}")


def apply_recipe(key: str) -> dict[str, Any]:
    """The defaults with *key*'s values laid over them. Keys a recipe names
    that are not parameters are dropped, so an old recipe never breaks a
    newer form."""
    values = defaults()
    values.update({k: v for k, v in get_recipe(key).values.items() if k in ALL_PARAMS})
    return values
