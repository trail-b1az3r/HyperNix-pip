"""instant_pot — one-shot end-to-end training pipeline.

Every other subsystem in hypernix is a named tool: the oven for
generation, the freezer for VRAM budgeting, the fridges for data
hygiene.  The instant pot is the counter-example: a single
``brew(recipe)`` call that does download → preheat → train → save,
optionally adding → convert → quantize on the end.

Use the instant pot when you already know the right defaults and
just want a trained GGUF out the other side.  Peel back to the
individual subsystems when the defaults don't fit.

``recipe`` is a plain dict so it can also be loaded from JSON / YAML:

    {
        "repo_id": "nix2.5",               # or "local_dir": "./snap"
        "dataset": "./corpus.txt",
        "out_dir": "./trained",
        "steps": 500,
        "batch_size": 1,
        "context_length": 1024,
        "lr": 3e-4,
        "device": "cuda",
        "dtype": "float16",
        "seed": 0,
        "freeze_embed": false,
        "quants": ["fp16", "q4_k_m"]       # optional; emits GGUFs too
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hypernix.models import old_oven
from hypernix.system import old_fridge


def brew(recipe: dict[str, Any]) -> Path:
    """Run the full pipeline described by ``recipe``; return the trained
    snapshot's path.

    Recognised keys:

    ``repo_id`` | ``local_dir``  — source snapshot.
    ``dataset``                   — raw-text training file (required).
    ``out_dir``                   — trained snapshot path (required).
    ``steps``, ``batch_size``, ``context_length``, ``lr``,
    ``log_every``, ``save_every``, ``seed``, ``quiet``,
    ``weight_decay``, ``grad_clip``
                                  — passed through to ``CodeOven.train``.
    ``device``, ``dtype``         — passed to ``old_oven.preheat``.
    ``freeze_embed``              — if True, freezes ``embed_tokens``
                                    before training (cheap fine-tune).
    ``quants``                    — optional list of quant aliases.
                                    When present, the trained snapshot
                                    is converted to GGUF and each quant
                                    is produced alongside.  Requires
                                    ``llama-quantize`` for k-quants.
    ``gguf_out_dir``              — where the GGUFs go (defaults to
                                    ``out_dir / gguf``).
    """
    dataset = recipe.get("dataset")
    if dataset is None:
        raise KeyError("instant_pot.brew: 'dataset' is required")
    out_dir = recipe.get("out_dir")
    if out_dir is None:
        raise KeyError("instant_pot.brew: 'out_dir' is required")
    # Pass 2 (v0.50): fail fast on a missing dataset file with a
    # message that points at the path the caller actually passed,
    # rather than letting train() raise a deeper "no chunks" error
    # twenty stack frames down.
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"instant_pot.brew: dataset {str(dataset_path)!r} does not "
            f"exist (resolved to {dataset_path.resolve()}).  Pass an "
            f"existing raw-text file via the recipe's 'dataset' key.",
        )

    # 1) Preheat the oven.
    oven = old_oven.preheat(
        repo_id=recipe.get("repo_id", "ray0rf1re/hyper-nix.1"),
        revision=recipe.get("revision"),
        local_dir=recipe.get("local_dir"),
        token=recipe.get("token"),
        device=recipe.get("device"),
        dtype=recipe.get("dtype", "float32"),
        quiet=recipe.get("quiet", True),
    )

    if recipe.get("freeze_embed"):
        old_fridge.freeze(oven.model, patterns=("embed_tokens",))

    compute_framework = None
    if recipe.get("distributed"):
        from hypernix.training.compute_framework import ComputeArch, ComputeFramework
        device_str = recipe.get("device", "cuda")
        arch = ComputeArch.CPU if device_str == "cpu" else ComputeArch.SINGLE_GPU
        compute_framework = ComputeFramework(
            local_rank=arch,
            use_ddp=recipe.get("use_ddp", False),
            use_fsdp=recipe.get("use_fsdp", False),
            use_lazy_suzan=recipe.get("use_lazy_suzan", False)
        )
        compute_framework.initialize()
        oven.model = compute_framework.prepare_model(oven.model)

    # 2) Train.
    # V4 by default. This was V3, and V3 is deprecated as of 0.72.6 — a
    # default HyperNix chose must not produce a warning the user has to
    # act on. `use_pressure_cooker_v3: true` still selects V3 on purpose.
    #
    # V4 only works here because it now accepts a bare `lr`: before, it
    # took a rate only inside a ScheduleConfig and raised TypeError the
    # moment `train()` handed it one.
    if recipe.get("use_pressure_cooker_v3", False):
        from hypernix.optimizers.pressure_cooker_v3 import PressureCookerV3
        opt_class = PressureCookerV3
    else:
        from hypernix.optimizers.pressure_cooker_v4 import PressureCookerV4
        opt_class = PressureCookerV4
    trained = oven.train(
        dataset, out_dir,
        steps=recipe.get("steps", 500),
        batch_size=recipe.get("batch_size", 1),
        context_length=recipe.get("context_length", 512),
        lr=recipe.get("lr", 3e-4),
        weight_decay=recipe.get("weight_decay", 0.1),
        grad_clip=recipe.get("grad_clip", 1.0),
        log_every=recipe.get("log_every", 10),
        save_every=recipe.get("save_every", 0),
        seed=recipe.get("seed"),
        quiet=recipe.get("quiet", False),
        optimizer_class=opt_class,
        use_abbicus=recipe.get("use_abbicus", True),
        compute_framework=compute_framework,
    )

    # 3) Optionally emit GGUFs.
    quants: list[str] = recipe.get("quants", []) or []
    if quants:
        from hypernix.quant.convert import convert_to_gguf
        from hypernix.quant.quantize import quantize_gguf

        gguf_dir = Path(recipe.get("gguf_out_dir") or Path(trained) / "gguf")
        gguf_dir.mkdir(parents=True, exist_ok=True)
        base = Path(trained).name
        produced: dict[str, Path] = {}
        # Always start with fp16 as the intermediate.
        fp16 = gguf_dir / f"{base}-fp16.gguf"
        convert_to_gguf(trained, fp16, dtype="fp16", arch_name="hypernix", name=base)
        produced["fp16"] = fp16
        for q in quants:
            if q in {"fp16", "f16"}:
                continue
            if q in {"fp32", "f32"}:
                out = gguf_dir / f"{base}-fp32.gguf"
                convert_to_gguf(trained, out, dtype="fp32", arch_name="hypernix",
                                name=base)
                produced["fp32"] = out
                continue
            out = gguf_dir / f"{base}-{q}.gguf"
            quantize_gguf(source_gguf=fp16, output_gguf=out, quant_type=q)
            produced[q] = out

    return Path(trained)
